import os
import json
import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import LOKA_STORAGE_DIR
from src.workflows.project_setup import setup_project_workspace
from src.workflows.timeline_extraction import extract_timeline_from_project

app = FastAPI(title="Video Project Timeline & Feedback UI")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up application storage directories
BACKEND_DIR = Path(__file__).resolve().parent
APP_STORAGE_DIR = Path(os.environ.get("LOKA_STORAGE_DIR", LOKA_STORAGE_DIR)).expanduser().resolve()

DATA_DIR = APP_STORAGE_DIR / "data"
ASSETS_DIR = APP_STORAGE_DIR / "assets"
# STATIC_DIR = BACKEND_DIR / "static"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
# STATIC_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ASSET_DIRS = {
    "style": "00_style",
    "characters": "01_characters",
    "props": "02_props",
    "locations": "03_locations",
    "audio": "04_audio",
    "references": "05_references",
    "clips": "06_clips/_final",
}

REQUIRED_PROJECT_DIRS = [
    "00_style",
    "01_characters",
    "02_props",
    "03_locations",
    "04_audio",
    "05_references",
    "06_clips/_final",
    "06_clips/_raw",
]

def project_data_dir(project_name: str) -> Path:
    return DATA_DIR / safe_project_name(project_name)

def project_assets_dir(project_name: str) -> Path:
    return ASSETS_DIR / safe_project_name(project_name)

def safe_project_name(project_name: str) -> str:
    if not project_name or project_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid project name")
    if Path(project_name).name != project_name:
        raise HTTPException(status_code=400, detail="Invalid project name")
    return project_name

def safe_upload_name(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")
    safe_name = Path(filename).name
    if safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid uploaded filename")
    return safe_name

def ensure_project_asset_tree(project_name: str) -> Path:
    project_dir = project_assets_dir(project_name)
    for rel_dir in REQUIRED_PROJECT_DIRS:
        (project_dir / rel_dir).mkdir(parents=True, exist_ok=True)
    project_data_dir(project_name).mkdir(parents=True, exist_ok=True)
    return project_dir

# Helper function to format file sizes
def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

# Scan assets folder to group files by subdirectories
def get_assets_list(assets_dir: str | Path, project_name: str) -> dict:
    assets_path = Path(assets_dir)
    if not assets_path.exists():
        return {}
        
    categories = {}
    for root, _, files in os.walk(assets_path):
        root_path = Path(root)
        # Calculate relative path from the assets root
        rel_dir = os.path.relpath(root_path, assets_path)
        if rel_dir == ".":
            continue
            
        # Group files under their respective top-level folders (e.g. 01_characters)
        parts = rel_dir.split(os.sep)
        top_category = parts[0]
        
        if top_category not in categories:
            categories[top_category] = []
            
        for file in files:
            if file.startswith("."):
                continue
                
            file_path = root_path / file
            size = file_path.stat().st_size
            ext = file_path.suffix.lower()
            
            # Determine file type
            if ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                file_type = "image"
            elif ext in [".mp4", ".mov", ".mkv", ".webm"]:
                file_type = "video"
            elif ext in [".mp3", ".wav", ".m4a", ".aac"]:
                file_type = "audio"
            else:
                file_type = "other"
                
            rel_to_assets_root = file_path.relative_to(ASSETS_DIR)
            url = f"/assets/{quote(rel_to_assets_root.as_posix())}"
            
            categories[top_category].append({
                "name": file,
                "path": os.path.relpath(file_path, assets_path),
                "url": url,
                "type": file_type,
                "size": format_size(size),
            })
            
    # Sort files alphabetically inside each category
    for cat in categories:
        categories[cat] = sorted(categories[cat], key=lambda x: x["name"])
        
    return categories

# Dynamic creation of raw_feedback_temp.json from aligned feedback
def ensure_raw_feedback(project_name: str) -> str:
    project_dir = project_data_dir(project_name)
    aligned_path = project_dir / "feedback.json"
    raw_path = project_dir / "raw_feedback_temp.json"
    
    if not aligned_path.exists():
        raise HTTPException(status_code=404, detail=f"Feedback JSON not found for project {project_name}")
        
    with aligned_path.open("r", encoding="utf-8") as f:
        aligned_data = json.load(f)
        
    raw_items = []
    for group in aligned_data:
        for item in group.get("feedback_items", []):
            raw_items.append({
                "timestamp": item.get("timestamp"),
                "category": item.get("category", "video"),
                "remark": item.get("remark")
            })
            
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_items, f, indent=2, ensure_ascii=False)
        
    return str(raw_path)

@app.get("/api/projects")
def list_projects():
    projects = []
    if DATA_DIR.exists():
        for item_path in DATA_DIR.iterdir():
            if item_path.is_dir():
                timeline_path = item_path / "timeline.json"
                if timeline_path.exists():
                    projects.append(item_path.name)
    return sorted(projects)

@app.post("/api/projects")
def create_project(project_name: str = Form(...)):
    project_name = safe_project_name(project_name.strip())
    timeline_path = project_data_dir(project_name) / "timeline.json"
    if timeline_path.exists():
        raise HTTPException(status_code=409, detail=f"Project {project_name} already exists")

    project_dir = ensure_project_asset_tree(project_name)
    return {
        "project_name": project_name,
        "assets_dir": str(project_dir),
        "required_directories": REQUIRED_PROJECT_DIRS,
    }

@app.post("/api/projects/{project_name}/assets/{category}")
async def upload_project_assets(project_name: str, category: str, files: list[UploadFile] = File(...)):
    project_name = safe_project_name(project_name)
    rel_dir = PROJECT_ASSET_DIRS.get(category)
    if rel_dir is None:
        valid = ", ".join(sorted(PROJECT_ASSET_DIRS))
        raise HTTPException(status_code=400, detail=f"Invalid asset category. Use one of: {valid}")

    ensure_project_asset_tree(project_name)
    target_dir = project_assets_dir(project_name) / rel_dir
    saved_files = []
    for upload in files:
        filename = safe_upload_name(upload.filename)
        destination = target_dir / filename
        with destination.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        saved_files.append(str(destination.relative_to(project_assets_dir(project_name))))

    return {"project_name": project_name, "category": category, "files": saved_files}

@app.post("/api/projects/{project_name}/premiere-package")
async def import_premiere_package(project_name: str, package: UploadFile = File(...)):
    project_name = safe_project_name(project_name)
    filename = safe_upload_name(package.filename)
    if Path(filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Premiere package must be a .zip file")

    ensure_project_asset_tree(project_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        package_path = Path(temp_dir) / filename
        with package_path.open("wb") as out_file:
            shutil.copyfileobj(package.file, out_file)

        try:
            _project_dir, prproj_path = setup_project_workspace(
                zip_path=str(package_path),
                project_name=project_name,
                assets_dir=str(ASSETS_DIR),
            )
            timeline_path = extract_timeline_from_project(
                prproj_path=prproj_path,
                project_name=project_name,
                output_base_dir=str(DATA_DIR),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to import Premiere package: {exc}") from exc

    return {"project_name": project_name, "timeline_path": timeline_path}

def find_clip_url(clip_name: str, assets_dir: str | Path) -> Optional[str]:
    clips_dir = Path(assets_dir) / "06_clips"
    if not clips_dir.exists():
        return None
    for root, _, files in os.walk(clips_dir):
        if clip_name in files:
            full_path = Path(root) / clip_name
            rel_path = full_path.relative_to(ASSETS_DIR)
            return f"/assets/{quote(rel_path.as_posix())}"
    return None

def save_output_to_prompts(project: str, provider: str = "unknown"):
    import datetime
    project_dir = project_data_dir(project)
    output_json = project_dir / "output.json"
    prompts_json = project_dir / "video_prompts.json"
    
    if not output_json.exists():
        return
        
    try:
        with output_json.open("r", encoding="utf-8") as f:
            output_data = json.load(f)
    except Exception as e:
        print(f"Error loading output.json: {e}")
        return
        
    if not isinstance(output_data, list) or len(output_data) == 0:
        return
        
    # Load prompts
    prompts_data = []
    if prompts_json.exists():
        try:
            with prompts_json.open("r", encoding="utf-8") as f:
                prompts_data = json.load(f)
        except Exception as e:
            print(f"Error loading video_prompts.json: {e}")
            
    # Update matching clips
    updated = False
    for gen_item in output_data:
        matched_clip = gen_item.get("matched_clip")
        if not matched_clip:
            continue
            
        prompt_text = gen_item.get("video_model_prompt", "")
        prompt_text_lower = prompt_text.lower()
        error_phrases = ["quota expired", "model usage in spike", "rate limit", "error:", "api error", "limit exceeded", "overloaded"]
        is_error = any(phrase in prompt_text_lower for phrase in error_phrases)
            
        found = False
        for p_item in prompts_data:
            clip_used = p_item.get("clip_used")
            if clip_used == matched_clip or (clip_used and matched_clip and os.path.basename(clip_used) == os.path.basename(matched_clip)):
                found = True
                if is_error:
                    p_item["latest_error"] = prompt_text
                    updated = True
                else:
                    p_item["latest_error"] = None
                    history = p_item.get("history", [])
                    if not isinstance(history, list):
                        history = []
                        
                    if len(history) == 0 and p_item.get("video_model_prompt"):
                        history.append({
                            "timestamp": p_item.get("created_at", datetime.datetime.now().isoformat()),
                            "provider": p_item.get("provider", "unknown"),
                            "video_model_prompt": p_item.get("video_model_prompt"),
                            "selected_assets": p_item.get("selected_assets", []),
                            "explanation": p_item.get("explanation"),
                            "quality_report": p_item.get("quality_report"),
                            "initial_frame_image_path": p_item.get("initial_frame_image_path"),
                            "initial_frame_prompt": p_item.get("initial_frame_prompt")
                        })
                        
                    new_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "provider": provider,
                        "video_model_prompt": gen_item.get("video_model_prompt"),
                        "selected_assets": gen_item.get("selected_assets", []),
                        "explanation": gen_item.get("explanation"),
                        "quality_report": gen_item.get("quality_report"),
                        "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
                        "initial_frame_prompt": gen_item.get("initial_frame_prompt")
                    }
                    history.append(new_entry)
                    p_item["history"] = history
                    
                    p_item["video_model_prompt"] = gen_item.get("video_model_prompt")
                    p_item["selected_assets"] = gen_item.get("selected_assets", [])
                    p_item["explanation"] = gen_item.get("explanation")
                    p_item["status"] = "success"
                    p_item["quality_report"] = gen_item.get("quality_report")
                    p_item["initial_frame_image_path"] = gen_item.get("initial_frame_image_path")
                    p_item["initial_frame_prompt"] = gen_item.get("initial_frame_prompt")
                    updated = True
                break
        
        if not found:
            if is_error:
                prompts_data.append({
                    "clip_used": matched_clip,
                    "category": gen_item.get("category", "video"),
                    "generation_type": gen_item.get("prompt_format", "complex"),
                    "video_model_prompt": "",
                    "selected_assets": [],
                    "status": "failed",
                    "latest_error": prompt_text,
                    "history": []
                })
            else:
                new_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "provider": provider,
                    "video_model_prompt": gen_item.get("video_model_prompt"),
                    "selected_assets": gen_item.get("selected_assets", []),
                    "explanation": gen_item.get("explanation"),
                    "quality_report": gen_item.get("quality_report"),
                    "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
                    "initial_frame_prompt": gen_item.get("initial_frame_prompt")
                }
                prompts_data.append({
                    "clip_used": matched_clip,
                    "category": gen_item.get("category", "video"),
                    "generation_type": gen_item.get("prompt_format", "complex"),
                    "video_model_prompt": gen_item.get("video_model_prompt"),
                    "selected_assets": gen_item.get("selected_assets", []),
                    "status": "success",
                    "explanation": gen_item.get("explanation"),
                    "quality_report": gen_item.get("quality_report"),
                    "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
                    "initial_frame_prompt": gen_item.get("initial_frame_prompt"),
                    "history": [new_entry]
                })
            updated = True
            
    if updated:
        try:
            with prompts_json.open("w", encoding="utf-8") as f:
                json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving video_prompts.json: {e}")

@app.get("/api/project/{project_name}")
def get_project_data(project_name: str):
    project_dir = project_data_dir(project_name)
    timeline_path = project_dir / "timeline.json"
    feedback_path = project_dir / "feedback.json"
    
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Project timeline not found")
        
    with timeline_path.open("r", encoding="utf-8") as f:
        timeline_data = json.load(f)
        
    if feedback_path.exists():
        with feedback_path.open("r", encoding="utf-8") as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
        
    # Map assets directory
    assets_dir = project_assets_dir(project_name)
        
    assets = get_assets_list(assets_dir, project_name)
    
    # Map timeline clips to include static URLs if they exist in 06_clips
    video_timeline = timeline_data.get("video_timeline", [])
    timeline_with_urls = []
    for clip in video_timeline:
        clip_url = find_clip_url(clip.get("clip", ""), assets_dir)
        timeline_with_urls.append({
            **clip,
            "clip_url": clip_url
        })
        
    # We assign unique sequential raw indexes to aligned feedback items 
    # to facilitate the execution of individual feedback items via main.py --index
    raw_index_counter = 0
    aligned_feedback_with_indexes = []
    
    for group in feedback_data:
        indexed_items = []
        for item in group.get("feedback_items", []):
            indexed_items.append({
                **item,
                "raw_index": raw_index_counter
            })
            raw_index_counter += 1
            
        aligned_feedback_with_indexes.append({
            **group,
            "feedback_items": indexed_items
        })
        
    # Load prompts list if it exists
    prompts_json = project_dir / "video_prompts.json"
    prompts_data = []
    if prompts_json.exists():
        try:
            with prompts_json.open("r", encoding="utf-8") as f:
                prompts_data = json.load(f)
        except Exception as e:
            print(f"Error loading prompts data: {e}")

    return {
        "project_name": project_name,
        "timeline": timeline_with_urls,
        "feedback": aligned_feedback_with_indexes,
        "assets": assets,
        "prompts": prompts_data,
        "summary": timeline_data.get("summary", {}),
        "sequence_name": timeline_data.get("sequence_name", ""),
        "total_duration_tc": timeline_data.get("total_duration_tc", ""),
        "total_duration_s": timeline_data.get("total_duration_s", 0)
    }

@app.get("/api/run-workflow")
async def run_workflow(project: str, index: int, provider: str = "gemini"):
    safe_project_name(project)
    # Prepare inputs
    try:
        raw_feedback_path = ensure_raw_feedback(project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate raw feedback mapping: {str(e)}")
        
    assets_dir = project_assets_dir(project)
    if not assets_dir.exists():
        raise HTTPException(status_code=404, detail=f"Assets not found for project {project}")
        
    project_dir = project_data_dir(project)
    timeline_path = project_dir / "timeline.json"
    output_json = project_dir / "output.json"
    output_report = project_dir / "output_report.md"
    
    cmd = [
        sys.executable,
        "main.py",
        "--index", str(index),
        "--feedback-path", str(raw_feedback_path),
        "--timeline-path", str(timeline_path),
        "--assets-dir", str(assets_dir),
        "--output-json", str(output_json),
        "--output-report", str(output_report),
        "--provider", provider
    ]
    
    async def log_generator():
        yield f"data: [START] Launching workflow subprocess for feedback index {index}...\n\n"
        yield f"data: Executing command: {' '.join(cmd)}\n\n\n"
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BACKEND_DIR)
            )
            
            # Helper to read streams concurrently
            async def read_stream(stream, prefix=""):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    yield f"data: {prefix}{decoded}\n\n"
            
            # Read stdout and stderr
            async for out_line in read_stream(process.stdout):
                yield out_line
                
            async for err_line in read_stream(process.stderr, "[STDERR] "):
                yield err_line
                
            rc = await process.wait()
            if rc == 0:
                save_output_to_prompts(project, provider)
            yield f"\ndata: [SUCCESS] Workflow execution finished with exit code {rc}\n\n"
            
        except Exception as e:
            yield f"data: [ERROR] Failed to run subprocess: {str(e)}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/workflow-result")
def get_workflow_result(project: str):
    output_json = project_data_dir(project) / "output.json"
    if not output_json.exists():
        raise HTTPException(status_code=404, detail="No workflow result found. Run a feedback workflow first.")
    with output_json.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse result JSON: {str(e)}")

# Mount Static Assets and Data folders
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

#* Serve the UI at root path - No need because the frontend is served via tauri react
# @app.get("/")
# async def serve_ui():
#     return FileResponse(STATIC_DIR / "index.html")

# Mount frontend files under /static
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
