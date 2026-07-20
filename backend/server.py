import os
import json
import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import LITE_MODEL, LOKA_STORAGE_DIR, OPENAI_LITE_MODEL
from src.workflows.project_setup import setup_project_workspace
from src.workflows.timeline_extraction import extract_timeline_from_project
from src.workflows.feedback_parsing import parse_and_align_feedback
from src.logging_utils import log_event

app = FastAPI(title="Video Project Timeline & Feedback UI")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_event(
            "http.error",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise

    log_event(
        "http.request",
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        user_agent=request.headers.get("user-agent", "")[:160],
    )
    return response

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

@app.post("/api/projects/{project_name}/feedback")
async def upload_project_feedback(project_name: str, file: UploadFile = File(...)):
    project_name = safe_project_name(project_name)
    filename = safe_upload_name(file.filename)
    
    # Check if timeline.json exists
    project_dir = project_data_dir(project_name)
    timeline_path = project_dir / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Timeline must be extracted before feedback can be parsed and aligned. Please import Premiere package first."
        )

    # Validate file extension
    ext = Path(filename).suffix.lower()
    if ext not in {".txt", ".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Feedback file must be a .txt, .csv, .xlsx, or .xls file")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set on the server.")

    from openai import OpenAI
    openai_client = OpenAI(api_key=openai_key)

    ensure_project_asset_tree(project_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / filename
        with temp_file_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        try:
            feedback_json_path = parse_and_align_feedback(
                feedback_file_path=str(temp_file_path),
                timeline_json_path=str(timeline_path),
                project_name=project_name,
                openai_client=openai_client,
                output_base_dir=str(DATA_DIR),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to parse and align feedback: {exc}") from exc

    return {"project_name": project_name, "feedback_path": feedback_json_path}

class ManualFeedbackRequest(BaseModel):
    clip_used: str
    category: str
    remark: str
    timestamp: Optional[str] = None


class ClipChatRequest(BaseModel):
    clip_index: int
    message: str
    provider: Literal["openai", "gemini"] = "openai"


class ClipMemoryRequest(BaseModel):
    clip_index: int
    text: str
    scope: Literal["clip", "project"] = "clip"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_chat_key(clip_name: str, clip_index: int) -> str:
    return f"{clip_name}::{clip_index}"


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chat_history_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "clip_chats.json"


def chat_memory_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "chat_memory.json"


def load_chat_history(project_name: str) -> dict:
    data = read_json_file(chat_history_path(project_name), {"clips": {}})
    if not isinstance(data, dict):
        return {"clips": {}}
    data.setdefault("clips", {})
    return data


def save_chat_history(project_name: str, history: dict) -> None:
    write_json_file(chat_history_path(project_name), history)


def load_chat_memory(project_name: str) -> dict:
    data = read_json_file(chat_memory_path(project_name), {"project": [], "clips": {}})
    if not isinstance(data, dict):
        return {"project": [], "clips": {}}
    data.setdefault("project", [])
    data.setdefault("clips", {})
    return data


def save_chat_memory(project_name: str, memory: dict) -> None:
    write_json_file(chat_memory_path(project_name), memory)


def make_chat_message(role: str, content: str, metadata: Optional[dict] = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "created_at": now_iso(),
        "metadata": metadata or {},
    }


def make_memory_item(text: str, *, scope: str, source: str = "chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "scope": scope,
        "text": text.strip(),
        "created_at": now_iso(),
        "source": source,
    }


def matching_prompt(project_data: dict, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip", "")
    basename = os.path.basename(clip_name)
    for prompt in project_data.get("prompts", []):
        prompt_clip = prompt.get("clip_used") or ""
        same_clip = prompt_clip == clip_name or os.path.basename(prompt_clip) == basename
        occurrence = prompt.get("clip_occurrence")
        if same_clip and (occurrence is None or occurrence == clip_index):
            return prompt
    return None


def matching_feedback(project_data: dict, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip", "")
    for group in project_data.get("feedback", []):
        occurrence = group.get("clip_occurrence")
        if group.get("clip_used") == clip_name and (occurrence is None or occurrence == clip_index):
            return group
    return None


def latest_prompt_version(prompt: Optional[dict]) -> Optional[dict]:
    if not prompt:
        return None
    history = prompt.get("history")
    if isinstance(history, list) and history:
        return history[-1]
    if prompt.get("video_model_prompt"):
        return prompt
    return None


def build_clip_chat_context(project_name: str, clip_index: int) -> dict:
    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    feedback = matching_feedback(project_data, clip, clip_index)
    prompt = matching_prompt(project_data, clip, clip_index)
    latest_version = latest_prompt_version(prompt)
    memory = load_chat_memory(project_name)
    key = clip_chat_key(clip.get("clip", ""), clip_index)

    return {
        "project": {
            "name": project_data.get("project_name", project_name),
            "sequence_name": project_data.get("sequence_name", ""),
            "total_duration_tc": project_data.get("total_duration_tc", ""),
            "total_duration_s": project_data.get("total_duration_s", 0),
            "clip_count": len(timeline),
        },
        "clip": clip,
        "clip_index": clip_index,
        "clip_key": key,
        "adjacent_clips": {
            "previous": timeline[clip_index - 1] if clip_index > 0 else None,
            "next": timeline[clip_index + 1] if clip_index < len(timeline) - 1 else None,
        },
        "feedback": feedback,
        "prompt": prompt,
        "latest_version": latest_version,
        "assets": project_data.get("assets", {}),
        "memory": {
            "project": memory.get("project", []),
            "clip": memory.get("clips", {}).get(key, []),
        },
    }


def compact_chat_context(context: dict) -> str:
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    selected_assets = latest_version.get("selected_assets") or []
    project_memory = [item.get("text", "") for item in context.get("memory", {}).get("project", [])][-8:]
    clip_memory = [item.get("text", "") for item in context.get("memory", {}).get("clip", [])][-12:]
    assets_by_category = {
        category: [asset.get("path") for asset in assets[:12]]
        for category, assets in (context.get("assets") or {}).items()
    }

    return json.dumps(
        {
            "project": context.get("project"),
            "clip": context.get("clip"),
            "clip_index": context.get("clip_index"),
            "adjacent_clips": context.get("adjacent_clips"),
            "feedback_items": feedback_items,
            "selected_assets": selected_assets,
            "latest_video_prompt": latest_version.get("video_model_prompt"),
            "latest_prompt_explanation": latest_version.get("explanation"),
            "quality_report": latest_version.get("quality_report"),
            "asset_library_sample": assets_by_category,
            "project_memory": project_memory,
            "clip_memory": clip_memory,
        },
        ensure_ascii=False,
        indent=2,
    )


def extract_memory_notes(message: str, assistant_text: str) -> list[dict]:
    notes = []
    for match in re.finditer(r"(?:remember|save to memory)(?: that)?\s*:?\s+(.+)", message, flags=re.IGNORECASE):
        note = match.group(1).strip()
        if note:
            notes.append(make_memory_item(note, scope="clip", source="user"))

    for line in assistant_text.splitlines():
        if line.lower().startswith("memory:"):
            note = line.split(":", 1)[1].strip()
            if note:
                notes.append(make_memory_item(note, scope="clip", source="assistant"))

    return notes


def infer_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    suggestions = []
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    if any(word in lower for word in ["workflow", "run", "execute"]) and feedback_items:
        suggestions.append({
            "type": "execute_workflow",
            "label": "Run Workflow",
            "feedback_index": feedback_items[0].get("raw_index"),
        })
    if any(word in lower for word in ["video", "generate", "seedance", "render"]) and latest_version.get("video_model_prompt"):
        suggestions.append({"type": "prepare_video", "label": "Prepare Video"})
    return suggestions


def fallback_chat_reply(message: str, context: dict) -> str:
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    selected_assets = latest_version.get("selected_assets") or []
    clip = context.get("clip") or {}
    project = context.get("project") or {}
    memory = context.get("memory") or {}
    lower = message.lower()

    if "memory" in lower:
        memory_count = len(memory.get("clip", [])) + len(memory.get("project", []))
        return (
            f"I found {memory_count} saved memory note(s) for this context. "
            "This build stores durable JSON memory now, and can later swap in Mem0, Letta, Graphiti/Zep, Cognee, or LangGraph/LangMem for semantic retrieval."
        )
    if "feedback" in lower:
        if not feedback_items:
            return "No feedback is attached to this clip yet."
        return "\n".join(f"#{item.get('raw_index')} {item.get('category')}: {item.get('remark')}" for item in feedback_items)
    if "asset" in lower:
        if selected_assets:
            return "Latest selected assets:\n" + "\n".join(f"- {os.path.basename(asset)}" for asset in selected_assets)
        return "No selected assets are attached to the latest plan yet. The project asset library is available in context."
    if "workflow" in lower or "run" in lower:
        if feedback_items:
            return f"The workflow can run against feedback index {feedback_items[0].get('raw_index')}. Use the suggested Run Workflow action."
        return "The existing workflow needs at least one feedback item for this clip."
    if "video" in lower or "generate" in lower:
        if latest_version.get("video_model_prompt"):
            return "A generated prompt is ready. Use Prepare Video to open the prompt, references, audio trim, and quality report."
        return "No generated prompt exists yet. Run the feedback workflow first, then prepare video generation."

    return (
        f"Clip #{context.get('clip_index', 0) + 1}/{project.get('clip_count')}: {clip.get('clip')}.\n"
        f"Timecode: {clip.get('start_tc')} to {clip.get('end_tc')} ({clip.get('duration_s', 0):.2f}s).\n"
        f"Feedback items: {len(feedback_items)}. Selected assets: {len(selected_assets)}. "
        f"Prompt ready: {'yes' if latest_version.get('video_model_prompt') else 'no'}."
    )


async def generate_chat_reply(provider: str, message: str, context: dict, messages: list[dict]) -> str:
    system_prompt = (
        "You are Loka15 Studio's clip assistant inside a video feedback and generation tool. "
        "Answer as a practical editor-facing collaborator. Use only the supplied context. "
        "You can discuss timeline, clip details, feedback, assets, prior prompt generations, quality reports, memory, workflow execution, and video-generation handoff. "
        "If the user asks to save a durable preference, include a final line starting with 'Memory:' followed by the exact note. "
        "Do not claim that you executed actions; the app will show action buttons separately."
    )
    context_text = compact_chat_context(context)
    history = [
        {"role": item.get("role"), "content": item.get("content")}
        for item in messages[-10:]
        if item.get("role") in {"user", "assistant"}
    ]

    try:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context JSON:\n{context_text}"},
                *history,
                {"role": "user", "content": message},
            ]
            response = client.responses.create(
                model=os.environ.get("OPENAI_CHAT_MODEL", OPENAI_LITE_MODEL),
                input=input_messages,
            )
            return response.output_text.strip()

        if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            from google import genai

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            prompt = (
                f"{system_prompt}\n\nContext JSON:\n{context_text}\n\n"
                f"Recent messages:\n{json.dumps(history, ensure_ascii=False)}\n\nUser: {message}"
            )
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_CHAT_MODEL", LITE_MODEL),
                contents=prompt,
            )
            return (response.text or "").strip()
    except Exception as exc:
        log_event(
            "clip_chat.llm_error",
            provider=provider,
            project=context.get("project", {}).get("name"),
            clip_index=context.get("clip_index"),
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )

    return fallback_chat_reply(message, context)

@app.post("/api/projects/{project_name}/feedback/item")
def add_manual_feedback(project_name: str, item: ManualFeedbackRequest):
    project_name = safe_project_name(project_name)
    project_dir = project_data_dir(project_name)
    feedback_path = project_dir / "feedback.json"
    
    # 1. Load existing feedback
    if feedback_path.exists():
        try:
            with feedback_path.open("r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load existing feedback: {str(e)}")
    else:
        feedback_data = []

    # 2. Add or find group for clip_used
    found_group = None
    for group in feedback_data:
        if group.get("clip_used") == item.clip_used:
            found_group = group
            break
            
    if not found_group:
        found_group = {
            "clip_used": item.clip_used,
            "previous_clip": None,
            "audio_used": None,
            "feedback_items": []
        }
        feedback_data.append(found_group)
        
    # 3. Add feedback item
    new_item = {
        "timestamp": item.timestamp,
        "category": item.category,
        "remark": item.remark
    }
    found_group["feedback_items"].append(new_item)
    
    # 4. Save feedback back
    try:
        with feedback_path.open("w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
        
    return {"status": "success", "project_name": project_name}

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
    handoff_fields = [
        "video_provider",
        "segmind_model",
        "segmind_payload_status",
        "segmind_payload",
        "segmind_prompt",
        "segmind_first_frame_url",
        "segmind_reference_images",
        "segmind_reference_videos",
        "segmind_reference_audios",
        "segmind_payload_error",
        "audio_reference_path",
        "trimmed_audio_path",
        "audio_trim_start_s",
        "audio_trim_end_s",
        "audio_trim_duration_s",
        "audio_trim_source",
        "audio_trim_error",
        "audio_used",
        "audio_path",
        "audio_url",
        "is_dialogue_active",
        "generate_audio",
        "ratio",
        "duration",
        "matched_clip",
        "clip_occurrence",
        "clip_start_tc",
        "clip_end_tc",
        "clip_start_s",
        "clip_end_s",
        "clip_duration_s",
    ]

    def copy_handoff_fields(target: dict, source: dict) -> None:
        for field in handoff_fields:
            if field in source:
                target[field] = source.get(field)

    def prompt_history_entry(item: dict, entry_provider: str) -> dict:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "provider": entry_provider,
            "video_model_prompt": item.get("video_model_prompt"),
            "selected_assets": item.get("selected_assets", []),
            "explanation": item.get("explanation"),
            "quality_report": item.get("quality_report"),
            "initial_frame_image_path": item.get("initial_frame_image_path"),
            "initial_frame_prompt": item.get("initial_frame_prompt"),
            "clip_frame_paths": item.get("clip_frame_paths", []),
        }
        copy_handoff_fields(entry, item)
        return entry
    
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
            same_clip = (
                clip_used == matched_clip
                or (clip_used and matched_clip and os.path.basename(clip_used) == os.path.basename(matched_clip))
            )
            existing_occurrence = p_item.get("clip_occurrence")
            generated_occurrence = gen_item.get("clip_occurrence")
            same_occurrence = (
                generated_occurrence is None
                or existing_occurrence is None
                or existing_occurrence == generated_occurrence
            )
            if same_clip and same_occurrence:
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
                        first_entry = prompt_history_entry(p_item, p_item.get("provider", "unknown"))
                        first_entry["timestamp"] = p_item.get("created_at", first_entry["timestamp"])
                        history.append(first_entry)
                        
                    new_entry = prompt_history_entry(gen_item, provider)
                    history.append(new_entry)
                    p_item["history"] = history
                    
                    p_item["video_model_prompt"] = gen_item.get("video_model_prompt")
                    p_item["selected_assets"] = gen_item.get("selected_assets", [])
                    p_item["explanation"] = gen_item.get("explanation")
                    p_item["status"] = "success"
                    p_item["quality_report"] = gen_item.get("quality_report")
                    p_item["initial_frame_image_path"] = gen_item.get("initial_frame_image_path")
                    p_item["initial_frame_prompt"] = gen_item.get("initial_frame_prompt")
                    p_item["clip_frame_paths"] = gen_item.get("clip_frame_paths", [])
                    copy_handoff_fields(p_item, gen_item)
                    updated = True
                break
        
        if not found:
            if is_error:
                prompts_data.append({
                    "clip_used": matched_clip,
                    "matched_clip": gen_item.get("matched_clip"),
                    "clip_occurrence": gen_item.get("clip_occurrence"),
                    "category": gen_item.get("category", "video"),
                    "generation_type": gen_item.get("prompt_format", "complex"),
                    "video_model_prompt": "",
                    "selected_assets": [],
                    "status": "failed",
                    "latest_error": prompt_text,
                    "history": []
                })
            else:
                new_entry = prompt_history_entry(gen_item, provider)
                prompt_record = {
                    "clip_used": matched_clip,
                    "matched_clip": gen_item.get("matched_clip"),
                    "clip_occurrence": gen_item.get("clip_occurrence"),
                    "category": gen_item.get("category", "video"),
                    "generation_type": gen_item.get("prompt_format", "complex"),
                    "video_model_prompt": gen_item.get("video_model_prompt"),
                    "selected_assets": gen_item.get("selected_assets", []),
                    "status": "success",
                    "explanation": gen_item.get("explanation"),
                    "quality_report": gen_item.get("quality_report"),
                    "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
                    "initial_frame_prompt": gen_item.get("initial_frame_prompt"),
                    "clip_frame_paths": gen_item.get("clip_frame_paths", []),
                    "history": [new_entry]
                }
                copy_handoff_fields(prompt_record, gen_item)
                prompts_data.append(prompt_record)
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


@app.get("/api/projects/{project_name}/chat/clip/{clip_index}")
def get_clip_chat(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    context = build_clip_chat_context(project_name, clip_index)
    history = load_chat_history(project_name)
    messages = history.get("clips", {}).get(context["clip_key"], [])

    if not messages:
        messages = [
            make_chat_message(
                "assistant",
                (
                    f"I am attached to clip #{clip_index + 1}. I can use timeline, feedback, assets, "
                    "saved memory, generated prompts, workflow execution, and video-generation handoff context."
                ),
                {"kind": "welcome"},
            )
        ]
        history.setdefault("clips", {})[context["clip_key"]] = messages
        save_chat_history(project_name, history)

    return {
        "project_name": project_name,
        "clip_index": clip_index,
        "clip_key": context["clip_key"],
        "messages": messages,
        "memory": context["memory"],
        "context": {
            "project": context["project"],
            "clip": context["clip"],
            "adjacent_clips": context["adjacent_clips"],
            "feedback_count": len((context.get("feedback") or {}).get("feedback_items", [])),
            "selected_asset_count": len((context.get("latest_version") or {}).get("selected_assets", []) or []),
            "prompt_ready": bool((context.get("latest_version") or {}).get("video_model_prompt")),
        },
    }


@app.post("/api/projects/{project_name}/chat/clip")
async def post_clip_chat(project_name: str, request: ClipChatRequest):
    project_name = safe_project_name(project_name)
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = build_clip_chat_context(project_name, request.clip_index)
    history = load_chat_history(project_name)
    clip_messages = history.setdefault("clips", {}).setdefault(context["clip_key"], [])

    user_message = make_chat_message("user", message_text)
    clip_messages.append(user_message)

    assistant_text = await generate_chat_reply(request.provider, message_text, context, clip_messages)
    saved_memories = extract_memory_notes(message_text, assistant_text)

    if saved_memories:
        memory = load_chat_memory(project_name)
        clip_memory = memory.setdefault("clips", {}).setdefault(context["clip_key"], [])
        existing = {item.get("text", "").strip().lower() for item in clip_memory}
        for item in saved_memories:
            if item["text"].strip().lower() not in existing:
                clip_memory.append(item)
                existing.add(item["text"].strip().lower())
        save_chat_memory(project_name, memory)
        context = build_clip_chat_context(project_name, request.clip_index)

    actions = infer_action_suggestions(message_text, context)
    assistant_message = make_chat_message(
        "assistant",
        assistant_text,
        {
            "provider": request.provider,
            "actions": actions,
            "saved_memory_ids": [item["id"] for item in saved_memories],
        },
    )
    clip_messages.append(assistant_message)
    save_chat_history(project_name, history)

    return {
        "project_name": project_name,
        "clip_index": request.clip_index,
        "clip_key": context["clip_key"],
        "messages": clip_messages,
        "assistant_message": assistant_message,
        "memory": context["memory"],
        "suggested_actions": actions,
    }


@app.post("/api/projects/{project_name}/chat/memory")
def add_clip_memory(project_name: str, request: ClipMemoryRequest):
    project_name = safe_project_name(project_name)
    context = build_clip_chat_context(project_name, request.clip_index)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Memory text cannot be empty")

    memory = load_chat_memory(project_name)
    item = make_memory_item(text, scope=request.scope, source="manual")
    if request.scope == "project":
        memory.setdefault("project", []).append(item)
    else:
        memory.setdefault("clips", {}).setdefault(context["clip_key"], []).append(item)
    save_chat_memory(project_name, memory)
    updated_context = build_clip_chat_context(project_name, request.clip_index)
    return {"memory": updated_context["memory"], "item": item}

@app.get("/api/run-workflow")
async def run_workflow(project: str, index: int, provider: str = "openai"):
    safe_project_name(project)
    log_event("workflow.request", project=project, index=index, provider=provider)
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
        workflow_started = time.perf_counter()
        log_event("workflow.subprocess.start", project=project, index=index, provider=provider, command=cmd)
        
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
                log_event(
                    "workflow.subprocess.finish",
                    project=project,
                    index=index,
                    provider=provider,
                    status="success",
                    exit_code=rc,
                    duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                )
                yield f"\ndata: [SUCCESS] Workflow execution finished with exit code {rc}\n\n"
            else:
                log_event(
                    "workflow.subprocess.finish",
                    project=project,
                    index=index,
                    provider=provider,
                    status="failed",
                    exit_code=rc,
                    duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                )
                yield f"\ndata: [ERROR] Workflow execution failed with exit code {rc}\n\n"
            
        except Exception as e:
            log_event(
                "workflow.subprocess.error",
                project=project,
                index=index,
                provider=provider,
                duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                error_type=type(e).__name__,
                error=str(e)[:500],
            )
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
