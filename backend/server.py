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
from urllib.parse import quote, unquote
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import LITE_MODEL, LOKA_STORAGE_DIR, OPENAI_LITE_MODEL
from src.workflows.project_setup import setup_project_workspace
from src.workflows.timeline_extraction import extract_timeline_from_project
from src.workflows.feedback_parsing import parse_and_align_feedback
from src.workflows.prompt_generation import extract_last_frame, get_video_duration
from src.workflows.clip_context import analyze_clip_context, clip_context_dir
from src.logging_utils import log_event
from src.chat_memory import ChatMemoryStore

APP_VERSION = os.environ.get("LOKA_APP_VERSION", "0.1.0")
BACKEND_STARTED_AT = datetime.now(timezone.utc)

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "loka15-backend",
        "version": APP_VERSION,
        "started_at": BACKEND_STARTED_AT.isoformat(),
    }


@app.get("/version")
def version():
    return {
        "service": "loka15-backend",
        "version": APP_VERSION,
        "python": sys.version.split()[0],
    }


@app.post("/shutdown")
async def shutdown(request: Request, background_tasks: BackgroundTasks):
    expected_token = os.environ.get("LOKA_BACKEND_SHUTDOWN_TOKEN")
    supplied_token = request.headers.get("x-loka-shutdown-token")
    if not expected_token or supplied_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid shutdown token")

    def stop_process():
        time.sleep(0.2)
        os._exit(0)

    background_tasks.add_task(stop_process)
    return {"status": "shutting_down"}

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
    for group_index, group in enumerate(aligned_data):
        group_id = f"{group.get('clip_used') or 'unmatched'}::{group.get('clip_occurrence', group_index)}"
        group_start = len(raw_items)
        for item in group.get("feedback_items", []):
            raw_items.append({
                "timestamp": item.get("timestamp"),
                "category": item.get("category", "video"),
                "remark": item.get("remark"),
                "group_id": group_id,
                "clip_used": group.get("clip_used"),
                "clip_occurrence": group.get("clip_occurrence"),
            })

        group_indexes = list(range(group_start, len(raw_items)))
        for raw_index in group_indexes:
            raw_items[raw_index]["sibling_raw_indexes"] = [
                other_index for other_index in group_indexes if other_index != raw_index
            ]
            
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


def chat_workflow_intents_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "chat_workflow_intents.json"


def load_chat_history(project_name: str) -> dict:
    data = read_json_file(chat_history_path(project_name), {"clips": {}})
    if not isinstance(data, dict):
        return {"clips": {}}
    data.setdefault("clips", {})
    return data


def save_chat_history(project_name: str, history: dict) -> None:
    write_json_file(chat_history_path(project_name), history)


def make_chat_message(role: str, content: str, metadata: Optional[dict] = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "created_at": now_iso(),
        "metadata": metadata or {},
    }


def static_url_for_path(path: str | None) -> Optional[str]:
    if not path:
        return None
    if re.match(r"^https?://", path):
        return path

    normalized = unquote(str(path).replace("\\", "/"))
    abs_path = Path(normalized).expanduser()
    try:
        if abs_path.is_absolute():
            resolved = abs_path.resolve()
            if resolved.is_relative_to(ASSETS_DIR):
                rel = resolved.relative_to(ASSETS_DIR)
                return f"/assets/{quote(rel.as_posix())}"
            if resolved.is_relative_to(DATA_DIR):
                rel = resolved.relative_to(DATA_DIR)
                return f"/data/{quote(rel.as_posix())}"
    except Exception:
        pass

    for marker, prefix in (("/assets/", "/assets/"), ("assets/", "/assets/"), ("/data/", "/data/"), ("data/", "/data/")):
        marker_index = normalized.find(marker)
        if marker_index != -1:
            clean = normalized[marker_index + (1 if marker.startswith("/") else 0):]
            if clean.startswith("assets/"):
                return f"/assets/{quote(clean.removeprefix('assets/'))}"
            if clean.startswith("data/"):
                return f"/data/{quote(clean.removeprefix('data/'))}"
            return f"{prefix}{quote(clean.removeprefix(prefix.lstrip('/')))}"
    return None


def preview_path_for_media(path: str | None) -> str | None:
    if not path:
        return None
    if re.match(r"^https?://", path):
        return path

    normalized = unquote(str(path).replace("\\", "/"))
    abs_path = Path(normalized).expanduser()
    try:
        if abs_path.is_absolute():
            resolved = abs_path.resolve()
            if resolved.is_relative_to(ASSETS_DIR):
                rel_to_assets = resolved.relative_to(ASSETS_DIR)
                parts = rel_to_assets.parts
                if len(parts) > 1:
                    return Path(*parts[1:]).as_posix()
                return rel_to_assets.as_posix()
            if resolved.is_relative_to(DATA_DIR):
                rel_to_data = resolved.relative_to(DATA_DIR)
                return f"/data/{quote(rel_to_data.as_posix())}"
    except Exception:
        pass

    if normalized.startswith("/assets/"):
        parts = normalized.removeprefix("/assets/").split("/", 1)
        return parts[1] if len(parts) == 2 else parts[0]
    if normalized.startswith("assets/"):
        parts = normalized.removeprefix("assets/").split("/", 1)
        return parts[1] if len(parts) == 2 else parts[0]
    if normalized.startswith("/data/") or normalized.startswith("data/"):
        return static_url_for_path(normalized)
    return normalized


def media_size_for_path(path: str | None) -> str:
    if not path or re.match(r"^https?://", path):
        return ""
    normalized = unquote(str(path).replace("\\", "/"))
    try:
        candidate = Path(normalized).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return format_size(candidate.stat().st_size)
        if normalized.startswith("/assets/"):
            rel = normalized.removeprefix("/assets/")
            parts = rel.split("/", 1)
            if len(parts) == 2:
                candidate = ASSETS_DIR / parts[0] / parts[1]
                if candidate.exists():
                    return format_size(candidate.stat().st_size)
        if normalized.startswith("assets/"):
            rel = normalized.removeprefix("assets/")
            parts = rel.split("/", 1)
            if len(parts) == 2:
                candidate = ASSETS_DIR / parts[0] / parts[1]
                if candidate.exists():
                    return format_size(candidate.stat().st_size)
    except Exception:
        return ""
    return ""


def infer_media_type(path_or_url: str, fallback: str = "other") -> str:
    lower = path_or_url.lower().split("?", 1)[0]
    ext = Path(lower).suffix
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if ext in {".mp4", ".mov", ".mkv", ".webm"}:
        return "video"
    if ext in {".mp3", ".wav", ".m4a", ".aac"}:
        return "audio"
    return fallback


def add_chat_media_item(
    media: list[dict],
    *,
    label: str,
    source: str,
    path: str | None,
    media_type: str | None = None,
    thumbnail_path: str | None = None,
) -> None:
    url = static_url_for_path(path)
    if not url:
        return
    if any(item.get("url") == url for item in media):
        return
    preview_path = preview_path_for_media(path)
    filename = os.path.basename(unquote(str(preview_path or path).split("?", 1)[0])) or label
    media.append({
        "type": media_type or infer_media_type(str(path)),
        "label": label,
        "source": source,
        "name": filename,
        "path": preview_path,
        "url": url,
        "size": media_size_for_path(path),
        "thumbnail_url": static_url_for_path(thumbnail_path),
    })


def wants_clip_media_gallery(text: str) -> bool:
    lower = text.lower()
    show_words = {"show", "display", "list", "view", "see", "open"}
    media_words = {"asset", "assets", "reference", "references", "image", "images", "video", "videos", "frames", "media"}
    return any(word in lower for word in show_words) and any(word in lower for word in media_words)


def build_clip_media_gallery(context: dict) -> list[dict]:
    media: list[dict] = []
    latest_version = context.get("latest_version") or {}
    feedback = context.get("feedback") or {}
    clip_frame_paths = latest_version.get("clip_frame_paths") or []

    for asset_path in latest_version.get("selected_assets") or []:
        add_chat_media_item(media, label="Selected asset", source="selected_assets", path=asset_path)

    initial_frame = latest_version.get("initial_frame_image_path")
    if initial_frame:
        add_chat_media_item(media, label="Initial frame", source="initial_frame", path=initial_frame, media_type="image")

    for frame_path in clip_frame_paths:
        add_chat_media_item(media, label="Extracted clip frame", source="clip_frames", path=frame_path, media_type="image")

    if latest_version.get("clip_segment_path"):
        add_chat_media_item(
            media,
            label="Analyzed clip segment",
            source="clip_context",
            path=latest_version.get("clip_segment_path"),
            media_type="video",
        )

    referenced_frames = list(latest_version.get("referenced_frames") or [])
    for item in feedback.get("feedback_items", []):
        referenced_frames.extend(item.get("referenced_frames") or [])
    referenced_frames.extend(feedback.get("referenced_frames") or [])
    for ref_frame in referenced_frames:
        label = "Referenced frame"
        if ref_frame.get("timestamp"):
            label = f"Referenced frame {ref_frame.get('timestamp')}"
        add_chat_media_item(
            media,
            label=label,
            source="referenced_frames",
            path=ref_frame.get("frame_path"),
            media_type="image",
        )

    return media


def memory_store(project_name: str) -> ChatMemoryStore:
    return ChatMemoryStore(chat_memory_path(project_name))


def matching_prompt(project_data: dict, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip", "")
    basename = os.path.basename(clip_name)
    same_clip_prompts = []
    for prompt in project_data.get("prompts", []):
        prompt_clip = prompt.get("clip_used") or ""
        same_clip = prompt_clip == clip_name or os.path.basename(prompt_clip) == basename
        if same_clip:
            same_clip_prompts.append(prompt)

    for prompt in same_clip_prompts:
        if prompt.get("clip_occurrence") == clip_index:
            return prompt
    for prompt in same_clip_prompts:
        if prompt.get("clip_occurrence") is None:
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


def load_clip_context(project_name: str, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip")
    if not clip_name:
        return None
    context_path = Path(clip_context_dir(str(DATA_DIR), project_name, clip_name, clip_index)) / "clip_context.json"
    if not context_path.exists():
        return None
    try:
        with context_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"Warning: Failed to load clip context {context_path}: {exc}")
    return None


def wants_clip_summary_or_analysis(text: str) -> bool:
    lower = text.lower()
    intent_words = {"summarize", "summary", "describe", "understand", "analysis", "analyze", "what happens", "what is happening"}
    clip_words = {"clip", "shot", "scene", "video"}
    return any(word in lower for word in intent_words) and any(word in lower for word in clip_words)


def summarize_clip_context(clip_context: dict) -> str:
    if not clip_context:
        return "This clip has not been analyzed yet."
    lines = []
    summary = clip_context.get("summary")
    if summary:
        lines.append(summary)
    for label, key in [
        ("Visible characters", "visible_characters"),
        ("Expressions", "expressions"),
        ("Gaze", "gaze"),
        ("Actions", "actions"),
        ("Blocking", "blocking"),
        ("Continuity notes", "continuity_notes"),
        ("Uncertainty", "uncertainty_flags"),
    ]:
        values = clip_context.get(key) or []
        if values:
            lines.append(f"{label}: " + "; ".join(str(value) for value in values))
    if clip_context.get("camera_framing"):
        lines.append(f"Camera/framing: {clip_context.get('camera_framing')}")
    if clip_context.get("location"):
        lines.append(f"Location: {clip_context.get('location')}")
    return "\n".join(lines) if lines else "The saved clip context has no summary details yet."


def ensure_clip_context_for_chat(project_name: str, context: dict, provider: str) -> tuple[Optional[dict], Optional[str]]:
    existing = context.get("clip_context")
    if existing:
        return existing, None
    if provider != "openai":
        return None, "Clip analysis on demand currently requires OpenAI."
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "Clip analysis has not been run yet, and OPENAI_API_KEY is not configured."

    from openai import OpenAI

    clip = context.get("clip") or {}
    clip_context = analyze_clip_context(
        project_name=project_name,
        clip_name=clip.get("clip"),
        clip_occurrence=context.get("clip_index"),
        clip_start_s=clip.get("start_s"),
        clip_end_s=clip.get("end_s"),
        clip_duration_s=clip.get("duration_s"),
        assets_dir=str(ASSETS_DIR),
        output_base_dir=str(DATA_DIR),
        client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        provider="openai",
        feedback_items=(context.get("feedback") or {}).get("feedback_items", []),
    )
    context["clip_context"] = clip_context
    return clip_context, None


def build_clip_chat_context(project_name: str, clip_index: int, query: str = "") -> dict:
    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    feedback = matching_feedback(project_data, clip, clip_index)
    prompt = matching_prompt(project_data, clip, clip_index)
    latest_version = latest_prompt_version(prompt)
    clip_context = load_clip_context(project_name, clip, clip_index)
    store = memory_store(project_name)
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
        "clip_context": clip_context,
        "assets": project_data.get("assets", {}),
        "memory": store.for_clip(key, query=query),
    }


def compact_chat_context(context: dict) -> str:
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    selected_assets = latest_version.get("selected_assets") or []
    project_memory = [item.get("text", "") for item in context.get("memory", {}).get("project", [])][-8:]
    clip_memory = [item.get("text", "") for item in context.get("memory", {}).get("clip", [])][-12:]
    relevant_memory = [
        {
            "text": item.get("text", ""),
            "scope": item.get("scope"),
            "relevance_score": item.get("relevance_score"),
            "relevance_reasons": item.get("relevance_reasons", []),
        }
        for item in context.get("memory", {}).get("relevant", [])
    ]
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
            "clip_context": context.get("clip_context"),
            "quality_report": latest_version.get("quality_report"),
            "asset_library_sample": assets_by_category,
            "project_memory": project_memory,
            "clip_memory": clip_memory,
            "relevant_memory": relevant_memory,
        },
        ensure_ascii=False,
        indent=2,
    )


def extract_memory_notes(message: str, assistant_text: str) -> list[str]:
    notes = []
    for match in re.finditer(r"(?:remember|save to memory)(?: that)?\s*:?\s+(.+)", message, flags=re.IGNORECASE):
        note = match.group(1).strip()
        if note:
            notes.append(note)

    for line in assistant_text.splitlines():
        if line.lower().startswith("memory:"):
            note = line.split(":", 1)[1].strip()
            if note:
                notes.append(note)

    return notes


def wants_previous_last_frame_continuity(text: str) -> bool:
    lower = text.lower()
    has_previous_clip = "previous clip" in lower or "prior clip" in lower or "last clip" in lower
    has_last_frame = "last frame" in lower or "final frame" in lower or "ending frame" in lower
    has_continuity = "continuity" in lower or "reference" in lower or "match" in lower
    return has_previous_clip and has_last_frame and has_continuity


def wants_autonomous_execution(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in ["execute", "run", "start"]) and any(
        word in lower for word in ["workflow", "pipeline", "generation"]
    )


def load_chat_workflow_intents(project_name: str) -> dict:
    data = read_json_file(chat_workflow_intents_path(project_name), {"feedback": {}})
    if not isinstance(data, dict):
        return {"feedback": {}}
    data.setdefault("feedback", {})
    return data


def save_chat_workflow_intents(project_name: str, intents: dict) -> None:
    write_json_file(chat_workflow_intents_path(project_name), intents)


def save_pending_workflow_intents(project_name: str, actions: list[dict], message: str, context: dict) -> None:
    workflow_actions = [
        action for action in actions
        if action.get("type") == "execute_workflow" and action.get("feedback_index") is not None
    ]
    if not workflow_actions:
        return

    intents = load_chat_workflow_intents(project_name)
    feedback_intents = intents.setdefault("feedback", {})
    for action in workflow_actions:
        feedback_index = str(action["feedback_index"])
        feedback_intents[feedback_index] = {
            "created_at": now_iso(),
            "clip_index": context.get("clip_index"),
            "clip_key": context.get("clip_key"),
            "user_message": message,
            "continuity_reference": action.get("continuity_reference"),
            "autonomous": bool(action.get("autonomous")),
            "status": "pending",
        }
    save_chat_workflow_intents(project_name, intents)


def infer_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    suggestions = []
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    if any(word in lower for word in ["workflow", "run", "execute"]) and feedback_items:
        action = {
            "type": "execute_workflow",
            "label": "Run Workflow",
            "feedback_index": feedback_items[0].get("raw_index"),
        }
        if wants_previous_last_frame_continuity(text):
            action["label"] = "Run With Previous Last Frame"
            action["continuity_reference"] = "previous_clip_last_frame"
        if wants_autonomous_execution(text):
            action["autonomous"] = True
        suggestions.append(action)
    if any(word in lower for word in ["video", "generate", "seedance", "render"]) and latest_version.get("video_model_prompt"):
        suggestions.append({"type": "prepare_video", "label": "Prepare Video"})
    return suggestions


def add_unique_action(actions: list[dict], action: dict) -> None:
    key = (action.get("type"), action.get("label"), action.get("feedback_index"), action.get("prompt"))
    existing_keys = {
        (item.get("type"), item.get("label"), item.get("feedback_index"), item.get("prompt"))
        for item in actions
    }
    if key not in existing_keys:
        actions.append(action)


def dynamic_chat_suggestions(message: str, context: dict, explicit_actions: list[dict]) -> list[dict]:
    lower = message.lower()
    actions = list(explicit_actions)
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    media = build_clip_media_gallery(context)

    if feedback_items:
        feedback_index = feedback_items[0].get("raw_index")
        workflow_label = "Run Workflow Again" if latest_version.get("video_model_prompt") else "Run Workflow"
        add_unique_action(actions, {
            "type": "execute_workflow",
            "label": workflow_label,
            "feedback_index": feedback_index,
        })

    if latest_version.get("video_model_prompt"):
        add_unique_action(actions, {"type": "prepare_video", "label": "Generate Video"})

    if media and not wants_clip_media_gallery(message):
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Show Assets",
            "prompt": "Show all assets and references for this clip.",
        })

    if "feedback" not in lower and feedback_items:
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Show Feedback",
            "prompt": "Show the feedback for this clip.",
        })

    if len(actions) > 4:
        return actions[:4]
    return actions


def fallback_chat_reply(message: str, context: dict) -> str:
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    selected_assets = latest_version.get("selected_assets") or []
    clip = context.get("clip") or {}
    project = context.get("project") or {}
    memory = context.get("memory") or {}
    lower = message.lower()

    if wants_clip_summary_or_analysis(message):
        if context.get("clip_context"):
            return summarize_clip_context(context.get("clip_context"))
        if context.get("clip_context_error"):
            return context.get("clip_context_error")
        return "This clip has not been analyzed yet. Ask me to analyze the clip with OpenAI, or run the workflow first."
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
            if wants_clip_media_gallery(message):
                return "Here are the assets and references currently attached to this clip."
            return "Latest selected assets:\n" + "\n".join(f"- {os.path.basename(asset)}" for asset in selected_assets)
        if wants_clip_media_gallery(message):
            return "I can show the current clip, but no selected assets or generated references are attached to this clip yet."
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
    if wants_clip_summary_or_analysis(message):
        if context.get("clip_context"):
            return summarize_clip_context(context.get("clip_context"))
        if context.get("clip_context_error"):
            return context.get("clip_context_error")
        return "This clip has not been analyzed yet. Ask me to analyze the clip with OpenAI, or run the workflow first."

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
        "clip_context_path",
        "clip_segment_path",
        "clip_context_summary",
        "clip_context_status",
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


def workflow_intent_for_feedback(project_name: str, feedback_index: int) -> dict:
    intents = load_chat_workflow_intents(project_name)
    intent = (intents.get("feedback") or {}).get(str(feedback_index), {})
    if intent.get("status") not in {"pending", None}:
        return {}
    return intent


def mark_workflow_intent_used(project_name: str, feedback_index: int, updates: dict) -> None:
    intents = load_chat_workflow_intents(project_name)
    feedback_intents = intents.setdefault("feedback", {})
    key = str(feedback_index)
    current = feedback_intents.get(key, {})
    current.update(updates)
    current["updated_at"] = now_iso()
    feedback_intents[key] = current
    save_chat_workflow_intents(project_name, intents)


def feedback_context_for_raw_index(project_name: str, feedback_index: int) -> Optional[dict]:
    project_data = get_project_data(project_name)
    for group in project_data.get("feedback", []):
        for item in group.get("feedback_items", []):
            if item.get("raw_index") == feedback_index:
                clip_index = group.get("clip_occurrence")
                if clip_index is None:
                    clip_name = group.get("clip_used")
                    for index, clip in enumerate(project_data.get("timeline", [])):
                        if clip.get("clip") == clip_name:
                            clip_index = index
                            break
                return {
                    "group": group,
                    "item": item,
                    "clip_index": clip_index,
                    "timeline": project_data.get("timeline", []),
                }
    return None


def resolve_clip_file_for_continuity(project_name: str, clip_name: str) -> Optional[Path]:
    project_assets = project_assets_dir(project_name)
    for rel_dir in ("06_clips/_final", "06_clips/_raw"):
        candidate = project_assets / rel_dir / clip_name
        if candidate.exists():
            return candidate
    return None


def prepare_continuity_reference_from_intent(project_name: str, feedback_index: int, intent: dict) -> tuple[Optional[str], Optional[str]]:
    if intent.get("continuity_reference") != "previous_clip_last_frame":
        return None, None

    feedback_context = feedback_context_for_raw_index(project_name, feedback_index)
    if not feedback_context:
        return None, "No feedback context found for continuity reference."

    clip_index = feedback_context.get("clip_index")
    timeline = feedback_context.get("timeline") or []
    if not isinstance(clip_index, int) or clip_index <= 0 or clip_index >= len(timeline):
        return None, "This clip has no previous timeline clip to use for continuity."

    previous_clip = timeline[clip_index - 1]
    previous_clip_name = previous_clip.get("clip")
    if not previous_clip_name:
        return None, "Previous clip name is missing from the timeline."

    previous_clip_path = resolve_clip_file_for_continuity(project_name, previous_clip_name)
    if not previous_clip_path:
        return None, f"Previous clip file not found for {previous_clip_name}."

    output_dir = project_data_dir(project_name) / "chat_continuity_frames" / f"feedback_{feedback_index}"
    duration_s = get_video_duration(str(previous_clip_path)) or float(previous_clip.get("duration_s") or 0.0) or 5.0
    frame_path = extract_last_frame(str(previous_clip_path), str(output_dir), duration_s)
    if not frame_path:
        return None, f"Could not extract the last frame from {previous_clip_name}."

    note = (
        f"Use the extracted last frame from previous clip '{previous_clip_name}' as the continuity "
        f"reference and first-frame anchor for the workflow."
    )
    mark_workflow_intent_used(
        project_name,
        feedback_index,
        {
            "status": "prepared",
            "continuity_frame_path": frame_path,
            "continuity_note": note,
            "previous_clip": previous_clip_name,
        },
    )
    return frame_path, note

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

    context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
    history = load_chat_history(project_name)
    clip_messages = history.setdefault("clips", {}).setdefault(context["clip_key"], [])

    user_message = make_chat_message("user", message_text)
    clip_messages.append(user_message)

    if wants_clip_summary_or_analysis(message_text):
        _clip_context, analysis_error = ensure_clip_context_for_chat(project_name, context, request.provider)
        if analysis_error:
            context["clip_context_error"] = analysis_error

    assistant_text = await generate_chat_reply(request.provider, message_text, context, clip_messages)
    memory_notes = extract_memory_notes(message_text, assistant_text)
    saved_memories = []

    if memory_notes:
        store = memory_store(project_name)
        for note in memory_notes:
            saved_memories.append(
                store.add(
                    note,
                    scope="clip",
                    clip_key=context["clip_key"],
                    source="chat",
                    tags=["chat"],
                    confidence=0.86,
                )
            )
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)

    explicit_actions = infer_action_suggestions(message_text, context)
    actions = dynamic_chat_suggestions(message_text, context, explicit_actions)
    save_pending_workflow_intents(project_name, actions, message_text, context)
    media = build_clip_media_gallery(context) if wants_clip_media_gallery(message_text) else []
    assistant_message = make_chat_message(
        "assistant",
        assistant_text,
        {
            "provider": request.provider,
            "actions": actions,
            "media": media,
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

    item = memory_store(project_name).add(
        text,
        scope=request.scope,
        clip_key=context["clip_key"] if request.scope == "clip" else None,
        source="manual",
        tags=["manual"],
        confidence=0.9,
    )
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
    continuity_frame_path = None
    continuity_note = None
    continuity_error = None
    intent = workflow_intent_for_feedback(project, index)
    if intent:
        continuity_frame_path, continuity_note = prepare_continuity_reference_from_intent(project, index, intent)
        if not continuity_frame_path and intent.get("continuity_reference"):
            continuity_error = "Continuity reference requested but could not be prepared."
            mark_workflow_intent_used(project, index, {"status": "failed", "error": continuity_error})
    
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
    if continuity_frame_path:
        cmd.extend(["--continuity-frame-path", continuity_frame_path])
    if continuity_note:
        cmd.extend(["--continuity-note", continuity_note])
    
    async def log_generator():
        yield f"data: [START] Launching workflow subprocess for feedback index {index}...\n\n"
        if continuity_frame_path:
            yield f"data: [CONTEXT] Prepared previous-clip last-frame continuity reference: {continuity_frame_path}\n\n"
        elif continuity_error:
            yield f"data: [CONTEXT] {continuity_error}\n\n"
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
    port = int(os.environ.get("LOKA_BACKEND_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="info")
