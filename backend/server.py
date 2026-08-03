import os
import json
import asyncio
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote, unquote
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import set_key, unset_key

from config.settings import (
    LITE_MODEL,
    LOADED_ENV_FILES,
    LOKA_STORAGE_DIR,
    OPENAI_LITE_MODEL,
    OS_ENV_KEYS_AT_START,
    SECRET_ENV_KEYS,
    app_config_env_path,
)
from src.workflows.project_setup import setup_project_workspace
from src.workflows.timeline_extraction import extract_timeline_from_project
from src.workflows.feedback_parsing import parse_and_align_feedback
from src.workflows.prompt_generation import extract_last_frame, get_video_duration
from src.workflows.clip_context import analyze_clip_context, clip_context_dir
from src.workflows.referenced_frames import extract_frame_at_offset
from src.generator.media import _frame_offsets_for_duration
from src.generator.client import _default_model_for_provider, generate_structured
from src.generator.validator import merge_learning_eval_report, run_learning_eval, run_quality_check
from src.schemas import PromptResult
from src.clustering import find_matching_clip_occurrence
from src.utils import parse_timestamp_to_seconds
from src.logging_utils import log_event
from src.chat_memory import ChatMemoryStore
from src.prompt_learning import PromptEvalCaseStore, PromptLearningStore
from scripts.generate_seedance_video import (
    SeedanceGenerationRecoveryError,
    SupabaseAssetUrlCache,
    build_segmind_payload,
    create_seedance_task,
    save_video_bytes,
)

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
APP_STORAGE_DIR = Path(LOKA_STORAGE_DIR).expanduser().resolve()

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


class RuntimeSecretsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    segmind_api_key: Optional[str] = None


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


def env_secret_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    config_env = app_config_env_path(APP_STORAGE_DIR)
    for key in SECRET_ENV_KEYS:
        configured = bool(os.environ.get(key))
        locked_by_os_env = key in OS_ENV_KEYS_AT_START
        if locked_by_os_env:
            source = "os_environment"
        elif configured:
            source = "app_or_dev_env_file"
        else:
            source = "missing"
        status[key] = {
            "configured": configured,
            "source": source,
            "locked_by_os_env": locked_by_os_env,
            "can_update": not locked_by_os_env,
        }
    return {
        "keys": status,
        "config_env_path": str(config_env),
    }


@app.get("/api/config/runtime")
def get_runtime_config():
    return {
        "app_storage_dir": str(APP_STORAGE_DIR),
        "data_dir": str(DATA_DIR),
        "assets_dir": str(ASSETS_DIR),
        "loaded_env_files": LOADED_ENV_FILES,
        "secrets": env_secret_status(),
    }


@app.post("/api/config/secrets")
def update_runtime_secrets(payload: RuntimeSecretsUpdate):
    updates = {
        "OPENAI_API_KEY": payload.openai_api_key,
        "GEMINI_API_KEY": payload.gemini_api_key,
        "SEGMIND_API_KEY": payload.segmind_api_key,
    }
    requested_updates = {key: value for key, value in updates.items() if value is not None}
    locked_keys = sorted(key for key in requested_updates if key in OS_ENV_KEYS_AT_START)
    if locked_keys:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot update keys controlled by OS environment: {', '.join(locked_keys)}",
        )

    config_env = app_config_env_path(APP_STORAGE_DIR)
    config_env.parent.mkdir(parents=True, exist_ok=True)
    if not config_env.exists():
        config_env.touch(mode=0o600)

    for key, value in requested_updates.items():
        normalized_value = value.strip()
        if normalized_value:
            set_key(str(config_env), key, normalized_value, quote_mode="always")
            os.environ[key] = normalized_value
        else:
            unset_key(str(config_env), key)
            os.environ.pop(key, None)

    try:
        config_env.chmod(0o600)
    except OSError:
        pass

    return {
        "status": "ok",
        "config_env_path": str(config_env),
        "secrets": env_secret_status(),
    }

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


class ClipChatActionRequest(BaseModel):
    clip_index: int
    action: dict
    message: str = ""
    provider: Literal["openai", "gemini"] = "openai"


class ClipMemoryRequest(BaseModel):
    clip_index: int
    text: str
    scope: Literal["clip", "project"] = "clip"


class GenerateClipVideoRequest(BaseModel):
    clip_index: int
    prompt_version_index: Optional[int] = None
    provider: str = "segmind"
    resolution: Literal["480p", "720p", "1080p", "4k"] = "720p"
    generate_audio: bool = False
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"] = "9:16"
    duration: int = 5


class WorkflowJobRequest(BaseModel):
    feedback_index: int
    provider: Literal["openai", "gemini"] = "openai"
    agent_run_id: Optional[str] = None


PROMPT_FEEDBACK_CATEGORIES = {
    "missed_feedback",
    "wrong_visual_detail",
    "wrong_character_or_wardrobe",
    "continuity_error",
    "bad_camera_instruction",
    "bad_audio_or_dialogue",
    "unsupported_assumption",
    "format_error",
    "too_vague",
    "too_verbose",
    "provider_incompatible",
    "other",
}
PROMPT_FEEDBACK_RATINGS = {"positive", "negative"}
PROMPT_FEEDBACK_STATUSES = {"open", "approved", "rejected", "resolved"}


class PromptFeedbackCreateRequest(BaseModel):
    clip_index: int
    clip_key: str
    prompt_id: str
    prompt_version_id: str
    rating: Literal["positive", "negative"]
    categories: list[str] = Field(default_factory=list)
    severity: int = 3
    comment: str = ""
    correction: str = ""
    remember_note: str = ""
    create_eval_case: bool = False
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptFeedbackPatchRequest(BaseModel):
    rating: Optional[Literal["positive", "negative"]] = None
    categories: Optional[list[str]] = None
    severity: Optional[int] = None
    comment: Optional[str] = None
    correction: Optional[str] = None
    remember_note: Optional[str] = None
    create_eval_case: Optional[bool] = None
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptLessonCreateRequest(BaseModel):
    scope: Literal["project", "clip"] = "project"
    clip_key: Optional[str] = None
    category: str = "other"
    lesson: str
    source_feedback_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class PromptLessonPatchRequest(BaseModel):
    scope: Optional[Literal["project", "clip"]] = None
    clip_key: Optional[str] = None
    category: Optional[str] = None
    lesson: Optional[str] = None
    source_feedback_ids: Optional[list[str]] = None
    confidence: Optional[float] = None
    positive_examples: Optional[list[str]] = None
    negative_examples: Optional[list[str]] = None
    archived: Optional[bool] = None


class PromptEvalCasePatchRequest(BaseModel):
    name: Optional[str] = None
    input: Optional[dict] = None
    expected_behavior: Optional[list[str]] = None
    failure_categories: Optional[list[str]] = None
    enabled: Optional[bool] = None


class PromptRevisionRequest(BaseModel):
    clip_index: int
    feedback_ids: list[str] = Field(default_factory=list)
    lesson_ids: list[str] = Field(default_factory=list)
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestRequest(BaseModel):
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestionResult(BaseModel):
    lesson: str
    category: str = "other"
    confidence: float = 0.75
    reasoning: str = ""


class ClipSelectionUpdate(BaseModel):
    active_prompt_version_id: Optional[str] = None
    active_generated_video_id: Optional[str] = None
    selected_assets: Optional[list[dict]] = None
    pinned_assets: Optional[list[dict]] = None
    selected_asset_ids: Optional[list[str]] = None
    selected_asset_paths: Optional[list[str]] = None
    pinned_asset_ids: Optional[list[str]] = None
    pinned_asset_paths: Optional[list[str]] = None


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


def atomic_write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def chat_history_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "clip_chats.json"


def chat_memory_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "chat_memory.json"


def chat_workflow_intents_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "chat_workflow_intents.json"


def chat_agent_runs_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "chat_agent_runs.json"


def prompt_feedback_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "prompt_feedback.json"


def prompt_lessons_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "prompt_lessons.json"


def prompt_eval_cases_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "prompt_eval_cases.json"


def project_jobs_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "jobs.json"


def clip_selections_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "clip_selections.json"


def project_events_path(project_name: str) -> Path:
    return project_data_dir(project_name) / "events.jsonl"


def load_project_events(project_name: str, *, limit: int = 200, clip_key: Optional[str] = None, event_type: Optional[str] = None) -> list[dict]:
    path = project_events_path(project_name)
    if not path.exists():
        return []
    events = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if clip_key and event.get("clip_key") != clip_key:
                    continue
                if event_type and event.get("type") != event_type:
                    continue
                events.append(event)
    except Exception:
        return []
    return events[-max(1, min(limit, 1000)):]


def append_project_event(
    project_name: str,
    event_type: str,
    *,
    actor: str = "system",
    clip_index: Optional[int] = None,
    clip_key: Optional[str] = None,
    entity: Optional[str] = None,
    entity_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    path = project_events_path(project_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_events = load_project_events(project_name, limit=1)
    previous = previous_events[-1] if previous_events else None
    sequence = int((previous or {}).get("sequence") or 0) + 1
    event = {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "sequence": sequence,
        "type": event_type,
        "actor": actor,
        "project_name": project_name,
        "clip_index": clip_index,
        "clip_key": clip_key,
        "entity": entity,
        "entity_id": entity_id,
        "payload": payload or {},
        "previous_event_hash": (previous or {}).get("event_hash"),
        "created_at": now_iso(),
    }
    event["event_hash"] = stable_json_hash({
        key: value
        for key, value in event.items()
        if key != "event_hash"
    })
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def load_chat_history(project_name: str) -> dict:
    data = read_json_file(chat_history_path(project_name), {"clips": {}})
    if not isinstance(data, dict):
        return {"clips": {}}
    data.setdefault("clips", {})
    return data


def save_chat_history(project_name: str, history: dict) -> None:
    write_json_file(chat_history_path(project_name), history)


def load_project_jobs(project_name: str) -> dict:
    data = read_json_file(project_jobs_path(project_name), {"schema_version": 1, "jobs": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "jobs": []}
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        jobs = []
    return {"schema_version": 1, "jobs": jobs}


def save_project_jobs(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("jobs", [])
    write_json_file(project_jobs_path(project_name), data)


def load_clip_selections(project_name: str) -> dict:
    data = read_json_file(clip_selections_path(project_name), {"schema_version": 1, "clips": {}})
    if not isinstance(data, dict):
        return {"schema_version": 1, "clips": {}}
    clips = data.get("clips")
    if not isinstance(clips, dict):
        clips = {}
    return {"schema_version": 1, "clips": clips}


def save_clip_selections(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("clips", {})
    write_json_file(clip_selections_path(project_name), data)


def load_prompt_feedback(project_name: str) -> dict:
    data = read_json_file(prompt_feedback_path(project_name), {"schema_version": 1, "items": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "items": []}
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    return {"schema_version": 1, "items": [item for item in items if isinstance(item, dict)]}


def save_prompt_feedback(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("items", [])
    atomic_write_json_file(prompt_feedback_path(project_name), data)


def _clean_prompt_feedback_categories(categories: list[str]) -> list[str]:
    cleaned = []
    for category in categories or []:
        if category not in PROMPT_FEEDBACK_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid prompt feedback category: {category}")
        if category not in cleaned:
            cleaned.append(category)
    return cleaned


def _validate_prompt_feedback_payload(payload: dict) -> dict:
    rating = payload.get("rating")
    if rating not in PROMPT_FEEDBACK_RATINGS:
        raise HTTPException(status_code=400, detail="Invalid prompt feedback rating")
    status = payload.get("status")
    if status and status not in PROMPT_FEEDBACK_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid prompt feedback status")

    try:
        severity = int(payload.get("severity", 3))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Severity must be a number from 1 to 5") from exc
    if severity < 1 or severity > 5:
        raise HTTPException(status_code=400, detail="Severity must be from 1 to 5")

    categories = _clean_prompt_feedback_categories(list(payload.get("categories") or []))
    comment = str(payload.get("comment") or "").strip()
    correction = str(payload.get("correction") or "").strip()
    remember_note = str(payload.get("remember_note") or "").strip()
    if rating == "negative" and not any([comment, correction, remember_note]):
        raise HTTPException(status_code=400, detail="Negative feedback needs a comment, correction, or remember note")

    prompt_version_id = str(payload.get("prompt_version_id") or "").strip()
    prompt_id = str(payload.get("prompt_id") or "").strip()
    clip_key = str(payload.get("clip_key") or "").strip()
    if not prompt_version_id:
        raise HTTPException(status_code=400, detail="Prompt version id is required")
    if not prompt_id:
        raise HTTPException(status_code=400, detail="Prompt id is required")
    if not clip_key:
        raise HTTPException(status_code=400, detail="Clip key is required")
    clip_index = payload.get("clip_index")
    if not isinstance(clip_index, int) or clip_index < 0:
        raise HTTPException(status_code=400, detail="Valid clip index is required")

    return {
        **payload,
        "clip_index": clip_index,
        "clip_key": clip_key,
        "prompt_id": prompt_id,
        "prompt_version_id": prompt_version_id,
        "rating": rating,
        "categories": categories,
        "severity": severity,
        "comment": comment,
        "correction": correction,
        "remember_note": remember_note,
        "create_eval_case": bool(payload.get("create_eval_case", False)),
        "status": status or ("approved" if rating == "positive" else "open"),
    }


def create_prompt_feedback_item(project_name: str, payload: dict, *, actor: str = "user") -> dict:
    payload = _validate_prompt_feedback_payload(payload)
    created_at = now_iso()
    item = {
        "id": str(uuid.uuid4()),
        "project_name": project_name,
        **payload,
        "created_at": created_at,
        "updated_at": created_at,
    }
    data = load_prompt_feedback(project_name)
    data.setdefault("items", []).append(item)
    save_prompt_feedback(project_name, data)
    eval_case = None
    if item.get("create_eval_case") and item.get("rating") == "negative":
        try:
            eval_case = create_eval_case_from_feedback_item(project_name, item)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_feedback_created",
        actor=actor,
        clip_index=item.get("clip_index"),
        clip_key=item.get("clip_key"),
        entity="prompt_feedback",
        entity_id=item.get("id"),
        payload={
            "prompt_id": item.get("prompt_id"),
            "prompt_version_id": item.get("prompt_version_id"),
            "rating": item.get("rating"),
            "status": item.get("status"),
            "categories": item.get("categories", []),
        },
    )
    if eval_case:
        append_project_event(
            project_name,
            "prompt_eval_case_created",
            actor=actor,
            clip_index=item.get("clip_index"),
            clip_key=item.get("clip_key"),
            entity="prompt_eval_case",
            entity_id=eval_case.get("id"),
            payload={
                "source_feedback_id": item.get("id"),
                "failure_categories": eval_case.get("failure_categories", []),
            },
        )
    return {"item": item, "eval_case": eval_case}


def prompt_feedback_items_for_clip(project_name: str, clip_index: Optional[int] = None) -> list[dict]:
    items = load_prompt_feedback(project_name).get("items", [])
    if clip_index is None:
        return items
    return [item for item in items if item.get("clip_index") == clip_index]


def prompt_feedback_summary_by_version(project_name: str, clip_index: int) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for item in prompt_feedback_items_for_clip(project_name, clip_index):
        version_id = str(item.get("prompt_version_id") or "")
        if not version_id:
            continue
        summary = summaries.setdefault(version_id, {
            "prompt_version_id": version_id,
            "status": "unreviewed",
            "total_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "open_negative_count": 0,
            "rejected_count": 0,
            "latest_feedback_at": None,
        })
        summary["total_count"] += 1
        if item.get("rating") == "positive":
            summary["positive_count"] += 1
        if item.get("rating") == "negative":
            summary["negative_count"] += 1
            if item.get("status") == "open":
                summary["open_negative_count"] += 1
        if item.get("status") == "rejected":
            summary["rejected_count"] += 1
        updated_at = item.get("updated_at") or item.get("created_at")
        if updated_at and (not summary.get("latest_feedback_at") or updated_at > summary["latest_feedback_at"]):
            summary["latest_feedback_at"] = updated_at

    for summary in summaries.values():
        if summary["rejected_count"]:
            summary["status"] = "rejected"
        elif summary["open_negative_count"]:
            summary["status"] = "needs_revision"
        elif summary["positive_count"] and not summary["open_negative_count"]:
            summary["status"] = "approved"
        else:
            summary["status"] = "unreviewed"
    return summaries


def find_prompt_feedback_item(project_name: str, feedback_id: str) -> Optional[dict]:
    return next((item for item in load_prompt_feedback(project_name).get("items", []) if item.get("id") == feedback_id), None)


def clip_state_for_lesson_sources(project_name: str, source_feedback_ids: list[str]) -> Optional[dict]:
    for feedback_id in source_feedback_ids or []:
        item = find_prompt_feedback_item(project_name, feedback_id)
        if item and isinstance(item.get("clip_index"), int):
            return build_clip_state(project_name, item["clip_index"])
    return None


def _prompt_context_for_feedback(project_name: str, feedback_item: dict) -> dict:
    clip_index = feedback_item.get("clip_index")
    if not isinstance(clip_index, int):
        return {}
    state = build_clip_state(project_name, clip_index)
    versions = (state.get("active_prompt") or {}).get("versions") or []
    version = next(
        (candidate for candidate in versions if candidate.get("prompt_version_id") == feedback_item.get("prompt_version_id")),
        None,
    )
    return {
        "clip": ((state.get("timeline") or {}).get("clip") or {}),
        "feedback_state": state.get("feedback_state"),
        "prompt_version": version or {},
    }


def _eval_case_context_for_feedback(project_name: str, feedback_item: dict) -> dict:
    context = _prompt_context_for_feedback(project_name, feedback_item)
    prompt_version = context.get("prompt_version") or {}
    feedback_state = context.get("feedback_state") or {}
    return {
        "clip_summary": (
            prompt_version.get("clip_context_summary")
            or (((context.get("clip") or {}).get("summary")) if isinstance(context.get("clip"), dict) else "")
            or ""
        ),
        "selected_assets": prompt_version.get("selected_assets") or [],
        "feedback_items": feedback_state.get("feedback_items") or [
            {
                "remark": feedback_item.get("comment") or feedback_item.get("correction") or feedback_item.get("remember_note") or "",
                "category": "video",
            }
        ],
    }


def create_eval_case_from_feedback_item(project_name: str, feedback_item: dict) -> dict:
    context = _eval_case_context_for_feedback(project_name, feedback_item)
    return prompt_eval_case_store(project_name).add_eval_case_from_feedback(
        feedback_item,
        clip_summary=context.get("clip_summary", ""),
        selected_assets=context.get("selected_assets", []),
        feedback_items=context.get("feedback_items", []),
    )


def _client_for_prompt_provider(provider: Literal["openai", "gemini"]):
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")
    from google import genai

    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _feedback_revision_text(feedback_items: list[dict]) -> str:
    return "\n".join(
        (
            f"- rating={item.get('rating')} severity={item.get('severity')} "
            f"categories={item.get('categories') or []}\n"
            f"  comment: {item.get('comment') or ''}\n"
            f"  correction: {item.get('correction') or ''}\n"
            f"  remember: {item.get('remember_note') or ''}"
        )
        for item in feedback_items
    )


def _lesson_revision_text(lessons: list[dict]) -> str:
    return "\n".join(
        f"- [{lesson.get('category')}] {lesson.get('lesson')}"
        for lesson in lessons
        if lesson.get("lesson")
    )


def _eval_case_revision_text(eval_cases: list[dict]) -> str:
    return "\n".join(
        f"- {case.get('name')}: {'; '.join(case.get('expected_behavior') or [])}"
        for case in eval_cases
        if case.get("enabled", True)
    )


def _compact_applied_eval_cases(eval_cases: list[dict]) -> list[dict]:
    return [
        {
            "id": case.get("id"),
            "name": case.get("name"),
            "failure_categories": case.get("failure_categories", []),
            "expected_behavior": case.get("expected_behavior", []),
        }
        for case in eval_cases
    ]


def _suggest_prompt_lesson(project_name: str, feedback_item: dict, provider: Literal["openai", "gemini"]) -> dict:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    context = _prompt_context_for_feedback(project_name, feedback_item)
    prompt_version = context.get("prompt_version") or {}
    prompt_text = str(prompt_version.get("video_model_prompt") or "")[:2400]
    prompt = (
        "Create one durable lesson from this prompt feedback. The lesson must help future prompt generation avoid "
        "repeating the same kind of mistake. Avoid copying the raw complaint, avoid overgeneralizing from one clip, "
        "and write a rule that is useful across similar prompt-generation tasks.\n\n"
        f"PROJECT: {project_name}\n"
        f"FEEDBACK CATEGORY: {json.dumps(feedback_item.get('categories') or [], ensure_ascii=False)}\n"
        f"RATING: {feedback_item.get('rating')}\n"
        f"SEVERITY: {feedback_item.get('severity')}\n"
        f"COMMENT: {feedback_item.get('comment') or ''}\n"
        f"CORRECTION: {feedback_item.get('correction') or ''}\n"
        f"REMEMBER NOTE: {feedback_item.get('remember_note') or ''}\n"
        f"CLIP: {json.dumps(context.get('clip') or {}, ensure_ascii=False)}\n"
        f"PROMPT EXCERPT:\n{prompt_text}"
    )
    try:
        suggestion = generate_structured(
            provider=provider,
            client=client,
            model=_default_model_for_provider(provider),
            contents=[prompt],
            schema=PromptLessonSuggestionResult,
            system_instruction=(
                "You convert user prompt feedback into one concise, editable lesson. "
                "Return only a reusable rule, a category, confidence, and brief reasoning."
            ),
            temperature=0.1,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to suggest lesson: {exc}") from exc

    category = suggestion.get("category") or (feedback_item.get("categories") or ["other"])[0]
    if category not in PROMPT_FEEDBACK_CATEGORIES:
        category = (feedback_item.get("categories") or ["other"])[0]
    try:
        confidence = max(0.0, min(1.0, float(suggestion.get("confidence", 0.75))))
    except Exception:
        confidence = 0.75
    return {
        "lesson": str(suggestion.get("lesson") or "").strip(),
        "category": category,
        "confidence": confidence,
        "reasoning": str(suggestion.get("reasoning") or "").strip(),
        "source_feedback_id": feedback_item.get("id"),
    }


def clip_selection_for_key(project_name: str, clip_key: str) -> dict:
    selections = load_clip_selections(project_name)
    selection = selections.get("clips", {}).get(clip_key)
    return selection if isinstance(selection, dict) else {}


def _asset_refs_from_ids_paths(asset_ids: list[str], asset_paths: list[str], *, source: str = "clip_selection") -> list[dict]:
    refs = []
    seen = set()
    max_count = max(len(asset_ids), len(asset_paths))
    for index in range(max_count):
        asset_id = asset_ids[index] if index < len(asset_ids) else None
        asset_path = asset_paths[index] if index < len(asset_paths) else None
        key = asset_id or asset_path
        if not key or key in seen:
            continue
        seen.add(key)
        refs.append({
            "asset_id": asset_id,
            "path": asset_path,
            "role": "selected_reference",
            "reason": "Persisted from legacy selected asset state.",
            "confidence": 0.75,
            "source": source,
        })
    return refs


def normalize_selection_asset_refs(selection: dict) -> list[dict]:
    raw_refs = selection.get("selected_assets")
    if raw_refs is None:
        raw_refs = selection.get("pinned_assets")
    if isinstance(raw_refs, list):
        refs = []
        for ref in raw_refs:
            if isinstance(ref, dict):
                asset_id = ref.get("asset_id")
                asset_path = ref.get("path") or ref.get("asset_path") or ref.get("selected_path")
                if not asset_id and not asset_path:
                    continue
                refs.append({
                    "asset_id": asset_id,
                    "path": asset_path,
                    "role": ref.get("role") or "selected_reference",
                    "reason": ref.get("reason") or "Selected for this clip.",
                    "confidence": float(ref.get("confidence") if ref.get("confidence") is not None else 0.8),
                    "source": ref.get("source") or "clip_selection",
                })
            elif isinstance(ref, str):
                refs.append({
                    "asset_id": None,
                    "path": ref,
                    "role": "selected_reference",
                    "reason": "Selected for this clip.",
                    "confidence": 0.75,
                    "source": "clip_selection",
                })
        return refs

    return _asset_refs_from_ids_paths(
        list(selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or []),
        list(selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or []),
    )


def _mirror_asset_ref_lists(selection: dict) -> None:
    refs = normalize_selection_asset_refs(selection)
    selection["selected_assets"] = refs
    selection["pinned_assets"] = refs
    selection["selected_asset_ids"] = [ref["asset_id"] for ref in refs if ref.get("asset_id")]
    selection["selected_asset_paths"] = [ref["path"] for ref in refs if ref.get("path")]
    selection["pinned_asset_ids"] = selection["selected_asset_ids"]
    selection["pinned_asset_paths"] = selection["selected_asset_paths"]


def update_clip_selection(project_name: str, clip_key: str, updates: dict) -> dict:
    selections = load_clip_selections(project_name)
    clips = selections.setdefault("clips", {})
    now = now_iso()
    current = clips.get(clip_key)
    if not isinstance(current, dict):
        current = {
            "clip_key": clip_key,
            "created_at": now,
        }
    current.update({
        key: value
        for key, value in updates.items()
        if key in {
            "active_prompt_version_id",
            "active_generated_video_id",
            "selected_assets",
            "pinned_assets",
            "selected_asset_ids",
            "selected_asset_paths",
            "pinned_asset_ids",
            "pinned_asset_paths",
        }
    })
    if "selected_asset_ids" in updates and "pinned_asset_ids" not in updates:
        current["pinned_asset_ids"] = updates["selected_asset_ids"]
    if "selected_asset_paths" in updates and "pinned_asset_paths" not in updates:
        current["pinned_asset_paths"] = updates["selected_asset_paths"]
    if any(key in updates for key in {
        "selected_assets",
        "pinned_assets",
        "selected_asset_ids",
        "selected_asset_paths",
        "pinned_asset_ids",
        "pinned_asset_paths",
    }):
        _mirror_asset_ref_lists(current)
    current["clip_key"] = clip_key
    current["updated_at"] = now
    clips[clip_key] = current
    save_clip_selections(project_name, selections)
    append_project_event(
        project_name,
        "clip_selection_updated",
        actor="chat_agent" if any(key in updates for key in {"selected_assets", "pinned_assets"}) else "backend",
        clip_key=clip_key,
        entity="clip_selection",
        entity_id=clip_key,
        payload={
            "updated_keys": sorted(updates.keys()),
            "active_prompt_version_id": current.get("active_prompt_version_id"),
            "active_generated_video_id": current.get("active_generated_video_id"),
            "selected_assets": current.get("selected_assets", []),
            "selected_asset_ids": current.get("selected_asset_ids", []),
            "selected_asset_paths": current.get("selected_asset_paths", []),
        },
    )
    return current


def create_project_job(
    project_name: str,
    job_type: str,
    *,
    clip_index: Optional[int] = None,
    feedback_index: Optional[int] = None,
    provider: str = "",
    payload: Optional[dict] = None,
    agent_run_id: Optional[str] = None,
) -> dict:
    data = load_project_jobs(project_name)
    now = now_iso()
    job = {
        "id": str(uuid.uuid4()),
        "type": job_type,
        "status": "queued",
        "project_name": project_name,
        "clip_index": clip_index,
        "feedback_index": feedback_index,
        "provider": provider,
        "payload": payload or {},
        "agent_run_id": agent_run_id,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
    }
    data.setdefault("jobs", []).append(job)
    save_project_jobs(project_name, data)
    append_project_event(
        project_name,
        "job_created",
        actor="agent" if agent_run_id else "backend",
        clip_index=clip_index,
        entity="job",
        entity_id=job["id"],
        payload={
            "job_type": job_type,
            "status": job["status"],
            "feedback_index": feedback_index,
            "provider": provider,
            "agent_run_id": agent_run_id,
        },
    )
    return job


def update_project_job(project_name: str, job_id: str, updates: dict) -> dict:
    data = load_project_jobs(project_name)
    updated = {}
    previous_status = None
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            previous_status = job.get("status")
            job.update(updates)
            job["updated_at"] = now_iso()
            updated = job
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    save_project_jobs(project_name, data)
    if "status" in updates and updates.get("status") != previous_status:
        append_project_event(
            project_name,
            "job_status_changed",
            actor="backend",
            clip_index=updated.get("clip_index"),
            entity="job",
            entity_id=job_id,
            payload={
                "job_type": updated.get("type"),
                "previous_status": previous_status,
                "status": updated.get("status"),
                "feedback_index": updated.get("feedback_index"),
                "agent_run_id": updated.get("agent_run_id"),
                "error": updated.get("error"),
            },
        )
    elif "result" in updates:
        append_project_event(
            project_name,
            "job_result_updated",
            actor="backend",
            clip_index=updated.get("clip_index"),
            entity="job",
            entity_id=job_id,
            payload={
                "job_type": updated.get("type"),
                "status": updated.get("status"),
                "has_self_evaluation": bool((updated.get("result") or {}).get("self_evaluation")),
            },
        )
    return updated


def append_project_job_log(project_name: str, job_id: str, message: str) -> dict:
    data = load_project_jobs(project_name)
    updated = {}
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            job.setdefault("logs", []).append({
                "timestamp": now_iso(),
                "message": message,
            })
            job["updated_at"] = now_iso()
            updated = job
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    save_project_jobs(project_name, data)
    return updated


def get_project_job(project_name: str, job_id: str) -> dict:
    data = load_project_jobs(project_name)
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            return job
    raise HTTPException(status_code=404, detail="Job not found")


def list_recent_project_jobs(project_name: str, limit: int = 50) -> list[dict]:
    data = load_project_jobs(project_name)
    jobs = list(data.get("jobs", []))
    jobs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return jobs[:limit]


def project_job_cancel_requested(project_name: str, job_id: str) -> bool:
    try:
        return bool(get_project_job(project_name, job_id).get("cancel_requested"))
    except HTTPException:
        return False


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


def generated_videos_for_version(version: Optional[dict]) -> list[dict]:
    if not version:
        return []
    videos = version.get("generated_videos")
    return videos if isinstance(videos, list) else []


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

    generated_videos = generated_videos_for_version(latest_version)
    if not generated_videos:
        generated_videos = generated_videos_for_version(context.get("prompt"))
    for index, video in enumerate(generated_videos):
        add_chat_media_item(
            media,
            label=video.get("label") or f"Generated video v{video.get('version') or index + 1}",
            source="generated_videos",
            path=video.get("path") or video.get("url"),
            media_type="video",
        )

    return media


def memory_store(project_name: str) -> ChatMemoryStore:
    return ChatMemoryStore(chat_memory_path(project_name))


def prompt_learning_store(project_name: str) -> PromptLearningStore:
    return PromptLearningStore(prompt_lessons_path(project_name))


def prompt_eval_case_store(project_name: str) -> PromptEvalCaseStore:
    return PromptEvalCaseStore(prompt_eval_cases_path(project_name))


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
    versions = prompt_versions_for_record(prompt)
    return versions[-1] if versions else None


def stable_state_id(prefix: str, *parts: Any) -> str:
    raw = "::".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def file_mtime_iso(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _asset_identity(project_name: str, category: str, asset: dict) -> dict:
    path = str(asset.get("path") or asset.get("name") or "")
    return {
        **asset,
        "asset_id": stable_state_id("asset", project_name, category, path),
        "category": category,
    }


def _asset_lookup_by_path(assets: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for category, items in (assets or {}).items():
        for asset in items:
            enriched = _asset_identity("", category, asset)
            candidates = {
                str(asset.get("path") or ""),
                str(asset.get("name") or ""),
                os.path.basename(str(asset.get("path") or "")),
                os.path.basename(str(asset.get("name") or "")),
            }
            for candidate in candidates:
                if candidate:
                    lookup[candidate.lower()] = enriched
    return lookup


def _normalize_selected_asset(project_name: str, asset_path: str, lookup: dict[str, dict], role: str = "selected_reference") -> dict:
    path_text = str(asset_path or "")
    basename = os.path.basename(path_text)
    known = lookup.get(path_text.lower()) or lookup.get(basename.lower())
    if known:
        return {
            **known,
            "asset_id": stable_state_id("asset", project_name, known.get("category"), known.get("path") or known.get("name")),
            "role": role,
            "source": "prompt_version",
            "selected_path": path_text,
        }
    return {
        "asset_id": stable_state_id("asset", project_name, path_text),
        "name": basename or path_text,
        "path": path_text,
        "url": static_url_for_path(path_text),
        "type": infer_media_type(path_text),
        "size": media_size_for_path(path_text),
        "category": None,
        "role": role,
        "source": "prompt_version",
        "selected_path": path_text,
        "missing": not bool(static_url_for_path(path_text)),
    }


def _selected_assets_from_selection(project_name: str, selection: dict, asset_lookup: dict[str, dict]) -> list[dict]:
    structured_refs = normalize_selection_asset_refs(selection)
    pinned_assets: list[dict] = []
    seen_ids: set[str] = set()

    for ref in structured_refs:
        ref_path = ref.get("path")
        ref_id = ref.get("asset_id")
        normalized = _normalize_selected_asset(project_name, ref_path or ref_id, asset_lookup, role=ref.get("role") or "selected_reference")
        if ref_id:
            normalized["asset_id"] = ref_id
        normalized["source"] = ref.get("source") or "clip_selection"
        normalized["role"] = ref.get("role") or normalized.get("role") or "selected_reference"
        normalized["reason"] = ref.get("reason") or "Selected for this clip."
        normalized["confidence"] = ref.get("confidence", 0.8)
        normalized["structured_ref"] = ref
        if normalized["asset_id"] not in seen_ids:
            pinned_assets.append(normalized)
            seen_ids.add(normalized["asset_id"])

    return pinned_assets


def _referenced_frames_for_state(feedback: Optional[dict], latest_version: Optional[dict], project_name: str) -> list[dict]:
    referenced_frames = list((latest_version or {}).get("referenced_frames") or [])
    feedback = feedback or {}
    for item in feedback.get("feedback_items", []):
        referenced_frames.extend(item.get("referenced_frames") or [])
    referenced_frames.extend(feedback.get("referenced_frames") or [])

    normalized = []
    seen = set()
    for ref in referenced_frames:
        frame_path = str(ref.get("frame_path") or "")
        key = frame_path or json.dumps(ref, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            **ref,
            "reference_id": stable_state_id("ref", project_name, ref.get("timestamp"), frame_path, ref.get("clip_used")),
            "url": static_url_for_path(frame_path),
        })
    return normalized


def _prompt_versions_for_state(prompt: Optional[dict], project_name: str, clip_index: int) -> list[dict]:
    if not prompt:
        return []
    versions = prompt_versions_for_record(prompt)
    feedback_summaries = prompt_feedback_summary_by_version(project_name, clip_index)
    prompt_id = stable_state_id(
        "prompt",
        project_name,
        prompt.get("clip_used") or prompt.get("matched_clip"),
        prompt.get("clip_occurrence"),
    )
    normalized = []
    for index, version in enumerate(versions):
        normalized.append({
            **version,
            "prompt_id": prompt_id,
            "prompt_version_id": stable_state_id(
                "version",
                prompt_id,
                index,
                version.get("timestamp"),
                version.get("video_model_prompt"),
            ),
            "version_index": index,
            "is_latest": index == len(versions) - 1,
            "clip_index": clip_index,
            "feedback_summary": feedback_summaries.get(
                stable_state_id(
                    "version",
                    prompt_id,
                    index,
                    version.get("timestamp"),
                    version.get("video_model_prompt"),
                ),
                {
                    "prompt_version_id": stable_state_id(
                        "version",
                        prompt_id,
                        index,
                        version.get("timestamp"),
                        version.get("video_model_prompt"),
                    ),
                    "status": "unreviewed",
                    "total_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "open_negative_count": 0,
                    "rejected_count": 0,
                    "latest_feedback_at": None,
                },
            ),
        })
    return normalized


def query_requests_latest_prompt(query: str = "") -> bool:
    lower = str(query or "").lower()
    has_prompt = "prompt" in lower or "prompts" in lower
    has_latest = any(token in lower for token in ["latest", "newest", "recent", "current", "last generated"])
    return has_prompt and has_latest


def has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def wants_prompt_display(text: str) -> bool:
    lower = text.lower()
    has_prompt = has_word(lower, "prompt") or has_word(lower, "prompts")
    has_display = any(has_word(lower, word) for word in ["show", "view", "display", "see", "give", "tell", "paste", "what"])
    return has_prompt and (has_display or query_requests_latest_prompt(text))


def format_latest_prompt_reply(context: dict) -> str:
    active_prompt = (context.get("clip_state") or {}).get("active_prompt") or {}
    latest_version = context.get("latest_version") or {}
    versions = active_prompt.get("versions") or []
    version_index = latest_version.get("version_index")
    if version_index is None and versions:
        version_index = len(versions) - 1
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    if not prompt_text:
        return "No generated prompt exists for this clip yet."
    label = f"v{int(version_index) + 1}" if version_index is not None else "latest"
    return f"Latest Prompt ({label})\n\n{prompt_text}"


def user_explicitly_requests_video_generation(text: str) -> bool:
    lower = text.lower()
    if wants_prompt_display(text):
        return False
    has_video_target = any(has_word(lower, word) for word in ["video", "clip", "render", "seedance"])
    has_generation_verb = any(has_word(lower, word) for word in ["generate", "render", "create", "make", "start"])
    return has_video_target and has_generation_verb


def promote_latest_prompt_selection(project_name: str, clip_index: Optional[int]) -> Optional[dict]:
    if clip_index is None:
        return None
    try:
        project_data = get_project_data(project_name)
        timeline = project_data.get("timeline", [])
        if clip_index < 0 or clip_index >= len(timeline):
            return None
        clip = timeline[clip_index]
        prompt = matching_prompt(project_data, clip, clip_index)
        prompt_versions = _prompt_versions_for_state(prompt, project_name, clip_index)
        if not prompt_versions:
            return None
        latest_version = prompt_versions[-1]
        clip_key = clip_chat_key(clip.get("clip", ""), clip_index)
        return update_clip_selection(
            project_name,
            clip_key,
            {"active_prompt_version_id": latest_version.get("prompt_version_id")},
        )
    except Exception as exc:
        print(f"Warning: could not promote latest prompt selection for {project_name} clip {clip_index}: {exc}")
        return None


def build_clip_freshness(
    project_name: str,
    *,
    clip: dict,
    clip_key: str,
    feedback_state: Optional[dict],
    prompt_versions: list[dict],
    active_version: Optional[dict],
    assets: dict,
    selected_assets: list[dict],
    pinned_assets: list[dict],
    referenced_frames: list[dict],
    normalized_videos: list[dict],
    active_video: Optional[dict],
    selection: dict,
    selection_stale_reasons: list[str],
) -> dict:
    source_paths = {
        "timeline": project_data_dir(project_name) / "timeline.json",
        "feedback": project_data_dir(project_name) / "feedback.json",
        "prompts": project_data_dir(project_name) / "video_prompts.json",
        "clip_selections": clip_selections_path(project_name),
        "agent_runs": chat_agent_runs_path(project_name),
        "memory": chat_memory_path(project_name),
        "prompt_feedback": prompt_feedback_path(project_name),
        "prompt_lessons": prompt_lessons_path(project_name),
        "prompt_eval_cases": prompt_eval_cases_path(project_name),
        "events": project_events_path(project_name),
    }
    fingerprints = {
        "timeline": stable_json_hash({
            "clip": clip.get("clip"),
            "start_s": clip.get("start_s"),
            "end_s": clip.get("end_s"),
            "duration_s": clip.get("duration_s"),
        }),
        "feedback": stable_json_hash(feedback_state or {}),
        "prompts": stable_json_hash({
            "prompt_version_ids": [version.get("prompt_version_id") for version in prompt_versions],
            "active_prompt_version_id": (active_version or {}).get("prompt_version_id"),
            "active_prompt_text": (active_version or {}).get("video_model_prompt"),
        }),
        "assets": stable_json_hash({
            "selected_assets": selected_assets,
            "pinned_assets": pinned_assets,
            "referenced_frames": referenced_frames,
            "available_asset_ids": sorted(
                stable_state_id("asset", project_name, category, asset.get("path") or asset.get("name"))
                for category, items in (assets or {}).items()
                for asset in items
            ),
        }),
        "videos": stable_json_hash({
            "generated_video_ids": [video.get("generated_video_id") for video in normalized_videos],
            "active_video_id": (active_video or {}).get("generated_video_id"),
        }),
        "selection": stable_json_hash({
            "active_prompt_version_id": selection.get("active_prompt_version_id"),
            "active_generated_video_id": selection.get("active_generated_video_id"),
            "selected_assets": normalize_selection_asset_refs(selection),
        }),
        "prompt_feedback": stable_json_hash(prompt_feedback_summary_by_version(project_name, active_version.get("clip_index", -1) if active_version else -1)),
        "prompt_lessons": stable_json_hash(prompt_learning_store(project_name).list_lessons(clip_key=clip_key, include_archived=True, limit=500)),
        "prompt_eval_cases": stable_json_hash(prompt_eval_case_store(project_name).list_eval_cases(clip_index=active_version.get("clip_index", -1) if active_version else -1, limit=500)),
    }
    fingerprints["state"] = stable_json_hash({
        key: value
        for key, value in fingerprints.items()
        if key != "state"
    })

    return {
        "source_files": {key: str(path) for key, path in source_paths.items()},
        "source_mtimes": {key: file_mtime_iso(path) for key, path in source_paths.items()},
        "derived_from": ["timeline", "feedback", "assets", "prompts", "clip_selections", "prompt_feedback", "prompt_lessons", "prompt_eval_cases", "memory", "agent_runs", "events"],
        "fingerprints": fingerprints,
        "state_hash": fingerprints["state"],
        "stale_reasons": selection_stale_reasons,
    }


def annotate_agent_run_freshness(run: dict, current_freshness: dict) -> dict:
    annotated = dict(run)
    previous = ((run.get("context_summary") or {}).get("freshness_snapshot") or {})
    previous_fingerprints = previous.get("fingerprints") or {}
    current_fingerprints = current_freshness.get("fingerprints") or {}
    stale_reasons = []

    for key in ["timeline", "feedback", "prompts", "assets", "videos", "selection", "prompt_feedback", "prompt_lessons", "prompt_eval_cases"]:
        if previous_fingerprints.get(key) and previous_fingerprints.get(key) != current_fingerprints.get(key):
            stale_reasons.append(f"{key}_changed")

    if previous.get("state_hash") and previous.get("state_hash") != current_freshness.get("state_hash") and not stale_reasons:
        stale_reasons.append("clip_state_changed")

    annotated["freshness"] = {
        "is_stale": bool(stale_reasons),
        "stale_reasons": stale_reasons,
        "captured_state_hash": previous.get("state_hash"),
        "current_state_hash": current_freshness.get("state_hash"),
        "captured_at": previous.get("captured_at"),
    }
    return annotated


def build_learning_state_for_clip(
    project_name: str,
    *,
    clip_index: int,
    clip_key: str,
    query: str,
    active_version: Optional[dict],
) -> dict:
    learning_store = prompt_learning_store(project_name)
    base_state = learning_store.for_clip(clip_key, query=query)
    project_lessons = base_state.get("project") or []
    clip_lessons = base_state.get("clip") or []
    relevant_lessons = base_state.get("relevant") or []
    eval_cases = prompt_eval_case_store(project_name).list_eval_cases(
        clip_index=clip_index,
        limit=20,
    )
    active_version_id = (active_version or {}).get("prompt_version_id")
    feedback_items = [
        item
        for item in prompt_feedback_items_for_clip(project_name, clip_index)
        if not active_version_id or item.get("prompt_version_id") == active_version_id
    ]
    open_issue_count = sum(
        1
        for item in feedback_items
        if item.get("rating") == "negative" and item.get("status") == "open"
    )
    quality_report = (active_version or {}).get("quality_report") or {}
    return {
        **base_state,
        "feedback_count": len(feedback_items),
        "open_issue_count": open_issue_count,
        "lessons": [*project_lessons, *clip_lessons],
        "relevant_lessons": relevant_lessons,
        "eval_cases": eval_cases,
        "latest_learning_report": quality_report.get("learning_eval"),
    }


def build_clip_state(project_name: str, clip_index: int, query: str = "") -> dict:
    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    clip_key = clip_chat_key(clip.get("clip", ""), clip_index)
    selection = clip_selection_for_key(project_name, clip_key)
    selection_stale_reasons = []
    feedback = matching_feedback(project_data, clip, clip_index)
    prompt = matching_prompt(project_data, clip, clip_index)
    prompt_versions = _prompt_versions_for_state(prompt, project_name, clip_index)
    default_version = prompt_versions[-1] if prompt_versions else None
    selected_prompt_version_id = selection.get("active_prompt_version_id")
    if selected_prompt_version_id and query_requests_latest_prompt(query):
        selection_stale_reasons.append("active_prompt_version_overridden_by_latest_query")
        selected_prompt_version_id = None
    selected_version = next(
        (version for version in prompt_versions if version.get("prompt_version_id") == selected_prompt_version_id),
        None,
    ) if selected_prompt_version_id else None
    if selected_prompt_version_id and not selected_version:
        selection_stale_reasons.append("active_prompt_version_missing")
    latest_version = selected_version or default_version
    assets = project_data.get("assets", {})
    asset_lookup = _asset_lookup_by_path(assets)
    selected_assets = [
        _normalize_selected_asset(project_name, asset_path, asset_lookup)
        for asset_path in (latest_version or {}).get("selected_assets", [])
    ]
    pinned_assets = _selected_assets_from_selection(project_name, selection, asset_lookup)
    referenced_frames = _referenced_frames_for_state(feedback, latest_version, project_name)
    generated_videos = generated_videos_for_version(latest_version)
    normalized_videos = [
        {
            **video,
            "generated_video_id": stable_state_id(
                "video",
                project_name,
                clip_key,
                video.get("version"),
                video.get("timestamp"),
                video.get("path") or video.get("url"),
            ),
        }
        for video in generated_videos
    ]
    default_video = normalized_videos[-1] if normalized_videos else None
    selected_video_id = selection.get("active_generated_video_id")
    selected_video = next(
        (video for video in normalized_videos if video.get("generated_video_id") == selected_video_id),
        None,
    ) if selected_video_id else None
    if selected_video_id and not selected_video:
        selection_stale_reasons.append("active_generated_video_missing")
    latest_video = selected_video or default_video
    clip_context = load_clip_context(project_name, clip, clip_index)
    store = memory_store(project_name)
    feedback_items = (feedback or {}).get("feedback_items", [])
    feedback_indexes = {item.get("raw_index") for item in feedback_items}
    recent_jobs = [
        job for job in list_recent_project_jobs(project_name)
        if job.get("clip_index") == clip_index
        or job.get("feedback_index") in feedback_indexes
    ][:10]

    feedback_state = {
        **(feedback or {}),
        "feedback_id": stable_state_id("feedback", project_name, clip_key),
        "feedback_items": [
            {
                **item,
                "feedback_item_id": stable_state_id(
                    "feedback_item",
                    project_name,
                    clip_key,
                    item.get("raw_index"),
                    item.get("timestamp"),
                    item.get("remark"),
                ),
            }
            for item in feedback_items
        ],
    } if feedback else None
    freshness = build_clip_freshness(
        project_name,
        clip=clip,
        clip_key=clip_key,
        feedback_state=feedback_state,
        prompt_versions=prompt_versions,
        active_version=latest_version,
        assets=assets,
        selected_assets=selected_assets,
        pinned_assets=pinned_assets,
        referenced_frames=referenced_frames,
        normalized_videos=normalized_videos,
        active_video=latest_video,
        selection=selection,
        selection_stale_reasons=selection_stale_reasons,
    )
    agent_runs = [
        annotate_agent_run_freshness(run, freshness)
        for run in recent_agent_runs_for_clip(project_name, clip_key)
    ]

    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "clip_key": clip_key,
        "clip_state_id": stable_state_id("clip_state", project_name, clip_key),
        "updated_at": now_iso(),
        "project": {
            "name": project_data.get("project_name", project_name),
            "sequence_name": project_data.get("sequence_name", ""),
            "total_duration_tc": project_data.get("total_duration_tc", ""),
            "total_duration_s": project_data.get("total_duration_s", 0),
            "clip_count": len(timeline),
        },
        "timeline": {
            "clip": clip,
            "adjacent_clips": {
                "previous": timeline[clip_index - 1] if clip_index > 0 else None,
                "next": timeline[clip_index + 1] if clip_index < len(timeline) - 1 else None,
            },
        },
        "feedback_state": feedback_state,
        "active_prompt": {
            "prompt": prompt,
            "prompt_id": prompt_versions[-1]["prompt_id"] if prompt_versions else None,
            "version": latest_version,
            "version_id": latest_version.get("prompt_version_id") if latest_version else None,
            "version_index": latest_version.get("version_index") if latest_version else None,
            "versions": prompt_versions,
            "prompt_ready": bool((latest_version or {}).get("video_model_prompt")),
        },
        "asset_state": {
            "available_assets": {
                category: [_asset_identity(project_name, category, asset) for asset in items]
                for category, items in assets.items()
            },
            "selected_assets": selected_assets,
            "pinned_assets": pinned_assets,
            "referenced_frames": referenced_frames,
            "missing_assets": [asset for asset in [*selected_assets, *pinned_assets] if asset.get("missing")],
        },
        "video_state": {
            "generated_videos": normalized_videos,
            "latest_video": latest_video,
            "active_video": latest_video,
            "active_video_id": latest_video.get("generated_video_id") if latest_video else None,
            "generation_attempts": (latest_version or {}).get("video_generation_attempts", []),
            "latest_attempt": (latest_version or {}).get("latest_video_generation_attempt"),
        },
        "selection_state": {
            "clip_key": clip_key,
            "active_prompt_version_id": selected_prompt_version_id,
            "resolved_prompt_version_id": latest_version.get("prompt_version_id") if latest_version else None,
            "active_generated_video_id": selected_video_id,
            "resolved_generated_video_id": latest_video.get("generated_video_id") if latest_video else None,
            "selected_assets": normalize_selection_asset_refs(selection),
            "selected_asset_ids": selection.get("selected_asset_ids") or selection.get("pinned_asset_ids") or [],
            "selected_asset_paths": selection.get("selected_asset_paths") or selection.get("pinned_asset_paths") or [],
            "pinned_assets": normalize_selection_asset_refs(selection),
            "pinned_asset_ids": selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or [],
            "pinned_asset_paths": selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or [],
            "resolved_assets": pinned_assets,
            "selection_source": "persisted" if selection else "default_latest",
            "stale_reasons": selection_stale_reasons,
            "updated_at": selection.get("updated_at"),
        },
        "analysis_state": {
            "clip_context": clip_context,
            "clip_context_ready": bool(clip_context),
        },
        "memory_state": store.for_clip(clip_key, query=query),
        "learning_state": build_learning_state_for_clip(
            project_name,
            clip_index=clip_index,
            clip_key=clip_key,
            query=query,
            active_version=latest_version,
        ),
        "agent_state": {
            "recent_runs": agent_runs,
            "pending_actions": [
                action
                for run in agent_runs
                for action in run.get("suggested_actions", [])
                if run.get("status") in {"ready", "awaiting_approval", "planned"}
            ][:8],
        },
        "job_state": {
            "recent_jobs": recent_jobs,
            "active_jobs": [job for job in recent_jobs if job.get("status") in {"queued", "running", "cancelling"}],
        },
        "freshness": freshness,
    }


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
    explicit_analysis_phrases = {
        "summarize",
        "summary",
        "describe",
        "understand",
        "analysis",
        "analyze",
        "what happens",
        "what is happening",
        "what do you see",
        "what can you see",
        "look at this",
        "watch this",
    }
    clip_words = {"clip", "shot", "scene", "video", "frame", "frames", "visual", "visible"}
    if any(phrase in lower for phrase in explicit_analysis_phrases):
        return True
    return any(word in lower for word in ["see", "visible", "happening", "shown"]) and any(word in lower for word in clip_words)


def wants_regenerated_clip_summary(text: str) -> bool:
    lower = text.lower()
    regeneration_words = {"regenerate", "refresh", "reanalyze", "re-analyze", "rerun", "re-run", "update"}
    summary_words = {"summary", "summarize", "analysis", "analyze", "context"}
    return any(word in lower for word in regeneration_words) and any(word in lower for word in summary_words)


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
    if clip_context.get("audio_transcript"):
        lines.append(f"Audio transcript: {clip_context.get('audio_transcript')}")
    elif clip_context.get("audio_status") and clip_context.get("audio_status") != "not_configured":
        lines.append(f"Audio status: {clip_context.get('audio_status')}")
    return "\n".join(lines) if lines else "The saved clip context has no summary details yet."


def ensure_clip_context_for_chat(project_name: str, context: dict, provider: str, force: bool = False) -> tuple[Optional[dict], Optional[str]]:
    existing = context.get("clip_context")
    if existing and not force:
        return existing, None
    if provider != "openai":
        return None, "Clip analysis on demand currently requires OpenAI."
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "Clip analysis has not been run yet, and OPENAI_API_KEY is not configured."

    from openai import OpenAI

    clip = context.get("clip") or {}
    feedback = context.get("feedback") or {}
    latest_version = context.get("latest_version") or {}
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
        audio_name=feedback.get("audio_used") or latest_version.get("audio_used"),
        audio_path=feedback.get("audio_path") or latest_version.get("audio_path") or latest_version.get("trimmed_audio_path"),
    )
    context["clip_context"] = clip_context
    return clip_context, None


def _safe_reference_frame_stem(timestamp: str, clip_name: str) -> str:
    safe_timestamp = re.sub(r"[^A-Za-z0-9._-]+", "_", timestamp.strip()).strip("._-") or "timestamp"
    safe_clip = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(clip_name).stem).strip("._-") or "clip"
    return f"ref_{safe_timestamp}_{safe_clip}"


def _reference_frame_output_path(project_name: str, timestamp: str, clip_name: str) -> Path:
    base_dir = project_data_dir(project_name) / "referenced_frames"
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_reference_frame_stem(timestamp, clip_name)
    candidate = base_dir / f"{stem}.jpg"
    if not candidate.exists():
        return candidate
    return candidate


def _resolve_raw_clip_path(project_name: str, clip_name: str) -> Optional[Path]:
    candidates = [
        project_assets_dir(project_name) / "06_clips" / "_raw" / clip_name,
        project_assets_dir(project_name) / "06_clips" / "_final" / clip_name,
        ASSETS_DIR / project_name / "06_clips" / "_raw" / clip_name,
        ASSETS_DIR / project_name / "06_clips" / "_final" / clip_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _feedback_group_matches_clip(group: dict, clip_name: str, clip_index: int) -> bool:
    occurrence = group.get("clip_occurrence")
    return group.get("clip_used") == clip_name and (occurrence is None or occurrence == clip_index)


def _append_unique_referenced_frame(target: dict, ref_frame: dict) -> bool:
    refs = target.setdefault("referenced_frames", [])
    frame_path = ref_frame.get("frame_path")
    for existing in refs:
        if existing.get("frame_path") == frame_path:
            existing.update({key: value for key, value in ref_frame.items() if value not in (None, "")})
            return False
    refs.append(ref_frame)
    return True


def _attach_reference_frame_to_feedback(project_name: str, current_clip_index: int, ref_frame: dict) -> None:
    feedback_path = project_data_dir(project_name) / "feedback.json"
    feedback_data = read_json_file(feedback_path, [])
    if not isinstance(feedback_data, list):
        feedback_data = []

    timeline = read_json_file(project_data_dir(project_name) / "timeline.json", {}).get("video_timeline", [])
    if current_clip_index < 0 or current_clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")
    current_clip_name = timeline[current_clip_index].get("clip")

    group = next(
        (candidate for candidate in feedback_data if _feedback_group_matches_clip(candidate, current_clip_name, current_clip_index)),
        None,
    )
    if group is None:
        group = {
            "clip_used": current_clip_name,
            "clip_occurrence": current_clip_index,
            "previous_clip": None,
            "audio_used": None,
            "feedback_items": [],
        }
        feedback_data.append(group)

    _append_unique_referenced_frame(group, ref_frame)
    feedback_items = group.setdefault("feedback_items", [])
    if feedback_items:
        _append_unique_referenced_frame(feedback_items[0], ref_frame)

    write_json_file(feedback_path, feedback_data)


def extract_reference_frame(
    project_name: str,
    current_clip_index: int,
    timestamp: str,
    reason: str = "",
    attach_to: str = "current_feedback",
) -> dict:
    timestamp = (timestamp or "").strip()
    if not timestamp:
        raise HTTPException(status_code=400, detail="A timestamp is required.")

    ref_sec = parse_timestamp_to_seconds(timestamp)
    if ref_sec is None:
        raise HTTPException(status_code=400, detail=f"Could not parse timestamp '{timestamp}'.")

    timeline_path = project_data_dir(project_name) / "timeline.json"
    timeline_data = read_json_file(timeline_path, {})
    video_timeline = timeline_data.get("video_timeline", [])
    clip_idx, matched_clip = find_matching_clip_occurrence(video_timeline, ref_sec)
    if not matched_clip:
        raise HTTPException(status_code=404, detail=f"Timestamp {timestamp} does not match any timeline clip.")

    clip_name = matched_clip.get("clip")
    clip_start_s = float(matched_clip.get("start_s") or 0.0)
    clip_duration_s = float(matched_clip.get("duration_s") or max(0.0, float(matched_clip.get("end_s") or 0.0) - clip_start_s))
    offset_s = max(0.0, min(ref_sec - clip_start_s, clip_duration_s))
    source_clip_path = _resolve_raw_clip_path(project_name, clip_name)
    if not source_clip_path:
        raise HTTPException(status_code=404, detail=f"Raw clip file not found for {clip_name}.")

    output_frame_path = _reference_frame_output_path(project_name, timestamp, clip_name)
    if not output_frame_path.exists():
        if not extract_frame_at_offset(str(source_clip_path), offset_s, str(output_frame_path)):
            raise HTTPException(status_code=500, detail=f"Could not extract a frame from {clip_name} at {timestamp}.")

    ref_frame = {
        "timestamp": timestamp,
        "reason": reason.strip() or f"Reference frame extracted from {timestamp}",
        "frame_path": str(output_frame_path.resolve()),
        "clip_used": clip_name,
        "clip_occurrence": clip_idx,
        "offset_s": offset_s,
        "source_clip_path": str(source_clip_path.resolve()),
        "attached_to": attach_to,
    }

    if attach_to == "current_feedback":
        _attach_reference_frame_to_feedback(project_name, current_clip_index, ref_frame)

    return {
        "status": "ok",
        "reference_frame": ref_frame,
        "message": f"Extracted frame at {timestamp} from {clip_name} and attached it as a reference.",
    }


REFERENCE_FRAME_TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def parse_explicit_reference_frame_request(text: str) -> Optional[dict]:
    lower = text.lower()
    timestamp_match = REFERENCE_FRAME_TIMESTAMP_PATTERN.search(text)
    if not timestamp_match:
        return None

    has_extract_intent = any(word in lower for word in ["extract", "grab", "capture", "pull", "save"])
    has_reference_intent = any(word in lower for word in ["reference", "attach", "add"])
    mentions_frame = "frame" in lower or "still" in lower
    if not (has_extract_intent and has_reference_intent and mentions_frame):
        return None

    timestamp = timestamp_match.group(0)
    reason = text.strip()
    return {
        "timestamp": timestamp,
        "reason": reason,
        "attach_to": "current_feedback",
    }


def _extract_quoted_value(text: str) -> Optional[str]:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    return match.group(1).strip() if match else None


def _parse_feedback_index(text: str, context: dict) -> Optional[int]:
    patterns = [
        r"(?:feedback|item)\s*#?\s*(\d+)",
        r"#(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    if len(feedback_items) == 1:
        raw_index = feedback_items[0].get("raw_index")
        return int(raw_index) if raw_index is not None else None
    return None


def _parse_prompt_version_id(text: str, context: dict) -> Optional[str]:
    versions = ((context.get("clip_state") or {}).get("active_prompt") or {}).get("versions") or []
    if not versions:
        return None
    lower = text.lower()
    direct_id = re.search(r"\bversion_[a-f0-9]{12}\b", text)
    if direct_id:
        return direct_id.group(0)
    if "latest" in lower:
        return versions[-1].get("prompt_version_id")
    match = re.search(r"(?:prompt\s*)?version\s*#?\s*(\d+)", lower)
    if match:
        requested = int(match.group(1))
        for index in (requested - 1, requested):
            if 0 <= index < len(versions):
                return versions[index].get("prompt_version_id")
    return None


def _asset_candidates_for_context(context: dict) -> list[dict]:
    candidates = []
    for category, assets in (context.get("assets") or {}).items():
        for asset in assets:
            enriched = _asset_identity(
                (context.get("project") or {}).get("name", ""),
                category,
                asset,
            )
            candidates.append(enriched)
    return candidates


def _parse_asset_selector(text: str, context: dict) -> dict:
    quoted = _extract_quoted_value(text)
    direct_id = re.search(r"\basset_[a-f0-9]{12}\b", text)
    if direct_id:
        return {"asset_id": direct_id.group(0)}
    if quoted:
        return {"asset_path": quoted}

    lower = text.lower()
    candidates = _asset_candidates_for_context(context)
    matches = []
    for asset in candidates:
        path = str(asset.get("path") or "")
        name = str(asset.get("name") or "")
        basename = os.path.basename(path or name).lower()
        if basename and basename in lower:
            matches.append(asset)
            continue
        stem = Path(basename).stem.lower()
        if stem and stem in lower:
            matches.append(asset)
    if len(matches) == 1:
        asset = matches[0]
        return {
            "asset_id": asset.get("asset_id"),
            "asset_path": asset.get("path") or asset.get("name"),
        }
    return {}


def _infer_asset_role(text: str) -> str:
    lower = text.lower()
    if "character" in lower or "person" in lower or "face" in lower or "costume" in lower:
        return "character_reference"
    if "style" in lower or "look" in lower or "mood" in lower or "lighting" in lower:
        return "style_reference"
    if "continuity" in lower or "match" in lower:
        return "continuity_reference"
    if "frame" in lower or "still" in lower:
        return "continuity_frame"
    if "prop" in lower or "object" in lower:
        return "prop_reference"
    return "selected_reference"


def _infer_asset_reason(text: str, role: str) -> str:
    quoted = _extract_quoted_value(text)
    if quoted:
        return f"User selected {quoted} as {role}."
    return f"User selected this asset as {role}."


def _state_mutation_actions_from_message(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    actions = []

    if any(word in lower for word in ["set", "use", "select", "switch"]) and "prompt" in lower and "version" in lower:
        version_id = _parse_prompt_version_id(text, context)
        actions.append({
            "type": "set_active_prompt_version",
            "label": "Set Active Prompt Version",
            "prompt_version_id": version_id,
        })

    asset_selector = _parse_asset_selector(text, context)
    asset_words = any(word in lower for word in ["asset", "image", "character", "prop"]) or (
        "reference" in lower and "frame" not in lower and "still" not in lower
    )
    if asset_words and any(word in lower for word in ["attach", "add", "pin", "use", "select"]):
        role = _infer_asset_role(text)
        actions.append({
            "type": "attach_asset",
            "label": "Attach Asset",
            "role": role,
            "reason": _infer_asset_reason(text, role),
            "confidence": 0.86 if asset_selector else 0.45,
            **asset_selector,
        })
    if asset_words and any(word in lower for word in ["detach", "remove", "unpin", "deselect"]):
        actions.append({
            "type": "detach_asset",
            "label": "Detach Asset",
            **asset_selector,
        })

    if "feedback" in lower and any(word in lower for word in ["resolve", "resolved", "done", "fixed", "close", "complete"]):
        actions.append({
            "type": "mark_feedback_resolved",
            "label": "Mark Feedback Resolved",
            "feedback_index": _parse_feedback_index(text, context),
        })

    reference_request = parse_explicit_reference_frame_request(text)
    if reference_request:
        actions.append({
            "type": "add_reference_frame",
            "label": "Add Reference Frame",
            **reference_request,
        })

    return actions


def build_clip_chat_context(project_name: str, clip_index: int, query: str = "") -> dict:
    clip_state = build_clip_state(project_name, clip_index, query=query)
    clip = (clip_state.get("timeline") or {}).get("clip") or {}
    feedback = clip_state.get("feedback_state")
    prompt = (clip_state.get("active_prompt") or {}).get("prompt")
    latest_version = (clip_state.get("active_prompt") or {}).get("version")
    clip_context = (clip_state.get("analysis_state") or {}).get("clip_context")
    key = clip_state.get("clip_key")
    project = clip_state.get("project") or {}

    return {
        "project": project,
        "clip": clip,
        "clip_index": clip_index,
        "clip_key": key,
        "clip_state": clip_state,
        "adjacent_clips": (clip_state.get("timeline") or {}).get("adjacent_clips", {}),
        "feedback": feedback,
        "prompt": prompt,
        "latest_version": latest_version,
        "clip_context": clip_context,
        "assets": (clip_state.get("asset_state") or {}).get("available_assets", {}),
        "memory": clip_state.get("memory_state", {}),
        "learning": clip_state.get("learning_state", {}),
        "agent_runs": (clip_state.get("agent_state") or {}).get("recent_runs", []),
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
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    latest_learning_report = learning.get("latest_learning_report") or {}
    assets_by_category = {
        category: [asset.get("path") for asset in assets[:12]]
        for category, assets in (context.get("assets") or {}).items()
    }

    return json.dumps(
        {
            "project": context.get("project"),
            "clip": context.get("clip"),
            "clip_index": context.get("clip_index"),
            "clip_state_id": (context.get("clip_state") or {}).get("clip_state_id"),
            "adjacent_clips": context.get("adjacent_clips"),
            "feedback_items": feedback_items,
            "selected_assets": selected_assets,
            "asset_state": {
                "selected_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("selected_assets", []),
                "pinned_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("pinned_assets", []),
                "referenced_frames": ((context.get("clip_state") or {}).get("asset_state") or {}).get("referenced_frames", []),
                "missing_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("missing_assets", []),
            },
            "selection_state": (context.get("clip_state") or {}).get("selection_state"),
            "latest_video_prompt": latest_version.get("video_model_prompt"),
            "active_prompt_ids": {
                "prompt_id": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("prompt_id"),
                "version_id": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("version_id"),
                "version_index": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("version_index"),
            },
            "generated_videos": generated_videos_for_version(latest_version),
            "video_state": (context.get("clip_state") or {}).get("video_state"),
            "latest_prompt_explanation": latest_version.get("explanation"),
            "clip_context": context.get("clip_context"),
            "quality_report": latest_version.get("quality_report"),
            "learning_state": {
                "feedback_count": learning.get("feedback_count", 0),
                "open_issue_count": learning.get("open_issue_count", 0),
                "lesson_count": len(learning.get("lessons") or []),
                "relevant_lesson_count": len(learning.get("relevant_lessons") or learning.get("relevant") or []),
                "eval_case_count": len(learning.get("eval_cases") or []),
                "latest_learning_passed": latest_learning_report.get("passed") if latest_learning_report else None,
                "latest_learning_failed_cases": latest_learning_report.get("failed_cases", []) if latest_learning_report else [],
                "relevant_lessons": [
                    {
                        "lesson": item.get("lesson"),
                        "category": item.get("category"),
                        "scope": item.get("scope"),
                        "relevance_score": item.get("relevance_score"),
                    }
                    for item in (learning.get("relevant_lessons") or learning.get("relevant") or [])[:6]
                ],
            },
            "asset_library_sample": assets_by_category,
            "project_memory": project_memory,
            "clip_memory": clip_memory,
            "relevant_memory": relevant_memory,
            "recent_agent_runs": [
                {
                    "id": run.get("id"),
                    "goal": run.get("goal"),
                    "status": run.get("status"),
                    "intent": run.get("intent"),
                    "autonomy_level": run.get("autonomy_level"),
                    "approval_required": run.get("approval_required"),
                    "freshness": run.get("freshness"),
                    "updated_at": run.get("updated_at"),
                }
                for run in context.get("agent_runs", [])[:3]
            ],
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


def normalize_chat_reply_markdown(text: str) -> str:
    """Remove accidental code fences around prompt prose before saving chat replies."""
    if not text:
        return text

    normalized = text.strip()
    full_fence = re.match(r"^```[\w-]*\s*\n([\s\S]*?)\n```$", normalized)
    if full_fence:
        return full_fence.group(1).strip()

    prompt_fence = re.compile(
        r"(^|\n)([^\n`]*(?:prompt|seedance|video model prompt)[^\n`]*:\s*)?\n?```(?:text|markdown|md)?\s*\n([\s\S]*?)\n```",
        re.IGNORECASE,
    )

    def replace_prompt_fence(match: re.Match) -> str:
        prefix = match.group(1) or ""
        label = (match.group(2) or "").strip()
        body = match.group(3).strip()
        if label:
            return f"{prefix}{label}\n{body}"
        return f"{prefix}{body}"

    return prompt_fence.sub(replace_prompt_fence, normalized).strip()


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


def load_chat_agent_runs(project_name: str) -> dict:
    data = read_json_file(chat_agent_runs_path(project_name), {"schema_version": 1, "runs": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "runs": []}
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    return {"schema_version": 1, "runs": runs}


def save_chat_agent_runs(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("runs", [])
    write_json_file(chat_agent_runs_path(project_name), data)


def recent_agent_runs_for_clip(project_name: str, clip_key: str, limit: int = 5) -> list[dict]:
    data = load_chat_agent_runs(project_name)
    runs = [
        run for run in data.get("runs", [])
        if run.get("clip_key") == clip_key
    ]
    runs.sort(key=lambda run: run.get("updated_at", ""), reverse=True)
    return runs[:limit]


def create_chat_agent_run(project_name: str, run: dict) -> dict:
    data = load_chat_agent_runs(project_name)
    now = now_iso()
    persisted = {
        "id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
        **run,
    }
    data.setdefault("runs", []).append(persisted)
    save_chat_agent_runs(project_name, data)
    append_project_event(
        project_name,
        "agent_run_created",
        actor="chat_agent",
        clip_index=persisted.get("clip_index"),
        clip_key=persisted.get("clip_key"),
        entity="agent_run",
        entity_id=persisted.get("id"),
        payload={
            "goal": persisted.get("goal"),
            "intent": persisted.get("intent"),
            "status": persisted.get("status"),
            "autonomy_level": persisted.get("autonomy_level"),
            "approval_required": persisted.get("approval_required"),
            "plan_step_count": len(persisted.get("plan_steps") or []),
            "state_hash": ((persisted.get("context_summary") or {}).get("freshness_snapshot") or {}).get("state_hash"),
        },
    )
    return persisted


def update_chat_agent_run(project_name: str, run_id: str, updates: dict) -> dict:
    data = load_chat_agent_runs(project_name)
    now = now_iso()
    updated_run = {}
    previous_status = None
    for run in data.get("runs", []):
        if run.get("id") == run_id:
            previous_status = run.get("status")
            run.update(updates)
            run["updated_at"] = now
            updated_run = run
            break
    if updated_run:
        save_chat_agent_runs(project_name, data)
        if "status" in updates and updates.get("status") != previous_status:
            append_project_event(
                project_name,
                "agent_run_status_changed",
                actor="chat_agent",
                clip_index=updated_run.get("clip_index"),
                clip_key=updated_run.get("clip_key"),
                entity="agent_run",
                entity_id=run_id,
                payload={
                    "previous_status": previous_status,
                    "status": updated_run.get("status"),
                    "intent": updated_run.get("intent"),
                    "self_evaluation_verdict": (updated_run.get("self_evaluation") or {}).get("verdict"),
                },
            )
    return updated_run


LEARNING_CHAT_ACTION_TYPES = {
    "save_prompt_feedback",
    "suggest_prompt_lesson",
    "approve_prompt_lesson",
    "revise_prompt_from_feedback",
    "run_prompt_learning_eval",
}


def _infer_prompt_feedback_category(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ["wardrobe", "character", "clothes", "costume"]):
        return "wrong_character_or_wardrobe"
    if any(word in lower for word in ["continuity", "same", "preserve", "consistent"]):
        return "continuity_error"
    if any(word in lower for word in ["camera", "pan", "zoom", "dolly", "shot"]):
        return "bad_camera_instruction"
    if any(word in lower for word in ["audio", "dialogue", "voice", "music"]):
        return "bad_audio_or_dialogue"
    if any(word in lower for word in ["invent", "unsupported", "assumption", "prop", "extra"]):
        return "unsupported_assumption"
    if any(word in lower for word in ["vague", "unclear", "specific"]):
        return "too_vague"
    if any(word in lower for word in ["verbose", "long"]):
        return "too_verbose"
    if any(word in lower for word in ["format", "json", "schema"]):
        return "format_error"
    if any(word in lower for word in ["provider", "seedance", "segmind", "incompatible"]):
        return "provider_incompatible"
    return "other"


def _active_prompt_ids_for_context(context: dict) -> tuple[Optional[str], Optional[str]]:
    active_prompt = (context.get("clip_state") or {}).get("active_prompt") or {}
    version = active_prompt.get("version") or {}
    prompt_id = active_prompt.get("prompt_id") or version.get("prompt_id")
    version_id = active_prompt.get("version_id") or version.get("prompt_version_id")
    return prompt_id, version_id


def _active_prompt_feedback_items(context: dict, *, negative_only: bool = False) -> list[dict]:
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    active_prompt_id, active_version_id = _active_prompt_ids_for_context(context)
    items = []
    for item in ((context.get("clip_state") or {}).get("learning_feedback_items") or []):
        if active_version_id and item.get("prompt_version_id") != active_version_id:
            continue
        if active_prompt_id and item.get("prompt_id") != active_prompt_id:
            continue
        if negative_only and item.get("rating") != "negative":
            continue
        items.append(item)
    if items:
        return items
    clip_index = context.get("clip_index")
    project_name = (context.get("project") or {}).get("name")
    if not project_name or clip_index is None:
        return []
    items = prompt_feedback_items_for_clip(project_name, int(clip_index))
    return [
        item for item in items
        if (not active_version_id or item.get("prompt_version_id") == active_version_id)
        and (not negative_only or item.get("rating") == "negative")
    ][: max(1, int(learning.get("feedback_count") or len(items) or 1))]


def _latest_feedback_for_learning_action(project_name: str, context: dict, action: dict) -> Optional[dict]:
    feedback_id = action.get("feedback_id")
    if feedback_id:
        return find_prompt_feedback_item(project_name, str(feedback_id))
    items = _active_prompt_feedback_items(context, negative_only=True) or _active_prompt_feedback_items(context)
    open_items = [item for item in items if item.get("status") == "open"]
    return (open_items or items or [None])[-1]


def infer_learning_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    actions: list[dict] = []
    prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    has_prompt = bool(prompt_version_id and ((context.get("latest_version") or {}).get("video_model_prompt")))
    category = _infer_prompt_feedback_category(text)

    if has_prompt and any(phrase in lower for phrase in [
        "save this as feedback",
        "save prompt feedback",
        "mark prompt as bad",
        "this prompt is bad",
        "prompt is wrong",
        "add feedback",
    ]):
        actions.append({
            "type": "save_prompt_feedback",
            "label": "Save Prompt Feedback",
            "prompt": text.strip(),
            "rating": "negative",
            "categories": [category],
            "severity": 4 if any(word in lower for word in ["bad", "wrong", "failed", "ignored"]) else 3,
            "create_eval_case": any(phrase in lower for phrase in ["eval", "regression", "test case"]),
            "prompt_id": prompt_id,
            "prompt_version_id": prompt_version_id,
        })

    if any(phrase in lower for phrase in ["suggest a lesson", "suggest lesson", "turn this into a lesson"]):
        actions.append({
            "type": "suggest_prompt_lesson",
            "label": "Suggest Lesson",
            "prompt": text.strip(),
        })

    if any(phrase in lower for phrase in ["approve lesson", "save as lesson", "remember this lesson", "app should remember", "remember this"]):
        actions.append({
            "type": "approve_prompt_lesson",
            "label": "Approve Lesson",
            "prompt": text.strip(),
            "lesson": text.strip(),
            "category": category,
            "scope": "project",
        })

    if has_prompt and any(phrase in lower for phrase in [
        "revise prompt",
        "revise the prompt",
        "improve prompt",
        "fix the prompt",
        "rewrite prompt",
    ]):
        actions.append({
            "type": "revise_prompt_from_feedback",
            "label": "Revise Prompt",
            "prompt": text.strip(),
            "prompt_version_id": prompt_version_id,
        })

    if has_prompt and any(phrase in lower for phrase in [
        "run learning eval",
        "learning eval",
        "evaluate learning",
        "check learned",
        "check prompt against lessons",
    ]):
        actions.append({
            "type": "run_prompt_learning_eval",
            "label": "Run Learning Eval",
            "prompt": text.strip(),
            "prompt_version_id": prompt_version_id,
        })

    return actions


def infer_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    suggestions = _state_mutation_actions_from_message(text, context)
    for action in infer_learning_action_suggestions(text, context):
        add_unique_action(suggestions, action)
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
    if user_explicitly_requests_video_generation(text) and latest_version.get("video_model_prompt"):
        action = {"type": "generate_video", "label": "Generate Video"}
        suggestions.append(action)
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
        add_unique_action(actions, {"type": "generate_video", "label": "Generate Video"})

    if context.get("clip_context") and wants_clip_summary_or_analysis(message):
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Regenerate Summary",
            "prompt": "Regenerate the clip summary by analyzing the clip frames and audio again.",
        })

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

    if len(actions) > 5:
        return actions[:5]
    return actions


def _action_goal(action: dict) -> str:
    if action.get("type") == "execute_workflow":
        return "Run the feedback workflow for this clip."
    if action.get("type") == "generate_video":
        return "Generate a video from the latest prompt."
    if action.get("type") == "save_prompt_feedback":
        return "Save prompt feedback for the active prompt version."
    if action.get("type") == "suggest_prompt_lesson":
        return "Suggest an editable lesson from prompt feedback."
    if action.get("type") == "approve_prompt_lesson":
        return "Save an approved prompt lesson for future prompts."
    if action.get("type") == "revise_prompt_from_feedback":
        return "Create a revised prompt version using feedback and lessons."
    if action.get("type") == "run_prompt_learning_eval":
        return "Evaluate the active prompt against lessons and eval cases."
    if action.get("type") == "prepare_video":
        return "Prepare the latest prompt and references for video generation."
    if action.get("type") == "send_message":
        return action.get("prompt") or "Ask a contextual follow-up question."
    return action.get("label") or "Continue the clip chat."


def _step_for_action(index: int, action: dict, autonomy_level: str) -> dict:
    action_type = action.get("type") or "send_message"
    risky = action_type in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}
    if risky and autonomy_level == "approval_required":
        status = "awaiting_approval"
    elif risky and autonomy_level == "full_autopilot":
        status = "dispatch_ready"
    else:
        status = "pending"
    return {
        "id": f"step-{index}",
        "label": action.get("label") or _action_goal(action),
        "status": status,
        "tool": action_type,
        "requires_approval": risky,
        "action": action,
    }


def _asks_for_permission_or_advice(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith(("can you", "could you", "should i", "should we", "can i", "could i", "would you"))


def plan_chat_agent_run(message: str, context: dict, explicit_actions: list[dict], available_actions: list[dict]) -> dict:
    lower = message.lower()
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}

    if wants_previous_last_frame_continuity(message):
        intent = "continuity_workflow"
        confidence = 0.92
    elif any(action.get("type") in {
        "set_active_prompt_version",
        "attach_asset",
        "detach_asset",
        "mark_feedback_resolved",
        "add_reference_frame",
    } for action in explicit_actions):
        intent = "state_mutation"
        confidence = 0.9
    elif any(action.get("type") == "execute_workflow" for action in explicit_actions):
        intent = "workflow_execution"
        confidence = 0.88
    elif any(action.get("type") == "generate_video" for action in explicit_actions):
        intent = "video_generation"
        confidence = 0.86
    elif any(action.get("type") in LEARNING_CHAT_ACTION_TYPES for action in explicit_actions):
        intent = "prompt_learning"
        confidence = 0.84
    elif wants_clip_summary_or_analysis(message):
        intent = "clip_analysis"
        confidence = 0.82
    elif "memory" in lower or "remember" in lower or "save to memory" in lower:
        intent = "memory_update"
        confidence = 0.78
    elif wants_clip_media_gallery(message):
        intent = "asset_review"
        confidence = 0.76
    elif any(phrase in lower for phrase in ["self evaluate", "self-evaluate", "self evaluation", "evaluate state", "check state"]):
        intent = "self_evaluation"
        confidence = 0.78
    elif any(word in lower for word in ["evaluate", "quality", "check prompt", "review prompt", "audit prompt"]):
        intent = "prompt_evaluation"
        confidence = 0.75
    elif "feedback" in lower:
        intent = "feedback_review"
        confidence = 0.74
    else:
        intent = "general_chat"
        confidence = 0.55

    risky_actions = [
        action for action in explicit_actions
        if action.get("type") in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}
    ]
    has_generate_video_action = any(action.get("type") == "generate_video" for action in risky_actions)
    has_learning_action = any(action.get("type") in LEARNING_CHAT_ACTION_TYPES for action in risky_actions)
    direct_execution = not has_generate_video_action and not has_learning_action and not _asks_for_permission_or_advice(message) and (
        wants_autonomous_execution(message)
        or any(has_word(lower, word) for word in ["render", "seedance"])
    )
    if risky_actions and direct_execution:
        autonomy_level = "full_autopilot"
        approval_required = False
        status = "ready"
    elif risky_actions:
        autonomy_level = "approval_required"
        approval_required = True
        status = "awaiting_approval"
    elif available_actions:
        autonomy_level = "suggest"
        approval_required = False
        status = "planned"
    else:
        autonomy_level = "manual"
        approval_required = False
        status = "planned"

    required_actions = [dict(action) for action in explicit_actions]
    action_plan = required_actions or [
        action for action in available_actions
        if action.get("type") in {"execute_workflow", "generate_video", "prepare_video", *LEARNING_CHAT_ACTION_TYPES}
    ][:2]

    steps = []
    if intent in {"general_chat", "feedback_review"}:
        steps.append({
            "id": "step-1",
            "label": "Answer from current clip context and saved project state.",
            "status": "completed",
            "tool": "chat_reply",
            "requires_approval": False,
            "action": None,
        })

    safe_step_by_intent = {
        "clip_analysis": ("inspect_clip_context", "Inspect saved clip context and analysis status."),
        "asset_review": ("search_assets", "Search attached clip assets and project media."),
        "memory_update": ("add_memory", "Persist any durable preference captured from chat."),
        "prompt_evaluation": ("evaluate_prompt", "Evaluate the latest prompt against feedback and available context."),
        "self_evaluation": ("self_evaluate", "Evaluate stale state, feedback coverage, prompt quality, assets, and video status."),
    }
    if intent in safe_step_by_intent:
        tool, label = safe_step_by_intent[intent]
        steps.append({
            "id": f"step-{len(steps) + 1}",
            "label": label,
            "status": "pending",
            "tool": tool,
            "requires_approval": False,
            "action": None,
        })

    for index, action in enumerate(action_plan, start=len(steps) + 1):
        steps.append(_step_for_action(index, action, autonomy_level))

    if not steps and feedback_items:
        steps.append({
            "id": "step-1",
            "label": "Keep the workflow action available for this feedback item.",
            "status": "pending",
            "tool": "execute_workflow",
            "requires_approval": True,
            "action": {
                "type": "execute_workflow",
                "label": "Run Workflow",
                "feedback_index": feedback_items[0].get("raw_index"),
            },
        })

    goal = message.strip()
    if not goal:
        goal = _action_goal(required_actions[0]) if required_actions else "Continue the clip chat."

    return {
        "goal": goal,
        "clip_index": context.get("clip_index"),
        "clip_key": context.get("clip_key"),
        "status": status,
        "intent": intent,
        "confidence": confidence,
        "autonomy_level": autonomy_level,
        "approval_required": approval_required,
        "plan_steps": steps,
        "required_actions": required_actions,
        "available_actions": available_actions,
        "suggested_actions": available_actions,
        "tool_results": [],
        "errors": [],
        "context_summary": {
            "feedback_count": len(feedback_items),
            "prompt_ready": bool(latest_version.get("video_model_prompt")),
            "clip_context_ready": bool(context.get("clip_context")),
            "freshness_snapshot": {
                **((context.get("clip_state") or {}).get("freshness") or {}),
                "captured_at": now_iso(),
            },
        },
    }


def apply_agent_run_action_policy(actions: list[dict], agent_run: dict) -> list[dict]:
    if not agent_run.get("approval_required"):
        return actions
    gated_actions = []
    for action in actions:
        next_action = dict(action)
        if next_action.get("type") in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}:
            next_action.pop("autonomous", None)
        gated_actions.append(next_action)
    return gated_actions


def strip_risky_autonomous_actions(actions: list[dict]) -> list[dict]:
    stripped = []
    for action in actions:
        next_action = dict(action)
        if next_action.get("type") in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}:
            next_action.pop("autonomous", None)
        stripped.append(next_action)
    return stripped


def search_assets_for_agent(context: dict, query: str, limit: int = 8) -> dict:
    query_tokens = {
        token for token in re.findall(r"[a-zA-Z0-9_'-]{2,}", query.lower())
        if token not in {"show", "list", "view", "see", "open", "asset", "assets", "media", "reference", "references"}
    }
    matches = []
    for category, assets in (context.get("assets") or {}).items():
        for asset in assets:
            haystack = f"{category} {asset.get('name', '')} {asset.get('path', '')} {asset.get('type', '')}".lower()
            score = sum(1 for token in query_tokens if token in haystack)
            if score or not query_tokens:
                matches.append({
                    "category": category,
                    "name": asset.get("name"),
                    "path": asset.get("path"),
                    "url": asset.get("url"),
                    "type": asset.get("type"),
                    "score": score,
                })
    matches.sort(key=lambda item: (item.get("score", 0), item.get("name") or ""), reverse=True)
    return {
        "status": "ok",
        "tool": "search_assets",
        "message": f"Found {len(matches[:limit])} matching asset(s).",
        "matches": matches[:limit],
    }


def inspect_clip_context_for_agent(context: dict) -> dict:
    clip_context = context.get("clip_context")
    if not clip_context:
        return {
            "status": "missing",
            "tool": "inspect_clip_context",
            "message": "No saved clip analysis exists yet.",
        }
    return {
        "status": "ok",
        "tool": "inspect_clip_context",
        "message": "Loaded saved clip analysis.",
        "summary": summarize_clip_context(clip_context),
        "clip_context_path": clip_context.get("clip_context_path"),
        "analysis_status": clip_context.get("status"),
    }


def memory_update_result_for_agent(saved_memories: list[dict]) -> dict:
    if not saved_memories:
        return {
            "status": "missing",
            "tool": "add_memory",
            "message": "No durable memory note was detected in this turn.",
        }
    return {
        "status": "ok",
        "tool": "add_memory",
        "message": f"Saved {len(saved_memories)} memory note(s).",
        "memory_ids": [item.get("id") for item in saved_memories],
        "memories": [{"id": item.get("id"), "text": item.get("text"), "scope": item.get("scope")} for item in saved_memories],
    }


def evaluate_prompt_for_agent(context: dict) -> dict:
    latest_version = context.get("latest_version") or {}
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    if not prompt_text:
        return {
            "status": "missing",
            "tool": "evaluate_prompt",
            "message": "No generated prompt exists yet for this clip.",
        }

    prompt_lower = prompt_text.lower()
    missed_feedback = []
    for item in feedback_items:
        remark = str(item.get("remark") or "").strip()
        tokens = [
            token for token in re.findall(r"[a-zA-Z0-9_'-]{4,}", remark.lower())
            if token not in {"make", "this", "that", "with", "from", "should", "clip", "shot", "video"}
        ]
        if tokens and not any(token in prompt_lower for token in tokens[:8]):
            missed_feedback.append(remark)

    quality_report = latest_version.get("quality_report") or {}
    suggestions = list(quality_report.get("suggestions") or [])
    if missed_feedback:
        suggestions.append("Review whether the prompt explicitly addresses every feedback remark.")
    if not latest_version.get("selected_assets"):
        suggestions.append("No selected assets are attached to this prompt version.")

    return {
        "status": "ok",
        "tool": "evaluate_prompt",
        "message": "Evaluated the latest prompt against feedback and handoff metadata.",
        "prompt_ready": True,
        "feedback_count": len(feedback_items),
        "missed_feedback": missed_feedback[:5],
        "quality_passed": quality_report.get("passed"),
        "suggestions": suggestions[:6],
    }


def clip_index_for_feedback_index(project_name: str, feedback_index: Optional[int]) -> Optional[int]:
    if feedback_index is None:
        return None
    project_data = get_project_data(project_name)
    for group in project_data.get("feedback", []):
        for item in group.get("feedback_items", []):
            if item.get("raw_index") == feedback_index:
                occurrence = group.get("clip_occurrence")
                if occurrence is not None:
                    return int(occurrence)
                clip_used = group.get("clip_used")
                for index, clip in enumerate(project_data.get("timeline", [])):
                    if clip.get("clip") == clip_used:
                        return index
    return None


def evaluate_agent_state_for_clip(project_name: str, clip_index: int, *, trigger: str = "manual", agent_run_id: Optional[str] = None) -> dict:
    state = build_clip_state(project_name, clip_index)
    freshness = state.get("freshness") or {}
    feedback_items = ((state.get("feedback_state") or {}).get("feedback_items") or [])
    active_prompt = state.get("active_prompt") or {}
    latest_version = active_prompt.get("version") or {}
    asset_state = state.get("asset_state") or {}
    video_state = state.get("video_state") or {}
    selection_state = state.get("selection_state") or {}

    unresolved_feedback = [
        item for item in feedback_items
        if not item.get("resolved") and item.get("status") != "resolved"
    ]
    prompt_report = evaluate_prompt_for_agent({
        "latest_version": latest_version,
        "feedback": {"feedback_items": feedback_items},
    })
    stale_runs = [
        run for run in (state.get("agent_state") or {}).get("recent_runs", [])
        if (run.get("freshness") or {}).get("is_stale")
        and (not agent_run_id or run.get("id") != agent_run_id)
    ]
    selected_refs = selection_state.get("selected_assets") or selection_state.get("pinned_assets") or []
    pinned_assets = asset_state.get("pinned_assets") or []
    missing_assets = asset_state.get("missing_assets") or []
    generated_videos = video_state.get("generated_videos") or []
    latest_attempt = video_state.get("latest_attempt") or {}
    active_video = video_state.get("active_video") or video_state.get("latest_video")

    checks = {
        "stale_state": {
            "status": "fail" if freshness.get("stale_reasons") or stale_runs else "pass",
            "stale_reasons": freshness.get("stale_reasons", []),
            "stale_run_count": len(stale_runs),
        },
        "feedback_coverage": {
            "status": "pass" if not unresolved_feedback else "needs_attention",
            "total_feedback": len(feedback_items),
            "unresolved_feedback": [
                {
                    "feedback_index": item.get("raw_index"),
                    "category": item.get("category"),
                    "remark": item.get("remark"),
                }
                for item in unresolved_feedback[:8]
            ],
        },
        "prompt_quality": {
            "status": (
                "missing" if prompt_report.get("status") == "missing"
                else "needs_attention" if prompt_report.get("missed_feedback") or prompt_report.get("quality_passed") is False
                else "pass"
            ),
            "prompt_version_id": active_prompt.get("version_id"),
            "prompt_ready": bool(active_prompt.get("prompt_ready")),
            "missed_feedback": prompt_report.get("missed_feedback", []),
            "suggestions": prompt_report.get("suggestions", []),
        },
        "asset_completeness": {
            "status": "fail" if missing_assets else "needs_attention" if not selected_refs and not pinned_assets else "pass",
            "structured_asset_count": len(selected_refs),
            "resolved_asset_count": len(pinned_assets),
            "missing_assets": missing_assets,
        },
        "video_status": {
            "status": (
                "failed" if latest_attempt.get("status") in {"failed", "recovery_failed"}
                else "pass" if active_video
                else "missing"
            ),
            "active_video_id": video_state.get("active_video_id"),
            "generated_video_count": len(generated_videos),
            "latest_attempt": latest_attempt,
        },
    }

    if checks["prompt_quality"]["status"] == "missing":
        next_action = {"type": "run_workflow", "reason": "No active prompt exists for this clip."}
    elif checks["stale_state"]["status"] == "fail":
        next_action = {"type": "review_state", "reason": "State changed since earlier agent work."}
    elif checks["asset_completeness"]["status"] in {"fail", "needs_attention"}:
        next_action = {"type": "attach_asset", "reason": "Assets are missing or not structured for this clip."}
    elif checks["prompt_quality"]["status"] == "needs_attention":
        next_action = {"type": "review_prompt", "reason": "Prompt may not cover all feedback or quality checks."}
    elif checks["video_status"]["status"] in {"missing", "failed"}:
        next_action = {"type": "generate_video", "reason": "Prompt is ready but no successful active video is available."}
    elif checks["feedback_coverage"]["status"] == "needs_attention":
        next_action = {"type": "mark_feedback_resolved", "reason": "Generated state exists; unresolved feedback should be reviewed."}
    else:
        next_action = {"type": "none", "reason": "Prompt, assets, feedback, and video state look ready."}

    blocking_statuses = {check["status"] for check in checks.values() if check.get("status") in {"fail", "failed", "missing"}}
    needs_attention = any(check.get("status") == "needs_attention" for check in checks.values())
    verdict = "blocked" if blocking_statuses else "needs_attention" if needs_attention else "passed"
    return {
        "status": "ok",
        "tool": "self_evaluate",
        "trigger": trigger,
        "clip_index": clip_index,
        "clip_key": state.get("clip_key"),
        "verdict": verdict,
        "message": f"Self-evaluation {verdict}. Next action: {next_action['type']}.",
        "checks": checks,
        "next_action": next_action,
        "state_hash": freshness.get("state_hash"),
        "evaluated_at": now_iso(),
    }


def self_evaluation_for_agent(project_name: str, context: dict, agent_run_id: Optional[str] = None) -> dict:
    return evaluate_agent_state_for_clip(
        project_name,
        int(context.get("clip_index") or 0),
        trigger="chat",
        agent_run_id=agent_run_id,
    )


def set_active_prompt_version_for_agent(project_name: str, context: dict, action: dict) -> dict:
    prompt_version_id = action.get("prompt_version_id")
    versions = ((context.get("clip_state") or {}).get("active_prompt") or {}).get("versions") or []
    match = next((version for version in versions if version.get("prompt_version_id") == prompt_version_id), None)
    if not prompt_version_id or not match:
        return {
            "status": "error",
            "tool": "set_active_prompt_version",
            "message": "Could not find the requested prompt version for this clip.",
        }

    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"active_prompt_version_id": prompt_version_id},
    )
    append_project_event(
        project_name,
        "active_prompt_version_set",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="prompt_version",
        entity_id=prompt_version_id,
        payload={
            "version_index": match.get("version_index"),
            "prompt_id": match.get("prompt_id"),
        },
    )
    return {
        "status": "ok",
        "tool": "set_active_prompt_version",
        "message": f"Set active prompt version to index {match.get('version_index')}.",
        "prompt_version_id": prompt_version_id,
        "version_index": match.get("version_index"),
        "selection": selection,
        "mutates_state": True,
    }


def _current_selection_lists(context: dict) -> tuple[list[str], list[str]]:
    selection = (context.get("clip_state") or {}).get("selection_state") or {}
    asset_ids = list(selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or [])
    asset_paths = list(selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or [])
    return asset_ids, asset_paths


def _current_selection_asset_refs(context: dict) -> list[dict]:
    selection = (context.get("clip_state") or {}).get("selection_state") or {}
    return normalize_selection_asset_refs(selection)


def _resolve_asset_action_target(context: dict, action: dict) -> dict:
    asset_id = action.get("asset_id")
    asset_path = action.get("asset_path")
    for asset in _asset_candidates_for_context(context):
        candidate_id = asset.get("asset_id")
        candidate_path = asset.get("path") or asset.get("name")
        if asset_id and candidate_id == asset_id:
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
        if asset_path and str(candidate_path) == str(asset_path):
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
        if asset_path and os.path.basename(str(candidate_path)) == os.path.basename(str(asset_path)):
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
    if asset_path:
        return {"asset_id": asset_id, "asset_path": asset_path, "asset": None}
    return {}


def attach_asset_for_agent(project_name: str, context: dict, action: dict) -> dict:
    target = _resolve_asset_action_target(context, action)
    if not target:
        return {
            "status": "error",
            "tool": "attach_asset",
            "message": "Could not identify which asset to attach.",
        }
    refs = _current_selection_asset_refs(context)
    target_id = target.get("asset_id")
    target_path = target.get("asset_path")
    next_ref = {
        "asset_id": target_id,
        "path": target_path,
        "role": action.get("role") or "selected_reference",
        "reason": action.get("reason") or "Selected from chat.",
        "confidence": float(action.get("confidence") if action.get("confidence") is not None else 0.82),
        "source": "chat_agent",
    }
    replaced = False
    for index, ref in enumerate(refs):
        if (target_id and ref.get("asset_id") == target_id) or (target_path and ref.get("path") == target_path):
            refs[index] = {**ref, **next_ref}
            replaced = True
            break
    if not replaced:
        refs.append(next_ref)
    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"selected_assets": refs},
    )
    append_project_event(
        project_name,
        "asset_attached",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="asset",
        entity_id=target.get("asset_id"),
        payload={
            "asset_path": target.get("asset_path"),
            "role": next_ref["role"],
            "reason": next_ref["reason"],
            "confidence": next_ref["confidence"],
        },
    )
    return {
        "status": "ok",
        "tool": "attach_asset",
        "message": f"Attached asset {os.path.basename(str(target.get('asset_path') or target.get('asset_id')))} to this clip.",
        "asset_id": target.get("asset_id"),
        "asset_path": target.get("asset_path"),
        "role": next_ref["role"],
        "reason": next_ref["reason"],
        "confidence": next_ref["confidence"],
        "selection": selection,
        "mutates_state": True,
    }


def detach_asset_for_agent(project_name: str, context: dict, action: dict) -> dict:
    target = _resolve_asset_action_target(context, action)
    if not target:
        return {
            "status": "error",
            "tool": "detach_asset",
            "message": "Could not identify which asset to detach.",
        }
    target_id = target.get("asset_id")
    target_path = target.get("asset_path")
    refs = _current_selection_asset_refs(context)
    next_refs = [
        ref for ref in refs
        if not (
            (target_id and ref.get("asset_id") == target_id)
            or (target_path and ref.get("path") == target_path)
            or (target_path and os.path.basename(str(ref.get("path") or "")) == os.path.basename(str(target_path)))
        )
    ]
    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"selected_assets": next_refs},
    )
    append_project_event(
        project_name,
        "asset_detached",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="asset",
        entity_id=target_id,
        payload={
            "asset_path": target_path,
            "remaining_asset_count": len(next_refs),
        },
    )
    return {
        "status": "ok",
        "tool": "detach_asset",
        "message": f"Detached asset {os.path.basename(str(target_path or target_id))} from this clip.",
        "asset_id": target_id,
        "asset_path": target_path,
        "selection": selection,
        "mutates_state": True,
    }


def mark_feedback_resolved_for_agent(project_name: str, context: dict, action: dict) -> dict:
    feedback_index = action.get("feedback_index")
    if feedback_index is None:
        return {
            "status": "error",
            "tool": "mark_feedback_resolved",
            "message": "Could not identify which feedback item to mark resolved.",
        }

    feedback_path = project_data_dir(project_name) / "feedback.json"
    feedback_data = read_json_file(feedback_path, [])
    if not isinstance(feedback_data, list):
        return {
            "status": "error",
            "tool": "mark_feedback_resolved",
            "message": "Feedback data is not available for this project.",
        }

    raw_index_counter = 0
    for group in feedback_data:
        for item in group.get("feedback_items", []):
            if raw_index_counter == int(feedback_index):
                item["status"] = "resolved"
                item["resolved"] = True
                item["resolved_at"] = now_iso()
                item["resolved_by"] = "chat_agent"
                write_json_file(feedback_path, feedback_data)
                append_project_event(
                    project_name,
                    "feedback_resolved",
                    actor="chat_agent",
                    clip_index=context.get("clip_index"),
                    clip_key=context.get("clip_key"),
                    entity="feedback_item",
                    entity_id=str(feedback_index),
                    payload={
                        "feedback_index": feedback_index,
                        "category": item.get("category"),
                        "remark": item.get("remark"),
                    },
                )
                return {
                    "status": "ok",
                    "tool": "mark_feedback_resolved",
                    "message": f"Marked feedback #{feedback_index} as resolved.",
                    "feedback_index": feedback_index,
                    "mutates_state": True,
                }
            raw_index_counter += 1

    return {
        "status": "error",
        "tool": "mark_feedback_resolved",
        "message": f"Feedback #{feedback_index} was not found.",
    }


def add_reference_frame_for_agent(project_name: str, context: dict, action: dict, existing_tool_results: list[dict]) -> dict:
    for result in existing_tool_results:
        if result.get("status") == "ok" and result.get("reference_frame"):
            append_project_event(
                project_name,
                "reference_frame_added",
                actor="chat_agent",
                clip_index=context.get("clip_index"),
                clip_key=context.get("clip_key"),
                entity="reference_frame",
                entity_id=(result.get("reference_frame") or {}).get("frame_path"),
                payload=result.get("reference_frame") or {},
            )
            return {
                **result,
                "tool": "add_reference_frame",
                "mutates_state": True,
            }
    try:
        result = extract_reference_frame(
            project_name=project_name,
            current_clip_index=int(context.get("clip_index") or 0),
            timestamp=str(action.get("timestamp") or ""),
            reason=str(action.get("reason") or ""),
            attach_to=str(action.get("attach_to") or "current_feedback"),
        )
        append_project_event(
            project_name,
            "reference_frame_added",
            actor="chat_agent",
            clip_index=context.get("clip_index"),
            clip_key=context.get("clip_key"),
            entity="reference_frame",
            entity_id=(result.get("reference_frame") or {}).get("frame_path"),
            payload=result.get("reference_frame") or {},
        )
        return {
            **result,
            "tool": "add_reference_frame",
            "mutates_state": True,
        }
    except HTTPException as exc:
        return {"status": "error", "tool": "add_reference_frame", "message": exc.detail}


def save_prompt_feedback_for_agent(project_name: str, context: dict, action: dict, message: str) -> dict:
    prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    if not prompt_id or not prompt_version_id:
        return {"status": "missing", "tool": "save_prompt_feedback", "message": "No active prompt version is available for feedback."}
    comment = str(action.get("comment") or action.get("prompt") or message or "").strip()
    correction = str(action.get("correction") or "").strip()
    remember_note = str(action.get("remember_note") or "").strip()
    rating = action.get("rating") if action.get("rating") in PROMPT_FEEDBACK_RATINGS else "negative"
    payload = {
        "clip_index": int(context.get("clip_index") or 0),
        "clip_key": context.get("clip_key") or "",
        "prompt_id": prompt_id,
        "prompt_version_id": prompt_version_id,
        "rating": rating,
        "categories": action.get("categories") or [_infer_prompt_feedback_category(comment or correction or remember_note)],
        "severity": int(action.get("severity") or (4 if rating == "negative" else 1)),
        "comment": comment,
        "correction": correction,
        "remember_note": remember_note,
        "create_eval_case": bool(action.get("create_eval_case")),
        "status": action.get("status"),
    }
    try:
        created = create_prompt_feedback_item(project_name, payload, actor="chat_agent")
    except HTTPException as exc:
        return {"status": "error", "tool": "save_prompt_feedback", "message": exc.detail}
    item = created["item"]
    return {
        "status": "ok",
        "tool": "save_prompt_feedback",
        "message": "Saved prompt feedback for the active prompt version.",
        "feedback_id": item.get("id"),
        "eval_case_id": (created.get("eval_case") or {}).get("id"),
        "mutates_state": True,
    }


def suggest_prompt_lesson_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    item = _latest_feedback_for_learning_action(project_name, context, action)
    if not item:
        return {"status": "missing", "tool": "suggest_prompt_lesson", "message": "No prompt feedback is available to turn into a lesson."}
    try:
        suggestion = _suggest_prompt_lesson(project_name, item, provider)  # type: ignore[arg-type]
    except HTTPException as exc:
        return {"status": "error", "tool": "suggest_prompt_lesson", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "suggest_prompt_lesson", "message": str(exc)}
    return {
        "status": "ok",
        "tool": "suggest_prompt_lesson",
        "message": "Suggested an editable prompt lesson. Review it before saving.",
        "feedback_id": item.get("id"),
        "suggestion": suggestion,
    }


def _lesson_text_from_action(action: dict, message: str) -> str:
    for key in ("lesson", "remember_note", "prompt"):
        text = str(action.get(key) or "").strip()
        if text:
            return re.sub(
                r"^(approve lesson|save as lesson|remember this lesson|remember this|app should remember)[:\s-]*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip() or text
    return message.strip()


def approve_prompt_lesson_for_agent(project_name: str, context: dict, action: dict, message: str) -> dict:
    lesson_text = _lesson_text_from_action(action, message)
    if not lesson_text:
        return {"status": "missing", "tool": "approve_prompt_lesson", "message": "No lesson text was provided."}
    source_feedback_ids = [
        str(value)
        for value in (action.get("source_feedback_ids") or [])
        if value
    ]
    if not source_feedback_ids:
        feedback_item = _latest_feedback_for_learning_action(project_name, context, action)
        if feedback_item:
            source_feedback_ids = [feedback_item.get("id")]
    scope = action.get("scope") if action.get("scope") in {"project", "clip"} else "project"
    try:
        lesson = prompt_learning_store(project_name).add_lesson(
            lesson_text,
            scope=scope,
            clip_key=context.get("clip_key") if scope == "clip" else None,
            category=str(action.get("category") or _infer_prompt_feedback_category(lesson_text)),
            source_feedback_ids=source_feedback_ids,
            confidence=float(action.get("confidence") or 0.84),
        )
    except ValueError as exc:
        return {"status": "error", "tool": "approve_prompt_lesson", "message": str(exc)}
    append_project_event(
        project_name,
        "prompt_lesson_created",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=lesson.get("clip_key"),
        entity="prompt_lesson",
        entity_id=lesson.get("id"),
        payload={
            "scope": lesson.get("scope"),
            "category": lesson.get("category"),
            "source_feedback_ids": lesson.get("source_feedback_ids", []),
        },
    )
    return {
        "status": "ok",
        "tool": "approve_prompt_lesson",
        "message": "Lesson saved. The app will remember and apply this approved lesson in future prompt work.",
        "lesson_id": lesson.get("id"),
        "lesson": lesson,
        "mutates_state": True,
    }


def revise_prompt_from_feedback_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    _prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    prompt_version_id = str(action.get("prompt_version_id") or prompt_version_id or "")
    if not prompt_version_id:
        return {"status": "missing", "tool": "revise_prompt_from_feedback", "message": "No active prompt version is available to revise."}
    feedback_ids = [
        str(value)
        for value in (action.get("feedback_ids") or [])
        if value
    ]
    if not feedback_ids:
        feedback_ids = [
            item.get("id")
            for item in _active_prompt_feedback_items(context, negative_only=True)
            if item.get("status") == "open" and item.get("id")
        ]
    lesson_ids = [
        str(value)
        for value in (action.get("lesson_ids") or [])
        if value
    ]
    if not lesson_ids:
        learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
        lesson_ids = [
            lesson.get("id")
            for lesson in (learning.get("relevant_lessons") or learning.get("relevant") or [])
            if lesson.get("id")
        ][:6]
    try:
        response = revise_prompt_from_feedback(
            project_name,
            prompt_version_id,
            PromptRevisionRequest(
                clip_index=int(context.get("clip_index") or 0),
                feedback_ids=feedback_ids,
                lesson_ids=lesson_ids,
                provider=provider,  # type: ignore[arg-type]
            ),
        )
    except HTTPException as exc:
        return {"status": "error", "tool": "revise_prompt_from_feedback", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "revise_prompt_from_feedback", "message": str(exc)}
    append_project_event(
        project_name,
        "prompt_version_revised_from_feedback",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="prompt_version",
        entity_id=(response.get("prompt_version") or {}).get("prompt_version_id"),
        payload={
            "source_prompt_version_id": prompt_version_id,
            "feedback_ids": feedback_ids,
            "lesson_ids": lesson_ids,
        },
    )
    return {
        "status": "ok",
        "tool": "revise_prompt_from_feedback",
        "message": "New prompt version created from feedback and approved lessons.",
        "prompt_version": response.get("prompt_version"),
        "quality_report": response.get("quality_report"),
        "learning_report": response.get("learning_report"),
        "mutates_state": True,
    }


def run_prompt_learning_eval_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    latest_version = context.get("latest_version") or {}
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    if not prompt_text:
        return {"status": "missing", "tool": "run_prompt_learning_eval", "message": "No active prompt text is available to evaluate."}
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    eval_cases = learning.get("eval_cases") or []
    lessons = learning.get("relevant_lessons") or learning.get("relevant") or learning.get("lessons") or []
    try:
        client = _client_for_prompt_provider(provider)  # type: ignore[arg-type]
        model = _default_model_for_provider(provider)
        report = run_learning_eval(
            provider=provider,
            client=client,
            model=model,
            prompt=prompt_text,
            eval_cases=eval_cases,
            lessons=lessons,
        )
    except HTTPException as exc:
        return {"status": "error", "tool": "run_prompt_learning_eval", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "run_prompt_learning_eval", "message": str(exc)}
    return {
        "status": "ok",
        "tool": "run_prompt_learning_eval",
        "message": "Ran learning eval for the active prompt.",
        "learning_eval": report,
    }


def execute_learning_action_for_agent(project_name: str, context: dict, action: dict, message: str, provider: str) -> dict:
    action_type = action.get("type")
    if action_type == "save_prompt_feedback":
        return save_prompt_feedback_for_agent(project_name, context, action, message)
    if action_type == "suggest_prompt_lesson":
        return suggest_prompt_lesson_for_agent(project_name, context, action, provider)
    if action_type == "approve_prompt_lesson":
        return approve_prompt_lesson_for_agent(project_name, context, action, message)
    if action_type == "revise_prompt_from_feedback":
        return revise_prompt_from_feedback_for_agent(project_name, context, action, provider)
    if action_type == "run_prompt_learning_eval":
        return run_prompt_learning_eval_for_agent(project_name, context, action, provider)
    return {"status": "error", "tool": action_type or "learning_action", "message": "Unsupported learning action."}


def execute_safe_agent_run_tools(project_name: str, run: dict, context: dict, message: str, saved_memories: list[dict], provider: str = "openai") -> dict:
    tool_results = list(run.get("tool_results") or [])
    errors = list(run.get("errors") or [])
    changed = False

    for step in run.get("plan_steps", []):
        if step.get("status") != "pending":
            continue
        tool = step.get("tool")
        try:
            if tool == "search_assets":
                result = search_assets_for_agent(context, message)
            elif tool == "inspect_clip_context":
                result = inspect_clip_context_for_agent(context)
            elif tool == "add_memory":
                result = memory_update_result_for_agent(saved_memories)
            elif tool == "evaluate_prompt":
                result = evaluate_prompt_for_agent(context)
            elif tool == "self_evaluate":
                result = self_evaluation_for_agent(project_name, context, run.get("id"))
            elif tool == "set_active_prompt_version":
                result = set_active_prompt_version_for_agent(project_name, context, step.get("action") or {})
            elif tool == "attach_asset":
                result = attach_asset_for_agent(project_name, context, step.get("action") or {})
            elif tool == "detach_asset":
                result = detach_asset_for_agent(project_name, context, step.get("action") or {})
            elif tool == "mark_feedback_resolved":
                result = mark_feedback_resolved_for_agent(project_name, context, step.get("action") or {})
            elif tool == "add_reference_frame":
                result = add_reference_frame_for_agent(project_name, context, step.get("action") or {}, tool_results)
            elif tool in LEARNING_CHAT_ACTION_TYPES:
                result = execute_learning_action_for_agent(project_name, context, step.get("action") or {"type": tool}, message, provider)
            elif tool in {"chat_reply", "send_message"}:
                result = {"status": "ok", "tool": tool, "message": "Chat reply completed."}
            else:
                continue
        except Exception as exc:
            result = {"status": "error", "tool": tool, "message": str(exc)}

        tool_results.append(result)
        if result.get("status") == "error":
            step["status"] = "failed"
            errors.append(str(result.get("message") or result))
        elif result.get("status") == "missing":
            step["status"] = "blocked"
        else:
            step["status"] = "completed"
        changed = True

    if changed:
        run["tool_results"] = tool_results
        run["errors"] = errors
        if errors:
            run["status"] = "failed"
        elif any(step.get("status") == "awaiting_approval" for step in run.get("plan_steps", [])):
            run["status"] = "awaiting_approval"
        elif any(step.get("status") == "dispatch_ready" for step in run.get("plan_steps", [])):
            run["status"] = "ready"
        elif all(step.get("status") in {"completed", "blocked"} for step in run.get("plan_steps", [])):
            run["status"] = "completed"
    return run


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
            return "A generated prompt is ready. Use Generate Video to render a versioned clip from the latest prompt."
        return "No generated prompt exists yet. Run the feedback workflow first, then prepare video generation."

    return (
        f"Clip #{context.get('clip_index', 0) + 1}/{project.get('clip_count')}: {clip.get('clip')}.\n"
        f"Timecode: {clip.get('start_tc')} to {clip.get('end_tc')} ({clip.get('duration_s', 0):.2f}s).\n"
        f"Feedback items: {len(feedback_items)}. Selected assets: {len(selected_assets)}. "
        f"Prompt ready: {'yes' if latest_version.get('video_model_prompt') else 'no'}."
    )


def reference_frame_tool_schema() -> dict:
    return {
        "type": "function",
        "name": "extract_reference_frame",
        "description": (
            "Extract a still frame from the project timeline at an explicit timestamp and attach it "
            "as a reference to the currently open clip chat. Use only when the user clearly asks to "
            "extract, grab, capture, save, add, or attach a frame/still/reference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "Timeline timestamp such as 00:44, 1:04, or 00:01:04.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short editor-facing reason for the reference frame.",
                },
                "attach_to": {
                    "type": "string",
                    "enum": ["current_feedback"],
                    "description": "Where to attach the extracted frame.",
                },
            },
            "required": ["timestamp", "reason", "attach_to"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _response_output_items(response: Any) -> list[Any]:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    return list(output or [])


def _output_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is None and isinstance(response, dict):
        text = response.get("output_text")
    return (text or "").strip()


def _tool_call_arguments(item: Any) -> dict:
    raw_args = _output_item_value(item, "arguments", "{}")
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tool_call_id(item: Any) -> str:
    return str(_output_item_value(item, "call_id") or _output_item_value(item, "id") or "")


def _extract_reference_frame_tool_calls(response: Any) -> list[Any]:
    calls = []
    for item in _response_output_items(response):
        if _output_item_value(item, "type") == "function_call" and _output_item_value(item, "name") == "extract_reference_frame":
            calls.append(item)
    return calls


def _run_reference_frame_tool_call(project_name: str, context: dict, arguments: dict) -> dict:
    return extract_reference_frame(
        project_name=project_name,
        current_clip_index=int(context.get("clip_index") or 0),
        timestamp=str(arguments.get("timestamp") or ""),
        reason=str(arguments.get("reason") or ""),
        attach_to=str(arguments.get("attach_to") or "current_feedback"),
    )


async def generate_chat_reply_with_tools(provider: str, message: str, context: dict, messages: list[dict]) -> tuple[str, list[dict]]:
    if wants_prompt_display(message):
        return format_latest_prompt_reply(context), []

    if wants_clip_summary_or_analysis(message):
        if context.get("clip_context"):
            return summarize_clip_context(context.get("clip_context")), []
        if context.get("clip_context_error"):
            return context.get("clip_context_error"), []
        return "This clip has not been analyzed yet. Ask me to analyze the clip with OpenAI, or run the workflow first.", []

    explicit_reference = parse_explicit_reference_frame_request(message)
    if explicit_reference:
        result = _run_reference_frame_tool_call(context.get("project", {}).get("name", ""), context, explicit_reference)
        return result["message"], [result]

    system_prompt = (
        "You are Loka15 Studio's clip assistant inside a video feedback and generation tool. "
        "Answer as a practical editor-facing collaborator. Use only the supplied context. "
        "You can discuss timeline, clip details, feedback, assets, prior prompt generations, quality reports, memory, workflow execution, and video-generation handoff. "
        "When discussing or revising prompts, consider learning_state.relevant_lessons, unresolved prompt feedback, eval cases, and the latest learning eval report from the supplied context. "
        "Do not claim the model has permanently learned; say the app will remember and apply approved lessons. "
        "If the user wants to save feedback, suggest or approve a lesson, revise a prompt from feedback, or run a learning eval, recommend the matching chat action instead of claiming it already happened. "
        "You can call extract_reference_frame when the user clearly asks to extract/grab/capture/save/add/attach a frame or still at an explicit timestamp as a reference for this clip. "
        "Format replies as normal Markdown. When showing prompt text, write it as plain paragraphs or bullets under a short heading; do not use ```text, ```markdown, or any Markdown code fence for prompts. "
        "Never wrap the whole reply in a Markdown code fence or indent prompt text as a code block. "
        "If the user asks to save a durable preference, include a final line starting with 'Memory:' followed by the exact note. "
        "Do not claim that you executed actions unless a tool result says the action completed."
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
                tools=[reference_frame_tool_schema()],
            )
            tool_calls = _extract_reference_frame_tool_calls(response)
            tool_results = []
            if tool_calls:
                tool_outputs = []
                for call in tool_calls:
                    call_id = _tool_call_id(call)
                    try:
                        result = _run_reference_frame_tool_call(
                            context.get("project", {}).get("name", ""),
                            context,
                            _tool_call_arguments(call),
                        )
                    except HTTPException as exc:
                        result = {"status": "error", "message": exc.detail}
                    except Exception as exc:
                        result = {"status": "error", "message": str(exc)}
                    tool_results.append(result)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    })

                followup = client.responses.create(
                    model=os.environ.get("OPENAI_CHAT_MODEL", OPENAI_LITE_MODEL),
                    input=tool_outputs,
                    previous_response_id=getattr(response, "id", None),
                )
                reply_text = _response_text(followup)
                if reply_text:
                    return normalize_chat_reply_markdown(reply_text), tool_results
                successful = [result for result in tool_results if result.get("status") == "ok"]
                if successful:
                    return normalize_chat_reply_markdown(successful[-1].get("message", "Reference frame extracted.")), tool_results
                return normalize_chat_reply_markdown(tool_results[-1].get("message", "Reference frame extraction failed.")), tool_results
            reply_text = _response_text(response)
            return normalize_chat_reply_markdown(reply_text or fallback_chat_reply(message, context)), []

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
            return normalize_chat_reply_markdown((response.text or "").strip()), []
    except Exception as exc:
        log_event(
            "clip_chat.llm_error",
            provider=provider,
            project=context.get("project", {}).get("name"),
            clip_index=context.get("clip_index"),
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )

    return normalize_chat_reply_markdown(fallback_chat_reply(message, context)), []


async def generate_chat_reply(provider: str, message: str, context: dict, messages: list[dict]) -> str:
    text, _tool_results = await generate_chat_reply_with_tools(provider, message, context, messages)
    return text

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


def resolve_timeline_audio_path(project_name: str, audio_name: str | None) -> Optional[Path]:
    if not audio_name:
        return None
    project_assets = project_assets_dir(project_name)
    clean_name = str(audio_name).replace("\\", "/")
    filename = Path(clean_name).name
    candidates = [
        project_assets / "04_audio" / audio_name,
        project_assets / "04_audio" / clean_name,
        project_assets / "04_audio" / filename,
        project_assets / filename,
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    audio_dir = project_assets / "04_audio"
    if filename and audio_dir.exists():
        filename_lower = filename.lower()
        for candidate in audio_dir.rglob("*"):
            try:
                if candidate.is_file() and candidate.name.lower() == filename_lower:
                    return candidate.resolve()
            except OSError:
                continue
    return None


def timeline_audio_with_urls(project_name: str, timeline_data: dict) -> dict:
    audio_timeline = timeline_data.get("audio_timeline") or {}
    dedicated = []
    for index, segment in enumerate(audio_timeline.get("dedicated_audio_tracks") or []):
        audio_path = resolve_timeline_audio_path(project_name, segment.get("clip"))
        dedicated.append({
            **segment,
            "audio_index": index,
            "audio_url": static_url_for_path(str(audio_path)) if audio_path else None,
            "audio_path": str(audio_path) if audio_path else None,
        })
    return {
        **audio_timeline,
        "dedicated_audio_tracks": dedicated,
    }


def _safe_waveform_stem(text: str) -> str:
    stem = Path(text or "audio").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "audio"


def _waveform_cache_path(project_name: str, audio_path: Path, segment: dict, bin_count: int) -> Path:
    try:
        stat = audio_path.stat()
        fingerprint_source = {
            "version": 2,
            "path": str(audio_path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "start_s": segment.get("start_s"),
            "end_s": segment.get("end_s"),
            "bins": bin_count,
        }
    except OSError:
        fingerprint_source = {
            "version": 2,
            "path": str(audio_path),
            "start_s": segment.get("start_s"),
            "end_s": segment.get("end_s"),
            "bins": bin_count,
        }
    digest = hashlib.sha1(json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return project_data_dir(project_name) / "audio_waveforms" / f"{_safe_waveform_stem(segment.get('clip', 'audio'))}_{digest}.json"


def _decode_waveform_samples(audio_path: Path, start_s: float, duration: float) -> bytes:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{max(start_s, 0.0):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "8000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=12)
    except subprocess.TimeoutExpired:
        return b""
    if completed.returncode != 0:
        return b""
    return completed.stdout


def _extract_waveform_peaks(audio_path: Path, segment: dict, bin_count: int) -> list[float]:
    sequence_start_s = max(float(segment.get("start_s") or 0.0), 0.0)
    end_s = max(float(segment.get("end_s") or 0.0), sequence_start_s)
    duration = end_s - sequence_start_s
    if duration <= 0:
        return []

    decoded = _decode_waveform_samples(audio_path, sequence_start_s, duration)
    if not decoded:
        decoded = _decode_waveform_samples(audio_path, 0.0, duration)
    if not decoded:
        return []

    samples = array("h")
    usable_bytes = len(decoded) - (len(decoded) % samples.itemsize)
    samples.frombytes(decoded[:usable_bytes])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return []

    samples_per_bin = max(1, len(samples) // max(1, bin_count))
    peaks = []
    for start in range(0, len(samples), samples_per_bin):
        chunk = samples[start:start + samples_per_bin]
        peak = max((abs(sample) for sample in chunk), default=0) / 32768
        peaks.append(round(min(1.0, peak), 4))
        if len(peaks) >= bin_count:
            break
    return peaks


def waveform_for_audio_segment(project_name: str, segment: dict, bin_count: int) -> dict:
    audio_path = resolve_timeline_audio_path(project_name, segment.get("clip"))
    base = {
        **segment,
        "audio_url": static_url_for_path(str(audio_path)) if audio_path else None,
        "peaks": [],
    }
    if not audio_path:
        return {**base, "error": "Audio file not found"}

    cache_path = _waveform_cache_path(project_name, audio_path, segment, bin_count)
    cached = read_json_file(cache_path, None) if cache_path.exists() else None
    if isinstance(cached, dict) and isinstance(cached.get("peaks"), list):
        return {**base, "peaks": cached["peaks"]}

    peaks = _extract_waveform_peaks(audio_path, segment, bin_count)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(cache_path, {"peaks": peaks})
    return {**base, "peaks": peaks}


def _segment_duration(segment: dict) -> float:
    return max(
        float(segment.get("duration_s") or 0.0),
        float(segment.get("end_s") or 0.0) - float(segment.get("start_s") or 0.0),
        0.0,
    )


def select_master_audio_segments(project_name: str, segments: list[dict], total_duration: float) -> list[dict]:
    if not segments:
        return []

    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, segment in enumerate(segments):
        clip_name = str(segment.get("clip") or "")
        if not clip_name:
            continue
        grouped.setdefault(clip_name, []).append((index, segment))

    candidates = []
    for clip_name, items in grouped.items():
        starts = [float(segment.get("start_s") or 0.0) for _index, segment in items]
        ends = [float(segment.get("end_s") or 0.0) for _index, segment in items]
        first_index = min(index for index, _segment in items)
        first_segment = min((segment for _index, segment in items), key=lambda item: float(item.get("start_s") or 0.0))
        start_s = min(starts) if starts else 0.0
        end_s = max(ends) if ends else start_s
        span = max(0.0, end_s - start_s)
        total_placed_duration = sum(_segment_duration(segment) for _index, segment in items)
        has_file = resolve_timeline_audio_path(project_name, clip_name) is not None
        name_score = 1 if re.search(r"\b(mix|master|final)\b", clip_name, re.IGNORECASE) else 0
        starts_at_head = 1 if start_s <= 1.0 else 0
        coverage = span / max(total_duration, 0.1)
        candidates.append({
            **first_segment,
            "audio_index": first_index,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": span,
            "master_group": True,
            "master_segment_count": len(items),
            "_score": (
                2 if has_file else 0,
                starts_at_head,
                name_score,
                min(coverage, 1.5),
                total_placed_duration,
                span,
            ),
        })

    candidates.sort(key=lambda item: item["_score"], reverse=True)
    selected = candidates[:1]
    for item in selected:
        item.pop("_score", None)
    return selected

PROMPT_VERSION_HANDOFF_FIELDS = [
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
    "audio_transcript",
    "dialogue_text",
    "dialogue_language",
    "dialogue_extraction_reasoning",
    "audio_used",
    "audio_path",
    "audio_url",
    "referenced_frames",
    "referenced_frame_paths",
    "referenced_frame_labels",
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
    "applied_prompt_lessons",
    "applied_prompt_eval_cases",
    "revision_source_prompt_version_id",
    "revision_feedback_ids",
    "revision_lesson_ids",
]


def copy_prompt_handoff_fields(target: dict, source: dict) -> None:
    for field in PROMPT_VERSION_HANDOFF_FIELDS:
        if field in source:
            target[field] = source.get(field)


def prompt_history_entry(item: dict, entry_provider: str) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": entry_provider,
        "video_model_prompt": item.get("video_model_prompt"),
        "selected_assets": item.get("selected_assets", []),
        "explanation": item.get("explanation"),
        "quality_report": item.get("quality_report"),
        "initial_frame_image_path": item.get("initial_frame_image_path"),
        "initial_frame_prompt": item.get("initial_frame_prompt"),
        "clip_frame_paths": item.get("clip_frame_paths", []),
    }
    copy_prompt_handoff_fields(entry, item)
    return entry


def _same_prompt_clip(prompt_record: dict, matched_clip: str, clip_occurrence: Optional[int]) -> bool:
    clip_used = prompt_record.get("clip_used")
    same_clip = (
        clip_used == matched_clip
        or (clip_used and matched_clip and os.path.basename(clip_used) == os.path.basename(matched_clip))
    )
    existing_occurrence = prompt_record.get("clip_occurrence")
    same_occurrence = (
        clip_occurrence is None
        or existing_occurrence is None
        or existing_occurrence == clip_occurrence
    )
    return bool(same_clip and same_occurrence)


def append_prompt_version_to_records(
    prompts_data: list[dict],
    gen_item: dict,
    provider: str = "unknown",
) -> dict:
    matched_clip = gen_item.get("matched_clip") or gen_item.get("clip_used")
    if not matched_clip:
        raise ValueError("matched_clip or clip_used is required to append a prompt version")
    clip_occurrence = gen_item.get("clip_occurrence")

    for prompt_record in prompts_data:
        if not _same_prompt_clip(prompt_record, matched_clip, clip_occurrence):
            continue
        prompt_record["latest_error"] = None
        history = prompt_record.get("history", [])
        if not isinstance(history, list):
            history = []
        if len(history) == 0 and prompt_record.get("video_model_prompt"):
            first_entry = prompt_history_entry(prompt_record, prompt_record.get("provider", "unknown"))
            first_entry["timestamp"] = prompt_record.get("created_at", first_entry["timestamp"])
            history.append(first_entry)
        new_entry = prompt_history_entry(gen_item, provider)
        history.append(new_entry)
        prompt_record["history"] = history
        prompt_record["video_model_prompt"] = gen_item.get("video_model_prompt")
        prompt_record["selected_assets"] = gen_item.get("selected_assets", [])
        prompt_record["explanation"] = gen_item.get("explanation")
        prompt_record["status"] = "success"
        prompt_record["quality_report"] = gen_item.get("quality_report")
        prompt_record["initial_frame_image_path"] = gen_item.get("initial_frame_image_path")
        prompt_record["initial_frame_prompt"] = gen_item.get("initial_frame_prompt")
        prompt_record["clip_frame_paths"] = gen_item.get("clip_frame_paths", [])
        prompt_record["referenced_frames"] = gen_item.get("referenced_frames", [])
        prompt_record["referenced_frame_paths"] = gen_item.get("referenced_frame_paths", [])
        prompt_record["referenced_frame_labels"] = gen_item.get("referenced_frame_labels", [])
        copy_prompt_handoff_fields(prompt_record, gen_item)
        return new_entry

    new_entry = prompt_history_entry(gen_item, provider)
    prompt_record = {
        "clip_used": matched_clip,
        "matched_clip": gen_item.get("matched_clip") or matched_clip,
        "clip_occurrence": gen_item.get("clip_occurrence"),
        "category": gen_item.get("category", "video"),
        "generation_type": gen_item.get("prompt_format", gen_item.get("generation_type", "complex")),
        "video_model_prompt": gen_item.get("video_model_prompt"),
        "selected_assets": gen_item.get("selected_assets", []),
        "status": "success",
        "explanation": gen_item.get("explanation"),
        "quality_report": gen_item.get("quality_report"),
        "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
        "initial_frame_prompt": gen_item.get("initial_frame_prompt"),
        "clip_frame_paths": gen_item.get("clip_frame_paths", []),
        "referenced_frames": gen_item.get("referenced_frames", []),
        "referenced_frame_paths": gen_item.get("referenced_frame_paths", []),
        "referenced_frame_labels": gen_item.get("referenced_frame_labels", []),
        "history": [new_entry],
    }
    copy_prompt_handoff_fields(prompt_record, gen_item)
    prompts_data.append(prompt_record)
    return new_entry


def append_prompt_version(project_name: str, clip_index: int, new_prompt_version: dict, provider: str = "unknown") -> dict:
    project_name = safe_project_name(project_name)
    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")
    clip = timeline[clip_index]
    prompts_path = project_data_dir(project_name) / "video_prompts.json"
    prompts_data = read_json_file(prompts_path, [])
    if not isinstance(prompts_data, list):
        prompts_data = []
    gen_item = {
        **new_prompt_version,
        "matched_clip": new_prompt_version.get("matched_clip") or new_prompt_version.get("clip_used") or clip.get("clip"),
        "clip_used": new_prompt_version.get("clip_used") or new_prompt_version.get("matched_clip") or clip.get("clip"),
        "clip_occurrence": new_prompt_version.get("clip_occurrence", clip_index),
        "category": new_prompt_version.get("category", "video"),
    }
    appended = append_prompt_version_to_records(prompts_data, gen_item, provider)
    write_json_file(prompts_path, prompts_data)
    return appended


def save_output_to_prompts(project: str, provider: str = "unknown"):
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
        if is_error:
            for p_item in prompts_data:
                if _same_prompt_clip(p_item, matched_clip, gen_item.get("clip_occurrence")):
                    p_item["latest_error"] = prompt_text
                    updated = True
                    found = True
                    break
        else:
            append_prompt_version_to_records(prompts_data, gen_item, provider)
            updated = True
            found = True
        
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
            updated = True
            
    if updated:
        try:
            with prompts_json.open("w", encoding="utf-8") as f:
                json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving video_prompts.json: {e}")


def build_workflow_subprocess_command(project: str, index: int, provider: str) -> tuple[list[str], Optional[str], Optional[str], Optional[str]]:
    safe_project_name(project)
    try:
        raw_feedback_path = ensure_raw_feedback(project)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate raw feedback mapping: {exc}") from exc

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
        "--provider", provider,
    ]
    if continuity_frame_path:
        cmd.extend(["--continuity-frame-path", continuity_frame_path])
    if continuity_note:
        cmd.extend(["--continuity-note", continuity_note])
    return cmd, continuity_frame_path, continuity_note, continuity_error


def terminal_job_status(status: str) -> bool:
    return status in {"succeeded", "failed", "cancelled"}


def agent_run_jobs_terminal(project_name: str, agent_run_id: str) -> bool:
    jobs = [
        job for job in load_project_jobs(project_name).get("jobs", [])
        if job.get("agent_run_id") == agent_run_id
    ]
    return bool(jobs) and all(terminal_job_status(str(job.get("status"))) for job in jobs)


def record_self_evaluation_after_job(project_name: str, job_id: str, clip_index: Optional[int], trigger: str) -> Optional[dict]:
    if clip_index is None:
        append_project_job_log(project_name, job_id, "[SELF_EVAL] Skipped: could not resolve clip index.")
        return None

    try:
        evaluation = evaluate_agent_state_for_clip(project_name, int(clip_index), trigger=trigger, agent_run_id=get_project_job(project_name, job_id).get("agent_run_id"))
    except Exception as exc:
        append_project_job_log(project_name, job_id, f"[SELF_EVAL] Skipped: {exc}")
        return None
    job = get_project_job(project_name, job_id)
    result = dict(job.get("result") or {})
    result["self_evaluation"] = evaluation
    update_project_job(project_name, job_id, {"result": result})
    append_project_job_log(project_name, job_id, f"[SELF_EVAL] {evaluation['message']}")
    append_project_event(
        project_name,
        "self_evaluation_completed",
        actor="chat_agent",
        clip_index=clip_index,
        clip_key=evaluation.get("clip_key"),
        entity="job",
        entity_id=job_id,
        payload={
            "trigger": trigger,
            "verdict": evaluation.get("verdict"),
            "next_action": evaluation.get("next_action"),
            "state_hash": evaluation.get("state_hash"),
            "agent_run_id": job.get("agent_run_id"),
        },
    )

    agent_run_id = job.get("agent_run_id")
    if agent_run_id:
        current_run = next(
            (run for run in load_chat_agent_runs(project_name).get("runs", []) if run.get("id") == agent_run_id),
            None,
        )
        if current_run:
            tool_results = list(current_run.get("tool_results") or [])
            tool_results.append(evaluation)
            updates = {
                "tool_results": tool_results,
                "self_evaluation": evaluation,
            }
            if agent_run_jobs_terminal(project_name, agent_run_id):
                updates["status"] = "completed" if evaluation.get("verdict") == "passed" else "needs_attention"
            update_chat_agent_run(project_name, agent_run_id, updates)
    return evaluation


async def execute_workflow_job(project: str, job_id: str) -> None:
    job = get_project_job(project, job_id)
    feedback_index = int(job.get("feedback_index"))
    provider = job.get("provider") or "openai"
    update_project_job(project, job_id, {"status": "running", "started_at": now_iso()})
    append_project_job_log(project, job_id, f"[START] Launching workflow subprocess for feedback index {feedback_index}.")
    workflow_started = time.perf_counter()

    try:
        cmd, continuity_frame_path, _continuity_note, continuity_error = build_workflow_subprocess_command(project, feedback_index, provider)
        if continuity_frame_path:
            append_project_job_log(project, job_id, f"[CONTEXT] Prepared continuity reference: {continuity_frame_path}")
        elif continuity_error:
            append_project_job_log(project, job_id, f"[CONTEXT] {continuity_error}")
        append_project_job_log(project, job_id, f"Executing command: {' '.join(cmd)}")
        log_event("workflow.job.start", project=project, index=feedback_index, provider=provider, job_id=job_id, command=cmd)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BACKEND_DIR),
        )

        async def read_stream(stream, prefix=""):
            while True:
                if project_job_cancel_requested(project, job_id):
                    process.terminate()
                    append_project_job_log(project, job_id, "[CANCEL] Cancellation requested.")
                    return "cancelled"
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                append_project_job_log(project, job_id, f"{prefix}{decoded}")
            return None

        cancelled = await read_stream(process.stdout)
        if not cancelled:
            cancelled = await read_stream(process.stderr, "[STDERR] ")
        rc = await process.wait()
        duration_ms = round((time.perf_counter() - workflow_started) * 1000, 2)

        if cancelled or project_job_cancel_requested(project, job_id):
            update_project_job(
                project,
                job_id,
                {
                    "status": "cancelled",
                    "finished_at": now_iso(),
                    "error": "Workflow job cancelled.",
                    "result": {"exit_code": rc, "duration_ms": duration_ms},
                },
            )
            return

        if rc == 0:
            save_output_to_prompts(project, provider)
            clip_index = clip_index_for_feedback_index(project, feedback_index)
            promoted_selection = promote_latest_prompt_selection(project, clip_index)
            update_project_job(
                project,
                job_id,
                {
                    "status": "succeeded",
                    "finished_at": now_iso(),
                    "result": {
                        "exit_code": rc,
                        "duration_ms": duration_ms,
                        "clip_index": clip_index,
                        "active_prompt_version_id": (promoted_selection or {}).get("active_prompt_version_id"),
                    },
                },
            )
            append_project_job_log(project, job_id, f"[SUCCESS] Workflow execution finished with exit code {rc}.")
            record_self_evaluation_after_job(project, job_id, clip_index, "workflow_job")
            log_event("workflow.job.finish", project=project, index=feedback_index, provider=provider, job_id=job_id, status="success", exit_code=rc, duration_ms=duration_ms)
        else:
            update_project_job(
                project,
                job_id,
                {
                    "status": "failed",
                    "finished_at": now_iso(),
                    "error": f"Workflow execution failed with exit code {rc}.",
                    "result": {"exit_code": rc, "duration_ms": duration_ms},
                },
            )
            append_project_job_log(project, job_id, f"[ERROR] Workflow execution failed with exit code {rc}.")
    except Exception as exc:
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": str(exc)})
        append_project_job_log(project, job_id, f"[ERROR] {exc}")
        log_event("workflow.job.error", project=project, index=feedback_index, provider=provider, job_id=job_id, error_type=type(exc).__name__, error=str(exc)[:500])


def execute_video_generation_job(project: str, job_id: str) -> None:
    job = get_project_job(project, job_id)
    payload = job.get("payload") or {}
    update_project_job(project, job_id, {"status": "running", "started_at": now_iso()})
    append_project_job_log(project, job_id, "[START] Video generation job started.")
    if project_job_cancel_requested(project, job_id):
        update_project_job(project, job_id, {"status": "cancelled", "finished_at": now_iso(), "error": "Video generation job cancelled before start."})
        return

    try:
        result = generate_clip_video(
            project,
            int(payload.get("clip_index") if payload.get("clip_index") is not None else job.get("clip_index")),
            payload.get("prompt_version_index"),
            resolution=payload.get("resolution", "720p"),
            generate_audio=bool(payload.get("generate_audio", False)),
            aspect_ratio=payload.get("aspect_ratio", "9:16"),
            duration=int(payload.get("duration", 5)),
        )
        update_project_job(project, job_id, {"status": "succeeded", "finished_at": now_iso(), "result": result})
        append_project_job_log(project, job_id, "[SUCCESS] Video generation finished.")
        record_self_evaluation_after_job(project, job_id, int(result.get("clip_index")), "video_generation_job")
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": detail})
        append_project_job_log(project, job_id, f"[ERROR] {detail}")
    except Exception as exc:
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": str(exc)})
        append_project_job_log(project, job_id, f"[ERROR] {exc}")


def dispatch_agent_run_jobs(project_name: str, agent_run: dict, background_tasks: BackgroundTasks, provider: str = "openai") -> dict:
    dispatched_jobs = []
    tool_results = list(agent_run.get("tool_results") or [])

    for step in agent_run.get("plan_steps", []):
        if step.get("status") != "dispatch_ready":
            continue
        action = step.get("action") or {}
        action_type = action.get("type")
        job = None
        if action_type == "execute_workflow" and action.get("feedback_index") is not None:
            job = create_project_job(
                project_name,
                "workflow",
                feedback_index=int(action.get("feedback_index")),
                provider=action.get("provider") or provider,
                payload=action,
                agent_run_id=agent_run.get("id"),
            )
            background_tasks.add_task(execute_workflow_job, project_name, job["id"])
        elif action_type == "generate_video":
            clip_index = int(agent_run.get("clip_index") or 0)
            payload = {
                "clip_index": clip_index,
                "prompt_version_index": action.get("prompt_version_index"),
                "provider": "segmind",
                "resolution": action.get("resolution", "720p"),
                "generate_audio": bool(action.get("generate_audio", False)),
                "aspect_ratio": action.get("aspect_ratio", "9:16"),
                "duration": int(action.get("duration", 5)),
            }
            job = create_project_job(
                project_name,
                "generate_video",
                clip_index=clip_index,
                provider="segmind",
                payload=payload,
                agent_run_id=agent_run.get("id"),
            )
            background_tasks.add_task(execute_video_generation_job, project_name, job["id"])

        if job:
            step["status"] = "queued"
            step["job_id"] = job["id"]
            dispatched_jobs.append(job)
            append_project_event(
                project_name,
                "agent_job_dispatched",
                actor="chat_agent",
                clip_index=agent_run.get("clip_index"),
                clip_key=agent_run.get("clip_key"),
                entity="job",
                entity_id=job["id"],
                payload={
                    "agent_run_id": agent_run.get("id"),
                    "job_type": job.get("type"),
                    "action_type": action_type,
                    "step_id": step.get("id"),
                    "feedback_index": job.get("feedback_index"),
                },
            )
            tool_results.append({
                "status": "queued",
                "tool": action_type,
                "job_id": job["id"],
                "message": f"Created {job['type']} job.",
            })

    if dispatched_jobs:
        agent_run["status"] = "running"
        agent_run["tool_results"] = tool_results
        agent_run["jobs"] = dispatched_jobs
    return agent_run


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


def _prompt_version_signature(version: dict) -> tuple:
    return (
        str(version.get("video_model_prompt") or "").strip(),
        str(version.get("initial_frame_prompt") or "").strip(),
        str(version.get("initial_frame_image_path") or ""),
        json.dumps(version.get("selected_assets") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(version.get("referenced_frames") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(version.get("referenced_frame_paths") or [], sort_keys=True, ensure_ascii=False),
    )


def _top_level_prompt_version(prompt: dict) -> Optional[dict]:
    if not prompt.get("video_model_prompt"):
        return None
    entry = {
        key: value
        for key, value in prompt.items()
        if key not in {"history"}
    }
    if not entry.get("timestamp"):
        entry["timestamp"] = prompt.get("updated_at") or prompt.get("created_at")
    return entry


def prompt_versions_for_record(prompt: dict) -> list[dict]:
    history = prompt.get("history")
    versions = []
    if isinstance(history, list) and history:
        versions = [version for version in history if isinstance(version, dict)]
    top_level_version = _top_level_prompt_version(prompt)
    if top_level_version:
        if not versions or _prompt_version_signature(versions[-1]) != _prompt_version_signature(top_level_version):
            versions.append(top_level_version)
        else:
            versions[-1] = {**versions[-1], **top_level_version}
    return versions


def _safe_video_stem(clip_name: str) -> str:
    stem = Path(clip_name or "clip").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "clip"


def _next_generated_video_path(project_name: str, clip_name: str, version_number: int) -> Path:
    output_dir = project_assets_dir(project_name) / "generated_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{_safe_video_stem(clip_name)}_v{version_number:03d}.mp4"
    candidate = output_dir / base
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{_safe_video_stem(clip_name)}_v{version_number:03d}_{suffix}.mp4"
        suffix += 1
    return candidate


def _video_asset_url(path: Path) -> str:
    rel = path.resolve().relative_to(ASSETS_DIR)
    return f"/assets/{quote(rel.as_posix())}"


def append_video_generation_attempt(
    prompt_record: dict,
    selected_version: dict,
    attempt: dict,
    *,
    is_latest_version: bool,
) -> None:
    selected_version.setdefault("video_generation_attempts", []).append(attempt)
    if selected_version is not prompt_record and is_latest_version:
        prompt_record["video_generation_attempts"] = selected_version["video_generation_attempts"]
    elif selected_version is prompt_record:
        prompt_record["video_generation_attempts"] = selected_version["video_generation_attempts"]
    prompt_record["latest_video_generation_attempt"] = attempt


def generate_clip_video(
    project_name: str,
    clip_index: int,
    prompt_version_index: Optional[int] = None,
    *,
    resolution: str = "720p",
    generate_audio: bool = False,
    aspect_ratio: str = "9:16",
    duration: int = 5,
) -> dict:
    project_name = safe_project_name(project_name)
    api_key = os.environ.get("SEGMIND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SEGMIND_API_KEY is not configured.")

    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    prompts_path = project_data_dir(project_name) / "video_prompts.json"
    prompts_data = read_json_file(prompts_path, [])
    if not isinstance(prompts_data, list):
        raise HTTPException(status_code=404, detail="Prompt history was not found.")

    prompt_record = matching_prompt({"prompts": prompts_data}, clip, clip_index)
    if not prompt_record:
        raise HTTPException(status_code=404, detail="No generated prompt exists for this clip.")

    versions = prompt_versions_for_record(prompt_record)
    if not versions:
        raise HTTPException(status_code=400, detail="No usable prompt version exists for this clip.")

    selected_index = prompt_version_index if prompt_version_index is not None else len(versions) - 1
    if selected_index < 0 or selected_index >= len(versions):
        raise HTTPException(status_code=400, detail="Prompt version index is out of range.")
    selected_version = versions[selected_index]
    if not selected_version.get("video_model_prompt"):
        raise HTTPException(status_code=400, detail="Selected prompt version has no video prompt.")

    existing_videos = generated_videos_for_version(selected_version)
    next_version_number = len(existing_videos) + 1
    output_path = _next_generated_video_path(project_name, clip.get("clip", "clip.mp4"), next_version_number)
    is_latest_version = selected_index == len(versions) - 1

    cache = SupabaseAssetUrlCache()
    payload = None
    try:
        payload = build_segmind_payload(
            item={**prompt_record, **selected_version, "duration": duration, "generate_audio": generate_audio},
            api_key=api_key,
            cache=cache,
            use_local_initial_frame=True,
            initial_image_url=None,
        )
        payload["resolution"] = resolution
        payload["generate_audio"] = generate_audio
        payload["aspect_ratio"] = aspect_ratio
        payload["duration"] = max(4, min(15, int(duration or 5)))
        result = create_seedance_task(api_key=api_key, payload=payload)
        video_bytes = (result.get("content") or {}).get("bytes")
        if not video_bytes:
            raise RuntimeError(f"Segmind response did not include video bytes: {result}")
        save_video_bytes(video_bytes, output_path)
    except HTTPException:
        raise
    except SeedanceGenerationRecoveryError as exc:
        failed_attempt = {
            "timestamp": now_iso(),
            "status": "recovery_failed",
            "provider": "segmind",
            "model": os.environ.get("SEEDANCE_MODEL", "seedance-2.0"),
            "request_id": exc.request_id,
            "clip_used": clip.get("clip"),
            "clip_occurrence": clip_index,
            "prompt_version_index": selected_index,
            "prompt_timestamp": selected_version.get("timestamp"),
            "error": str(exc),
            "error_status_code": exc.status_code,
            "duration": (payload or {}).get("duration"),
            "resolution": (payload or {}).get("resolution"),
            "generate_audio": (payload or {}).get("generate_audio"),
            "ratio": (payload or {}).get("aspect_ratio"),
            "recoverable": bool(exc.request_id),
        }
        append_video_generation_attempt(
            prompt_record,
            selected_version,
            failed_attempt,
            is_latest_version=is_latest_version,
        )
        write_json_file(prompts_path, prompts_data)
        log_event(
            "video_generation.recovery_failed",
            project=project_name,
            clip_index=clip_index,
            prompt_version_index=selected_index,
            request_id=exc.request_id,
            status_code=exc.status_code,
            error=str(exc)[:500],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Segmind charged/submitted the job, but the app could not retrieve the result.",
                "request_id": exc.request_id,
                "status_code": exc.status_code,
                "recoverable": bool(exc.request_id),
                "next_step": "Do not regenerate immediately. Use this request id with Segmind support or retry recovery when available.",
            },
        ) from exc
    except Exception as exc:
        log_event(
            "video_generation.error",
            project=project_name,
            clip_index=clip_index,
            prompt_version_index=selected_index,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise HTTPException(status_code=500, detail=f"Video generation failed: {exc}") from exc

    video_entry = {
        "version": next_version_number,
        "timestamp": now_iso(),
        "provider": "segmind",
        "model": os.environ.get("SEEDANCE_MODEL", "seedance-2.0"),
        "request_id": result.get("id"),
        "clip_used": clip.get("clip"),
        "clip_occurrence": clip_index,
        "prompt_version_index": selected_index,
        "prompt_timestamp": selected_version.get("timestamp"),
        "path": str(output_path.resolve()),
        "url": _video_asset_url(output_path),
        "label": f"Generated video v{next_version_number}",
        "source_output_url": (result.get("content") or {}).get("video_url"),
        "duration": payload.get("duration"),
        "resolution": payload.get("resolution"),
        "generate_audio": payload.get("generate_audio"),
        "ratio": payload.get("aspect_ratio"),
        "storage_root": str(ASSETS_DIR.resolve()),
    }

    successful_attempt = {
        "timestamp": video_entry["timestamp"],
        "status": "succeeded",
        "provider": video_entry["provider"],
        "model": video_entry["model"],
        "request_id": video_entry["request_id"],
        "clip_used": video_entry["clip_used"],
        "clip_occurrence": video_entry["clip_occurrence"],
        "prompt_version_index": selected_index,
        "prompt_timestamp": video_entry["prompt_timestamp"],
        "path": video_entry["path"],
        "url": video_entry["url"],
        "duration": video_entry["duration"],
        "resolution": video_entry["resolution"],
        "generate_audio": video_entry["generate_audio"],
        "ratio": video_entry["ratio"],
    }
    append_video_generation_attempt(
        prompt_record,
        selected_version,
        successful_attempt,
        is_latest_version=is_latest_version,
    )
    selected_version.setdefault("generated_videos", []).append(video_entry)
    if selected_version is not prompt_record and is_latest_version:
        prompt_record["generated_videos"] = selected_version["generated_videos"]
    elif selected_version is prompt_record:
        prompt_record["generated_videos"] = selected_version["generated_videos"]
    prompt_record["latest_generated_video"] = video_entry

    write_json_file(prompts_path, prompts_data)
    log_event(
        "video_generation.success",
        project=project_name,
        clip_index=clip_index,
        prompt_version_index=selected_index,
        generated_video=video_entry["path"],
    )
    return {
        "project_name": project_name,
        "clip_index": clip_index,
        "prompt_version_index": selected_index,
        "video": video_entry,
        "generated_videos": selected_version["generated_videos"],
    }


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
        "audio_timeline": timeline_audio_with_urls(project_name, timeline_data),
        "summary": timeline_data.get("summary", {}),
        "sequence_name": timeline_data.get("sequence_name", ""),
        "total_duration_tc": timeline_data.get("total_duration_tc", ""),
        "total_duration_s": timeline_data.get("total_duration_s", 0),
        "fps": timeline_data.get("fps"),
        "frame_size": timeline_data.get("frame_size"),
    }


def _safe_timeline_filmstrip_stem(clip_name: str, clip_index: int) -> str:
    stem = Path(clip_name or "clip").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "clip"
    return f"{clip_index:04d}_{safe_stem}"


@app.get("/api/projects/{project_name}/clips/{clip_index}/filmstrip")
def get_clip_filmstrip(project_name: str, clip_index: int, frames: int = 4):
    project_name = safe_project_name(project_name)
    frame_count = max(1, min(int(frames or 4), 8))
    project_dir = project_data_dir(project_name)
    timeline_path = project_dir / "timeline.json"

    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Project timeline not found")

    timeline_data = read_json_file(timeline_path, {})
    video_timeline = timeline_data.get("video_timeline", [])
    if clip_index < 0 or clip_index >= len(video_timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = video_timeline[clip_index]
    clip_name = clip.get("clip", "")
    duration_s = float(clip.get("duration_s") or max(0.0, float(clip.get("end_s") or 0.0) - float(clip.get("start_s") or 0.0)) or 0.0)
    source_clip_path = _resolve_raw_clip_path(project_name, clip_name)

    if not source_clip_path:
        return {"clip_index": clip_index, "frames": [], "error": "Clip media file not found"}

    cache_dir = project_dir / "timeline_filmstrips" / _safe_timeline_filmstrip_stem(clip_name, clip_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets = _frame_offsets_for_duration(duration_s, frame_count)
    filmstrip_frames = []

    for index, offset_s in enumerate(offsets, start=1):
        frame_path = cache_dir / f"frame_{index:02d}.jpg"
        if not frame_path.exists():
            extract_frame_at_offset(str(source_clip_path), offset_s, str(frame_path))
        frame_url = static_url_for_path(str(frame_path)) if frame_path.exists() else None
        if frame_url:
            filmstrip_frames.append({"offset_s": offset_s, "url": frame_url})

    return {"clip_index": clip_index, "frames": filmstrip_frames}


@app.get("/api/projects/{project_name}/audio-waveform")
def get_project_audio_waveform(project_name: str, bins: int = 600, mode: str = "master"):
    project_name = safe_project_name(project_name)
    project_dir = project_data_dir(project_name)
    timeline_path = project_dir / "timeline.json"

    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Project timeline not found")

    timeline_data = read_json_file(timeline_path, {})
    segments = (timeline_data.get("audio_timeline") or {}).get("dedicated_audio_tracks") or []
    total_duration = max(float(timeline_data.get("total_duration_s") or 0.0), 0.1)
    waveform_mode = "all" if str(mode).lower() == "all" else "master"
    if waveform_mode == "master":
        segments = select_master_audio_segments(project_name, segments, total_duration)
    total_bins = max(120, min(int(bins or 600), 900 if waveform_mode == "master" else 1800))
    waveform_segments = []

    for index, segment in enumerate(segments):
        duration = max(float(segment.get("duration_s") or 0.0), float(segment.get("end_s") or 0.0) - float(segment.get("start_s") or 0.0), 0.0)
        segment_bins = max(12, min(total_bins, round((duration / total_duration) * total_bins)))
        waveform_segments.append({
            **waveform_for_audio_segment(project_name, {**segment, "audio_index": segment.get("audio_index", index)}, segment_bins),
            "audio_index": segment.get("audio_index", index),
        })

    return {
        "project_name": project_name,
        "bins": total_bins,
        "mode": waveform_mode,
        "segments": waveform_segments,
    }


@app.get("/api/projects/{project_name}/clips/{clip_index}/state")
def get_clip_state(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    return build_clip_state(project_name, clip_index)


@app.get("/api/projects/{project_name}/prompt-feedback")
def get_prompt_feedback(project_name: str, clip_index: Optional[int] = None):
    project_name = safe_project_name(project_name)
    items = prompt_feedback_items_for_clip(project_name, clip_index)
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "items": items,
        "summaries": prompt_feedback_summary_by_version(project_name, clip_index) if clip_index is not None else {},
    }


@app.get("/api/projects/{project_name}/prompt-lessons")
def get_prompt_lessons(
    project_name: str,
    clip_key: Optional[str] = None,
    category: Optional[str] = None,
    include_archived: bool = False,
):
    project_name = safe_project_name(project_name)
    lessons = prompt_learning_store(project_name).list_lessons(
        clip_key=clip_key,
        category=category,
        include_archived=include_archived,
    )
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_key": clip_key,
        "category": category,
        "lessons": lessons,
    }


@app.get("/api/projects/{project_name}/prompt-eval-cases")
def get_prompt_eval_cases(
    project_name: str,
    clip_index: Optional[int] = None,
    enabled: Optional[bool] = None,
):
    project_name = safe_project_name(project_name)
    cases = prompt_eval_case_store(project_name).list_eval_cases(
        clip_index=clip_index,
        enabled=enabled,
    )
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "enabled": enabled,
        "cases": cases,
    }


@app.post("/api/projects/{project_name}/prompt-lessons")
def create_prompt_lesson(project_name: str, request: PromptLessonCreateRequest):
    project_name = safe_project_name(project_name)
    try:
        lesson = prompt_learning_store(project_name).add_lesson(
            request.lesson,
            scope=request.scope,
            clip_key=request.clip_key,
            category=request.category,
            source_feedback_ids=request.source_feedback_ids,
            confidence=request.confidence,
            positive_examples=request.positive_examples,
            negative_examples=request.negative_examples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_lesson_created",
        actor="user",
        clip_key=lesson.get("clip_key"),
        entity="prompt_lesson",
        entity_id=lesson.get("id"),
        payload={
            "scope": lesson.get("scope"),
            "category": lesson.get("category"),
            "source_feedback_ids": lesson.get("source_feedback_ids", []),
        },
    )
    return {
        "lesson": lesson,
        "clip_state": clip_state_for_lesson_sources(project_name, lesson.get("source_feedback_ids", [])),
    }


@app.patch("/api/projects/{project_name}/prompt-lessons/{lesson_id}")
def patch_prompt_lesson(project_name: str, lesson_id: str, request: PromptLessonPatchRequest):
    project_name = safe_project_name(project_name)
    try:
        lesson = prompt_learning_store(project_name).update_lesson(
            lesson_id,
            request.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prompt lesson was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_lesson_updated",
        actor="user",
        clip_key=lesson.get("clip_key"),
        entity="prompt_lesson",
        entity_id=lesson.get("id"),
        payload={
            "scope": lesson.get("scope"),
            "category": lesson.get("category"),
            "archived": lesson.get("archived"),
        },
    )
    return {
        "lesson": lesson,
        "clip_state": clip_state_for_lesson_sources(project_name, lesson.get("source_feedback_ids", [])),
    }


@app.post("/api/projects/{project_name}/prompts/{prompt_version_id}/revise-from-feedback")
def revise_prompt_from_feedback(project_name: str, prompt_version_id: str, request: PromptRevisionRequest):
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, request.clip_index)
    clip_key = state.get("clip_key")
    versions = (state.get("active_prompt") or {}).get("versions") or []
    source_version = next(
        (version for version in versions if version.get("prompt_version_id") == prompt_version_id),
        None,
    )
    if not source_version:
        raise HTTPException(status_code=404, detail="Prompt version was not found for this clip.")
    source_prompt = str(source_version.get("video_model_prompt") or "").strip()
    if not source_prompt:
        raise HTTPException(status_code=400, detail="Selected prompt version has no prompt text.")

    feedback_items = [
        item
        for item in load_prompt_feedback(project_name).get("items", [])
        if item.get("clip_index") == request.clip_index
        and item.get("prompt_version_id") == prompt_version_id
    ]
    if request.feedback_ids:
        requested_feedback_ids = set(request.feedback_ids)
        selected_feedback = [item for item in feedback_items if item.get("id") in requested_feedback_ids]
        if len(selected_feedback) != len(requested_feedback_ids):
            raise HTTPException(status_code=400, detail="One or more feedback ids do not belong to this prompt version and clip.")
    else:
        selected_feedback = [
            item
            for item in feedback_items
            if item.get("rating") == "negative" and item.get("status") == "open"
        ]
    if not selected_feedback:
        raise HTTPException(status_code=400, detail="No matching prompt feedback was selected for revision.")

    lesson_store = prompt_learning_store(project_name)
    all_lessons = lesson_store.list_lessons(include_archived=False, limit=500)
    if request.lesson_ids:
        requested_lesson_ids = set(request.lesson_ids)
        selected_lessons = [lesson for lesson in all_lessons if lesson.get("id") in requested_lesson_ids]
        if len(selected_lessons) != len(requested_lesson_ids):
            raise HTTPException(status_code=400, detail="One or more lesson ids were not found.")
        for lesson in selected_lessons:
            if lesson.get("scope") == "clip" and lesson.get("clip_key") != clip_key:
                raise HTTPException(status_code=400, detail="One or more lesson ids belong to another clip.")
    else:
        query = " ".join(
            " ".join(str(item.get(key) or "") for key in ("comment", "correction", "remember_note"))
            for item in selected_feedback
        )
        selected_lessons = [
            candidate.item
            for candidate in lesson_store.retrieve_lessons(
                query=query,
                clip_key=clip_key,
                limit=6,
            )
        ]

    eval_case_candidates = prompt_eval_case_store(project_name).retrieve_eval_cases(
        query=" ".join(
            " ".join(str(item.get(key) or "") for key in ("comment", "correction", "remember_note"))
            for item in selected_feedback
        ),
        clip_key=clip_key,
        clip_index=request.clip_index,
        limit=6,
    )
    eval_cases = [candidate.item for candidate in eval_case_candidates]

    client = _client_for_prompt_provider(request.provider)
    model = _default_model_for_provider(request.provider)
    clip = (state.get("timeline") or {}).get("clip") or {}
    clip_context = (state.get("analysis_state") or {}).get("clip_context") or {}
    selected_assets = source_version.get("selected_assets") or []
    previous_quality_report = source_version.get("quality_report") or {}
    revision_prompt = (
        "Revise this existing AI video generation prompt using the supplied user feedback. "
        "Create a new improved version; do not overwrite history. Keep all correct details from the original prompt, "
        "preserve clip continuity, and change only what is necessary to satisfy feedback, lessons, and eval cases.\n\n"
        f"PROJECT: {project_name}\n"
        f"CLIP: {json.dumps(clip, ensure_ascii=False)}\n"
        f"CLIP_CONTEXT: {json.dumps(clip_context, ensure_ascii=False)}\n"
        f"SELECTED_ASSETS: {json.dumps(selected_assets, ensure_ascii=False)}\n\n"
        f"ORIGINAL_PROMPT:\n{source_prompt}\n\n"
        f"USER_FEEDBACK_TO_FIX:\n{_feedback_revision_text(selected_feedback)}\n\n"
        f"APPROVED_OR_RELEVANT_LESSONS:\n{_lesson_revision_text(selected_lessons) or 'None'}\n\n"
        f"RELEVANT_EVAL_CASE_EXPECTATIONS:\n{_eval_case_revision_text(eval_cases) or 'None'}\n\n"
        f"PREVIOUS_QUALITY_REPORT:\n{json.dumps(previous_quality_report, ensure_ascii=False)}\n\n"
        "Return a revised prompt and a short explanation of what changed. The revised prompt must remain English-only, "
        "provider-compatible, concrete, and detailed."
    )
    try:
        revision = generate_structured(
            provider=request.provider,
            client=client,
            model=model,
            contents=[revision_prompt],
            schema=PromptResult,
            system_instruction=(
                "You are a careful AI video prompt editor. You revise existing prompts using explicit user feedback, "
                "approved lessons, and eval case expectations while preserving correct continuity and visual details."
            ),
            temperature=0.15,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to revise prompt: {exc}") from exc

    revised_prompt = str(revision.get("video_model_prompt") or "").strip()
    if not revised_prompt:
        raise HTTPException(status_code=500, detail="Prompt revision returned empty prompt text.")

    quality_report = run_quality_check(
        provider=request.provider,
        client=client,
        model=model,
        prompt=revised_prompt,
        feedback_items=[
            {"remark": item.get("correction") or item.get("comment") or item.get("remember_note") or ""}
            for item in selected_feedback
        ],
        selected_assets=selected_assets,
        has_clip=bool(source_version.get("clip_frame_paths")),
    )
    learning_report = run_learning_eval(
        provider=request.provider,
        client=client,
        model=model,
        prompt=revised_prompt,
        eval_cases=eval_cases,
        lessons=selected_lessons,
    )
    quality_report = merge_learning_eval_report(quality_report, learning_report)

    new_version_payload = {
        **source_version,
        "timestamp": now_iso(),
        "provider": request.provider,
        "video_model_prompt": revised_prompt,
        "explanation": revision.get("explanation") or "Revised from prompt feedback.",
        "quality_report": quality_report,
        "revision_source_prompt_version_id": prompt_version_id,
        "revision_feedback_ids": [item.get("id") for item in selected_feedback],
        "revision_lesson_ids": [lesson.get("id") for lesson in selected_lessons],
        "applied_prompt_lessons": [
            {
                "id": lesson.get("id"),
                "scope": lesson.get("scope"),
                "category": lesson.get("category"),
                "lesson": lesson.get("lesson"),
            }
            for lesson in selected_lessons
        ],
        "applied_prompt_eval_cases": _compact_applied_eval_cases(eval_cases),
    }
    append_prompt_version(project_name, request.clip_index, new_version_payload, provider=request.provider)
    refreshed_state = build_clip_state(project_name, request.clip_index)
    latest_version = ((refreshed_state.get("active_prompt") or {}).get("versions") or [])[-1]
    update_clip_selection(project_name, clip_key, {"active_prompt_version_id": latest_version.get("prompt_version_id")})
    refreshed_state = build_clip_state(project_name, request.clip_index)
    prompt_version = (refreshed_state.get("active_prompt") or {}).get("version") or latest_version
    append_project_event(
        project_name,
        "prompt_version_revised_from_feedback",
        actor="user",
        clip_index=request.clip_index,
        clip_key=clip_key,
        entity="prompt_version",
        entity_id=prompt_version.get("prompt_version_id"),
        payload={
            "source_prompt_version_id": prompt_version_id,
            "feedback_ids": [item.get("id") for item in selected_feedback],
            "lesson_ids": [lesson.get("id") for lesson in selected_lessons],
            "eval_case_ids": [case.get("id") for case in eval_cases],
        },
    )
    return {
        "prompt_version": prompt_version,
        "quality_report": quality_report,
        "learning_report": learning_report,
        "clip_state": refreshed_state,
    }


@app.post("/api/projects/{project_name}/prompt-feedback")
def create_prompt_feedback(project_name: str, request: PromptFeedbackCreateRequest):
    project_name = safe_project_name(project_name)
    created = create_prompt_feedback_item(project_name, request.model_dump(), actor="user")
    item = created["item"]
    response = {
        "item": item,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }
    if created.get("eval_case"):
        response["eval_case"] = created["eval_case"]
    return response


@app.post("/api/projects/{project_name}/prompt-feedback/{feedback_id}/eval-case")
def create_prompt_eval_case_from_feedback(project_name: str, feedback_id: str):
    project_name = safe_project_name(project_name)
    item = find_prompt_feedback_item(project_name, feedback_id)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")
    if item.get("rating") != "negative":
        raise HTTPException(status_code=400, detail="Eval cases can only be created from negative feedback.")
    try:
        eval_case = create_eval_case_from_feedback_item(project_name, item)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_eval_case_created",
        actor="user",
        clip_index=item.get("clip_index"),
        clip_key=item.get("clip_key"),
        entity="prompt_eval_case",
        entity_id=eval_case.get("id"),
        payload={
            "source_feedback_id": item.get("id"),
            "failure_categories": eval_case.get("failure_categories", []),
        },
    )
    return {
        "eval_case": eval_case,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }


@app.post("/api/projects/{project_name}/prompt-feedback/{feedback_id}/suggest-lesson")
def suggest_prompt_lesson(project_name: str, feedback_id: str, request: PromptLessonSuggestRequest):
    project_name = safe_project_name(project_name)
    item = find_prompt_feedback_item(project_name, feedback_id)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")
    suggestion = _suggest_prompt_lesson(project_name, item, request.provider)
    return {
        "project_name": project_name,
        "feedback_id": feedback_id,
        "suggestion": suggestion,
    }


@app.patch("/api/projects/{project_name}/prompt-eval-cases/{case_id}")
def patch_prompt_eval_case(project_name: str, case_id: str, request: PromptEvalCasePatchRequest):
    project_name = safe_project_name(project_name)
    try:
        eval_case = prompt_eval_case_store(project_name).update_eval_case(
            case_id,
            request.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prompt eval case was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_eval_case_updated",
        actor="user",
        clip_index=eval_case.get("clip_index"),
        clip_key=eval_case.get("clip_key"),
        entity="prompt_eval_case",
        entity_id=eval_case.get("id"),
        payload={"enabled": eval_case.get("enabled")},
    )
    return {
        "eval_case": eval_case,
        "clip_state": build_clip_state(project_name, eval_case.get("clip_index", 0)),
    }


@app.patch("/api/projects/{project_name}/prompt-feedback/{feedback_id}")
def patch_prompt_feedback(project_name: str, feedback_id: str, request: PromptFeedbackPatchRequest):
    project_name = safe_project_name(project_name)
    data = load_prompt_feedback(project_name)
    items = data.setdefault("items", [])
    item = next((candidate for candidate in items if candidate.get("id") == feedback_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")

    updates = request.model_dump(exclude_none=True)
    allowed = {
        "rating",
        "categories",
        "severity",
        "comment",
        "correction",
        "remember_note",
        "create_eval_case",
        "status",
    }
    next_item = {**item, **{key: value for key, value in updates.items() if key in allowed}}
    validated = _validate_prompt_feedback_payload(next_item)
    item.update(validated)
    item["updated_at"] = now_iso()
    save_prompt_feedback(project_name, data)
    append_project_event(
        project_name,
        "prompt_feedback_updated",
        actor="user",
        clip_index=item.get("clip_index"),
        clip_key=item.get("clip_key"),
        entity="prompt_feedback",
        entity_id=item.get("id"),
        payload={
            "prompt_id": item.get("prompt_id"),
            "prompt_version_id": item.get("prompt_version_id"),
            "rating": item.get("rating"),
            "status": item.get("status"),
            "categories": item.get("categories", []),
        },
    )
    return {
        "item": item,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }


@app.get("/api/projects/{project_name}/events")
def get_project_events(project_name: str, limit: int = 200, clip_key: Optional[str] = None, event_type: Optional[str] = None):
    project_name = safe_project_name(project_name)
    return {
        "project_name": project_name,
        "events": load_project_events(project_name, limit=limit, clip_key=clip_key, event_type=event_type),
    }


@app.get("/api/projects/{project_name}/clips/{clip_index}/selection")
def get_clip_selection(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, clip_index)
    return {
        "selection": clip_selection_for_key(project_name, state["clip_key"]),
        "clip_state": state,
    }


@app.patch("/api/projects/{project_name}/clips/{clip_index}/selection")
def patch_clip_selection(project_name: str, clip_index: int, request: ClipSelectionUpdate):
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, clip_index)
    updates = request.model_dump(exclude_none=True)
    selection = update_clip_selection(project_name, state["clip_key"], updates)
    return {
        "selection": selection,
        "clip_state": build_clip_state(project_name, clip_index),
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
        "agent_runs": context["agent_runs"],
        "context": {
            "project": context["project"],
            "clip": context["clip"],
            "clip_state": context["clip_state"],
            "learning_state": context.get("learning", {}),
            "adjacent_clips": context["adjacent_clips"],
            "feedback_count": len((context.get("feedback") or {}).get("feedback_items", [])),
            "selected_asset_count": len((context.get("latest_version") or {}).get("selected_assets", []) or []),
            "prompt_ready": bool((context.get("latest_version") or {}).get("video_model_prompt")),
        },
    }


@app.post("/api/projects/{project_name}/chat/clip")
async def post_clip_chat(project_name: str, request: ClipChatRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
    history = load_chat_history(project_name)
    clip_messages = history.setdefault("clips", {}).setdefault(context["clip_key"], [])

    user_message = make_chat_message("user", message_text)
    clip_messages.append(user_message)

    analysis_requested = wants_clip_summary_or_analysis(message_text)
    regenerate_summary = wants_regenerated_clip_summary(message_text)
    if analysis_requested:
        _clip_context, analysis_error = ensure_clip_context_for_chat(
            project_name,
            context,
            request.provider,
            force=regenerate_summary,
        )
        if analysis_error:
            context["clip_context_error"] = analysis_error

    assistant_text, tool_results = await generate_chat_reply_with_tools(request.provider, message_text, context, clip_messages)
    if tool_results:
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
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
    agent_run = plan_chat_agent_run(message_text, context, explicit_actions, actions)
    actions = apply_agent_run_action_policy(actions, agent_run)
    agent_run["available_actions"] = actions
    agent_run["suggested_actions"] = actions
    agent_run["tool_results"] = tool_results
    agent_run = execute_safe_agent_run_tools(project_name, agent_run, context, message_text, saved_memories, provider=request.provider)
    mutation_results = [
        result for result in agent_run.get("tool_results", [])
        if isinstance(result, dict) and result.get("mutates_state")
    ]
    if mutation_results:
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
        successful_mutations = [result for result in mutation_results if result.get("status") == "ok"]
        if successful_mutations:
            assistant_text = "\n".join(result.get("message", "State updated.") for result in successful_mutations)
    if any(result.get("status") == "error" for result in tool_results if isinstance(result, dict)):
        agent_run["status"] = "failed"
        agent_run["errors"] = [
            str(result.get("message") or result)
            for result in tool_results
            if isinstance(result, dict) and result.get("status") == "error"
        ]
    agent_run = create_chat_agent_run(project_name, agent_run)
    agent_run = dispatch_agent_run_jobs(project_name, agent_run, background_tasks, provider=request.provider)
    if agent_run.get("jobs"):
        actions = strip_risky_autonomous_actions(actions)
        agent_run["available_actions"] = actions
        agent_run["suggested_actions"] = actions
        agent_run = update_chat_agent_run(
            project_name,
            agent_run["id"],
            {
                "status": agent_run.get("status"),
                "plan_steps": agent_run.get("plan_steps", []),
                "tool_results": agent_run.get("tool_results", []),
                "available_actions": actions,
                "suggested_actions": actions,
                "jobs": agent_run.get("jobs", []),
            },
        )
    save_pending_workflow_intents(project_name, actions, message_text, context)
    media = build_clip_media_gallery(context) if (wants_clip_media_gallery(message_text) or tool_results or analysis_requested) else []
    assistant_message = make_chat_message(
        "assistant",
        assistant_text,
        {
            "provider": request.provider,
            "actions": actions,
            "media": media,
            "saved_memory_ids": [item["id"] for item in saved_memories],
            "tool_results": tool_results,
            "agent_run": agent_run,
            "clip_state_id": context.get("clip_state", {}).get("clip_state_id"),
            "prompt_version_id": (context.get("clip_state", {}).get("active_prompt") or {}).get("version_id"),
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
        "agent_run": agent_run,
        "clip_state": context["clip_state"],
    }


@app.post("/api/projects/{project_name}/chat/clip/action")
def execute_clip_chat_action(project_name: str, request: ClipChatActionRequest):
    project_name = safe_project_name(project_name)
    action = dict(request.action or {})
    action_type = action.get("type")
    if action_type not in LEARNING_CHAT_ACTION_TYPES:
        raise HTTPException(status_code=400, detail="This endpoint only executes prompt learning chat actions.")
    context = build_clip_chat_context(project_name, request.clip_index, query=request.message or action.get("prompt") or action.get("label") or "")
    run = plan_chat_agent_run(
        request.message or action.get("label") or action_type,
        context,
        [action],
        [action],
    )
    for step in run.get("plan_steps", []):
        if step.get("tool") == action_type:
            step["status"] = "pending"
            step["requires_approval"] = False
    run["approval_required"] = False
    run["autonomy_level"] = "approved_action"
    run["status"] = "planned"
    run = execute_safe_agent_run_tools(
        project_name,
        run,
        context,
        request.message or action.get("prompt") or action.get("label") or "",
        [],
        provider=request.provider,
    )
    run = create_chat_agent_run(project_name, run)
    updated_context = build_clip_chat_context(project_name, request.clip_index, query=request.message or action.get("prompt") or "")
    return {
        "project_name": project_name,
        "clip_index": request.clip_index,
        "clip_key": updated_context["clip_key"],
        "action_result": (run.get("tool_results") or [{}])[-1],
        "agent_run": run,
        "clip_state": updated_context["clip_state"],
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


@app.post("/api/projects/{project_name}/generate-video")
def post_generate_clip_video(project_name: str, request: GenerateClipVideoRequest):
    return generate_clip_video(
        project_name,
        request.clip_index,
        request.prompt_version_index,
        resolution=request.resolution,
        generate_audio=request.generate_audio,
        aspect_ratio=request.aspect_ratio,
        duration=request.duration,
    )


@app.get("/api/projects/{project_name}/jobs")
def get_project_jobs(project_name: str):
    project_name = safe_project_name(project_name)
    return {"project_name": project_name, "jobs": list_recent_project_jobs(project_name)}


@app.get("/api/projects/{project_name}/jobs/{job_id}")
def get_project_job_endpoint(project_name: str, job_id: str):
    project_name = safe_project_name(project_name)
    return get_project_job(project_name, job_id)


@app.post("/api/projects/{project_name}/jobs/{job_id}/cancel")
def cancel_project_job(project_name: str, job_id: str):
    project_name = safe_project_name(project_name)
    job = get_project_job(project_name, job_id)
    if job.get("status") in {"succeeded", "failed", "cancelled"}:
        return job
    return update_project_job(project_name, job_id, {"cancel_requested": True, "status": "cancelling"})


@app.post("/api/projects/{project_name}/jobs/workflow")
def create_workflow_job(project_name: str, request: WorkflowJobRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    job = create_project_job(
        project_name,
        "workflow",
        feedback_index=request.feedback_index,
        provider=request.provider,
        payload={"feedback_index": request.feedback_index, "provider": request.provider},
        agent_run_id=request.agent_run_id,
    )
    background_tasks.add_task(execute_workflow_job, project_name, job["id"])
    return job


@app.post("/api/projects/{project_name}/jobs/generate-video")
def create_generate_video_job(project_name: str, request: GenerateClipVideoRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    payload = request.model_dump()
    job = create_project_job(
        project_name,
        "generate_video",
        clip_index=request.clip_index,
        provider=request.provider,
        payload=payload,
    )
    background_tasks.add_task(execute_video_generation_job, project_name, job["id"])
    return job


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
