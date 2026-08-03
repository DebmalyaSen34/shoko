import os
import json
import hashlib
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from fastapi import HTTPException

from config.settings import LOKA_STORAGE_DIR

APP_VERSION = os.environ.get("LOKA_APP_VERSION", "0.1.0")
BACKEND_STARTED_AT = datetime.now(timezone.utc)

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_STORAGE_DIR = Path(LOKA_STORAGE_DIR).expanduser().resolve()

DATA_DIR = APP_STORAGE_DIR / "data"
ASSETS_DIR = APP_STORAGE_DIR / "assets"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_chat_key(clip_name: str, clip_index: int) -> str:
    return f"{clip_name}::{clip_index}"


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
