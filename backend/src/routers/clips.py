import re
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.storage_paths import (
    project_data_dir,
    read_json_file,
    safe_project_name,
)
from src.project_manager import static_url_for_path
from src.reference_frames import _resolve_raw_clip_path
from src.clip_state import build_clip_state, clip_selection_for_key, update_clip_selection
from src.generator.media import _frame_offsets_for_duration
from src.workflows.referenced_frames import extract_frame_at_offset

router = APIRouter(tags=["clips"])


class ClipSelectionUpdate(BaseModel):
    active_prompt_version_id: Optional[str] = None
    active_generated_video_id: Optional[str] = None
    selected_assets: Optional[list[dict]] = None
    pinned_assets: Optional[list[dict]] = None
    selected_asset_ids: Optional[list[str]] = None
    selected_asset_paths: Optional[list[str]] = None
    pinned_asset_ids: Optional[list[str]] = None
    pinned_asset_paths: Optional[list[str]] = None


def _safe_timeline_filmstrip_stem(clip_name: str, clip_index: int) -> str:
    stem = Path(clip_name or "clip").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "clip"
    return f"{clip_index:04d}_{safe_stem}"


@router.get("/api/projects/{project_name}/clips/{clip_index}/filmstrip")
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


@router.get("/api/projects/{project_name}/clips/{clip_index}/state")
def get_clip_state(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    return build_clip_state(project_name, clip_index)


@router.get("/api/projects/{project_name}/clips/{clip_index}/selection")
def get_clip_selection(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, clip_index)
    return {
        "selection": clip_selection_for_key(project_name, state["clip_key"]),
        "clip_state": state,
    }


@router.patch("/api/projects/{project_name}/clips/{clip_index}/selection")
def patch_clip_selection(project_name: str, clip_index: int, request: ClipSelectionUpdate):
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, clip_index)
    updates = request.model_dump(exclude_none=True)
    selection = update_clip_selection(project_name, state["clip_key"], updates)
    return {
        "selection": selection,
        "clip_state": build_clip_state(project_name, clip_index),
    }
