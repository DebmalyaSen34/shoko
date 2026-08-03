import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.storage_paths import (
    ASSETS_DIR,
    DATA_DIR,
    PROJECT_ASSET_DIRS,
    REQUIRED_PROJECT_DIRS,
    ensure_project_asset_tree,
    project_assets_dir,
    project_data_dir,
    read_json_file,
    safe_project_name,
    safe_upload_name,
)
from src.project_manager import (
    find_clip_url,
    get_assets_list,
    get_project_data,
    load_project_events,
    select_master_audio_segments,
    timeline_audio_with_urls,
    waveform_for_audio_segment,
)
from src.workflows.project_setup import setup_project_workspace
from src.workflows.timeline_extraction import extract_timeline_from_project
from src.workflows.feedback_parsing import parse_and_align_feedback

router = APIRouter(tags=["projects"])


class ManualFeedbackRequest(BaseModel):
    clip_used: str
    category: str
    remark: str
    timestamp: Optional[str] = None


@router.get("/api/projects")
def list_projects():
    projects = []
    if DATA_DIR.exists():
        for item_path in DATA_DIR.iterdir():
            if item_path.is_dir():
                timeline_path = item_path / "timeline.json"
                if timeline_path.exists():
                    projects.append(item_path.name)
    return sorted(projects)


@router.post("/api/projects")
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


@router.post("/api/projects/{project_name}/assets/{category}")
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


@router.post("/api/projects/{project_name}/premiere-package")
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


@router.post("/api/projects/{project_name}/feedback")
async def upload_project_feedback(project_name: str, file: UploadFile = File(...)):
    project_name = safe_project_name(project_name)
    filename = safe_upload_name(file.filename)

    project_dir = project_data_dir(project_name)
    timeline_path = project_dir / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Timeline must be extracted before feedback can be parsed and aligned. Please import Premiere package first."
        )

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


@router.post("/api/projects/{project_name}/feedback/item")
def add_manual_feedback(project_name: str, item: ManualFeedbackRequest):
    project_name = safe_project_name(project_name)
    project_dir = project_data_dir(project_name)
    feedback_path = project_dir / "feedback.json"

    if feedback_path.exists():
        try:
            with feedback_path.open("r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load existing feedback: {str(e)}")
    else:
        feedback_data = []

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

    new_item = {
        "timestamp": item.timestamp,
        "category": item.category,
        "remark": item.remark
    }
    found_group["feedback_items"].append(new_item)

    try:
        with feedback_path.open("w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    return {"status": "success", "project_name": project_name}


@router.get("/api/project/{project_name}")
def get_project_data_endpoint(project_name: str):
    return get_project_data(project_name)


@router.get("/api/projects/{project_name}/audio-waveform")
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


@router.get("/api/projects/{project_name}/events")
def get_project_events(project_name: str, limit: int = 200, clip_key: Optional[str] = None, event_type: Optional[str] = None):
    project_name = safe_project_name(project_name)
    return {
        "project_name": project_name,
        "events": load_project_events(project_name, limit=limit, clip_key=clip_key, event_type=event_type),
    }
