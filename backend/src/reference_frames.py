import os
import re
import json
from pathlib import Path
from typing import Any, Optional
from fastapi import HTTPException

from src.storage_paths import (
    ASSETS_DIR,
    atomic_write_json_file,
    now_iso,
    project_assets_dir,
    project_data_dir,
    read_json_file,
    stable_state_id,
    write_json_file,
)
from src.clip_state import _asset_identity
from src.clustering import find_matching_clip_occurrence
from src.utils import parse_timestamp_to_seconds
from src.workflows.referenced_frames import extract_frame_at_offset


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
