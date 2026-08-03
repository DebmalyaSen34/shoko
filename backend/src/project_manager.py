import os
import re
import json
import uuid
import hashlib
import sys
import subprocess
from array import array
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, unquote
from fastapi import HTTPException

from src.storage_paths import (
    ASSETS_DIR,
    DATA_DIR,
    atomic_write_json_file,
    now_iso,
    project_assets_dir,
    project_data_dir,
    project_events_path,
    read_json_file,
    stable_json_hash,
    write_json_file,
)


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

    assets_dir = project_assets_dir(project_name)
    assets = get_assets_list(assets_dir, project_name)

    video_timeline = timeline_data.get("video_timeline", [])
    timeline_with_urls = []
    for clip in video_timeline:
        clip_url = find_clip_url(clip.get("clip", ""), assets_dir)
        timeline_with_urls.append({
            **clip,
            "clip_url": clip_url
        })

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


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_assets_list(assets_dir: str | Path, project_name: str) -> dict:
    assets_path = Path(assets_dir)
    if not assets_path.exists():
        return {}

    categories = {}
    for root, _, files in os.walk(assets_path):
        root_path = Path(root)
        rel_dir = os.path.relpath(root_path, assets_path)
        if rel_dir == ".":
            continue

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

    for cat in categories:
        categories[cat] = sorted(categories[cat], key=lambda x: x["name"])

    return categories


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


def load_project_events(
    project_name: str, *, limit: int = 200, clip_key: Optional[str] = None, event_type: Optional[str] = None
) -> list[dict]:
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
        key: value for key, value in event.items() if key != "event_hash"
    })
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


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
            if resolved.is_relative_to(storage_paths.ASSETS_DIR):
                rel = resolved.relative_to(storage_paths.ASSETS_DIR)
                return f"/assets/{quote(rel.as_posix())}"
            if resolved.is_relative_to(storage_paths.DATA_DIR):
                rel = resolved.relative_to(storage_paths.DATA_DIR)
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
            if resolved.is_relative_to(storage_paths.ASSETS_DIR):
                rel_to_assets = resolved.relative_to(storage_paths.ASSETS_DIR)
                parts = rel_to_assets.parts
                if len(parts) > 1:
                    return Path(*parts[1:]).as_posix()
                return rel_to_assets.as_posix()
            if resolved.is_relative_to(storage_paths.DATA_DIR):
                rel_to_data = resolved.relative_to(storage_paths.DATA_DIR)
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
