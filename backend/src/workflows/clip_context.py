import base64
import json
import mimetypes
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from src.generator.client import Provider, generate_structured
from src.generator.media import _frame_offsets_for_duration
from src.generator.upload import OpenAIFileReference, _openai_file_reference
from config.settings import OPENAI_REASONING_MODEL


class ClipContextResult(BaseModel):
    summary: str = Field(..., description="Concise user-visible summary of what is happening in the clip segment.")
    visible_characters: List[str] = Field(default_factory=list, description="Characters visibly present in the segment.")
    expressions: List[str] = Field(default_factory=list, description="Observed facial expressions or emotional state changes.")
    gaze: List[str] = Field(default_factory=list, description="Observed eye line, gaze direction, or camera-looking behavior.")
    actions: List[str] = Field(default_factory=list, description="Observed physical actions and gestures.")
    blocking: List[str] = Field(default_factory=list, description="Character placement and interaction within the frame.")
    camera_framing: str = Field("", description="Camera angle, shot size, movement, and composition notes.")
    location: Optional[str] = Field(None, description="Visible or inferred location/set.")
    continuity_notes: List[str] = Field(default_factory=list, description="Continuity notes useful for regeneration.")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Ambiguities or low-confidence observations.")


def _file_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _safe_slug(text: str) -> str:
    stem = os.path.splitext(os.path.basename(text or "clip"))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "clip"


def clip_context_dir(output_base_dir: str, project_name: str, clip_name: str, clip_occurrence: Optional[int]) -> str:
    occurrence = "unknown" if clip_occurrence is None else str(clip_occurrence)
    folder = f"{occurrence}_{_safe_slug(clip_name)}"
    return os.path.abspath(os.path.join(output_base_dir, project_name, "analysis", "clip_context", folder))


def trim_clip_segment(source_path: str, output_path: str, duration_s: float, source_offset_s: float = 0.0) -> bool:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = max(float(duration_s or 0.0), 0.0)
    if duration <= 0:
        return False
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(float(source_offset_s or 0.0), 0.0):.3f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        output_path,
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode == 0 and os.path.exists(output_path):
        return True

    fallback = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(float(source_offset_s or 0.0), 0.0):.3f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        output_path,
    ]
    completed = subprocess.run(fallback, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    return completed.returncode == 0 and os.path.exists(output_path)


def extract_context_frames(video_path: str, output_dir: str, duration_s: float) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    for index, offset in enumerate(_frame_offsets_for_duration(duration_s), start=1):
        frame_path = os.path.join(output_dir, f"frame_{index:03d}.jpg")
        command = [
            "ffmpeg", "-y", "-ss", f"{offset:.3f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "3", frame_path
        ]
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode == 0 and os.path.exists(frame_path):
            frame_paths.append(frame_path)
    return frame_paths


def resolve_audio_path(project_assets_dir: str, audio_name: Optional[str], audio_path: Optional[str]) -> Optional[str]:
    candidates = []
    if audio_path:
        candidates.append(audio_path)
        clean_audio_path = audio_path.replace("\\", "/")
        filename = os.path.basename(clean_audio_path)
        if filename:
            candidates.append(os.path.join(project_assets_dir, "04_audio", filename))
            candidates.append(os.path.join(project_assets_dir, filename))

    if audio_name:
        clean_audio_name = audio_name.replace("\\", "/")
        filename = os.path.basename(clean_audio_name)
        candidates.append(os.path.join(project_assets_dir, "04_audio", audio_name))
        candidates.append(os.path.join(project_assets_dir, "04_audio", clean_audio_name))
        if filename:
            candidates.append(os.path.join(project_assets_dir, "04_audio", filename))
            candidates.append(os.path.join(project_assets_dir, filename))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return audio_path


def extract_audio_segment(input_audio_path: str, output_audio_path: str, start_s: float, end_s: float) -> bool:
    duration = max(0.0, float(end_s or 0.0) - float(start_s or 0.0))
    if duration <= 0:
        return False
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(float(start_s or 0.0), 0.0):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        input_audio_path,
        "-acodec",
        "libmp3lame" if output_audio_path.endswith(".mp3") else "pcm_s16le",
        output_audio_path,
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    return completed.returncode == 0 and os.path.exists(output_audio_path)


def transcribe_audio_segment(client, audio_path: str, model: Optional[str] = None) -> str:
    with open(audio_path, "rb") as file:
        response = client.audio.transcriptions.create(
            model=model or os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
            file=file,
        )
    if isinstance(response, dict):
        return str(response.get("text") or "").strip()
    return str(getattr(response, "text", "") or "").strip()


def _status_for_context(segment_available: bool, segment_attached: bool, frames: List[str], error: Optional[str]) -> str:
    if error and not frames:
        return "failed"
    if segment_available and segment_attached:
        return "video_and_frames"
    if frames:
        return "frames_only"
    if segment_available:
        return "segment_only"
    return "missing_source"


def analyze_clip_context(
    *,
    project_name: str,
    clip_name: str,
    clip_occurrence: Optional[int],
    clip_start_s: Optional[float],
    clip_end_s: Optional[float],
    clip_duration_s: Optional[float],
    assets_dir: str,
    output_base_dir: str,
    client,
    provider: Provider = "openai",
    model: str = OPENAI_REASONING_MODEL,
    feedback_items: Optional[List[dict]] = None,
    audio_name: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> dict:
    context_dir = clip_context_dir(output_base_dir, project_name, clip_name, clip_occurrence)
    frames_dir = os.path.join(context_dir, "frames")
    audio_dir = os.path.join(context_dir, "audio")
    context_path = os.path.join(context_dir, "clip_context.json")
    segment_path = os.path.join(context_dir, "segment.mp4")
    audio_segment_path = os.path.join(audio_dir, "segment_audio.mp3")
    os.makedirs(context_dir, exist_ok=True)

    duration = max(float(clip_duration_s or 0.0), 0.0)
    if duration <= 0 and clip_start_s is not None and clip_end_s is not None:
        duration = max(0.0, float(clip_end_s) - float(clip_start_s))

    abs_assets_dir = os.path.abspath(assets_dir)
    source_candidates = [
        os.path.join(abs_assets_dir, project_name, "06_clips", "_raw", clip_name),
        os.path.join(abs_assets_dir, "06_clips", "_raw", clip_name),
    ]
    source_path = next((path for path in source_candidates if os.path.exists(path)), source_candidates[0])
    if not os.path.exists(source_path):
        result = {
            "status": "missing_source",
            "summary": "",
            "clip_context_path": os.path.abspath(context_path),
            "clip_segment_path": None,
            "segment_path": None,
            "frame_paths": [],
            "error": f"Raw clip file not found: {source_path}",
        }
        with open(context_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        return result

    source_offset_s = 0.0
    segment_available = trim_clip_segment(source_path, segment_path, duration, source_offset_s)
    frame_source = segment_path if segment_available else source_path
    frame_paths = extract_context_frames(frame_source, frames_dir, duration)
    audio_status = "not_configured"
    audio_error = None
    audio_transcript = ""
    resolved_audio_path = resolve_audio_path(os.path.join(abs_assets_dir, project_name), audio_name, audio_path)
    if resolved_audio_path and os.path.exists(resolved_audio_path):
        audio_status = "missing_segment"
        if clip_start_s is not None and clip_end_s is not None:
            try:
                if extract_audio_segment(resolved_audio_path, audio_segment_path, float(clip_start_s), float(clip_end_s)):
                    audio_status = "segment_ready"
                    if provider == "openai":
                        try:
                            audio_transcript = transcribe_audio_segment(client, audio_segment_path)
                            audio_status = "transcribed" if audio_transcript else "transcribed_empty"
                        except Exception as exc:
                            audio_status = "transcription_failed"
                            audio_error = f"Audio transcription failed: {exc}"
                else:
                    audio_error = (
                        f"Failed to trim audio from {float(clip_start_s):.3f}s "
                        f"to {float(clip_end_s):.3f}s."
                    )
            except Exception as exc:
                audio_status = "failed"
                audio_error = f"Audio segment extraction failed: {exc}"
        else:
            audio_status = "missing_timeline_bounds"
            audio_error = "Audio path exists, but clip start/end seconds are unavailable."
    elif audio_name or audio_path:
        audio_status = "missing_source"
        audio_error = f"Audio file not found: {audio_path or audio_name}"

    contents = []
    uploaded_ref: Optional[OpenAIFileReference] = None
    segment_attached = False
    upload_error = None
    if provider == "openai" and segment_available:
        upload_error = "OpenAI clip analysis uses extracted frames because current vision models do not accept video input directly."

    for frame_path in frame_paths:
        contents.append({
            "type": "input_image",
            "image_url": _file_data_url(frame_path),
            "detail": "auto",
        })

    feedback_items = feedback_items or []
    prompt = (
        "Analyze the provided chronological video frames and any matching audio transcript. "
        "Return only user-visible observations that can help improve an AI video prompt. "
        "Do not include hidden reasoning. Treat the frames as visual evidence and the transcript as audio/dialogue evidence.\n\n"
        f"CLIP: {clip_name}\n"
        f"CLIP_OCCURRENCE: {clip_occurrence}\n"
        f"TIMELINE_BOUNDS_SECONDS: {clip_start_s} to {clip_end_s}\n"
        f"SEGMENT_DURATION_SECONDS: {duration}\n"
        f"AUDIO_STATUS: {audio_status}\n"
        f"AUDIO_TRANSCRIPT: {audio_transcript or '[none]'}\n"
        f"AUDIO_ERROR: {audio_error or '[none]'}\n"
        f"FEEDBACK_CONTEXT: {json.dumps(feedback_items, ensure_ascii=False)}"
    )
    contents.append(prompt)

    analysis = {
        "summary": "",
        "visible_characters": [],
        "expressions": [],
        "gaze": [],
        "actions": [],
        "blocking": [],
        "camera_framing": "",
        "location": None,
        "continuity_notes": [],
        "uncertainty_flags": [],
    }
    analysis_error = upload_error
    if contents[:-1]:
        try:
            analysis = generate_structured(
                provider=provider,
                client=client,
                model=model,
                contents=contents,
                schema=ClipContextResult,
                system_instruction=(
                    "You are a precise film editor creating a user-visible clip understanding summary. "
                    "Describe only what is visible or strongly inferable from the supplied segment/frames."
                ),
                temperature=0.1,
            )
        except Exception as exc:
            analysis_error = f"Clip context analysis failed: {exc}"

    if uploaded_ref and uploaded_ref.file_id:
        try:
            client.files.delete(uploaded_ref.file_id)
        except Exception:
            pass

    status = _status_for_context(segment_available, segment_attached, frame_paths, analysis_error)
    result = {
        **analysis,
        "status": status,
        "clip_context_path": os.path.abspath(context_path),
        "clip_segment_path": os.path.abspath(segment_path) if segment_available else None,
        "segment_path": os.path.abspath(segment_path) if segment_available else None,
        "frame_paths": [os.path.abspath(path) for path in frame_paths],
        "clip_used": clip_name,
        "clip_occurrence": clip_occurrence,
        "clip_start_s": clip_start_s,
        "clip_end_s": clip_end_s,
        "clip_duration_s": duration,
        "source_clip_path": os.path.abspath(source_path),
        "source_offset_s": source_offset_s,
        "source_offset_note": "Timeline parser currently stores sequence bounds and clip duration, not source media in/out. Segment uses source offset 0.",
        "audio_used": audio_name,
        "audio_path": os.path.abspath(resolved_audio_path) if resolved_audio_path and os.path.exists(resolved_audio_path) else resolved_audio_path,
        "audio_segment_path": os.path.abspath(audio_segment_path) if os.path.exists(audio_segment_path) else None,
        "audio_transcript": audio_transcript,
        "audio_status": audio_status,
        "audio_error": audio_error,
        "source_feedback_timestamps": [item.get("timestamp") for item in feedback_items],
        "source_feedback_indexes": [item.get("raw_index") for item in feedback_items if item.get("raw_index") is not None],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": analysis_error,
    }
    with open(context_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result
