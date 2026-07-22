import os
import sys
# Add project root to sys.path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import base64
import mimetypes
import re
import subprocess
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from src.generator.client import generate_structured
from src.generator.media import _frame_offsets_for_duration
from src.schemas import PromptResult
from src.workflows.clip_context import analyze_clip_context
from config.settings import OPENAI_REASONING_MODEL
from scripts.generate_seedance_video import (
    SupabaseAssetUrlCache,
    attach_prepared_segmind_payload,
)

class DialogueAssessmentResult(BaseModel):
    is_dialogue_active: bool = Field(..., description="True if there is active spoken dialogue/speech occurring in this clip segment.")
    reasoning: str = Field(..., description="Reasoning for whether dialogue is active or not.")

def assess_dialogue_activity(
    openai_client,
    openai_model: str,
    frame_paths: List[str],
    remarks: List[str]
) -> bool:
    """Uses OpenAI to visually and textually assess if dialogue is active in the clip frames."""
    if not frame_paths:
        return False

    contents = []
    # Add frames as input images
    for path in frame_paths[:5]: # Limit to 5 frames
        contents.append({
            "type": "input_image",
            "image_url": _file_data_url(path),
            "detail": "auto"
        })

    prompt_text = (
        "You are an expert film editor and director.\n"
        "Analyze the sequential frames of the clip and the client feedback remarks.\n"
        f"Client Feedback Remarks: {json.dumps(remarks)}\n\n"
        "Determine if active spoken dialogue or character speech is occurring in this video segment.\n"
        "Set is_dialogue_active to true if: (1) a character's mouth is clearly open/speaking/delivering a line across the frames; or (2) the feedback remarks explicitly mention dialogue changes, quotes, pronunciation, or voice acting adjustments.\n"
        "Set is_dialogue_active to false if: the character is silent, standing, moving without speaking, or the edit is purely visual (e.g. framing, camera movements, pose edits) and no dialogue is mentioned or visually happening."
    )
    contents.append(prompt_text)

    try:
        response = generate_structured(
            provider="openai",
            client=openai_client,
            model=openai_model,
            contents=contents,
            schema=DialogueAssessmentResult,
            system_instruction="You are a precise post-production supervisor checking if dialogue/speech is active in a video clip segment."
        )
        is_active = response.get("is_dialogue_active", False)
        print(f"Dialogue assessment result: {is_active} (Reason: {response.get('reasoning')})")
        return is_active
    except Exception as e:
        print(f"Error during dialogue assessment: {e}. Falling back to False.")
        return False

class ContinuityVerificationResult(BaseModel):
    requires_previous_clip_continuity: bool = Field(..., description="True if these two frames share direct visual continuity (location, wardrobe, characters, action sequence).")
    reasoning: str = Field(..., description="Reasoning for whether continuity is required.")

def verify_visual_continuity(
    openai_client,
    openai_model: str,
    prev_last_frame_path: str,
    curr_first_frame_path: str,
    remarks: List[str]
) -> bool:
    """Uses OpenAI to visually compare boundary frames and verify if continuity is active."""
    contents = [
        {
            "type": "input_image",
            "image_url": _file_data_url(prev_last_frame_path),
            "detail": "auto"
        },
        {
            "type": "input_image",
            "image_url": _file_data_url(curr_first_frame_path),
            "detail": "auto"
        }
    ]
    prompt_text = (
        "You are an expert film editor and director.\n"
        "Analyze the last frame of the previous shot (Image 1) and the first frame of the current shot (Image 2).\n"
        f"Client Feedback Remarks: {json.dumps(remarks)}\n\n"
        "Determine if these two shots share direct visual continuity across the edit cut.\n"
        "They share continuity if:\n"
        "1. They occur in the same location setup and time of day.\n"
        "2. The character looks, postures, and wardrobe match.\n"
        "3. The action in Shot 2 is a direct visual sequence continuation of Shot 1.\n"
        "Set requires_previous_clip_continuity to true if they match visually. Otherwise, set it to false."
    )
    contents.append(prompt_text)

    try:
        response = generate_structured(
            provider="openai",
            client=openai_client,
            model=openai_model,
            contents=contents,
            schema=ContinuityVerificationResult,
            system_instruction="You are a precise post-production editor verifying visual continuity across edit boundaries."
        )
        res = response.get("requires_previous_clip_continuity", False)
        print(f"Continuity verification result: {res} (Reason: {response.get('reasoning')})")
        return res
    except Exception as e:
        print(f"Error during continuity verification: {e}. Falling back to False.")
        return False

# Helper to encode file to base64 Data URL
def _file_data_url(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def _resolve_audio_path(project_assets_dir: str, audio_name: Optional[str], audio_path: Optional[str]) -> Optional[str]:
    """Resolve a plan audio path, falling back to the project's durable audio asset location."""
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
            return candidate
    return audio_path

# FFMPEG Frame extraction helpers
def extract_frames_per_second(video_path: str, output_dir: str, duration_s: float) -> List[str]:
    """Extract adaptive, full-clip frames while tolerating individual ffmpeg failures."""
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    for index, offset in enumerate(_frame_offsets_for_duration(duration_s), start=1):
        frame_path = os.path.join(output_dir, f"frame_{index:03d}.jpg")
        command = [
            "ffmpeg", "-y", "-ss", f"{offset:.3f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "3", frame_path
        ]
        try:
            completed = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False
            )
            if completed.returncode == 0 and os.path.exists(frame_path):
                frame_paths.append(frame_path)
        except Exception as exc:
            print(f"Warning: FFMPEG frame extraction failed for offset {offset}: {exc}")

    return frame_paths

def get_video_duration(video_path: str) -> float:
    """Gets the duration of a video file using ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        if completed.returncode == 0:
            return float(completed.stdout.strip())
    except Exception:
        pass
    return 0.0

def extract_last_frame(video_path: str, output_dir: str, duration_s: float) -> Optional[str]:
    """Extract the last frame of the video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    duration = max(float(duration_s or 0.0), 0.0)
    offset = max(0.0, duration - 0.1)
    frame_path = os.path.join(output_dir, "last_frame.jpg")
    command = [
        "ffmpeg", "-y", "-ss", f"{offset:.3f}", "-i", video_path,
        "-frames:v", "1", "-q:v", "3", frame_path
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False
        )
        if completed.returncode == 0 and os.path.exists(frame_path):
            return frame_path
    except Exception as exc:
        print(f"Warning: FFMPEG last frame extraction failed: {exc}")
    return None

def extract_audio_segment(input_audio_path: str, output_audio_path: str, start_s: float, end_s: float) -> bool:
    """Extract and transcode a slice of an audio file using ffmpeg."""
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        return False
    command = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-i", input_audio_path,
        "-acodec", "libmp3lame" if output_audio_path.endswith(".mp3") else "pcm_s16le",
        output_audio_path
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False
        )
        return completed.returncode == 0 and os.path.exists(output_audio_path)
    except Exception as exc:
        print(f"Warning: FFMPEG audio segment extraction failed: {exc}")
        return False

# Resolve preproduction asset files matching characters and locations
def resolve_reference_assets(assets_dir: str, characters: List[str], location: Optional[str]) -> List[str]:
    """Finds files under assets_dir that match the character names or location and have image extensions."""
    matched = []
    characters_lower = [c.lower() for c in characters if c]
    location_lower = location.lower() if location else None
    IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}

    # We walk character folders and location folders recursively
    for root, _, files in os.walk(assets_dir):
        # Avoid walking final/raw clips folders
        if "06_clips" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            file_lower = file.lower()
            # Check characters
            for char in characters_lower:
                if char in file_lower:
                    matched.append(os.path.join(root, file))
                    break
            # Check location
            if location_lower and location_lower in file_lower:
                matched.append(os.path.join(root, file))

    # Return unique resolved files
    return sorted(list(set(matched)))


def _asset_matches_subject(path: str, subject: str) -> bool:
    subject_lower = (subject or "").strip().lower()
    if not subject_lower:
        return False
    return subject_lower in os.path.basename(path).lower()


def _missing_required_subject_sheets(selected_assets: List[str], subjects: List[str]) -> List[str]:
    missing = []
    for subject in subjects:
        if not any(_asset_matches_subject(path, subject) for path in selected_assets):
            missing.append(subject)
    return missing


def _infer_explicitly_requested_subjects(remarks: List[str], characters: List[str]) -> List[str]:
    action_pattern = re.compile(r"\b(show|add|bring|introduce|include|insert|appear|enters?|comes? in)\b")
    clauses = [
        clause.strip().lower()
        for remark in remarks
        for clause in re.split(r"[.;\n]", remark or "")
        if clause.strip()
    ]

    requested = []
    for clause in clauses:
        action_match = action_pattern.search(clause)
        if not action_match:
            continue
        candidates = []
        for character in characters:
            if not character:
                continue
            character_match = re.search(rf"\b{re.escape(character.lower())}\b", clause[action_match.end():])
            if character_match:
                candidates.append((character_match.start(), character))
        if candidates:
            _position, character = min(candidates, key=lambda item: item[0])
            if character not in requested:
                requested.append(character)
    return requested


# 3. Main Workflow Function
def generate_video_prompts_from_plan(
    plan_json_path: str,
    project_name: str,
    assets_dir: str,
    openai_client,
    openai_model: str = OPENAI_REASONING_MODEL,
    output_base_dir: str = "data"
) -> str:
    """
    Generates video prompts and constructs the final Seedance 2.0 payloads.
    Resolves assets, extracts clip frame context, handles previous clip continuity,
    integrates direct Base64 audio sync (Way A), and outputs video_prompts.json.
    """
    abs_plan_path = os.path.abspath(plan_json_path)
    if not os.path.exists(abs_plan_path):
        raise FileNotFoundError(f"Generation plan JSON not found at: {abs_plan_path}")

    with open(abs_plan_path, 'r', encoding='utf-8') as f:
        plan_items = json.load(f)

    abs_assets_dir = os.path.abspath(assets_dir)
    project_assets_dir = os.path.join(abs_assets_dir, project_name)
    results = []

    # Keep track of generated prompts by clip name to supply as continuity context
    previous_clip_prompts = {}
    segmind_cache = SupabaseAssetUrlCache()

    for item in plan_items:
        clip_name = item.get("clip_used")
        generation_type = item.get("generation_type", "none")
        remarks = item.get("remarks_to_process", [])

        # Skip if generation_type is none
        if generation_type == "none" or not clip_name:
            results.append({
                "clip_used": clip_name,
                "clip_occurrence": item.get("clip_occurrence"),
                "generation_type": "none",
                "video_model_prompt": "",
                "video_provider": "segmind",
                "segmind_payload_status": "skipped",
                "segmind_payload_error": "No visual generation requested for this segment.",
                "status": "none"
            })
            continue

        clip_start_s = item.get("clip_start_s", 0.0)
        clip_end_s = item.get("clip_end_s", 5.0)
        clip_duration = clip_end_s - clip_start_s

        # Setup persistent frames directory under the prompts output path
        clip_frames_dir = os.path.abspath(os.path.join(output_base_dir, project_name, "clip_frames", os.path.splitext(clip_name)[0]))
        os.makedirs(clip_frames_dir, exist_ok=True)

        contents = []
        feedback_context_items = [
            {"timestamp": timestamp, "remark": remark}
            for timestamp, remark in zip(item.get("source_feedback_timestamps", []), remarks)
        ] or [{"remark": remark} for remark in remarks]
        clip_context = analyze_clip_context(
            project_name=project_name,
            clip_name=clip_name,
            clip_occurrence=item.get("clip_occurrence"),
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            clip_duration_s=clip_duration,
            assets_dir=assets_dir,
            output_base_dir=output_base_dir,
            client=openai_client,
            provider="openai",
            model=openai_model,
            feedback_items=feedback_context_items,
        )

        # 2. Extract current clip frames (1 frame/sec) for visual guidance
        current_clip_path = os.path.join(project_assets_dir, "06_clips", "_raw", clip_name)
        current_frames = clip_context.get("frame_paths") or []
        if not current_frames and os.path.exists(current_clip_path):
            current_frames = extract_frames_per_second(current_clip_path, clip_frames_dir, clip_duration)
        for path in current_frames[:5]: # Limit to 5 frames
            contents.append({
                "type": "input_image",
                "image_url": _file_data_url(path),
                "detail": "auto"
            })

        selected_assets = []
        absent_requested_subjects = [
            subject for subject in item.get("absent_requested_subjects", [])
            if subject
        ]
        if not absent_requested_subjects:
            absent_requested_subjects = _infer_explicitly_requested_subjects(
                remarks,
                item.get("characters_present", []),
            )

        # 1. Resolve preproduction reference sheets after current frames are known.
        if generation_type == "complex":
            characters = item.get("characters_present", [])
            for subject in absent_requested_subjects:
                if subject not in characters:
                    characters.append(subject)
            loc = item.get("location")
            selected_assets = resolve_reference_assets(project_assets_dir, characters, loc)
            missing_subject_sheets = _missing_required_subject_sheets(
                selected_assets,
                absent_requested_subjects,
            )
            if missing_subject_sheets:
                warning = (
                    "Missing required character sheet(s) for explicitly requested absent subject(s): "
                    + ", ".join(missing_subject_sheets)
                )
                results.append({
                    "clip_used": clip_name,
                    "clip_occurrence": item.get("clip_occurrence"),
                    "category": "video",
                    "generation_type": generation_type,
                    "video_model_prompt": "",
                    "selected_assets": selected_assets,
                    "clip_frame_paths": current_frames,
                    "clip_context_path": clip_context.get("clip_context_path"),
                    "clip_segment_path": clip_context.get("clip_segment_path"),
                    "clip_context_summary": clip_context.get("summary", ""),
                    "clip_context_status": clip_context.get("status"),
                    "referenced_frames": [],
                    "referenced_frame_paths": [],
                    "referenced_frame_labels": [],
                    "audio_used": item.get("audio_used"),
                    "audio_path": item.get("audio_path"),
                    "is_dialogue_active": False,
                    "generate_audio": False,
                    "has_reference_audio": False,
                    "ratio": "9:16",
                    "duration": 5,
                    "status": "warning",
                    "quality_warning": warning,
                    "prompt_generation_review_required": True,
                    "absent_requested_subjects": absent_requested_subjects,
                    "missing_required_subject_sheets": missing_subject_sheets,
                    "explanation": warning,
                    "video_provider": "segmind",
                    "segmind_payload_status": "skipped",
                    "segmind_payload_error": warning,
                })
                continue

            for path in selected_assets[:3]: # Limit to 3 assets
                contents.append({
                    "type": "input_image",
                    "image_url": _file_data_url(path),
                    "detail": "auto"
                })

        # 2.5. Attach referenced frames from other clips/timestamps.
        # Labels make intercut/cutaway requests addressable in the generated prompt.
        referenced_frames = item.get("referenced_frames", [])
        attached_referenced_frames = []
        referenced_frame_paths = []
        for ref_frame in referenced_frames:
            ref_path = ref_frame.get("frame_path")
            if ref_path and os.path.exists(ref_path):
                if ref_path in referenced_frame_paths:
                    continue
                ref_label = f"@ref{len(attached_referenced_frames) + 1}"
                attached_ref = dict(ref_frame)
                attached_ref["label"] = ref_label
                attached_referenced_frames.append(attached_ref)
                referenced_frame_paths.append(ref_path)
                contents.append({
                    "type": "input_image",
                    "image_url": _file_data_url(ref_path),
                    "detail": "auto"
                })

        # 3. Handle previous clip continuity (last frame as first_frame role)
        requires_continuity = item.get("requires_previous_clip_continuity", False)
        prev_clip_name = item.get("previous_clip")
        first_frame_url = None
        last_frame_path = None

        if requires_continuity and prev_clip_name:
            prev_clip_path = os.path.join(project_assets_dir, "06_clips", "_raw", prev_clip_name)
            if os.path.exists(prev_clip_path):
                # Query actual previous clip duration to avoid seeking past the end
                prev_duration = get_video_duration(prev_clip_path) or 5.0
                potential_last_frame_path = extract_last_frame(prev_clip_path, clip_frames_dir, prev_duration)
                if potential_last_frame_path and current_frames:
                    # Let OpenAI visually compare the last frame of the previous clip
                    # and the first frame of the current clip to decide if continuity is actually required
                    is_continuity_valid = verify_visual_continuity(
                        openai_client=openai_client,
                        openai_model=openai_model,
                        prev_last_frame_path=potential_last_frame_path,
                        curr_first_frame_path=current_frames[0],
                        remarks=remarks
                    )
                    if is_continuity_valid:
                        last_frame_path = potential_last_frame_path
                        first_frame_url = _file_data_url(last_frame_path)
                        # Add to contents to show the model the visual cut frame
                        contents.append({
                            "type": "input_image",
                            "image_url": first_frame_url,
                            "detail": "auto"
                        })
                    else:
                        print(f"Visual continuity rejected by OpenAI for transition from {prev_clip_name} to {clip_name}.")
                        # Clean up the unused extracted frame file
                        try:
                            os.remove(potential_last_frame_path)
                        except Exception:
                            pass

        # 4. Dialogue and Audio Base64 setup (Way A)
        is_dialogue = item.get("is_dialogue_active", False)
        audio_name = item.get("audio_used")
        audio_path = _resolve_audio_path(project_assets_dir, audio_name, item.get("audio_path"))
        audio_url = None
        trimmed_audio_path = None
        audio_trim_error = None
        audio_trim_start_s = clip_start_s
        audio_trim_end_s = clip_end_s
        audio_trim_duration_s = max(0.0, clip_end_s - clip_start_s)

        # Trim from the full sequence mix using timeline cut times. Do not
        # transcribe or fall back to the full mix; the trimmed file is the
        # source of truth for dialogue timing and cadence.
        if audio_trim_duration_s > 0 and audio_path and os.path.exists(audio_path):
            clean_audio_identifier = os.path.basename((audio_name or audio_path or "audio").replace("\\", "/"))
            audio_stem = os.path.splitext(clean_audio_identifier)[0]
            trimmed_audio_name = f"trimmed_{audio_stem}.mp3"
            trimmed_audio_path = os.path.join(clip_frames_dir, trimmed_audio_name)
            success = extract_audio_segment(
                input_audio_path=audio_path,
                output_audio_path=trimmed_audio_path,
                start_s=audio_trim_start_s,
                end_s=audio_trim_end_s
            )
            if success and os.path.exists(trimmed_audio_path):
                audio_url = _file_data_url(trimmed_audio_path)
            else:
                audio_trim_error = (
                    f"Failed to trim audio from {audio_trim_start_s:.3f}s "
                    f"to {audio_trim_end_s:.3f}s for {clip_name}."
                )
                print(f"Warning: {audio_trim_error}")

        if audio_url:
            is_dialogue = True
        else:
            is_dialogue = False

        # 5. Build prompt instruction text
        clip_context_for_prompt = {
            key: clip_context.get(key)
            for key in [
                "summary",
                "visible_characters",
                "expressions",
                "gaze",
                "actions",
                "blocking",
                "camera_framing",
                "location",
                "continuity_notes",
                "uncertainty_flags",
                "status",
            ]
        }
        prompt_instruction = (
            f"You are modifying the video clip: \"{clip_name}\" "
            f"(Duration: {clip_duration:.2f}s, Segment: {item.get('clip_start_tc')} to {item.get('clip_end_tc')}).\n\n"
            f"Saved Clip Understanding Context: {json.dumps(clip_context_for_prompt, ensure_ascii=False)}\n\n"
            f"Client Feedback Remarks: {json.dumps(remarks)}\n\n"
            f"Tasks:\n"
            f"1. Analyze the provided current clip frames (Image references) showing the starting layout, camera positioning, and composition.\n"
        )

        if attached_referenced_frames:
            prompt_instruction += "   Referenced frames from other parts of the video are attached as labeled input images:\n"
            for ref_frame in attached_referenced_frames:
                clip_label = ref_frame.get("clip_used") or "unknown clip"
                timestamp_label = ref_frame.get("timestamp") or "unknown timestamp"
                reason = ref_frame.get("reason") or "visual reference requested by feedback"
                prompt_instruction += (
                    f"   - {ref_frame['label']}: frame from \"{clip_label}\" at {timestamp_label} "
                    f"(reason: {reason}; path: {ref_frame.get('frame_path')})\n"
                )
            prompt_instruction += (
                "   When the feedback asks for an intercut, cutaway, insert, or reaction from another clip, "
                "explicitly name the matching @ref label in the final prompt and describe that reference frame's "
                "character, expression, composition, and source clip so the video model knows exactly what shot to cut to.\n"
            )

        if selected_assets:
            prompt_instruction += "2. Analyze the preproduction reference sheets (Image references) to ensure character wardrobe and location geometry are strictly consistent.\n"
        else:
            prompt_instruction += "2. This is a simple camera or transition edit; do not worry about character/location reference sheets.\n"

        if absent_requested_subjects:
            prompt_instruction += (
                "   The following characters are explicitly requested by feedback but are not present in the current clip frames: "
                f"{', '.join(absent_requested_subjects)}. Use their matching character sheet reference(s) for identity, wardrobe, scale, and styling. "
                "Introduce them only to satisfy the client feedback, while preserving the current shot's camera, blocking, lighting, and continuity.\n"
            )

        if item.get("compound_feedback"):
            timestamps = item.get("source_feedback_timestamps", [])
            prompt_instruction += (
                "   This is a compound same-clip edit. Merge all client remarks into one coherent staged action instead of treating them as separate alternatives. "
                f"Source feedback timestamps: {json.dumps(timestamps)}.\n"
            )

        if first_frame_url:
            prev_prompt_desc = previous_clip_prompts.get(prev_clip_name, "n/A")
            prompt_instruction += (
                f"3. Maintain strict visual continuity with the previous shot (Last Frame attached as reference image). "
                f"The previous shot prompt was: \"{prev_prompt_desc}\". Ensure lighting, setting layout, and character wardrobe match seamlessly across the cut.\n"
            )

        if audio_url and is_dialogue:
            prompt_instruction += (
                "4. Dialogue/audio is active in this shot. The video model will receive a trimmed reference audio "
                f"segment from timeline {audio_trim_start_s:.3f}s to {audio_trim_end_s:.3f}s. "
                "In the final prompt, include one concise instruction that facial performance, lip movement, pauses, "
                "and delivery synchronize to the supplied reference audio. Do not quote or invent transcript text.\n"
            )

        prompt_instruction += (
            "\nGenerate a highly detailed cinematic text prompt for a video generation model (like Runway Gen-3, Sora, or Seedance 2.0) "
            "that revises this shot to implement the client's remarks.\n\n"
            "IMPORTANT: The preproduction assets and referenced images/frames provided to the video model are low-resolution and compressed "
            "(processed via 'magick -quality 15 -colors 64 -resize 256x256'). You must include a specific instruction in the final generated prompt "
            "directing the video model to ignore the pixelation, low resolution, or color artifacts of these reference inputs, and to upscale and enhance "
            "them to generate a high-quality, sharp, photorealistic, non-pixelated cinematic output."
        )

        contents.append(prompt_instruction)

        # 6. Invoke OpenAI to generate the prompt
        system_instruction = (
            "You are a professional film director and AI video prompt engineer. "
            "You generate extremely detailed, consistent cinematic prompts for video generation models, "
            "ensuring client edits are precisely followed and preproduction design assets are visually respected."
        )

        llm_response = generate_structured(
            provider="openai",
            client=openai_client,
            model=openai_model,
            contents=contents,
            schema=PromptResult,
            system_instruction=system_instruction
        )

        video_prompt = llm_response.get("video_model_prompt", "")
        explanation = llm_response.get("explanation", "")

        # Save this prompt in memory for subsequent clips' continuity checks
        previous_clip_prompts[clip_name] = video_prompt

        # Append result payload
        result_payload = {
            "clip_used": clip_name,
            "clip_occurrence": item.get("clip_occurrence"),
            "category": "video",
            "generation_type": generation_type,
            "video_model_prompt": video_prompt,
            "selected_assets": selected_assets,
            "clip_frame_paths": current_frames,
            "clip_context_path": clip_context.get("clip_context_path"),
            "clip_segment_path": clip_context.get("clip_segment_path"),
            "clip_context_summary": clip_context.get("summary", ""),
            "clip_context_status": clip_context.get("status"),
            "referenced_frames": [{k: v for k, v in ref.items() if k != "label"} for ref in attached_referenced_frames],
            "referenced_frame_paths": referenced_frame_paths,
            "referenced_frame_labels": [ref_frame["label"] for ref_frame in attached_referenced_frames],
            "audio_used": audio_name,
            "audio_path": audio_path,
            "audio_url": audio_url,
            "audio_reference_path": trimmed_audio_path if audio_url else None,
            "trimmed_audio_path": trimmed_audio_path if audio_url else None,
            "audio_trim_start_s": audio_trim_start_s if audio_url else None,
            "audio_trim_end_s": audio_trim_end_s if audio_url else None,
            "audio_trim_duration_s": audio_trim_duration_s if audio_url else None,
            "audio_trim_source": "sequence_timeline" if audio_url else None,
            "audio_trim_error": audio_trim_error,
            "is_dialogue_active": is_dialogue,
            "generate_audio": False,
            "has_reference_audio": bool(audio_url),
            "ratio": "9:16",
            "duration": 5,
            "status": "success",
            "quality_warning": None,
            "prompt_generation_review_required": False,
            "compound_feedback": bool(item.get("compound_feedback")),
            "source_feedback_timestamps": item.get("source_feedback_timestamps", []),
            "absent_requested_subjects": absent_requested_subjects,
            "missing_required_subject_sheets": [],
            "explanation": explanation
        }
        if first_frame_url and last_frame_path:
            result_payload["first_frame_url"] = first_frame_url
            result_payload["initial_frame_image_path"] = last_frame_path

        result_payload = attach_prepared_segmind_payload(
            result_payload,
            cache=segmind_cache,
        )
        results.append(result_payload)

    # Save output prompts
    abs_output_base = os.path.abspath(output_base_dir)
    dest_dir = os.path.join(abs_output_base, project_name)
    os.makedirs(dest_dir, exist_ok=True)

    output_prompts_path = os.path.join(dest_dir, "video_prompts.json")
    with open(output_prompts_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return os.path.abspath(output_prompts_path)

if __name__ == "__main__":
    # Example usage
    plan_path = "data/project-red-and-green/generation_plan_short.json"
    project = "project-red-and-green"
    assets_directory = "assets"

    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompts_path = generate_video_prompts_from_plan(
        plan_json_path=plan_path,
        project_name=project,
        assets_dir=assets_directory,
        openai_client=openai_client
    )
    print(f"Video prompts generated at: {prompts_path}")
