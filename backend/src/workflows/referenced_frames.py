import os
import re
import sys
import json
import subprocess
from pydantic import BaseModel, Field
from typing import List, Optional

from src.generator.client import generate_structured, _detect_provider, _default_model_for_provider
from src.clustering import find_matching_clip_occurrence
from src.utils import parse_timestamp_to_seconds

class ReferencedTimestamp(BaseModel):
    timestamp: str = Field(..., description="The referenced timestamp in HH:MM:SS or MM:SS format mentioned in the remark that refers to another part of the video.")
    reason: str = Field(..., description="The description of what this referenced timestamp contains (e.g. 'Vir's evil smile reaction', 'the breaking glass').")

class FeedbackReferences(BaseModel):
    referenced_timestamps: List[ReferencedTimestamp] = Field(default_factory=list, description="List of timestamps referenced in the feedback remark that refer to another part of the video sequence.")

def extract_frame_at_offset(clip_path: str, offset_s: float, output_frame_path: str) -> bool:
    """Extract a single frame from clip_path at offset_s using ffmpeg."""
    os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
    command = [
        "ffmpeg", "-y", "-ss", f"{offset_s:.3f}", "-i", clip_path,
        "-frames:v", "1", "-q:v", "3", output_frame_path
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False
        )
        return completed.returncode == 0 and os.path.exists(output_frame_path)
    except Exception as exc:
        print(f"Warning: FFMPEG frame extraction failed for {clip_path} at offset {offset_s}: {exc}")
        return False

def analyze_and_extract_referenced_frames(
    feedback_json_path: str,
    timeline_json_path: str,
    project_name: str,
    assets_dir: str,
    client,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_base_dir: str = "data"
) -> str:
    """
    Analyzes feedback.json for remarks mentioning other timestamps, extracts those frames,
    and updates feedback.json with referenced frame metadata.
    """
    abs_feedback_path = os.path.abspath(feedback_json_path)
    abs_timeline_path = os.path.abspath(timeline_json_path)

    if not os.path.exists(abs_feedback_path):
        raise FileNotFoundError(f"Feedback JSON file not found at: {abs_feedback_path}")
    if not os.path.exists(abs_timeline_path):
        raise FileNotFoundError(f"Timeline JSON file not found at: {abs_timeline_path}")

    # Load inputs
    with open(abs_feedback_path, 'r', encoding='utf-8') as f:
        feedback_data = json.load(f)
    with open(abs_timeline_path, 'r', encoding='utf-8') as f:
        timeline_data = json.load(f)

    modified = False

    # Clean up existing duplicates in feedback_data (idempotency safety)
    for segment in feedback_data:
        if "referenced_frames" in segment:
            unique_refs = []
            for ref in segment["referenced_frames"]:
                if ref.get("frame_path") not in [r.get("frame_path") for r in unique_refs]:
                    unique_refs.append(ref)
            if len(unique_refs) != len(segment["referenced_frames"]):
                segment["referenced_frames"] = unique_refs
                modified = True
        
        for item in segment.get("feedback_items", []):
            if "referenced_frames" in item:
                unique_refs = []
                for ref in item["referenced_frames"]:
                    if ref.get("frame_path") not in [r.get("frame_path") for r in unique_refs]:
                        unique_refs.append(ref)
                if len(unique_refs) != len(item["referenced_frames"]):
                    item["referenced_frames"] = unique_refs
                    modified = True

    video_timeline = timeline_data.get("video_timeline", [])
    project_assets_dir = os.path.abspath(os.path.join(assets_dir, project_name))
    referenced_frames_dir = os.path.abspath(os.path.join(output_base_dir, project_name, "referenced_frames"))

    # Determine provider and model
    actual_provider = provider or _detect_provider(client)
    actual_model = model or _default_model_for_provider(actual_provider)

    print(f"Analyzing remarks in feedback for referenced timestamps using {actual_provider} / {actual_model}...")

    # Regex for fast check of timestamp mentions (e.g. 00:44, 1:04, etc.)
    timestamp_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')

    for segment in feedback_data:
        segment_ref_frames = []
        feedback_items = segment.get("feedback_items", [])
        
        for item in feedback_items:
            remark = item.get("remark", "")
            own_ts = item.get("timestamp")

            # Fast regex pre-check
            matches = timestamp_pattern.findall(remark)
            # Filter out own timestamp to avoid self-reference check
            other_matches = [m for m in matches if m != own_ts]

            if not other_matches:
                continue

            # Query LLM for structured analysis of referenced timestamps
            system_instruction = (
                "You are an expert film post-production supervisor. Analyze the editing feedback remark.\n"
                "Determine if the remark explicitly references any other timestamps (different from its own timestamp) "
                "to point to visual details, characters, actions, or assets from other parts of the video sequence.\n"
                "If it does reference other timestamps, extract them and the reason or description of the element."
            )
            prompt = (
                f"Remark at timestamp {own_ts or 'none'}: \"{remark}\"\n\n"
                f"Extract any referenced timestamps mentioned in the remark that point to other parts of the sequence."
            )

            try:
                response_json = generate_structured(
                    provider=actual_provider,
                    client=client,
                    model=actual_model,
                    contents=[prompt],
                    schema=FeedbackReferences,
                    system_instruction=system_instruction,
                    temperature=0.1
                )
                
                ref_ts_list = response_json.get("referenced_timestamps", [])
                
                for ref in ref_ts_list:
                    ref_ts = ref.get("timestamp")
                    reason = ref.get("reason", "")

                    if not ref_ts or ref_ts == own_ts:
                        continue

                    # Parse timestamp to seconds
                    ref_sec = parse_timestamp_to_seconds(ref_ts)
                    if ref_sec is None:
                        continue

                    # Find which clip contains this timestamp
                    clip_idx, matched_clip = find_matching_clip_occurrence(video_timeline, ref_sec)
                    if not matched_clip:
                        print(f"Warning: Referenced timestamp {ref_ts} does not match any clip on the timeline.")
                        continue

                    # Calculate offset
                    clip_name = matched_clip["clip"]
                    clip_start_s = matched_clip["start_s"]
                    clip_duration_s = matched_clip["duration_s"]
                    offset_s = max(0.0, min(ref_sec - clip_start_s, clip_duration_s))

                    # Deduplicate/clean existing referenced frames in item that have same timestamp but different clip
                    if "referenced_frames" in item:
                        original_len = len(item["referenced_frames"])
                        item["referenced_frames"] = [
                            f for f in item["referenced_frames"]
                            if f.get("timestamp") != ref_ts or f.get("clip_used") == clip_name
                        ]
                        if len(item["referenced_frames"]) != original_len:
                            modified = True

                    # Resolve path to source clip
                    clip_path = os.path.join(project_assets_dir, "06_clips", "_raw", clip_name)
                    if not os.path.exists(clip_path):
                        print(f"Warning: Raw clip file '{clip_path}' not found. Cannot extract referenced frame.")
                        continue

                    # Output path for frame
                    safe_ref_ts = ref_ts.replace(":", "_").replace(" ", "")
                    frame_filename = f"ref_{safe_ref_ts}_{os.path.splitext(clip_name)[0]}.jpg"
                    output_frame_path = os.path.join(referenced_frames_dir, frame_filename)

                    print(f"Extracting frame for referenced timestamp {ref_ts} (clip: {clip_name}, offset: {offset_s:.2f}s) to: {output_frame_path}")
                    
                    if extract_frame_at_offset(clip_path, offset_s, output_frame_path):
                        ref_frame_metadata = {
                            "timestamp": ref_ts,
                            "reason": reason,
                            "frame_path": os.path.abspath(output_frame_path),
                            "clip_used": clip_name,
                            "offset_s": offset_s
                        }
                        if not any(f["frame_path"] == ref_frame_metadata["frame_path"] for f in segment_ref_frames):
                            segment_ref_frames.append(ref_frame_metadata)
                        # Add metadata directly to the item as well
                        if "referenced_frames" not in item:
                            item["referenced_frames"] = []
                        if not any(f["frame_path"] == ref_frame_metadata["frame_path"] for f in item["referenced_frames"]):
                            item["referenced_frames"].append(ref_frame_metadata)
                            modified = True

            except Exception as e:
                print(f"Error analyzing referenced timestamps for remark: {remark}. Error: {e}")

        # Rebuild segment["referenced_frames"] to keep it in sync with items
        original_segment_refs = segment.get("referenced_frames", [])
        new_segment_refs = []
        for it in feedback_items:
            for ref_f in it.get("referenced_frames", []):
                if not any(f["frame_path"] == ref_f["frame_path"] for f in new_segment_refs):
                    new_segment_refs.append(ref_f)
        
        if "referenced_frames" in segment or new_segment_refs:
            if [f["frame_path"] for f in original_segment_refs] != [f["frame_path"] for f in new_segment_refs]:
                segment["referenced_frames"] = new_segment_refs
                modified = True

    if modified:
        with open(abs_feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        print(f"Updated feedback JSON saved to: {abs_feedback_path}")

    return abs_feedback_path
