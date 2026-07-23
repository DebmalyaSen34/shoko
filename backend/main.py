import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Optional
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

from src.generator import Provider, generate_video_prompts_batch
from src.selector import scan_visual_reference_assets
from src.parser import parse_prproj_to_json
from src.categorizer import process_feedback
from src.clustering import cluster_feedback_by_clip
from config.settings import DEFAULT_FEEDBACK_JSON_PATH, DEFAULT_ASSETS_DIR, DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_REPORT, DEFAULT_TIMELINE_PATH, OPENAI_REASONING_MODEL

# Workflows for the full agentic loop
from src.workflows.timeline_extraction import extract_timeline_from_project
from src.workflows.feedback_parsing import parse_and_align_feedback
from src.workflows.referenced_frames import analyze_and_extract_referenced_frames
from src.workflows.generation_planner import plan_generation_workflow
from src.workflows.prompt_generation import (
    extract_audio_segment,
    extract_dialogue_only,
    generate_video_prompts_from_plan,
    transcribe_audio_dialogue,
)
from src.workflows.clip_context import analyze_clip_context
from src.workflows.project_setup import setup_project_workspace
from scripts.generate_seedance_video import SupabaseAssetUrlCache, attach_prepared_segmind_payload

load_dotenv()

AGENTIC_STAGES = ("setup", "timeline", "feedback", "references", "plan", "prompts")


def _project_output_dir(output_base_dir: str, project_name: str) -> str:
    return os.path.abspath(os.path.join(output_base_dir, project_name))


def _agentic_artifact_paths(output_base_dir: str, project_name: str) -> dict[str, str]:
    project_dir = _project_output_dir(output_base_dir, project_name)
    return {
        "project_dir": project_dir,
        "timeline": os.path.join(project_dir, "timeline.json"),
        "feedback": os.path.join(project_dir, "feedback.json"),
        "references": os.path.join(project_dir, "referenced_frames"),
        "plan": os.path.join(project_dir, "generation_plan.json"),
        "prompts": os.path.join(project_dir, "video_prompts.json"),
        "manifest": os.path.join(project_dir, "agent_run_manifest.json"),
    }


def _resolve_audio_path(assets_dir: str, audio_name: Optional[str]) -> Optional[str]:
    if not audio_name:
        return None
    candidates = [
        os.path.join(assets_dir, "04_audio", audio_name),
        os.path.join(os.getcwd(), "assets", os.path.basename(assets_dir), "04_audio", audio_name),
        os.path.abspath(os.path.join("assets", os.path.basename(assets_dir), "04_audio", audio_name)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _matching_audio_segment(timeline_data: dict, clip_start_s: Optional[float]) -> dict:
    audio_tracks = (timeline_data.get("audio_timeline") or {}).get("dedicated_audio_tracks", [])
    if clip_start_s is None:
        return {}
    for audio in audio_tracks:
        if audio.get("start_s", 0) <= clip_start_s < audio.get("end_s", 0):
            return audio
    return audio_tracks[0] if audio_tracks else {}


def _attach_legacy_audio_reference(
    item: dict,
    *,
    timeline_data: dict,
    assets_dir: str,
    output_json: str,
) -> dict:
    if item.get("audio_reference_path") and item.get("trimmed_audio_path"):
        item["has_reference_audio"] = True
        item["is_dialogue_active"] = bool(item.get("dialogue_text") or item.get("audio_url"))
        item.setdefault("generate_audio", False)
        return item

    clip_start_s = item.get("clip_start_s")
    clip_end_s = item.get("clip_end_s")
    audio_segment = _matching_audio_segment(timeline_data, clip_start_s)
    audio_name = audio_segment.get("clip")
    audio_path = _resolve_audio_path(assets_dir, audio_name)
    item["audio_used"] = audio_name
    item["audio_path"] = audio_path

    if clip_start_s is None or clip_end_s is None:
        item["audio_trim_error"] = "Clip timeline bounds are missing; audio reference was not prepared."
        return item
    if not audio_path:
        item["audio_trim_error"] = f"Audio source file not found for {audio_name or 'timeline audio'}."
        return item

    trim_start_s = float(clip_start_s)
    trim_end_s = float(clip_end_s)
    trim_duration_s = max(0.0, trim_end_s - trim_start_s)
    if trim_duration_s <= 0:
        item["audio_trim_error"] = "Clip audio trim duration is zero."
        return item

    clip_slug = os.path.splitext(os.path.basename(str(item.get("clip_used") or "clip")))[0]
    safe_clip_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in clip_slug).strip("._-")
    audio_dir = os.path.join(os.path.dirname(output_json), "audio_references", safe_clip_slug or "clip")
    os.makedirs(audio_dir, exist_ok=True)
    trimmed_audio_path = os.path.abspath(
        os.path.join(audio_dir, f"trimmed_{os.path.splitext(os.path.basename(audio_name))[0]}.mp3")
    )

    if extract_audio_segment(audio_path, trimmed_audio_path, trim_start_s, trim_end_s):
        item.update(
            {
                "audio_reference_path": trimmed_audio_path,
                "trimmed_audio_path": trimmed_audio_path,
                "audio_trim_start_s": trim_start_s,
                "audio_trim_end_s": trim_end_s,
                "audio_trim_duration_s": trim_duration_s,
                "audio_trim_source": "sequence_timeline",
                "audio_trim_error": None,
                "has_reference_audio": True,
            }
        )
    else:
        item["audio_trim_error"] = (
            f"Failed to trim audio from {trim_start_s:.3f}s to {trim_end_s:.3f}s for {item.get('clip_used')}."
        )
    return item


def _prepare_legacy_dialogue_context(
    cluster: dict,
    *,
    timeline_data: dict,
    assets_dir: str,
    output_json: str,
    openai_client: Optional[OpenAI],
    openai_model: str,
) -> dict:
    clip = cluster.get("matched_clip") or {}
    clip_name = clip.get("clip")
    clip_start_s = clip.get("start_s")
    clip_end_s = clip.get("end_s")
    audio_segment = _matching_audio_segment(timeline_data, clip_start_s)
    audio_name = audio_segment.get("clip")
    audio_path = _resolve_audio_path(assets_dir, audio_name)

    context = {
        "audio_used": audio_name,
        "audio_path": audio_path,
        "audio_transcript": "",
        "dialogue_text": "",
        "dialogue_language": "none",
        "dialogue_extraction_reasoning": "",
        "is_dialogue_active": False,
        "has_reference_audio": False,
    }
    if clip_start_s is None or clip_end_s is None:
        context["audio_trim_error"] = "Clip timeline bounds are missing; audio reference was not prepared."
        cluster["dialogue_context"] = context
        return context
    if not audio_path:
        context["audio_trim_error"] = f"Audio source file not found for {audio_name or 'timeline audio'}."
        cluster["dialogue_context"] = context
        return context

    trim_start_s = float(clip_start_s)
    trim_end_s = float(clip_end_s)
    trim_duration_s = max(0.0, trim_end_s - trim_start_s)
    if trim_duration_s <= 0:
        context["audio_trim_error"] = "Clip audio trim duration is zero."
        cluster["dialogue_context"] = context
        return context

    clip_slug = os.path.splitext(os.path.basename(str(clip_name or "clip")))[0]
    safe_clip_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in clip_slug).strip("._-")
    audio_dir = os.path.join(os.path.dirname(output_json), "audio_references", safe_clip_slug or "clip")
    os.makedirs(audio_dir, exist_ok=True)
    trimmed_audio_path = os.path.abspath(
        os.path.join(audio_dir, f"trimmed_{os.path.splitext(os.path.basename(audio_name or 'audio'))[0]}.mp3")
    )

    if not extract_audio_segment(audio_path, trimmed_audio_path, trim_start_s, trim_end_s):
        context["audio_trim_error"] = (
            f"Failed to trim audio from {trim_start_s:.3f}s to {trim_end_s:.3f}s for {clip_name}."
        )
        cluster["dialogue_context"] = context
        return context

    context.update(
        {
            "audio_reference_path": trimmed_audio_path,
            "trimmed_audio_path": trimmed_audio_path,
            "audio_trim_start_s": trim_start_s,
            "audio_trim_end_s": trim_end_s,
            "audio_trim_duration_s": trim_duration_s,
            "audio_trim_source": "sequence_timeline",
            "audio_trim_error": None,
            "has_reference_audio": True,
        }
    )
    if openai_client is None:
        context["audio_trim_error"] = "OPENAI_API_KEY is required to transcribe dialogue for lip sync."
        cluster["dialogue_context"] = context
        return context

    try:
        audio_transcript = transcribe_audio_dialogue(openai_client, trimmed_audio_path)
        dialogue_result = extract_dialogue_only(
            openai_client,
            openai_model,
            audio_transcript,
            [item.get("remark", "") for item in cluster.get("feedback_items", [])],
        )
        context.update(
            {
                "audio_transcript": audio_transcript,
                "dialogue_text": dialogue_result.get("dialogue_text", "") if dialogue_result.get("has_dialogue") else "",
                "dialogue_language": dialogue_result.get("language", "none"),
                "dialogue_extraction_reasoning": dialogue_result.get("reasoning", ""),
                "is_dialogue_active": bool(dialogue_result.get("has_dialogue")),
            }
        )
    except Exception as exc:
        context["audio_trim_error"] = f"Audio transcription failed: {exc}"

    cluster["dialogue_context"] = context
    return context


def _feedback_lanes(category: str) -> set[str]:
    normalized = (category or "video").lower()
    if normalized == "both":
        return {"video", "audio"}
    if normalized == "audio":
        return {"audio"}
    return {"video"}


def _same_generation_group(clicked: dict, candidate: dict) -> bool:
    clicked_group = clicked.get("group_id")
    candidate_group = candidate.get("group_id")
    if clicked_group and candidate_group:
        same_group = clicked_group == candidate_group
    else:
        same_group = (
            clicked.get("clip_used") == candidate.get("clip_used")
            and clicked.get("clip_occurrence") == candidate.get("clip_occurrence")
        )
    return same_group and bool(
        _feedback_lanes(clicked.get("category", "video"))
        & _feedback_lanes(candidate.get("category", "video"))
    )


def _expand_feedback_selection(feedback_list: list[dict], index: Optional[int]) -> list[tuple[int, dict]]:
    if index is None:
        return list(enumerate(feedback_list))
    if index < 0 or index >= len(feedback_list):
        print(f"Error: Index {index} is out of bounds.", file=sys.stderr)
        sys.exit(1)

    clicked = feedback_list[index]
    sibling_indexes = clicked.get("sibling_raw_indexes")
    if isinstance(sibling_indexes, list):
        selected_indexes = {
            sibling_index
            for sibling_index in sibling_indexes + [index]
            if isinstance(sibling_index, int) and 0 <= sibling_index < len(feedback_list)
        }
        selected_indexes = {
            sibling_index
            for sibling_index in selected_indexes
            if _same_generation_group(clicked, feedback_list[sibling_index])
        }
    else:
        selected_indexes = {
            candidate_index
            for candidate_index, candidate in enumerate(feedback_list)
            if _same_generation_group(clicked, candidate)
        }

    if not selected_indexes:
        selected_indexes = {index}
    return [(candidate_index, feedback_list[candidate_index]) for candidate_index in sorted(selected_indexes)]


def _normalize_stage(stage: Optional[str], argument_name: str) -> Optional[str]:
    if stage is None:
        return None
    normalized = stage.lower()
    if normalized not in AGENTIC_STAGES:
        valid = ", ".join(AGENTIC_STAGES)
        raise ValueError(f"{argument_name} must be one of: {valid}")
    return normalized


def _normalize_force_stages(force_stage: Optional[str | list[str] | tuple[str, ...]]) -> set[str]:
    if force_stage is None:
        return set()
    if isinstance(force_stage, str):
        values = [force_stage]
    else:
        values = list(force_stage)
    return {
        normalized
        for value in values
        if (normalized := _normalize_stage(value, "force_stage")) is not None
    }


def _selected_agentic_stages(stage: Optional[str], from_stage: Optional[str]) -> list[str]:
    if stage and from_stage:
        raise ValueError("Use either stage or from_stage, not both.")
    if stage:
        return [stage]
    if from_stage:
        start_index = AGENTIC_STAGES.index(from_stage) + 1
        return list(AGENTIC_STAGES[start_index:])
    return list(AGENTIC_STAGES)


def _record_stage(
    manifest: dict,
    stage: str,
    status: str,
    artifact: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    entry = {"status": status}
    if artifact:
        entry["artifact"] = os.path.abspath(artifact)
    if reason:
        entry["reason"] = reason
    manifest["stages"][stage] = entry


def _write_agentic_manifest(
    manifest: dict,
    manifest_path: str,
    final_output: str,
    final_output_path: Optional[str],
) -> None:
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_output"] = final_output
    manifest["final_output_path"] = os.path.abspath(final_output_path) if final_output_path else None
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def _artifact_ready(manifest: dict, stage: str, path: str) -> bool:
    stage_status = manifest.get("stages", {}).get(stage, {}).get("status")
    return stage_status in {"ran", "reused"} or os.path.exists(path)

def write_markdown_report(results: list, report_path: str):
    """Write a clean, readable report summarizing the agent outcomes."""
    md = [
        "# Video Feedback Agent Analysis & Prompt Generation Report\n",
        "This report summarizes matched timeline segments, associated assets, and prompts generated to address client feedback.\n"
    ]

    for i, r in enumerate(results):
        cat = r["category"].upper()
        clip = r["matched_clip"] or "None"
        start = r["clip_start_tc"] or "N/A"
        end = r["clip_end_tc"] or "N/A"
        dur = r["clip_duration_s"] or 0.0

        md.append(f"### Cluster {i + 1}: `{clip}` ({cat})")
        md.append("**Client Feedback**:\n")
        for item in r.get("feedback_items", []):
            timestamp = item.get("timestamp") or "No Timestamp"
            md.append(f"- [{timestamp}] {item.get('remark', '')}")
        md.append("")
        
        md.append("| Detail | Value |")
        md.append("| --- | --- |")
        md.append(f"| **Timeline Clip** | `{clip}` |")
        md.append(f"| **Clip Duration** | {dur}s ({start} to {end}) |")
        if cat != "AUDIO":
            md.append(f"| **Selected Context Assets** | {', '.join([os.path.basename(f) for f in r.get('selected_assets', [])]) or 'None'} |")
            md.append(f"| **Prompt Format** | {r.get('prompt_format') or 'plain_text'} |")
        md.append("\n")
        
        if cat == "AUDIO":
            md.append("#### Generated Audio Instruction")
            md.append(f"```text\n{r.get('audio_instruction')}\n```")
        else:
            reference_legend = r.get("reference_legend")
            if reference_legend:
                md.append("#### Seedance 2.0 Reference Legend")
                md.append(f"```text\n{reference_legend}\n```")
            initial_frame_prompt = r.get("initial_frame_prompt")
            if initial_frame_prompt:
                md.append("#### Generated Initial Frame Prompt")
                md.append(f"```text\n{initial_frame_prompt}\n```")
            initial_frame_image_path = r.get("initial_frame_image_path")
            if initial_frame_image_path:
                md.append("#### Generated Initial Frame Image")
                md.append(f"![Initial Frame](file://{initial_frame_image_path})\n")
            clip_frame_paths = r.get("clip_frame_paths") or []
            if clip_frame_paths:
                md.append("#### Extracted Clip Frames")
                for index, path in enumerate(clip_frame_paths, start=1):
                    md.append(f"![Frame {index}](file://{path})")
                md.append("")
            md.append("#### Generated Seedance 2.0 Prompt")
            md.append(f"```text\n{r.get('video_model_prompt')}\n```")
            
            quality_report = r.get("quality_report")
            if quality_report:
                md.append("#### Prompt Quality Check Report")
                passed_str = "✅ PASSED" if quality_report.get("passed") else "⚠️ WARNING / FAILED"
                md.append(f"**Overall Status**: {passed_str}\n")
                
                md.append("| Check Category | Evaluation |")
                md.append("| --- | --- |")
                md.append(f"| **Feedback Adherence** | {quality_report.get('feedback_adherence')} |")
                md.append(f"| **Clothing Consistency** | {quality_report.get('clothing_consistency')} |")
                md.append(f"| **Vague Feedback Resolution** | {quality_report.get('vague_feedback_resolution')} |")
                
                forbidden_words = quality_report.get("forbidden_terms_found", [])
                forbidden_str = ", ".join([f"`{w}`" for w in forbidden_words]) if forbidden_words else "None"
                md.append(f"| **Forbidden/Negative Terms Found** | {forbidden_str} |")
                md.append("")
                
                suggestions = quality_report.get("suggestions", [])
                if suggestions:
                    md.append("**Actionable Suggestions for Improvement**:")
                    for sug in suggestions:
                        md.append(f"- {sug}")
                    md.append("")
            
        md.append(f"**Explanation**: {r.get('explanation')}\n")
        md.append("-" * 40 + "\n")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Generated Markdown report saved to: {report_path}")

def run_pipeline(
    feedback_path: str,
    timeline_path: str,
    assets_dir: str,
    output_json: str,
    output_report: str,
    index: Optional[int] = None,
    batch_size: int = 5,
    provider: Provider = "openai",
    initial_frames_dir: Optional[str] = None,
    video_frames_dir: Optional[str] = None,
    continuity_frame_path: Optional[str] = None,
    continuity_note: Optional[str] = None
):
    """Orchestrates the dynamic feedback agent workflow."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Check API key
    provider_label = "OpenAI" if provider == "openai" else "Gemini"
    print(f"[Step 2/6] Initializing {provider_label} Client... ", end="", flush=True)
    api_key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.environ.get(api_key_name)
    if not api_key:
        print(f"\nError: {api_key_name} environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if provider == "openai":
        client = OpenAI(api_key=api_key)
    else:
        client = genai.Client(api_key=api_key)
    transcription_client = client if provider == "openai" else None
    if transcription_client is None and os.environ.get("OPENAI_API_KEY"):
        transcription_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Verify file existence
    for filepath in [feedback_path, timeline_path]:
        if not os.path.exists(filepath):
            print(f"\nError: Required file '{filepath}' not found.", file=sys.stderr)
            sys.exit(1)
            
    # Load inputs
    with open(feedback_path, 'r') as f:
        feedback_list = json.load(f)
    with open(timeline_path, 'r') as f:
        timeline_data = json.load(f)
        
    video_timeline = timeline_data.get("video_timeline", [])
    print("Done")

    # 1. Scan visual references without spending a model request
    print(f"[Step 3/6] Scanning visual references in '{assets_dir}'... ", end="", flush=True)
    reference_assets = scan_visual_reference_assets(assets_dir)
    print(f"Done (Discovered {len(reference_assets)} references)")
    
    # 2. Determine target remarks
    remarks_to_process = _expand_feedback_selection(feedback_list, index)
    if index is not None:
        print(f"[Step 4/6] Setting target remark to index {index} ({len(remarks_to_process)} item(s) to process)...")
    else:
        print(f"[Step 4/6] Processing all {len(feedback_list)} feedback items...")
        
    feedback_to_process = [item for _, item in remarks_to_process]
    clusters = cluster_feedback_by_clip(feedback_to_process, video_timeline)
    video_clusters = [cluster for cluster in clusters if cluster["category"] == "video"]
    frame_size = timeline_data.get("frame_size", "unknown")
    project_data_dir_for_output = os.path.dirname(output_json)
    inferred_project_name = os.path.basename(project_data_dir_for_output) or "project"
    inferred_output_base_dir = os.path.dirname(project_data_dir_for_output) or "data"
    for cluster in video_clusters:
        cluster["frame_size"] = frame_size
        clip = cluster.get("matched_clip")
        clip_name = clip.get("clip") if clip else None
        if clip_name:
            cluster["clip_context"] = analyze_clip_context(
                project_name=inferred_project_name,
                clip_name=clip_name,
                clip_occurrence=cluster.get("clip_occurrence"),
                clip_start_s=clip.get("start_s") if clip else None,
                clip_end_s=clip.get("end_s") if clip else None,
                clip_duration_s=clip.get("duration_s") if clip else None,
                assets_dir=assets_dir,
                output_base_dir=inferred_output_base_dir,
                client=client,
                provider=provider,
                feedback_items=cluster.get("feedback_items", []),
            )
        if continuity_frame_path:
            cluster["continuity_reference_frame_path"] = os.path.abspath(continuity_frame_path)
            cluster["continuity_reference_note"] = continuity_note or (
                "Use this previous clip last frame as the continuity reference and first-frame anchor."
            )
        _prepare_legacy_dialogue_context(
            cluster,
            timeline_data=timeline_data,
            assets_dir=assets_dir,
            output_json=output_json,
            openai_client=transcription_client,
            openai_model=OPENAI_REASONING_MODEL,
        )
    planned_requests = (
        (len(video_clusters) + batch_size - 1) // batch_size
        if video_clusters
        else 0
    )
    print(
        f"[Step 5/6] Processing {len(video_clusters)} video cluster(s) "
        f"in {planned_requests} {provider_label} request batch(es)..."
    )

    batch_results = []
    if video_clusters:
        batch_results = generate_video_prompts_batch(
            client=client,
            clusters=video_clusters,
            reference_assets=reference_assets,
            assets_dir=assets_dir,
            batch_size=batch_size,
            provider=provider,
            initial_frames_dir=initial_frames_dir or os.path.join(
                os.path.dirname(output_json),
                "initial_frames",
            ),
            video_frames_dir=video_frames_dir or os.path.join(
                os.path.dirname(output_json),
                "video_frames",
            ),
            generate_initial_frame=False,
        )

    results = []
    segmind_cache = SupabaseAssetUrlCache()
    for cluster, generated in zip(video_clusters, batch_results):
        feedback_items = cluster["feedback_items"]
        clip = cluster["matched_clip"]
        clip_name = clip.get("clip") if clip else None
        clip_duration_s = clip.get("duration_s") if clip else None

        result_item = {
            "category": "video",
            "clip_used": clip_name,
            "generation_type": generated.get("prompt_format") or "complex",
            "feedback_items": feedback_items,
            "matched_clip": clip_name,
            "clip_occurrence": cluster["clip_occurrence"],
            "clip_start_tc": clip.get("start_tc") if clip else None,
            "clip_end_tc": clip.get("end_tc") if clip else None,
            "clip_start_s": clip.get("start_s") if clip else None,
            "clip_end_s": clip.get("end_s") if clip else None,
            "clip_duration_s": clip_duration_s,
            "selected_assets": generated.get("selected_assets", []),
            "prompt_format": generated.get("prompt_format"),
            "reference_legend": generated.get("reference_legend", ""),
            "initial_frame_prompt": generated.get("initial_frame_prompt", ""),
            "initial_frame_image_path": generated.get("initial_frame_image_path", ""),
            "clip_frame_paths": generated.get("clip_frame_paths", []),
            "clip_context_path": (cluster.get("clip_context") or {}).get("clip_context_path"),
            "clip_segment_path": (cluster.get("clip_context") or {}).get("clip_segment_path"),
            "clip_context_summary": (cluster.get("clip_context") or {}).get("summary", ""),
            "clip_context_status": (cluster.get("clip_context") or {}).get("status"),
            "video_model_prompt": generated.get("video_model_prompt"),
            "explanation": generated.get("explanation"),
            "ratio": "9:16",
            "duration": 5,
            "generate_audio": False,
            "status": generated.get("status"),
            "quality_warning": generated.get("quality_warning"),
            "quality_report": generated.get("quality_report"),
        }
        dialogue_context = cluster.get("dialogue_context") or {}
        result_item.update({
            "audio_used": dialogue_context.get("audio_used"),
            "audio_path": dialogue_context.get("audio_path"),
            "audio_reference_path": dialogue_context.get("audio_reference_path"),
            "trimmed_audio_path": dialogue_context.get("trimmed_audio_path"),
            "audio_trim_start_s": dialogue_context.get("audio_trim_start_s"),
            "audio_trim_end_s": dialogue_context.get("audio_trim_end_s"),
            "audio_trim_duration_s": dialogue_context.get("audio_trim_duration_s"),
            "audio_trim_source": dialogue_context.get("audio_trim_source"),
            "audio_trim_error": dialogue_context.get("audio_trim_error"),
            "audio_transcript": dialogue_context.get("audio_transcript", ""),
            "dialogue_text": dialogue_context.get("dialogue_text", ""),
            "dialogue_language": dialogue_context.get("dialogue_language", "none"),
            "dialogue_extraction_reasoning": dialogue_context.get("dialogue_extraction_reasoning", ""),
            "is_dialogue_active": bool(dialogue_context.get("is_dialogue_active")),
            "has_reference_audio": bool(dialogue_context.get("has_reference_audio")),
        })
        result_item = _attach_legacy_audio_reference(
            result_item,
            timeline_data=timeline_data,
            assets_dir=assets_dir,
            output_json=output_json,
        )
        result_item = attach_prepared_segmind_payload(
            result_item,
            cache=segmind_cache,
        )
        results.append(result_item)
    # 4. Save results
    print("\n[Step 6/6] Writing output reports... ", end="", flush=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    write_markdown_report(results, output_report)
    print("Done")
    print("\nPipeline execution completed successfully.")

def run_agentic_loop(
    prproj_path: Optional[str],
    raw_feedback_path: Optional[str],
    project_name: str,
    assets_dir: str,
    output_base_dir: str = "data",
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    project_package_path: Optional[str] = None,
    stage: Optional[str] = None,
    from_stage: Optional[str] = None,
    force_stage: Optional[str | list[str] | tuple[str, ...]] = None,
) -> str:
    """Executes an artifact-driven agentic loop using the modular workflows."""
    print(f"=== Starting Agentic Loop for Project: {project_name} ===")

    stage = _normalize_stage(stage, "stage")
    from_stage = _normalize_stage(from_stage, "from_stage")
    force_stages = _normalize_force_stages(force_stage)
    selected_stages = _selected_agentic_stages(stage, from_stage)
    artifact_paths = _agentic_artifact_paths(output_base_dir, project_name)
    os.makedirs(artifact_paths["project_dir"], exist_ok=True)

    manifest = {
        "project_name": project_name,
        "provider": provider,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "stage" if stage else "from_stage" if from_stage else "full",
        "requested_stage": stage,
        "requested_from_stage": from_stage,
        "force_stages": sorted(force_stages),
        "stages": {},
    }

    # Initialize clients
    provider_label = "OpenAI" if provider == "openai" else "Gemini"
    api_key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.environ.get(api_key_name)
    if not api_key:
        print(f"\nError: {api_key_name} environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if provider == "openai":
        openai_client = OpenAI(api_key=api_key)
        client = openai_client
    else:
        client = genai.Client(api_key=api_key)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Warning: OPENAI_API_KEY is not set. Some OpenAI-specific workflow steps may fail.")
            openai_client = None
        else:
            openai_client = OpenAI(api_key=openai_key)

    if project_package_path and "setup" not in selected_stages:
        print("[Loop setup] Preparing project workspace from package...")
        _project_dir, prproj_path = setup_project_workspace(
            zip_path=project_package_path,
            project_name=project_name,
            assets_dir=assets_dir,
        )
        _record_stage(manifest, "setup", "ran", _project_dir, "preflight for selected stage")

    final_output = "none"
    final_output_path = None
    timeline_json_path = artifact_paths["timeline"]
    feedback_json_path = artifact_paths["feedback"]
    plan_json_path = artifact_paths["plan"]
    prompts_json_path = artifact_paths["prompts"]

    for current_stage in AGENTIC_STAGES:
        if current_stage not in selected_stages:
            if current_stage in manifest["stages"]:
                continue
            artifact = artifact_paths.get(current_stage)
            status = "reused" if from_stage == current_stage and artifact and os.path.exists(artifact) else "skipped"
            _record_stage(manifest, current_stage, status, artifact, "outside selected stage range")
            continue

        if current_stage == "setup":
            if project_package_path:
                print("[Loop setup] Preparing project workspace from package...")
                _project_dir, prproj_path = setup_project_workspace(
                    zip_path=project_package_path,
                    project_name=project_name,
                    assets_dir=assets_dir,
                )
                _record_stage(manifest, current_stage, "ran", _project_dir)
                final_output = current_stage
                final_output_path = _project_dir
            else:
                _record_stage(manifest, current_stage, "skipped", reason="no project package provided")
            continue

        if current_stage == "timeline":
            should_reuse = os.path.exists(timeline_json_path) and current_stage not in force_stages
            if should_reuse:
                print(f"Timeline JSON already exists for project '{project_name}'. Reusing it.")
                _record_stage(manifest, current_stage, "reused", timeline_json_path)
            else:
                if not prproj_path:
                    raise ValueError("prproj_path is required to extract the timeline.")
                print("[Loop timeline] Extracting timeline from Premiere Pro project...")
                timeline_json_path = extract_timeline_from_project(
                    prproj_path=prproj_path,
                    project_name=project_name,
                    output_base_dir=output_base_dir,
                )
                _record_stage(manifest, current_stage, "ran", timeline_json_path)
            final_output = current_stage
            final_output_path = timeline_json_path
            continue

        if current_stage == "feedback":
            should_reuse = os.path.exists(feedback_json_path) and current_stage not in force_stages
            if should_reuse:
                print(f"Feedback JSON already exists for project '{project_name}'. Reusing it.")
                _record_stage(manifest, current_stage, "reused", feedback_json_path)
            else:
                if not openai_client:
                    raise ValueError("OPENAI_API_KEY is required for feedback parsing.")
                if not raw_feedback_path:
                    raise ValueError("raw_feedback_path is required to parse feedback.")
                if not _artifact_ready(manifest, "timeline", timeline_json_path):
                    raise FileNotFoundError(f"Timeline JSON not found at: {timeline_json_path}")
                print("[Loop feedback] Parsing raw feedback and aligning to timeline...")
                feedback_json_path = parse_and_align_feedback(
                    feedback_file_path=raw_feedback_path,
                    timeline_json_path=timeline_json_path,
                    project_name=project_name,
                    openai_client=openai_client,
                    openai_model=model,
                    output_base_dir=output_base_dir,
                )
                _record_stage(manifest, current_stage, "ran", feedback_json_path)
            final_output = current_stage
            final_output_path = feedback_json_path
            continue

        if current_stage == "references":
            if not _artifact_ready(manifest, "feedback", feedback_json_path):
                raise FileNotFoundError(f"Feedback JSON not found at: {feedback_json_path}")
            if not _artifact_ready(manifest, "timeline", timeline_json_path):
                raise FileNotFoundError(f"Timeline JSON not found at: {timeline_json_path}")
            print("[Loop references] Checking cross-referenced timestamps and extracting frames...")
            feedback_json_path = analyze_and_extract_referenced_frames(
                feedback_json_path=feedback_json_path,
                timeline_json_path=timeline_json_path,
                project_name=project_name,
                assets_dir=assets_dir,
                client=client,
                provider=provider,
                model=model,
                output_base_dir=output_base_dir,
            )
            _record_stage(manifest, current_stage, "ran", feedback_json_path)
            final_output = current_stage
            final_output_path = feedback_json_path
            continue

        if current_stage == "plan":
            if not openai_client:
                raise ValueError("OPENAI_API_KEY is required for generation planning.")
            if not _artifact_ready(manifest, "feedback", feedback_json_path):
                raise FileNotFoundError(f"Feedback JSON not found at: {feedback_json_path}")
            should_reuse = (
                from_stage == current_stage
                and os.path.exists(plan_json_path)
                and current_stage not in force_stages
            )
            if should_reuse:
                print(f"Generation plan already exists for project '{project_name}'. Reusing it.")
                _record_stage(manifest, current_stage, "reused", plan_json_path)
            else:
                print("[Loop plan] Creating video generation plan...")
                plan_json_path = plan_generation_workflow(
                    feedback_json_path=feedback_json_path,
                    project_name=project_name,
                    openai_client=openai_client,
                    openai_model=model,
                    output_base_dir=output_base_dir,
                )
                _record_stage(manifest, current_stage, "ran", plan_json_path)
            final_output = current_stage
            final_output_path = plan_json_path
            continue

        if current_stage == "prompts":
            if not openai_client:
                raise ValueError("OPENAI_API_KEY is required for prompt generation.")
            if not _artifact_ready(manifest, "plan", plan_json_path):
                raise FileNotFoundError(f"Generation plan JSON not found at: {plan_json_path}")
            should_reuse = (
                from_stage == current_stage
                and os.path.exists(prompts_json_path)
                and current_stage not in force_stages
            )
            if should_reuse:
                print(f"Video prompts already exist for project '{project_name}'. Reusing them.")
                _record_stage(manifest, current_stage, "reused", prompts_json_path)
            else:
                print("[Loop prompts] Generating video prompts from plan...")
                prompts_json_path = generate_video_prompts_from_plan(
                    plan_json_path=plan_json_path,
                    project_name=project_name,
                    assets_dir=assets_dir,
                    openai_client=openai_client,
                    openai_model=model,
                    output_base_dir=output_base_dir,
                )
                _record_stage(manifest, current_stage, "ran", prompts_json_path)
            final_output = current_stage
            final_output_path = prompts_json_path

    _write_agentic_manifest(manifest, artifact_paths["manifest"], final_output, final_output_path)
    print(f"Agentic loop finished at stage '{final_output}'. Manifest: {artifact_paths['manifest']}")
    if not final_output_path:
        raise ValueError("No agentic stage produced an output.")
    return final_output_path

def preprocessing( prproj_file_path: str, timeline_json_path: str, feedback_path: str, feedback_json_path: str) -> tuple[str, str]:
    """Performs preprocessing steps: parses the PRPROJ file to JSON, processes feedback, and ensures required files exist."""
    print("[Step 1/6] Performing preprocessing: parsing project file and processing feedback... ", end="", flush=True)
    
    # check if timeline_json_path exists, if not, parse the prproj file
    if not os.path.exists(timeline_json_path):
        print("\nTimeline JSON file not found. Parsing PRPROJ file...", end="", flush=True)
        parse_prproj_to_json(prproj_file_path, timeline_json_path)

    if not os.path.exists(feedback_json_path):
        print("\nFeedback JSON file not found. Processing feedback...", end="", flush=True)
        process_feedback(feedback_path, feedback_json_path)

    print("Done")
    
    return timeline_json_path, feedback_json_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the Seedance prompt generation pipeline."""

    parser = argparse.ArgumentParser(
        description="Generate Seedance prompts from feedback and timeline data."
    )
    parser.add_argument("--feedback-path", default=DEFAULT_FEEDBACK_JSON_PATH)
    parser.add_argument("--timeline-path", default=DEFAULT_TIMELINE_PATH)
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-report", default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--initial-frames-dir", default=None, help="Directory to save initial frame images. Defaults to a subdirectory of the output JSON path.")
    parser.add_argument("--video-frames-dir", default=None, help="Directory to save extracted video frames. Defaults to a subdirectory of the output JSON path.")
    parser.add_argument("--continuity-frame-path", default=None, help="Optional previous-clip last frame to use as continuity and first-frame anchor.")
    parser.add_argument("--continuity-note", default=None, help="Instruction describing how the continuity frame should be used.")
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        default=os.environ.get("MODEL_PROVIDER", "openai"),
        help="Model provider to use for prompt generation.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process only one feedback item by zero-based index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of video clusters per provider batch. Use 1 for single-video tests.",
    )
    parser.add_argument(
        "--prproj-path",
        default=None,
        help="Path to the Premiere Pro project (.prproj) file to run the full agentic loop."
    )
    parser.add_argument(
        "--project-package-path",
        default=None,
        help="Path to a zipped Premiere project package to prepare before running the agentic loop."
    )
    parser.add_argument(
        "--project-name",
        default="project-red-and-green",
        help="Name of the project to analyze."
    )
    parser.add_argument(
        "--output-base-dir",
        default="data",
        help="Base directory for agentic loop artifacts."
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Run the artifact-driven agentic workflow instead of the legacy prompt pipeline."
    )
    parser.add_argument(
        "--agentic-stage",
        choices=AGENTIC_STAGES,
        default=None,
        help="Run exactly one agentic stage for quick manual testing."
    )
    parser.add_argument(
        "--agentic-from",
        choices=AGENTIC_STAGES,
        default=None,
        help="Reuse the named stage artifact and run the stages after it."
    )
    parser.add_argument(
        "--force-stage",
        choices=AGENTIC_STAGES,
        action="append",
        default=[],
        help="Force an agentic stage to rerun even if its artifact exists. Can be repeated."
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()

    if args.agentic or args.prproj_path or args.project_package_path:
        run_agentic_loop(
            prproj_path=args.prproj_path,
            raw_feedback_path=args.feedback_path,
            project_name=args.project_name,
            assets_dir=args.assets_dir,
            output_base_dir=args.output_base_dir,
            provider=args.provider,
            model=OPENAI_REASONING_MODEL,
            project_package_path=args.project_package_path,
            stage=args.agentic_stage,
            from_stage=args.agentic_from,
            force_stage=args.force_stage,
        )
    else:
        run_pipeline(
            feedback_path=args.feedback_path,
            timeline_path=args.timeline_path,
            assets_dir=args.assets_dir,
            output_json=args.output_json,
            output_report=args.output_report,
            index=args.index,
            batch_size=args.batch_size,
            provider=args.provider,
            initial_frames_dir=args.initial_frames_dir,
            video_frames_dir=args.video_frames_dir,
            continuity_frame_path=args.continuity_frame_path,
            continuity_note=args.continuity_note,
        )
