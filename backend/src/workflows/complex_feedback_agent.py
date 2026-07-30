import json
import os
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.clustering import find_matching_clip_occurrence
from src.utils import parse_timestamp_to_seconds
from src.workflows.referenced_frames import extract_frame_at_offset


class ComplexReferencePlan(BaseModel):
    timestamp: Optional[str] = Field(
        None,
        description="Timeline timestamp for the source reference frame, or null when unresolved.",
    )
    reason: str = Field(
        "",
        description="Why this source frame should influence the target clip.",
    )
    clip_used: Optional[str] = Field(
        None,
        description="Source clip that contains the referenced frame.",
    )
    offset_s: Optional[float] = Field(
        None,
        description="Seconds into the source clip for the referenced frame.",
    )
    frame_path: Optional[str] = Field(
        None,
        description="Extracted local frame path when available.",
    )
    usage: Literal[
        "expression_anchor",
        "pose_anchor",
        "continuity_anchor",
        "cutaway_insert",
        "style_reference",
        "unspecified_visual_anchor",
    ] = Field(
        "unspecified_visual_anchor",
        description="How prompt generation should use this reference.",
    )


class ComplexFeedbackPlan(BaseModel):
    feedback_index: Optional[int] = Field(
        None,
        description="Index of the target feedback segment in feedback.json.",
    )
    target_clip: Optional[str] = Field(None, description="Clip receiving the edit.")
    target_start_s: Optional[float] = Field(None, description="Target clip timeline start.")
    target_end_s: Optional[float] = Field(None, description="Target clip timeline end.")
    remarks: list[str] = Field(default_factory=list, description="Client remarks being planned.")
    requires_cross_reference: bool = Field(
        False,
        description="True when another timestamp, clip, or visual source must be inspected.",
    )
    references: list[ComplexReferencePlan] = Field(default_factory=list)
    generation_strategy: Literal[
        "normal_pipeline",
        "attach_reference_frame",
        "match_reference_continuity",
        "request_user_selection",
    ] = "normal_pipeline"
    needs_user_approval: bool = Field(
        False,
        description="True when the agent is unsure or would expand scope/cost.",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Planner confidence from 0 to 1.",
    )
    evidence: list[str] = Field(default_factory=list)


class ComplexFeedbackInvestigationResult(BaseModel):
    plans: list[ComplexFeedbackPlan] = Field(default_factory=list)


REFERENCE_HINT_PATTERN = re.compile(
    r"\b("
    r"use|match|copy|same|similar|like|from|earlier|later|previous|next|"
    r"timeframe|time\s*frame|frame|shot|clip|reference|expression|pose|"
    r"reaction|continuity|cutaway|insert"
    r")\b",
    re.IGNORECASE,
)
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def complex_feedback_agent_available() -> bool:
    try:
        import agents  # noqa: F401
    except Exception:
        return False
    return True


def _remarks_for_segment(segment: dict) -> list[str]:
    return [
        str(item.get("remark") or "").strip()
        for item in segment.get("feedback_items", [])
        if str(item.get("remark") or "").strip()
    ]


def segment_needs_complex_reference_investigation(segment: dict) -> bool:
    text = " ".join(_remarks_for_segment(segment))
    if not text:
        return False
    return bool(TIMESTAMP_PATTERN.search(text) and REFERENCE_HINT_PATTERN.search(text))


def complex_feedback_mode_from_env() -> str:
    explicit_mode = os.environ.get("LOKA_COMPLEX_FEEDBACK_MODE", "").strip().lower()
    if explicit_mode in {"auto", "agent", "deterministic", "off"}:
        return explicit_mode
    legacy_enabled = os.environ.get("LOKA_USE_COMPLEX_FEEDBACK_AGENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return "agent" if legacy_enabled else "auto"


def _reference_usage(reason: str, remark_text: str) -> str:
    text = f"{reason} {remark_text}".lower()
    if any(word in text for word in ("expression", "smile", "reaction", "look")):
        return "expression_anchor"
    if any(word in text for word in ("pose", "hand", "body", "gesture", "standing")):
        return "pose_anchor"
    if any(word in text for word in ("continuity", "match", "same", "previous", "next")):
        return "continuity_anchor"
    if any(word in text for word in ("cutaway", "insert", "intercut")):
        return "cutaway_insert"
    return "unspecified_visual_anchor"


def build_deterministic_complex_feedback_plans(
    feedback_data: list[dict],
    timeline_data: dict,
    *,
    project_name: str,
    assets_dir: str,
    output_base_dir: str,
    extract_frames: bool = False,
) -> ComplexFeedbackInvestigationResult:
    """Build a local, testable baseline plan for explicit timestamp references."""
    video_timeline = timeline_data.get("video_timeline", [])
    project_assets_dir = os.path.abspath(os.path.join(assets_dir, project_name))
    referenced_frames_dir = os.path.abspath(os.path.join(output_base_dir, project_name, "referenced_frames"))
    plans: list[ComplexFeedbackPlan] = []

    for feedback_index, segment in enumerate(feedback_data):
        remarks = _remarks_for_segment(segment)
        if not segment_needs_complex_reference_investigation(segment):
            continue

        references: list[ComplexReferencePlan] = []
        remark_text = " ".join(remarks)
        own_timestamps = {
            str(item.get("timestamp"))
            for item in segment.get("feedback_items", [])
            if item.get("timestamp")
        }
        for timestamp in TIMESTAMP_PATTERN.findall(remark_text):
            if timestamp in own_timestamps:
                continue
            ref_sec = parse_timestamp_to_seconds(timestamp)
            if ref_sec is None:
                continue
            _clip_index, matched_clip = find_matching_clip_occurrence(video_timeline, ref_sec)
            if not matched_clip:
                references.append(
                    ComplexReferencePlan(
                        timestamp=timestamp,
                        reason="Timestamp was mentioned as a visual reference but does not match the timeline.",
                    )
                )
                continue

            clip_name = matched_clip.get("clip")
            clip_start_s = float(matched_clip.get("start_s") or 0.0)
            clip_duration_s = float(matched_clip.get("duration_s") or 0.0)
            offset_s = max(0.0, min(ref_sec - clip_start_s, clip_duration_s))
            frame_path = os.path.abspath(
                os.path.join(
                    referenced_frames_dir,
                    f"ref_{timestamp.replace(':', '_')}_{os.path.splitext(str(clip_name))[0]}.jpg",
                )
            )

            if extract_frames and clip_name:
                clip_path = os.path.join(project_assets_dir, "06_clips", "_raw", str(clip_name))
                if not os.path.exists(clip_path) or not extract_frame_at_offset(clip_path, offset_s, frame_path):
                    frame_path = None

            references.append(
                ComplexReferencePlan(
                    timestamp=timestamp,
                    reason=f"Explicit cross-timeframe visual reference from feedback: {remark_text}",
                    clip_used=clip_name,
                    offset_s=offset_s,
                    frame_path=frame_path if extract_frames else None,
                    usage=_reference_usage("", remark_text),
                )
            )

        plans.append(
            ComplexFeedbackPlan(
                feedback_index=feedback_index,
                target_clip=segment.get("clip_used"),
                target_start_s=segment.get("clip_start_s"),
                target_end_s=segment.get("clip_end_s"),
                remarks=remarks,
                requires_cross_reference=bool(references),
                references=references,
                generation_strategy="attach_reference_frame" if references else "request_user_selection",
                needs_user_approval=not references,
                confidence=0.7 if references else 0.2,
                evidence=["Detected timestamp plus cross-reference wording in feedback."],
            )
        )

    return ComplexFeedbackInvestigationResult(plans=plans)


def feedback_items_to_complex_segments(feedback_items: list[dict], timeline_data: dict) -> list[dict]:
    video_timeline = timeline_data.get("video_timeline", [])
    segments = []
    for index, item in enumerate(feedback_items):
        _occurrence, clip = find_matching_clip_occurrence(
            video_timeline,
            parse_timestamp_to_seconds(item.get("timestamp")),
        )
        segments.append(
            {
                "clip_used": clip.get("clip") if clip else item.get("clip_used"),
                "clip_start_s": clip.get("start_s") if clip else item.get("clip_start_s"),
                "clip_end_s": clip.get("end_s") if clip else item.get("clip_end_s"),
                "feedback_items": [dict(item)],
                "_flat_feedback_index": index,
            }
        )
    return segments


def _materialize_reference_frames(
    investigation: ComplexFeedbackInvestigationResult,
    timeline_data: dict,
    *,
    project_name: str,
    assets_dir: str,
    output_base_dir: str,
) -> ComplexFeedbackInvestigationResult:
    video_timeline = timeline_data.get("video_timeline", [])
    project_assets_dir = os.path.abspath(os.path.join(assets_dir, project_name))
    referenced_frames_dir = os.path.abspath(os.path.join(output_base_dir, project_name, "referenced_frames"))

    for plan in investigation.plans:
        for ref in plan.references:
            if ref.frame_path or not ref.timestamp:
                continue
            ref_sec = parse_timestamp_to_seconds(ref.timestamp)
            if ref_sec is None:
                continue
            _clip_index, matched_clip = find_matching_clip_occurrence(video_timeline, ref_sec)
            if not matched_clip:
                continue
            clip_name = str(ref.clip_used or matched_clip.get("clip") or "")
            if not clip_name:
                continue
            clip_start_s = float(matched_clip.get("start_s") or 0.0)
            clip_duration_s = float(matched_clip.get("duration_s") or 0.0)
            offset_s = ref.offset_s
            if offset_s is None:
                offset_s = max(0.0, min(ref_sec - clip_start_s, clip_duration_s))
            frame_path = os.path.abspath(
                os.path.join(
                    referenced_frames_dir,
                    f"ref_{ref.timestamp.replace(':', '_')}_{os.path.splitext(clip_name)[0]}.jpg",
                )
            )
            clip_path = os.path.join(project_assets_dir, "06_clips", "_raw", clip_name)
            if os.path.exists(clip_path) and extract_frame_at_offset(clip_path, float(offset_s), frame_path):
                ref.clip_used = clip_name
                ref.offset_s = float(offset_s)
                ref.frame_path = frame_path

    return investigation


def _make_complex_feedback_agent_tools(feedback_data: list[dict], timeline_data: dict):
    from agents import function_tool

    @function_tool
    def list_candidate_segments() -> str:
        """Return feedback segments that appear to need cross-timeframe reference planning."""
        candidates = []
        for index, segment in enumerate(feedback_data):
            if segment_needs_complex_reference_investigation(segment):
                candidates.append(
                    {
                        "feedback_index": index,
                        "target_clip": segment.get("clip_used"),
                        "target_start_s": segment.get("clip_start_s"),
                        "target_end_s": segment.get("clip_end_s"),
                        "remarks": _remarks_for_segment(segment),
                    }
                )
        return json.dumps(candidates, ensure_ascii=False)

    @function_tool
    def resolve_timeline_timestamp(timestamp: str) -> str:
        """Resolve a sequence timestamp to the containing source clip and clip-local offset."""
        ref_sec = parse_timestamp_to_seconds(timestamp)
        if ref_sec is None:
            return json.dumps({"timestamp": timestamp, "error": "invalid timestamp"})
        _clip_index, matched_clip = find_matching_clip_occurrence(
            timeline_data.get("video_timeline", []),
            ref_sec,
        )
        if not matched_clip:
            return json.dumps({"timestamp": timestamp, "error": "timestamp outside video timeline"})
        clip_start_s = float(matched_clip.get("start_s") or 0.0)
        clip_duration_s = float(matched_clip.get("duration_s") or 0.0)
        return json.dumps(
            {
                "timestamp": timestamp,
                "clip_used": matched_clip.get("clip"),
                "offset_s": max(0.0, min(ref_sec - clip_start_s, clip_duration_s)),
                "clip_start_s": matched_clip.get("start_s"),
                "clip_end_s": matched_clip.get("end_s"),
            },
            ensure_ascii=False,
        )

    return [list_candidate_segments, resolve_timeline_timestamp]


def run_complex_feedback_agent(
    feedback_data: list[dict],
    timeline_data: dict,
    *,
    model: str,
) -> ComplexFeedbackInvestigationResult:
    """Run the optional OpenAI Agents SDK planner for complex feedback references."""
    try:
        from agents import Agent, Runner
    except Exception as exc:
        raise RuntimeError("openai-agents is not installed. Install it to run the SDK pilot.") from exc

    agent = Agent(
        name="Complex Feedback Investigator",
        model=model,
        instructions=(
            "You are a post-production planning agent for Loka15 Studio. "
            "Investigate only feedback that asks to use, match, copy, or reference another "
            "timestamp, clip, shot, frame, expression, pose, reaction, or continuity source. "
            "Use tools to inspect candidate feedback segments and resolve referenced timestamps. "
            "Do not perform media extraction, video generation, file writes, or external uploads. "
            "Return a structured plan that can be consumed by the deterministic backend pipeline."
        ),
        tools=_make_complex_feedback_agent_tools(feedback_data, timeline_data),
        output_type=ComplexFeedbackInvestigationResult,
    )
    result = Runner.run_sync(
        agent,
        "Create complex cross-timeframe reference plans for the supplied project artifacts.",
    )
    return result.final_output


def _merge_reference_metadata(segment: dict, item: dict, ref: ComplexReferencePlan) -> bool:
    if not ref.frame_path:
        return False
    metadata = {
        "timestamp": ref.timestamp,
        "reason": ref.reason,
        "frame_path": os.path.abspath(ref.frame_path),
        "clip_used": ref.clip_used,
        "offset_s": ref.offset_s,
        "usage": ref.usage,
        "source": "complex_feedback_agent",
    }
    item.setdefault("referenced_frames", [])
    segment.setdefault("referenced_frames", [])
    changed = False
    if not any(existing.get("frame_path") == metadata["frame_path"] for existing in item["referenced_frames"]):
        item["referenced_frames"].append(metadata)
        changed = True
    if not any(existing.get("frame_path") == metadata["frame_path"] for existing in segment["referenced_frames"]):
        segment["referenced_frames"].append(metadata)
        changed = True
    return changed


def apply_complex_feedback_plans_to_feedback(
    feedback_data: list[dict],
    investigation: ComplexFeedbackInvestigationResult,
) -> bool:
    """Attach resolved agent reference metadata to feedback items without changing media files."""
    changed = False
    for plan in investigation.plans:
        if plan.feedback_index is None or plan.feedback_index >= len(feedback_data):
            continue
        segment = feedback_data[plan.feedback_index]
        feedback_items = segment.get("feedback_items", [])
        if not feedback_items:
            continue
        target_item = feedback_items[0]
        for ref in plan.references:
            changed = _merge_reference_metadata(segment, target_item, ref) or changed
        if plan.requires_cross_reference:
            segment["complex_reference_plan"] = plan.model_dump()
            changed = True
    return changed


def investigate_complex_feedback_references(
    feedback_json_path: str,
    timeline_json_path: str,
    project_name: str,
    assets_dir: str,
    *,
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    output_base_dir: str = "data",
    use_agents_sdk: bool = False,
) -> str:
    """
    Investigate complex cross-timeframe feedback and attach resolved references to feedback.json.

    The OpenAI Agents SDK is used only when explicitly enabled and installed. Frame extraction
    and JSON mutation remain deterministic backend work.
    """
    abs_feedback_path = os.path.abspath(feedback_json_path)
    abs_timeline_path = os.path.abspath(timeline_json_path)
    if not os.path.exists(abs_feedback_path):
        raise FileNotFoundError(f"Feedback JSON file not found at: {abs_feedback_path}")
    if not os.path.exists(abs_timeline_path):
        raise FileNotFoundError(f"Timeline JSON file not found at: {abs_timeline_path}")

    with open(abs_feedback_path, "r", encoding="utf-8") as file:
        feedback_data = json.load(file)
    with open(abs_timeline_path, "r", encoding="utf-8") as file:
        timeline_data = json.load(file)

    if provider == "openai" and use_agents_sdk:
        investigation = run_complex_feedback_agent(feedback_data, timeline_data, model=model)
        investigation = _materialize_reference_frames(
            investigation,
            timeline_data,
            project_name=project_name,
            assets_dir=assets_dir,
            output_base_dir=output_base_dir,
        )
    else:
        investigation = build_deterministic_complex_feedback_plans(
            feedback_data,
            timeline_data,
            project_name=project_name,
            assets_dir=assets_dir,
            output_base_dir=output_base_dir,
            extract_frames=True,
        )

    changed = apply_complex_feedback_plans_to_feedback(feedback_data, investigation)
    if changed:
        with open(abs_feedback_path, "w", encoding="utf-8") as file:
            json.dump(feedback_data, file, ensure_ascii=False, indent=2)
    return abs_feedback_path


def enrich_feedback_items_with_complex_references(
    feedback_items: list[dict],
    timeline_data: dict,
    project_name: str,
    assets_dir: str,
    *,
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    output_base_dir: str = "data",
    mode: Optional[str] = None,
) -> tuple[list[dict], bool, str]:
    """Enrich flat feedback items used by the legacy pipeline with cross-timeframe references."""
    resolved_mode = mode or complex_feedback_mode_from_env()
    if resolved_mode == "off":
        return feedback_items, False, resolved_mode

    segments = feedback_items_to_complex_segments(feedback_items, timeline_data)
    if not any(segment_needs_complex_reference_investigation(segment) for segment in segments):
        return feedback_items, False, resolved_mode

    should_use_sdk = (
        provider == "openai"
        and resolved_mode in {"auto", "agent"}
        and (resolved_mode == "agent" or complex_feedback_agent_available())
    )
    if should_use_sdk:
        investigation = run_complex_feedback_agent(segments, timeline_data, model=model)
        investigation = _materialize_reference_frames(
            investigation,
            timeline_data,
            project_name=project_name,
            assets_dir=assets_dir,
            output_base_dir=output_base_dir,
        )
    elif resolved_mode == "agent":
        raise RuntimeError("LOKA_COMPLEX_FEEDBACK_MODE=agent requires provider=openai and openai-agents installed.")
    else:
        investigation = build_deterministic_complex_feedback_plans(
            segments,
            timeline_data,
            project_name=project_name,
            assets_dir=assets_dir,
            output_base_dir=output_base_dir,
            extract_frames=True,
        )

    changed = apply_complex_feedback_plans_to_feedback(segments, investigation)
    if not changed:
        return feedback_items, False, resolved_mode

    enriched = [dict(item) for item in feedback_items]
    for segment in segments:
        flat_index = segment.get("_flat_feedback_index")
        if flat_index is None or flat_index >= len(enriched):
            continue
        item_refs = (segment.get("feedback_items") or [{}])[0].get("referenced_frames")
        if item_refs:
            enriched[flat_index]["referenced_frames"] = item_refs
        if segment.get("complex_reference_plan"):
            enriched[flat_index]["complex_reference_plan"] = segment["complex_reference_plan"]
    return enriched, True, resolved_mode
