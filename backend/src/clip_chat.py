import os
import re
import json
import uuid
from typing import Any, Literal, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field

from pathlib import Path

LEARNING_CHAT_ACTION_TYPES = {
    "save_prompt_feedback",
    "suggest_prompt_lesson",
    "approve_prompt_lesson",
    "revise_prompt_from_feedback",
    "run_prompt_learning_eval",
}
from src.prompt_feedback import PROMPT_FEEDBACK_RATINGS
import src.storage_paths as storage_paths
from src.storage_paths import (
    chat_agent_runs_path,
    chat_history_path,
    chat_workflow_intents_path,
    clip_chat_key,
    file_mtime_iso,
    now_iso,
    project_assets_dir,
    project_data_dir,
    read_json_file,
    stable_json_hash,
    stable_state_id,
    write_json_file,
)
from src.project_manager import (
    add_chat_media_item,
    append_project_event,
    build_clip_media_gallery,
    format_size,
    generated_videos_for_version,
    infer_media_type,
    make_chat_message,
    media_size_for_path,
    preview_path_for_media,
    static_url_for_path,
    wants_clip_media_gallery,
)
from src.clip_state import (
    _asset_identity,
    _selected_assets_from_selection,
    annotate_agent_run_freshness,
    build_clip_freshness,
    clip_selection_for_key,
    format_latest_prompt_reply,
    has_word,
    memory_store,
    normalize_selection_asset_refs,
    prompt_eval_case_store,
    prompt_learning_store,
    query_requests_latest_prompt,
    update_clip_selection,
    user_explicitly_requests_video_generation,
    wants_prompt_display,
)
from src.prompt_feedback import (
    PROMPT_FEEDBACK_CATEGORIES,
    _client_for_prompt_provider,
    _compact_applied_eval_cases,
    _eval_case_revision_text,
    _feedback_revision_text,
    _lesson_revision_text,
    _suggest_prompt_lesson,
    create_eval_case_from_feedback_item,
    revise_prompt_from_feedback,
    create_prompt_feedback_item,
    find_prompt_feedback_item,
    load_prompt_feedback,
    prompt_feedback_items_for_clip,
    prompt_feedback_summary_by_version,
    save_prompt_feedback,
)
from src.reference_frames import (
    _infer_asset_reason,
    _infer_asset_role,
    _parse_asset_selector,
    _parse_feedback_index,
    _parse_prompt_version_id,
    extract_reference_frame,
    parse_explicit_reference_frame_request,
)
from src.workflows.clip_context import analyze_clip_context, clip_context_dir
from src.generator.client import _default_model_for_provider, generate_structured
from src.generator.validator import merge_learning_eval_report, run_learning_eval, run_quality_check


def load_chat_history(project_name: str) -> dict:
    data = read_json_file(chat_history_path(project_name), {"clips": {}})
    if not isinstance(data, dict):
        return {"clips": {}}
    data.setdefault("clips", {})
    return data


def save_chat_history(project_name: str, history: dict) -> None:
    write_json_file(chat_history_path(project_name), history)


def load_chat_workflow_intents(project_name: str) -> dict:
    data = read_json_file(chat_workflow_intents_path(project_name), {"schema_version": 1, "feedback": {}})
    if not isinstance(data, dict):
        return {"schema_version": 1, "feedback": {}}
    return {"schema_version": 1, "feedback": data.get("feedback") if isinstance(data.get("feedback"), dict) else {}}


def save_chat_workflow_intents(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("feedback", {})
    write_json_file(chat_workflow_intents_path(project_name), data)


def save_pending_workflow_intents(project_name: str, actions: list[dict], message: str, context: dict) -> None:
    workflow_actions = [
        action for action in actions
        if action.get("type") == "execute_workflow" and action.get("feedback_index") is not None
    ]
    if not workflow_actions:
        return

    intents = load_chat_workflow_intents(project_name)
    feedback_intents = intents.setdefault("feedback", {})
    for action in workflow_actions:
        feedback_index = str(action["feedback_index"])
        feedback_intents[feedback_index] = {
            "created_at": now_iso(),
            "clip_index": context.get("clip_index"),
            "clip_key": context.get("clip_key"),
            "user_message": message,
            "continuity_reference": action.get("continuity_reference"),
            "autonomous": bool(action.get("autonomous")),
            "status": "pending",
        }
    save_chat_workflow_intents(project_name, intents)


def load_chat_agent_runs(project_name: str) -> dict:
    data = read_json_file(chat_agent_runs_path(project_name), {"schema_version": 1, "runs": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "runs": []}
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    return {"schema_version": 1, "runs": [run for run in runs if isinstance(run, dict)]}


def save_chat_agent_runs(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("runs", [])
    write_json_file(chat_agent_runs_path(project_name), data)


def recent_agent_runs_for_clip(project_name: str, clip_key: str, limit: int = 10) -> list[dict]:
    runs = load_chat_agent_runs(project_name).get("runs", [])
    matching = [run for run in runs if run.get("clip_key") == clip_key]
    matching.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return matching[:limit]


def create_chat_agent_run(project_name: str, run: dict) -> dict:
    data = load_chat_agent_runs(project_name)
    now = now_iso()
    persisted = {
        "id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
        **run,
    }
    data.setdefault("runs", []).append(persisted)
    save_chat_agent_runs(project_name, data)
    append_project_event(
        project_name,
        "agent_run_created",
        actor="chat_agent",
        clip_index=persisted.get("clip_index"),
        clip_key=persisted.get("clip_key"),
        entity="agent_run",
        entity_id=persisted["id"],
        payload={"intent": persisted.get("intent"), "status": persisted.get("status")},
    )
    return persisted


def update_chat_agent_run(project_name: str, run_id: str, updates: dict) -> dict:
    data = load_chat_agent_runs(project_name)
    updated = {}
    previous_status = None
    for run in data.get("runs", []):
        if run.get("id") == run_id:
            previous_status = run.get("status")
            run.update(updates)
            run["updated_at"] = now_iso()
            updated = run
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Agent run not found")
    save_chat_agent_runs(project_name, data)
    if "status" in updates and updates.get("status") != previous_status:
        append_project_event(
            project_name,
            "agent_run_status_changed",
            actor="chat_agent",
            clip_index=((updated.get("context_summary") or {}).get("clip_index")),
            clip_key=updated.get("clip_key"),
            entity="chat_agent_run",
            entity_id=run_id,
            payload={
                "previous_status": previous_status,
                "status": updated.get("status"),
            },
        )
    return updated


def matching_prompt(project_data: dict, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip", "")
    basename = os.path.basename(clip_name)
    same_clip_prompts = []
    for prompt in project_data.get("prompts", []):
        prompt_clip = prompt.get("clip_used") or ""
        same_clip = prompt_clip == clip_name or os.path.basename(prompt_clip) == basename
        if same_clip:
            same_clip_prompts.append(prompt)

    for prompt in same_clip_prompts:
        if prompt.get("clip_occurrence") == clip_index:
            return prompt
    for prompt in same_clip_prompts:
        if prompt.get("clip_occurrence") is None:
            return prompt
    return None


def matching_feedback(project_data: dict, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip", "")
    for group in project_data.get("feedback", []):
        occurrence = group.get("clip_occurrence")
        if group.get("clip_used") == clip_name and (occurrence is None or occurrence == clip_index):
            return group
    return None


def latest_prompt_version(prompt: Optional[dict]) -> Optional[dict]:
    if not prompt:
        return None
    from src.job_manager import prompt_versions_for_record
    versions = prompt_versions_for_record(prompt)
    return versions[-1] if versions else None


def load_clip_context(project_name: str, clip: dict, clip_index: int) -> Optional[dict]:
    clip_name = clip.get("clip")
    if not clip_name:
        return None
    from src.workflows.clip_context import clip_context_dir
    context_path = Path(clip_context_dir(str(storage_paths.DATA_DIR), project_name, clip_name, clip_index)) / "clip_context.json"
    if not context_path.exists():
        return None
    try:
        with context_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"Warning: Failed to load clip context {context_path}: {exc}")
    return None


def wants_clip_summary_or_analysis(text: str) -> bool:
    lower = text.lower()
    explicit_analysis_phrases = {
        "summarize",
        "summary",
        "describe",
        "understand",
        "analysis",
        "analyze",
        "what happens",
        "what is happening",
        "what do you see",
        "what can you see",
        "look at this",
        "watch this",
    }
    clip_words = {"clip", "shot", "scene", "video", "frame", "frames", "visual", "visible"}
    if any(phrase in lower for phrase in explicit_analysis_phrases):
        return True
    return any(word in lower for word in ["see", "visible", "happening", "shown"]) and any(word in lower for word in clip_words)


def wants_regenerated_clip_summary(text: str) -> bool:
    lower = text.lower()
    regeneration_words = {"regenerate", "refresh", "reanalyze", "re-analyze", "rerun", "re-run", "update"}
    summary_words = {"summary", "summarize", "analysis", "analyze", "context"}
    return any(word in lower for word in regeneration_words) and any(word in lower for word in summary_words)


def summarize_clip_context(clip_context: dict) -> str:
    if not clip_context:
        return "This clip has not been analyzed yet."
    lines = []
    summary = clip_context.get("summary")
    if summary:
        lines.append(summary)
    for label, key in [
        ("Visible characters", "visible_characters"),
        ("Expressions", "expressions"),
        ("Gaze", "gaze"),
        ("Actions", "actions"),
        ("Blocking", "blocking"),
        ("Continuity notes", "continuity_notes"),
        ("Uncertainty", "uncertainty_flags"),
    ]:
        values = clip_context.get(key) or []
        if values:
            lines.append(f"{label}: " + "; ".join(str(value) for value in values))
    if clip_context.get("camera_framing"):
        lines.append(f"Camera/framing: {clip_context.get('camera_framing')}")
    if clip_context.get("location"):
        lines.append(f"Location: {clip_context.get('location')}")
    if clip_context.get("audio_transcript"):
        lines.append(f"Audio transcript: {clip_context.get('audio_transcript')}")
    elif clip_context.get("audio_status") and clip_context.get("audio_status") != "not_configured":
        lines.append(f"Audio status: {clip_context.get('audio_status')}")
    return "\n".join(lines) if lines else "The saved clip context has no summary details yet."


def ensure_clip_context_for_chat(project_name: str, context: dict, provider: str, force: bool = False) -> tuple[Optional[dict], Optional[str]]:
    existing = context.get("clip_context")
    if existing and not force:
        return existing, None
    if provider != "openai":
        return None, "Clip analysis on demand currently requires OpenAI."
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "Clip analysis has not been run yet, and OPENAI_API_KEY is not configured."

    from openai import OpenAI
    from src.workflows.clip_context import analyze_clip_context

    clip = context.get("clip") or {}
    feedback = context.get("feedback") or {}
    latest_version = context.get("latest_version") or {}
    clip_context = analyze_clip_context(
        project_name=project_name,
        clip_name=clip.get("clip"),
        clip_occurrence=context.get("clip_index"),
        clip_start_s=clip.get("start_s"),
        clip_end_s=clip.get("end_s"),
        clip_duration_s=clip.get("duration_s"),
        assets_dir=str(storage_paths.ASSETS_DIR),
        output_base_dir=str(storage_paths.DATA_DIR),
        client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        provider="openai",
        feedback_items=(context.get("feedback") or {}).get("feedback_items", []),
        audio_name=feedback.get("audio_used") or latest_version.get("audio_used"),
        audio_path=feedback.get("audio_path") or latest_version.get("audio_path") or latest_version.get("trimmed_audio_path"),
    )
    context["clip_context"] = clip_context
    return clip_context, None


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


def _state_mutation_actions_from_message(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    actions = []

    if any(word in lower for word in ["set", "use", "select", "switch"]) and "prompt" in lower and "version" in lower:
        version_id = _parse_prompt_version_id(text, context)
        actions.append({
            "type": "set_active_prompt_version",
            "label": "Set Active Prompt Version",
            "prompt_version_id": version_id,
        })

    asset_selector = _parse_asset_selector(text, context)
    asset_words = any(word in lower for word in ["asset", "image", "character", "prop"]) or (
        "reference" in lower and "frame" not in lower and "still" not in lower
    )
    if asset_words and any(word in lower for word in ["attach", "add", "pin", "use", "select"]):
        role = _infer_asset_role(text)
        actions.append({
            "type": "attach_asset",
            "label": "Attach Asset",
            "role": role,
            "reason": _infer_asset_reason(text, role),
            "confidence": 0.86 if asset_selector else 0.45,
            **asset_selector,
        })
    if asset_words and any(word in lower for word in ["detach", "remove", "unpin", "deselect"]):
        actions.append({
            "type": "detach_asset",
            "label": "Detach Asset",
            **asset_selector,
        })

    if "feedback" in lower and any(word in lower for word in ["resolve", "resolved", "done", "fixed", "close", "complete"]):
        actions.append({
            "type": "mark_feedback_resolved",
            "label": "Mark Feedback Resolved",
            "feedback_index": _parse_feedback_index(text, context),
        })

    reference_request = parse_explicit_reference_frame_request(text)
    if reference_request:
        actions.append({
            "type": "add_reference_frame",
            "label": "Add Reference Frame",
            **reference_request,
        })

    return actions


def build_clip_chat_context(project_name: str, clip_index: int, query: str = "") -> dict:
    from src.clip_state import build_clip_state
    clip_state = build_clip_state(project_name, clip_index, query=query)
    timeline = clip_state.get("timeline") or {}
    active_prompt = clip_state.get("active_prompt") or {}
    asset_state = clip_state.get("asset_state") or {}

    return {
        "project": clip_state.get("project", {}),
        "clip": timeline.get("clip", {}),
        "clip_index": clip_index,
        "clip_key": clip_state.get("clip_key", ""),
        "clip_state": clip_state,
        "adjacent_clips": timeline.get("adjacent_clips", {}),
        "feedback": clip_state.get("feedback_state"),
        "prompt": active_prompt.get("prompt"),
        "latest_version": active_prompt.get("version"),
        "clip_context": (clip_state.get("analysis_state") or {}).get("clip_context"),
        "assets": asset_state.get("available_assets", {}),
        "selected_assets": asset_state.get("selected_assets", []),
        "pinned_assets": asset_state.get("pinned_assets", []),
        "referenced_frames": asset_state.get("referenced_frames", []),
        "generated_videos": (clip_state.get("video_state") or {}).get("generated_videos", []),
        "memory": clip_state.get("memory_state", {}),
        "learning": clip_state.get("learning_state", {}),
        "agent_runs": (clip_state.get("agent_state") or {}).get("recent_runs", []),
        "freshness": clip_state.get("freshness"),
    }


def generated_videos_for_version(version: Optional[dict]) -> list[dict]:
    if not version:
        return []
    videos = version.get("generated_videos")
    return videos if isinstance(videos, list) else []


def compact_chat_context(context: dict) -> str:
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    selected_assets = latest_version.get("selected_assets") or []
    project_memory = [item.get("text", "") for item in context.get("memory", {}).get("project", [])][-8:]
    clip_memory = [item.get("text", "") for item in context.get("memory", {}).get("clip", [])][-12:]
    relevant_memory = [
        {
            "text": item.get("text", ""),
            "scope": item.get("scope"),
            "relevance_score": item.get("relevance_score"),
            "relevance_reasons": item.get("relevance_reasons", []),
        }
        for item in context.get("memory", {}).get("relevant", [])
    ]
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    latest_learning_report = learning.get("latest_learning_report") or {}
    assets_by_category = {
        category: [asset.get("path") for asset in assets[:12]]
        for category, assets in (context.get("assets") or {}).items()
    }

    return json.dumps(
        {
            "project": context.get("project"),
            "clip": context.get("clip"),
            "clip_index": context.get("clip_index"),
            "clip_state_id": (context.get("clip_state") or {}).get("clip_state_id"),
            "adjacent_clips": context.get("adjacent_clips"),
            "feedback_items": feedback_items,
            "selected_assets": selected_assets,
            "asset_state": {
                "selected_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("selected_assets", []),
                "pinned_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("pinned_assets", []),
                "referenced_frames": ((context.get("clip_state") or {}).get("asset_state") or {}).get("referenced_frames", []),
                "missing_assets": ((context.get("clip_state") or {}).get("asset_state") or {}).get("missing_assets", []),
            },
            "selection_state": (context.get("clip_state") or {}).get("selection_state"),
            "latest_video_prompt": latest_version.get("video_model_prompt"),
            "active_prompt_ids": {
                "prompt_id": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("prompt_id"),
                "version_id": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("version_id"),
                "version_index": ((context.get("clip_state") or {}).get("active_prompt") or {}).get("version_index"),
            },
            "generated_videos": generated_videos_for_version(latest_version),
            "video_state": (context.get("clip_state") or {}).get("video_state"),
            "latest_prompt_explanation": latest_version.get("explanation"),
            "clip_context": context.get("clip_context"),
            "quality_report": latest_version.get("quality_report"),
            "learning_state": {
                "feedback_count": learning.get("feedback_count", 0),
                "open_issue_count": learning.get("open_issue_count", 0),
                "lesson_count": len(learning.get("lessons") or []),
                "relevant_lesson_count": len(learning.get("relevant_lessons") or learning.get("relevant") or []),
                "eval_case_count": len(learning.get("eval_cases") or []),
                "latest_learning_passed": latest_learning_report.get("passed") if latest_learning_report else None,
                "latest_learning_failed_cases": latest_learning_report.get("failed_cases", []) if latest_learning_report else [],
                "relevant_lessons": [
                    {
                        "lesson": item.get("lesson"),
                        "category": item.get("category"),
                        "scope": item.get("scope"),
                        "relevance_score": item.get("relevance_score"),
                    }
                    for item in (learning.get("relevant_lessons") or learning.get("relevant") or [])[:6]
                ],
            },
            "asset_library_sample": assets_by_category,
            "project_memory": project_memory,
            "clip_memory": clip_memory,
            "relevant_memory": relevant_memory,
            "recent_agent_runs": [
                {
                    "id": run.get("id"),
                    "goal": run.get("goal"),
                    "status": run.get("status"),
                    "intent": run.get("intent"),
                    "autonomy_level": run.get("autonomy_level"),
                    "approval_required": run.get("approval_required"),
                    "freshness": run.get("freshness"),
                    "updated_at": run.get("updated_at"),
                }
                for run in context.get("agent_runs", [])[:3]
            ],
        },
        ensure_ascii=False,
    )


def extract_memory_notes(message: str, assistant_text: str = "") -> list[str]:
    notes = []
    for match in re.finditer(r"(?:remember|save to memory)(?: that)?\s*:?\s+(.+)", message, flags=re.IGNORECASE):
        note = match.group(1).strip()
        if note:
            notes.append(note)

    for line in (assistant_text or "").splitlines():
        if line.lower().startswith("memory:"):
            note = line.split(":", 1)[1].strip()
            if note:
                notes.append(note)

    return notes


def normalize_chat_reply_markdown(text: str) -> str:
    """Remove accidental code fences around prompt prose before saving chat replies."""
    if not text:
        return text

    normalized = text.strip()
    full_fence = re.match(r"^```[\w-]*\s*\n([\s\S]*?)\n```$", normalized)
    if full_fence:
        return full_fence.group(1).strip()

    prompt_fence = re.compile(
        r"(^|\n)([^\n`]*(?:prompt|seedance|video model prompt)[^\n`]*:\s*)?\n?```(?:text|markdown|md)?\s*\n([\s\S]*?)\n```",
        re.IGNORECASE,
    )

    def replace_prompt_fence(match: re.Match) -> str:
        prefix = match.group(1) or ""
        label = (match.group(2) or "").strip()
        body = match.group(3).strip()
        if label:
            return f"{prefix}{label}\n{body}"
        return f"{prefix}{body}"

    return prompt_fence.sub(replace_prompt_fence, normalized).strip()


def wants_previous_last_frame_continuity(text: str) -> bool:
    lower = text.lower()
    has_continuity = any(word in lower for word in ["continuity", "match", "align", "flow", "seamless"])
    has_frame = any(word in lower for word in ["frame", "shot", "clip", "last frame", "previous clip"])
    return has_continuity and has_frame


def wants_autonomous_execution(text: str) -> bool:
    lower = text.lower()
    triggers = [
        "run this",
        "do it",
        "execute",
        "start generation",
        "generate video",
        "run the workflow",
        "automatically",
        "go ahead",
    ]
    return any(trigger in lower for trigger in triggers)


def _infer_prompt_feedback_category(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ["wardrobe", "character", "clothes", "costume"]):
        return "wrong_character_or_wardrobe"
    if any(word in lower for word in ["continuity", "same", "preserve", "consistent"]):
        return "continuity_error"
    if any(word in lower for word in ["camera", "pan", "zoom", "dolly", "shot"]):
        return "bad_camera_instruction"
    if any(word in lower for word in ["audio", "dialogue", "voice", "music"]):
        return "bad_audio_or_dialogue"
    if any(word in lower for word in ["invent", "unsupported", "assumption", "prop", "extra"]):
        return "unsupported_assumption"
    if any(word in lower for word in ["vague", "unclear", "specific"]):
        return "too_vague"
    if any(word in lower for word in ["verbose", "long"]):
        return "too_verbose"
    if any(word in lower for word in ["format", "json", "schema"]):
        return "format_error"
    if any(word in lower for word in ["provider", "seedance", "segmind", "incompatible"]):
        return "provider_incompatible"
    return "other"


def _active_prompt_ids_for_context(context: dict) -> tuple[Optional[str], Optional[str]]:
    active_prompt = (context.get("clip_state") or {}).get("active_prompt") or {}
    version = active_prompt.get("version") or {}
    prompt_id = active_prompt.get("prompt_id") or version.get("prompt_id")
    version_id = active_prompt.get("version_id") or version.get("prompt_version_id")
    return prompt_id, version_id


def _active_prompt_feedback_items(context: dict, *, negative_only: bool = False) -> list[dict]:
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    active_prompt_id, active_version_id = _active_prompt_ids_for_context(context)
    items = []
    for item in ((context.get("clip_state") or {}).get("learning_feedback_items") or []):
        if active_version_id and item.get("prompt_version_id") != active_version_id:
            continue
        if active_prompt_id and item.get("prompt_id") != active_prompt_id:
            continue
        if negative_only and item.get("rating") != "negative":
            continue
        items.append(item)
    if items:
        return items
    clip_index = context.get("clip_index")
    project_name = (context.get("project") or {}).get("name")
    if not project_name or clip_index is None:
        return []
    items = prompt_feedback_items_for_clip(project_name, int(clip_index))
    return [
        item for item in items
        if (not active_version_id or item.get("prompt_version_id") == active_version_id)
        and (not negative_only or item.get("rating") == "negative")
    ][: max(1, int(learning.get("feedback_count") or len(items) or 1))]


def _latest_feedback_for_learning_action(project_name: str, context: dict, action: dict) -> Optional[dict]:
    feedback_id = action.get("feedback_id")
    if feedback_id:
        return find_prompt_feedback_item(project_name, str(feedback_id))
    items = _active_prompt_feedback_items(context, negative_only=True) or _active_prompt_feedback_items(context)
    open_items = [item for item in items if item.get("status") == "open"]
    return (open_items or items or [None])[-1]


def infer_learning_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    actions: list[dict] = []
    prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    has_prompt = bool(prompt_version_id and ((context.get("latest_version") or {}).get("video_model_prompt")))
    category = _infer_prompt_feedback_category(text)

    if has_prompt and any(phrase in lower for phrase in [
        "save this as feedback",
        "save prompt feedback",
        "mark prompt as bad",
        "this prompt is bad",
        "prompt is wrong",
        "add feedback",
    ]):
        actions.append({
            "type": "save_prompt_feedback",
            "label": "Save Prompt Feedback",
            "prompt": text.strip(),
            "rating": "negative",
            "categories": [category],
            "severity": 4 if any(word in lower for word in ["bad", "wrong", "failed", "ignored"]) else 3,
            "create_eval_case": any(phrase in lower for phrase in ["eval", "regression", "test case"]),
            "prompt_id": prompt_id,
            "prompt_version_id": prompt_version_id,
        })

    if any(phrase in lower for phrase in ["suggest a lesson", "suggest lesson", "turn this into a lesson"]):
        actions.append({
            "type": "suggest_prompt_lesson",
            "label": "Suggest Lesson",
            "prompt": text.strip(),
        })

    if any(phrase in lower for phrase in ["approve lesson", "save as lesson", "remember this lesson", "app should remember", "remember this"]):
        actions.append({
            "type": "approve_prompt_lesson",
            "label": "Approve Lesson",
            "prompt": text.strip(),
            "lesson": text.strip(),
            "category": category,
            "scope": "project",
        })

    if has_prompt and any(phrase in lower for phrase in [
        "revise prompt",
        "revise the prompt",
        "improve prompt",
        "fix the prompt",
        "rewrite prompt",
    ]):
        actions.append({
            "type": "revise_prompt_from_feedback",
            "label": "Revise Prompt",
            "prompt": text.strip(),
            "prompt_version_id": prompt_version_id,
        })

    if has_prompt and any(phrase in lower for phrase in [
        "run learning eval",
        "learning eval",
        "evaluate learning",
        "check learned",
        "check prompt against lessons",
    ]):
        actions.append({
            "type": "run_prompt_learning_eval",
            "label": "Run Learning Eval",
            "prompt": text.strip(),
            "prompt_version_id": prompt_version_id,
        })

    return actions


def infer_action_suggestions(text: str, context: dict) -> list[dict]:
    lower = text.lower()
    suggestions = _state_mutation_actions_from_message(text, context)
    for action in infer_learning_action_suggestions(text, context):
        add_unique_action(suggestions, action)
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    if any(word in lower for word in ["workflow", "run", "execute"]) and feedback_items:
        action = {
            "type": "execute_workflow",
            "label": "Run Workflow",
            "feedback_index": feedback_items[0].get("raw_index"),
        }
        if wants_previous_last_frame_continuity(text):
            action["label"] = "Run With Previous Last Frame"
            action["continuity_reference"] = "previous_clip_last_frame"
        if wants_autonomous_execution(text):
            action["autonomous"] = True
        suggestions.append(action)
    if user_explicitly_requests_video_generation(text) and latest_version.get("video_model_prompt"):
        action = {"type": "generate_video", "label": "Generate Video"}
        suggestions.append(action)
    return suggestions


def add_unique_action(actions: list[dict], action: dict) -> None:
    key = (action.get("type"), action.get("label"), action.get("feedback_index"), action.get("prompt"))
    existing_keys = {
        (item.get("type"), item.get("label"), item.get("feedback_index"), item.get("prompt"))
        for item in actions
    }
    if key not in existing_keys:
        actions.append(action)


def dynamic_chat_suggestions(message: str, context: dict, explicit_actions: list[dict]) -> list[dict]:
    lower = message.lower()
    actions = list(explicit_actions)
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}
    media = build_clip_media_gallery(context)

    if feedback_items:
        feedback_index = feedback_items[0].get("raw_index")
        workflow_label = "Run Workflow Again" if latest_version.get("video_model_prompt") else "Run Workflow"
        add_unique_action(actions, {
            "type": "execute_workflow",
            "label": workflow_label,
            "feedback_index": feedback_index,
        })

    if latest_version.get("video_model_prompt"):
        add_unique_action(actions, {"type": "generate_video", "label": "Generate Video"})

    if context.get("clip_context") and wants_clip_summary_or_analysis(message):
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Regenerate Summary",
            "prompt": "Regenerate the clip summary by analyzing the clip frames and audio again.",
        })

    if media and not wants_clip_media_gallery(message):
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Show Assets",
            "prompt": "Show all assets and references for this clip.",
        })

    if "feedback" not in lower and feedback_items:
        add_unique_action(actions, {
            "type": "send_message",
            "label": "Show Feedback",
            "prompt": "Show the feedback for this clip.",
        })

    if len(actions) > 5:
        return actions[:5]
    return actions


def _action_goal(action: dict) -> str:
    if action.get("type") == "execute_workflow":
        return "Run the feedback workflow for this clip."
    if action.get("type") == "generate_video":
        return "Generate a video from the latest prompt."
    if action.get("type") == "save_prompt_feedback":
        return "Save prompt feedback for the active prompt version."
    if action.get("type") == "suggest_prompt_lesson":
        return "Suggest an editable lesson from prompt feedback."
    if action.get("type") == "approve_prompt_lesson":
        return "Save an approved prompt lesson for future prompts."
    if action.get("type") == "revise_prompt_from_feedback":
        return "Create a revised prompt version using feedback and lessons."
    if action.get("type") == "run_prompt_learning_eval":
        return "Evaluate the active prompt against lessons and eval cases."
    if action.get("type") == "prepare_video":
        return "Prepare the latest prompt and references for video generation."
    if action.get("type") == "send_message":
        return action.get("prompt") or "Ask a contextual follow-up question."
    return action.get("label") or "Continue the clip chat."


def _step_for_action(index: int, action: dict, autonomy_level: str) -> dict:
    action_type = action.get("type") or "send_message"
    risky = action_type in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}
    if risky and autonomy_level == "approval_required":
        status = "awaiting_approval"
    elif risky and autonomy_level == "full_autopilot":
        status = "dispatch_ready"
    else:
        status = "pending"
    return {
        "id": f"step-{index}",
        "label": action.get("label") or _action_goal(action),
        "status": status,
        "tool": action_type,
        "requires_approval": risky,
        "action": action,
    }


def _asks_for_permission_or_advice(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith(("can you", "could you", "should i", "should we", "can i", "could i", "would you"))


def plan_chat_agent_run(message: str, context: dict, explicit_actions: list[dict], available_actions: list[dict]) -> dict:
    lower = message.lower()
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    latest_version = context.get("latest_version") or {}

    if wants_previous_last_frame_continuity(message):
        intent = "continuity_workflow"
        confidence = 0.92
    elif any(action.get("type") in {
        "set_active_prompt_version",
        "attach_asset",
        "detach_asset",
        "mark_feedback_resolved",
        "add_reference_frame",
    } for action in explicit_actions):
        intent = "state_mutation"
        confidence = 0.9
    elif any(action.get("type") == "execute_workflow" for action in explicit_actions):
        intent = "workflow_execution"
        confidence = 0.88
    elif any(action.get("type") == "generate_video" for action in explicit_actions):
        intent = "video_generation"
        confidence = 0.86
    elif any(action.get("type") in LEARNING_CHAT_ACTION_TYPES for action in explicit_actions):
        intent = "prompt_learning"
        confidence = 0.84
    elif wants_clip_summary_or_analysis(message):
        intent = "clip_analysis"
        confidence = 0.82
    elif "memory" in lower or "remember" in lower or "save to memory" in lower:
        intent = "memory_update"
        confidence = 0.78
    elif wants_clip_media_gallery(message):
        intent = "asset_review"
        confidence = 0.76
    elif any(phrase in lower for phrase in ["self evaluate", "self-evaluate", "self evaluation", "evaluate state", "check state"]):
        intent = "self_evaluation"
        confidence = 0.78
    elif any(word in lower for word in ["evaluate", "quality", "check prompt", "review prompt", "audit prompt"]):
        intent = "prompt_evaluation"
        confidence = 0.75
    elif "feedback" in lower:
        intent = "feedback_review"
        confidence = 0.74
    else:
        intent = "general_chat"
        confidence = 0.55

    risky_actions = [
        action for action in explicit_actions
        if action.get("type") in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}
    ]
    has_generate_video_action = any(action.get("type") == "generate_video" for action in risky_actions)
    has_learning_action = any(action.get("type") in LEARNING_CHAT_ACTION_TYPES for action in risky_actions)
    direct_execution = not has_generate_video_action and not has_learning_action and not _asks_for_permission_or_advice(message) and (
        wants_autonomous_execution(message)
        or any(has_word(lower, word) for word in ["render", "seedance"])
    )
    if risky_actions and direct_execution:
        autonomy_level = "full_autopilot"
        approval_required = False
        status = "ready"
    elif risky_actions:
        autonomy_level = "approval_required"
        approval_required = True
        status = "awaiting_approval"
    elif available_actions:
        autonomy_level = "suggest"
        approval_required = False
        status = "planned"
    else:
        autonomy_level = "manual"
        approval_required = False
        status = "planned"

    required_actions = [dict(action) for action in explicit_actions]
    action_plan = required_actions or [
        action for action in available_actions
        if action.get("type") in {"execute_workflow", "generate_video", "prepare_video", *LEARNING_CHAT_ACTION_TYPES}
    ][:2]

    steps = []
    if intent in {"general_chat", "feedback_review"}:
        steps.append({
            "id": "step-1",
            "label": "Answer from current clip context and saved project state.",
            "status": "completed",
            "tool": "chat_reply",
            "requires_approval": False,
            "action": None,
        })

    safe_step_by_intent = {
        "clip_analysis": ("inspect_clip_context", "Inspect saved clip context and analysis status."),
        "asset_review": ("search_assets", "Search attached clip assets and project media."),
        "memory_update": ("add_memory", "Persist any durable preference captured from chat."),
        "prompt_evaluation": ("evaluate_prompt", "Evaluate the latest prompt against feedback and available context."),
        "self_evaluation": ("self_evaluate", "Evaluate stale state, feedback coverage, prompt quality, assets, and video status."),
    }
    if intent in safe_step_by_intent:
        tool, label = safe_step_by_intent[intent]
        steps.append({
            "id": f"step-{len(steps) + 1}",
            "label": label,
            "status": "pending",
            "tool": tool,
            "requires_approval": False,
            "action": None,
        })

    for index, action in enumerate(action_plan, start=len(steps) + 1):
        steps.append(_step_for_action(index, action, autonomy_level))

    if not steps and feedback_items:
        steps.append({
            "id": "step-1",
            "label": "Keep the workflow action available for this feedback item.",
            "status": "pending",
            "tool": "execute_workflow",
            "requires_approval": True,
            "action": {
                "type": "execute_workflow",
                "label": "Run Workflow",
                "feedback_index": feedback_items[0].get("raw_index"),
            },
        })

    goal = message.strip()
    if not goal:
        goal = _action_goal(required_actions[0]) if required_actions else "Continue the clip chat."

    return {
        "goal": goal,
        "clip_index": context.get("clip_index"),
        "clip_key": context.get("clip_key"),
        "status": status,
        "intent": intent,
        "confidence": confidence,
        "autonomy_level": autonomy_level,
        "approval_required": approval_required,
        "plan_steps": steps,
        "required_actions": required_actions,
        "available_actions": available_actions,
        "suggested_actions": available_actions,
        "tool_results": [],
        "errors": [],
        "context_summary": {
            "feedback_count": len(feedback_items),
            "prompt_ready": bool(latest_version.get("video_model_prompt")),
            "clip_context_ready": bool(context.get("clip_context")),
            "freshness_snapshot": {
                **((context.get("clip_state") or {}).get("freshness") or {}),
                "captured_at": now_iso(),
            },
        },
    }


def apply_agent_run_action_policy(actions: list[dict], agent_run: dict) -> list[dict]:
    if not agent_run.get("approval_required"):
        return actions
    gated_actions = []
    for action in actions:
        next_action = dict(action)
        if next_action.get("type") in {"execute_workflow", "generate_video", *LEARNING_CHAT_ACTION_TYPES}:
            next_action.pop("autonomous", None)
        gated_actions.append(next_action)
    return gated_actions


def strip_risky_autonomous_actions(suggested_actions: list[dict]) -> list[dict]:
    filtered = []
    for action in suggested_actions:
        action_type = action.get("type")
        if action_type in {"execute_workflow", "generate_video"}:
            continue
        filtered.append(action)
    return filtered


def search_assets_for_agent(context: dict, query: str, limit: int = 8) -> dict:
    query_tokens = {
        token for token in re.findall(r"[a-zA-Z0-9_'-]{2,}", query.lower())
        if token not in {"show", "list", "view", "see", "open", "asset", "assets", "media", "reference", "references"}
    }
    matches = []
    for category, assets in (context.get("assets") or {}).items():
        for asset in assets:
            haystack = f"{category} {asset.get('name', '')} {asset.get('path', '')} {asset.get('type', '')}".lower()
            score = sum(1 for token in query_tokens if token in haystack)
            if score or not query_tokens:
                matches.append({
                    "category": category,
                    "name": asset.get("name"),
                    "path": asset.get("path"),
                    "url": asset.get("url"),
                    "type": asset.get("type"),
                    "score": score,
                })
    matches.sort(key=lambda item: (item.get("score", 0), item.get("name") or ""), reverse=True)
    return {
        "status": "ok",
        "tool": "search_assets",
        "message": f"Found {len(matches[:limit])} matching asset(s).",
        "matches": matches[:limit],
    }


def inspect_clip_context_for_agent(context: dict) -> dict:
    clip_context = context.get("clip_context")
    if not clip_context:
        return {
            "status": "missing",
            "tool": "inspect_clip_context",
            "message": "No saved clip analysis exists yet.",
        }
    return {
        "status": "ok",
        "tool": "inspect_clip_context",
        "message": "Loaded saved clip analysis.",
        "summary": summarize_clip_context(clip_context),
        "clip_context_path": clip_context.get("clip_context_path"),
        "analysis_status": clip_context.get("status"),
    }


def memory_update_result_for_agent(saved_memories: list[dict]) -> dict:
    if not saved_memories:
        return {
            "status": "missing",
            "tool": "add_memory",
            "message": "No durable memory note was detected in this turn.",
        }
    return {
        "status": "ok",
        "tool": "add_memory",
        "message": f"Saved {len(saved_memories)} memory note(s).",
        "memory_ids": [item.get("id") for item in saved_memories],
        "memories": [{"id": item.get("id"), "text": item.get("text"), "scope": item.get("scope")} for item in saved_memories],
    }


def evaluate_prompt_for_agent(context: dict) -> dict:
    latest_version = context.get("latest_version") or {}
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    feedback_items = (context.get("feedback") or {}).get("feedback_items", [])
    if not prompt_text:
        return {
            "status": "missing",
            "tool": "evaluate_prompt",
            "message": "No generated prompt exists yet for this clip.",
        }

    prompt_lower = prompt_text.lower()
    missed_feedback = []
    for item in feedback_items:
        remark = str(item.get("remark") or "").strip()
        tokens = [
            token for token in re.findall(r"[a-zA-Z0-9_'-]{4,}", remark.lower())
            if token not in {"make", "this", "that", "with", "from", "should", "clip", "shot", "video"}
        ]
        if tokens and not any(token in prompt_lower for token in tokens[:8]):
            missed_feedback.append(remark)

    quality_report = latest_version.get("quality_report") or {}
    suggestions = list(quality_report.get("suggestions") or [])
    if missed_feedback:
        suggestions.append("Review whether the prompt explicitly addresses every feedback remark.")
    if not latest_version.get("selected_assets"):
        suggestions.append("No selected assets are attached to this prompt version.")

    return {
        "status": "ok",
        "tool": "evaluate_prompt",
        "message": "Evaluated the latest prompt against feedback and handoff metadata.",
        "prompt_ready": True,
        "feedback_count": len(feedback_items),
        "missed_feedback": missed_feedback[:5],
        "quality_passed": quality_report.get("passed"),
        "suggestions": suggestions[:6],
    }


def clip_index_for_feedback_index(project_name: str, feedback_index: int) -> int:
    feedback_data = read_json_file(project_data_dir(project_name) / "feedback.json", [])
    if not isinstance(feedback_data, list):
        return 0
    timeline = read_json_file(project_data_dir(project_name) / "timeline.json", {}).get("video_timeline", [])
    for group in feedback_data:
        for item in group.get("feedback_items", []):
            if item.get("raw_index") == feedback_index:
                occurrence = group.get("clip_occurrence")
                if isinstance(occurrence, int) and 0 <= occurrence < len(timeline):
                    return occurrence
                clip_used = group.get("clip_used")
                for idx, clip in enumerate(timeline):
                    if clip.get("clip") == clip_used:
                        return idx
    return 0


def evaluate_agent_state_for_clip(project_name: str, clip_index: int, *, trigger: str = "manual", agent_run_id: Optional[str] = None) -> dict:
    from src.clip_state import build_clip_state

    state = build_clip_state(project_name, clip_index)
    freshness = state.get("freshness") or {}
    feedback_items = ((state.get("feedback_state") or {}).get("feedback_items") or [])
    active_prompt = state.get("active_prompt") or {}
    latest_version = active_prompt.get("version") or {}
    asset_state = state.get("asset_state") or {}
    video_state = state.get("video_state") or {}
    selection_state = state.get("selection_state") or {}

    unresolved_feedback = [
        item for item in feedback_items
        if not item.get("resolved") and item.get("status") != "resolved"
    ]
    prompt_report = evaluate_prompt_for_agent({
        "latest_version": latest_version,
        "feedback": {"feedback_items": feedback_items},
    })
    stale_runs = [
        run for run in (state.get("agent_state") or {}).get("recent_runs", [])
        if (run.get("freshness") or {}).get("is_stale")
        and (not agent_run_id or run.get("id") != agent_run_id)
    ]
    selected_refs = selection_state.get("selected_assets") or selection_state.get("pinned_assets") or []
    pinned_assets = asset_state.get("pinned_assets") or []
    missing_assets = asset_state.get("missing_assets") or []
    generated_videos = video_state.get("generated_videos") or []
    latest_attempt = video_state.get("latest_attempt") or {}
    active_video = video_state.get("active_video") or video_state.get("latest_video")

    checks = {
        "stale_state": {
            "status": "fail" if freshness.get("stale_reasons") or stale_runs else "pass",
            "stale_reasons": freshness.get("stale_reasons", []),
            "stale_run_count": len(stale_runs),
        },
        "feedback_coverage": {
            "status": "pass" if not unresolved_feedback else "needs_attention",
            "total_feedback": len(feedback_items),
            "unresolved_feedback": [
                {
                    "feedback_index": item.get("raw_index"),
                    "category": item.get("category"),
                    "remark": item.get("remark"),
                }
                for item in unresolved_feedback[:8]
            ],
        },
        "prompt_quality": {
            "status": (
                "missing" if prompt_report.get("status") == "missing"
                else "needs_attention" if prompt_report.get("missed_feedback") or prompt_report.get("quality_passed") is False
                else "pass"
            ),
            "prompt_version_id": active_prompt.get("version_id"),
            "prompt_ready": bool(active_prompt.get("prompt_ready")),
            "missed_feedback": prompt_report.get("missed_feedback", []),
            "suggestions": prompt_report.get("suggestions", []),
        },
        "asset_completeness": {
            "status": "fail" if missing_assets else "needs_attention" if not selected_refs and not pinned_assets else "pass",
            "structured_asset_count": len(selected_refs),
            "resolved_asset_count": len(pinned_assets),
            "missing_assets": missing_assets,
        },
        "video_status": {
            "status": (
                "failed" if latest_attempt.get("status") in {"failed", "recovery_failed"}
                else "pass" if active_video
                else "missing"
            ),
            "active_video_id": video_state.get("active_video_id"),
            "generated_video_count": len(generated_videos),
            "latest_attempt": latest_attempt,
        },
    }

    if checks["prompt_quality"]["status"] == "missing":
        next_action = {"type": "run_workflow", "reason": "No active prompt exists for this clip."}
    elif checks["stale_state"]["status"] == "fail":
        next_action = {"type": "review_state", "reason": "State changed since earlier agent work."}
    elif checks["asset_completeness"]["status"] in {"fail", "needs_attention"}:
        next_action = {"type": "attach_asset", "reason": "Assets are missing or not structured for this clip."}
    elif checks["prompt_quality"]["status"] == "needs_attention":
        next_action = {"type": "review_prompt", "reason": "Prompt may not cover all feedback or quality checks."}
    elif checks["video_status"]["status"] in {"missing", "failed"}:
        next_action = {"type": "generate_video", "reason": "Prompt is ready but no successful active video is available."}
    elif checks["feedback_coverage"]["status"] == "needs_attention":
        next_action = {"type": "mark_feedback_resolved", "reason": "Generated state exists; unresolved feedback should be reviewed."}
    else:
        next_action = {"type": "none", "reason": "Prompt, assets, feedback, and video state look ready."}

    blocking_statuses = {check["status"] for check in checks.values() if check.get("status") in {"fail", "failed", "missing"}}
    needs_attention = any(check.get("status") == "needs_attention" for check in checks.values())
    verdict = "blocked" if blocking_statuses else "needs_attention" if needs_attention else "passed"
    return {
        "status": "ok",
        "tool": "self_evaluate",
        "trigger": trigger,
        "clip_index": clip_index,
        "clip_key": state.get("clip_key"),
        "verdict": verdict,
        "message": f"Self-evaluation {verdict}. Next action: {next_action['type']}.",
        "checks": checks,
        "next_action": next_action,
        "state_hash": freshness.get("state_hash"),
        "evaluated_at": now_iso(),
    }


def self_evaluation_for_agent(project_name: str, context: dict, agent_run_id: Optional[str] = None) -> dict:
    return evaluate_agent_state_for_clip(
        project_name,
        int(context.get("clip_index") or 0),
        trigger="chat",
        agent_run_id=agent_run_id,
    )


def set_active_prompt_version_for_agent(project_name: str, context: dict, action: dict) -> dict:
    prompt_version_id = action.get("prompt_version_id")
    versions = ((context.get("clip_state") or {}).get("active_prompt") or {}).get("versions") or []
    match = next((version for version in versions if version.get("prompt_version_id") == prompt_version_id), None)
    if not prompt_version_id or not match:
        return {
            "status": "error",
            "tool": "set_active_prompt_version",
            "message": "Could not find the requested prompt version for this clip.",
        }

    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"active_prompt_version_id": prompt_version_id},
    )
    append_project_event(
        project_name,
        "active_prompt_version_set",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="prompt_version",
        entity_id=prompt_version_id,
        payload={
            "version_index": match.get("version_index"),
            "prompt_id": match.get("prompt_id"),
        },
    )
    return {
        "status": "ok",
        "tool": "set_active_prompt_version",
        "message": f"Set active prompt version to index {match.get('version_index')}.",
        "prompt_version_id": prompt_version_id,
        "version_index": match.get("version_index"),
        "selection": selection,
        "mutates_state": True,
    }


def _current_selection_lists(context: dict) -> tuple[list[dict], list[dict]]:
    clip_state = context.get("clip_state") or {}
    selection_state = clip_state.get("selection_state") or {}
    selected_refs = list(selection_state.get("selected_assets") or [])
    pinned_refs = list(selection_state.get("pinned_assets") or [])
    return selected_refs, pinned_refs


def _current_selection_asset_refs(context: dict) -> list[dict]:
    selected_refs, pinned_refs = _current_selection_lists(context)
    refs = selected_refs or pinned_refs
    if not refs and context.get("selected_assets"):
        return normalize_selection_asset_refs({"selected_assets": context.get("selected_assets")})
    return refs


def _resolve_asset_action_target(context: dict, asset_id: Optional[str], asset_path: Optional[str]) -> Optional[dict]:
    available = context.get("assets") or {}
    for category, items in available.items():
        for asset in items:
            enriched = _asset_identity((context.get("project") or {}).get("name", ""), category, asset)
            if asset_id and enriched.get("asset_id") == asset_id:
                return enriched
            if asset_path:
                candidate_path = str(asset.get("path") or asset.get("name") or "").lower()
                candidate_base = os.path.basename(candidate_path)
                target_path = str(asset_path).lower()
                target_base = os.path.basename(target_path)
                if candidate_path == target_path or candidate_base == target_base:
                    return enriched
    if asset_path:
        return {
            "asset_id": stable_state_id("asset", (context.get("project") or {}).get("name", ""), asset_path),
            "name": os.path.basename(asset_path),
            "path": asset_path,
        }
    return None


def _current_selection_lists(context: dict) -> tuple[list[str], list[str]]:
    selection = (context.get("clip_state") or {}).get("selection_state") or {}
    asset_ids = list(selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or [])
    asset_paths = list(selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or [])
    return asset_ids, asset_paths


def _current_selection_asset_refs(context: dict) -> list[dict]:
    selection = (context.get("clip_state") or {}).get("selection_state") or {}
    return normalize_selection_asset_refs(selection)


def _resolve_asset_action_target(context: dict, action: dict) -> dict:
    asset_id = action.get("asset_id")
    asset_path = action.get("asset_path")
    for asset in _asset_candidates_for_context(context):
        candidate_id = asset.get("asset_id")
        candidate_path = asset.get("path") or asset.get("name")
        if asset_id and candidate_id == asset_id:
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
        if asset_path and str(candidate_path) == str(asset_path):
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
        if asset_path and os.path.basename(str(candidate_path)) == os.path.basename(str(asset_path)):
            return {"asset_id": candidate_id, "asset_path": candidate_path, "asset": asset}
    if asset_path:
        return {"asset_id": asset_id, "asset_path": asset_path, "asset": None}
    return {}


def attach_asset_for_agent(project_name: str, context: dict, action: dict) -> dict:
    target = _resolve_asset_action_target(context, action)
    if not target:
        return {
            "status": "error",
            "tool": "attach_asset",
            "message": "Could not identify which asset to attach.",
        }
    refs = _current_selection_asset_refs(context)
    target_id = target.get("asset_id")
    target_path = target.get("asset_path")
    next_ref = {
        "asset_id": target_id,
        "path": target_path,
        "role": action.get("role") or "selected_reference",
        "reason": action.get("reason") or "Selected from chat.",
        "confidence": float(action.get("confidence") if action.get("confidence") is not None else 0.82),
        "source": "chat_agent",
    }
    replaced = False
    for index, ref in enumerate(refs):
        if (target_id and ref.get("asset_id") == target_id) or (target_path and ref.get("path") == target_path):
            refs[index] = {**ref, **next_ref}
            replaced = True
            break
    if not replaced:
        refs.append(next_ref)
    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"selected_assets": refs},
    )
    append_project_event(
        project_name,
        "asset_attached",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="asset",
        entity_id=target.get("asset_id"),
        payload={
            "asset_path": target.get("asset_path"),
            "role": next_ref["role"],
            "reason": next_ref["reason"],
            "confidence": next_ref["confidence"],
        },
    )
    return {
        "status": "ok",
        "tool": "attach_asset",
        "message": f"Attached asset {os.path.basename(str(target.get('asset_path') or target.get('asset_id')))} to this clip.",
        "asset_id": target.get("asset_id"),
        "asset_path": target.get("asset_path"),
        "role": next_ref["role"],
        "reason": next_ref["reason"],
        "confidence": next_ref["confidence"],
        "selection": selection,
        "mutates_state": True,
    }


def detach_asset_for_agent(project_name: str, context: dict, action: dict) -> dict:
    target = _resolve_asset_action_target(context, action)
    if not target:
        return {
            "status": "error",
            "tool": "detach_asset",
            "message": "Could not identify which asset to detach.",
        }
    target_id = target.get("asset_id")
    target_path = target.get("asset_path")
    refs = _current_selection_asset_refs(context)
    next_refs = [
        ref for ref in refs
        if not (
            (target_id and ref.get("asset_id") == target_id)
            or (target_path and ref.get("path") == target_path)
            or (target_path and os.path.basename(str(ref.get("path") or "")) == os.path.basename(str(target_path)))
        )
    ]
    selection = update_clip_selection(
        project_name,
        context.get("clip_key"),
        {"selected_assets": next_refs},
    )
    append_project_event(
        project_name,
        "asset_detached",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="asset",
        entity_id=target_id,
        payload={
            "asset_path": target_path,
            "remaining_asset_count": len(next_refs),
        },
    )
    return {
        "status": "ok",
        "tool": "detach_asset",
        "message": f"Detached asset {os.path.basename(str(target_path or target_id))} from this clip.",
        "asset_id": target_id,
        "asset_path": target_path,
        "selection": selection,
        "mutates_state": True,
    }


def mark_feedback_resolved_for_agent(project_name: str, context: dict, action: dict) -> dict:
    feedback_index = action.get("feedback_index")
    if feedback_index is None:
        return {
            "status": "error",
            "tool": "mark_feedback_resolved",
            "message": "Could not identify which feedback item to mark resolved.",
        }

    feedback_path = project_data_dir(project_name) / "feedback.json"
    feedback_data = read_json_file(feedback_path, [])
    if not isinstance(feedback_data, list):
        return {
            "status": "error",
            "tool": "mark_feedback_resolved",
            "message": "Feedback data is not available for this project.",
        }

    raw_index_counter = 0
    for group in feedback_data:
        for item in group.get("feedback_items", []):
            if raw_index_counter == int(feedback_index):
                item["status"] = "resolved"
                item["resolved"] = True
                item["resolved_at"] = now_iso()
                item["resolved_by"] = "chat_agent"
                write_json_file(feedback_path, feedback_data)
                append_project_event(
                    project_name,
                    "feedback_resolved",
                    actor="chat_agent",
                    clip_index=context.get("clip_index"),
                    clip_key=context.get("clip_key"),
                    entity="feedback_item",
                    entity_id=str(feedback_index),
                    payload={
                        "feedback_index": feedback_index,
                        "category": item.get("category"),
                        "remark": item.get("remark"),
                    },
                )
                return {
                    "status": "ok",
                    "tool": "mark_feedback_resolved",
                    "message": f"Marked feedback #{feedback_index} as resolved.",
                    "feedback_index": feedback_index,
                    "mutates_state": True,
                }
            raw_index_counter += 1

    return {
        "status": "error",
        "tool": "mark_feedback_resolved",
        "message": f"Feedback #{feedback_index} was not found.",
    }


def add_reference_frame_for_agent(project_name: str, context: dict, action: dict, existing_tool_results: list[dict]) -> dict:
    for result in existing_tool_results:
        if result.get("status") == "ok" and result.get("reference_frame"):
            ref_frame = dict(result.get("reference_frame") or {})
            append_project_event(
                project_name,
                "reference_frame_added",
                actor="chat_agent",
                clip_index=context.get("clip_index"),
                clip_key=context.get("clip_key"),
                entity="reference_frame",
                entity_id=ref_frame.get("frame_path"),
                payload=ref_frame,
            )
            return {
                "status": "ok",
                "tool": "add_reference_frame",
                "reference_frame": ref_frame,
                "message": result.get("message"),
                "mutates_state": True,
            }
    try:
        result = extract_reference_frame(
            project_name=project_name,
            current_clip_index=int(context.get("clip_index") or 0),
            timestamp=str(action.get("timestamp") or ""),
            reason=str(action.get("reason") or ""),
            attach_to=str(action.get("attach_to") or "current_feedback"),
        )
        ref_frame = dict(result.get("reference_frame") or {})
        append_project_event(
            project_name,
            "reference_frame_added",
            actor="chat_agent",
            clip_index=context.get("clip_index"),
            clip_key=context.get("clip_key"),
            entity="reference_frame",
            entity_id=ref_frame.get("frame_path"),
            payload=ref_frame,
        )
        return {
            "status": "ok",
            "tool": "add_reference_frame",
            "reference_frame": ref_frame,
            "message": result.get("message"),
            "mutates_state": True,
        }
    except HTTPException as exc:
        return {"status": "error", "tool": "add_reference_frame", "message": exc.detail}


def save_prompt_feedback_for_agent(project_name: str, context: dict, action: dict, message: str) -> dict:
    prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    if not prompt_id or not prompt_version_id:
        return {"status": "missing", "tool": "save_prompt_feedback", "message": "No active prompt version is available for feedback."}
    comment = str(action.get("comment") or action.get("prompt") or message or "").strip()
    correction = str(action.get("correction") or "").strip()
    remember_note = str(action.get("remember_note") or "").strip()
    rating = action.get("rating") if action.get("rating") in PROMPT_FEEDBACK_RATINGS else "negative"
    payload = {
        "clip_index": int(context.get("clip_index") or 0),
        "clip_key": context.get("clip_key") or "",
        "prompt_id": prompt_id,
        "prompt_version_id": prompt_version_id,
        "rating": rating,
        "categories": action.get("categories") or [_infer_prompt_feedback_category(comment or correction or remember_note)],
        "severity": int(action.get("severity") or (4 if rating == "negative" else 1)),
        "comment": comment,
        "correction": correction,
        "remember_note": remember_note,
        "create_eval_case": bool(action.get("create_eval_case")),
        "status": action.get("status"),
    }
    try:
        created = create_prompt_feedback_item(project_name, payload, actor="chat_agent")
    except HTTPException as exc:
        return {"status": "error", "tool": "save_prompt_feedback", "message": exc.detail}
    item = created["item"]
    return {
        "status": "ok",
        "tool": "save_prompt_feedback",
        "message": "Saved prompt feedback for the active prompt version.",
        "feedback_id": item.get("id"),
        "eval_case_id": (created.get("eval_case") or {}).get("id"),
        "mutates_state": True,
    }


def suggest_prompt_lesson_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    item = _latest_feedback_for_learning_action(project_name, context, action)
    if not item:
        return {"status": "missing", "tool": "suggest_prompt_lesson", "message": "No prompt feedback is available to turn into a lesson."}
    try:
        suggestion = _suggest_prompt_lesson(project_name, item, provider)  # type: ignore[arg-type]
    except HTTPException as exc:
        return {"status": "error", "tool": "suggest_prompt_lesson", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "suggest_prompt_lesson", "message": str(exc)}
    return {
        "status": "ok",
        "tool": "suggest_prompt_lesson",
        "message": "Suggested an editable prompt lesson. Review it before saving.",
        "feedback_id": item.get("id"),
        "suggestion": suggestion,
    }


def _lesson_text_from_action(action: dict, message: str) -> str:
    for key in ("lesson", "remember_note", "prompt"):
        text = str(action.get(key) or "").strip()
        if text:
            return re.sub(
                r"^(approve lesson|save as lesson|remember this lesson|remember this|app should remember)[:\s-]*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip() or text
    return message.strip()


def approve_prompt_lesson_for_agent(project_name: str, context: dict, action: dict, message: str) -> dict:
    lesson_text = _lesson_text_from_action(action, message)
    if not lesson_text:
        return {"status": "missing", "tool": "approve_prompt_lesson", "message": "No lesson text was provided."}
    source_feedback_ids = [
        str(value)
        for value in (action.get("source_feedback_ids") or [])
        if value
    ]
    if not source_feedback_ids:
        feedback_item = _latest_feedback_for_learning_action(project_name, context, action)
        if feedback_item:
            source_feedback_ids = [feedback_item.get("id")]
    scope = action.get("scope") if action.get("scope") in {"project", "clip"} else "project"
    try:
        lesson = prompt_learning_store(project_name).add_lesson(
            lesson_text,
            scope=scope,
            clip_key=context.get("clip_key") if scope == "clip" else None,
            category=str(action.get("category") or _infer_prompt_feedback_category(lesson_text)),
            source_feedback_ids=source_feedback_ids,
            confidence=float(action.get("confidence") or 0.84),
        )
    except ValueError as exc:
        return {"status": "error", "tool": "approve_prompt_lesson", "message": str(exc)}
    append_project_event(
        project_name,
        "prompt_lesson_created",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=lesson.get("clip_key"),
        entity="prompt_lesson",
        entity_id=lesson.get("id"),
        payload={
            "scope": lesson.get("scope"),
            "category": lesson.get("category"),
            "source_feedback_ids": lesson.get("source_feedback_ids", []),
        },
    )
    return {
        "status": "ok",
        "tool": "approve_prompt_lesson",
        "message": "Lesson saved. The app will remember and apply this approved lesson in future prompt work.",
        "lesson_id": lesson.get("id"),
        "lesson": lesson,
        "mutates_state": True,
    }

def revise_prompt_from_feedback_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    _prompt_id, prompt_version_id = _active_prompt_ids_for_context(context)
    prompt_version_id = str(action.get("prompt_version_id") or prompt_version_id or "")
    if not prompt_version_id:
        return {"status": "missing", "tool": "revise_prompt_from_feedback", "message": "No active prompt version is available to revise."}
    feedback_ids = [
        str(value)
        for value in (action.get("feedback_ids") or [])
        if value
    ]
    if not feedback_ids:
        feedback_ids = [
            item.get("id")
            for item in _active_prompt_feedback_items(context, negative_only=True)
            if item.get("status") == "open" and item.get("id")
        ]
    lesson_ids = [
        str(value)
        for value in (action.get("lesson_ids") or [])
        if value
    ]
    if not lesson_ids:
        learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
        lesson_ids = [
            lesson.get("id")
            for lesson in (learning.get("relevant_lessons") or learning.get("relevant") or [])
            if lesson.get("id")
        ][:6]
    try:
        response = revise_prompt_from_feedback(
            project_name,
            prompt_version_id,
            PromptRevisionRequest(
                clip_index=int(context.get("clip_index") or 0),
                feedback_ids=feedback_ids,
                lesson_ids=lesson_ids,
                provider=provider,  # type: ignore[arg-type]
            ),
        )
    except HTTPException as exc:
        return {"status": "error", "tool": "revise_prompt_from_feedback", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "revise_prompt_from_feedback", "message": str(exc)}
    append_project_event(
        project_name,
        "prompt_version_revised_from_feedback",
        actor="chat_agent",
        clip_index=context.get("clip_index"),
        clip_key=context.get("clip_key"),
        entity="prompt_version",
        entity_id=(response.get("prompt_version") or {}).get("prompt_version_id"),
        payload={
            "source_prompt_version_id": prompt_version_id,
            "feedback_ids": feedback_ids,
            "lesson_ids": lesson_ids,
        },
    )
    return {
        "status": "ok",
        "tool": "revise_prompt_from_feedback",
        "message": "New prompt version created from feedback and approved lessons.",
        "prompt_version": response.get("prompt_version"),
        "quality_report": response.get("quality_report"),
        "learning_report": response.get("learning_report"),
        "mutates_state": True,
    }


def run_prompt_learning_eval_for_agent(project_name: str, context: dict, action: dict, provider: str) -> dict:
    latest_version = context.get("latest_version") or {}
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    if not prompt_text:
        return {"status": "missing", "tool": "run_prompt_learning_eval", "message": "No active prompt text is available to evaluate."}
    learning = context.get("learning") or ((context.get("clip_state") or {}).get("learning_state") or {})
    eval_cases = learning.get("eval_cases") or []
    lessons = learning.get("relevant_lessons") or learning.get("relevant") or learning.get("lessons") or []
    try:
        client = _client_for_prompt_provider(provider)  # type: ignore[arg-type]
        model = _default_model_for_provider(provider)
        report = run_learning_eval(
            provider=provider,
            client=client,
            model=model,
            prompt=prompt_text,
            eval_cases=eval_cases,
            lessons=lessons,
        )
    except HTTPException as exc:
        return {"status": "error", "tool": "run_prompt_learning_eval", "message": exc.detail}
    except Exception as exc:
        return {"status": "error", "tool": "run_prompt_learning_eval", "message": str(exc)}
    return {
        "status": "ok",
        "tool": "run_prompt_learning_eval",
        "message": "Ran learning eval for the active prompt.",
        "learning_eval": report,
    }


def execute_learning_action_for_agent(project_name: str, context: dict, action: dict, message: str, provider: str) -> dict:
    action_type = action.get("type")
    if action_type == "save_prompt_feedback":
        return save_prompt_feedback_for_agent(project_name, context, action, message)
    if action_type == "suggest_prompt_lesson":
        return suggest_prompt_lesson_for_agent(project_name, context, action, provider)
    if action_type == "approve_prompt_lesson":
        return approve_prompt_lesson_for_agent(project_name, context, action, message)
    if action_type == "revise_prompt_from_feedback":
        return revise_prompt_from_feedback_for_agent(project_name, context, action, provider)
    if action_type == "run_prompt_learning_eval":
        return run_prompt_learning_eval_for_agent(project_name, context, action, provider)
    return {"status": "error", "tool": action_type or "learning_action", "message": "Unsupported learning action."}


def execute_safe_agent_run_tools(project_name: str, run: dict, context: dict, message: str, saved_memories: list[dict], provider: str = "openai") -> dict:
    tool_results = list(run.get("tool_results") or [])
    errors = list(run.get("errors") or [])
    changed = False

    for step in run.get("plan_steps", []):
        if step.get("status") != "pending":
            continue
        tool = step.get("tool")
        try:
            if tool == "search_assets":
                result = search_assets_for_agent(context, message)
            elif tool == "inspect_clip_context":
                result = inspect_clip_context_for_agent(context)
            elif tool == "add_memory":
                result = memory_update_result_for_agent(saved_memories)
            elif tool == "evaluate_prompt":
                result = evaluate_prompt_for_agent(context)
            elif tool == "self_evaluate":
                result = self_evaluation_for_agent(project_name, context, run.get("id"))
            elif tool == "set_active_prompt_version":
                result = set_active_prompt_version_for_agent(project_name, context, step.get("action") or {})
            elif tool == "attach_asset":
                result = attach_asset_for_agent(project_name, context, step.get("action") or {})
            elif tool == "detach_asset":
                result = detach_asset_for_agent(project_name, context, step.get("action") or {})
            elif tool == "mark_feedback_resolved":
                result = mark_feedback_resolved_for_agent(project_name, context, step.get("action") or {})
            elif tool == "add_reference_frame":
                result = add_reference_frame_for_agent(project_name, context, step.get("action") or {}, tool_results)
            elif tool in LEARNING_CHAT_ACTION_TYPES:
                result = execute_learning_action_for_agent(project_name, context, step.get("action") or {"type": tool}, message, provider)
            elif tool in {"chat_reply", "send_message"}:
                result = {"status": "ok", "tool": tool, "message": "Chat reply completed."}
            else:
                continue
        except Exception as exc:
            result = {"status": "error", "tool": tool, "message": str(exc)}

        tool_results.append(result)
        if result.get("status") == "error":
            step["status"] = "failed"
            errors.append(str(result.get("message") or result))
        elif result.get("status") == "missing":
            step["status"] = "blocked"
        else:
            step["status"] = "completed"
        changed = True

    if changed:
        run["tool_results"] = tool_results
        run["errors"] = errors
        if errors:
            run["status"] = "failed"
        elif any(step.get("status") == "awaiting_approval" for step in run.get("plan_steps", [])):
            run["status"] = "awaiting_approval"
        elif any(step.get("status") == "dispatch_ready" for step in run.get("plan_steps", [])):
            run["status"] = "ready"
        elif all(step.get("status") in {"completed", "blocked"} for step in run.get("plan_steps", [])):
            run["status"] = "completed"
    return run


def fallback_chat_reply(user_message: str, context: dict, action_suggestions: Optional[list[dict]] = None) -> str:
    lines = []
    clip = context.get("clip") or {}
    clip_name = clip.get("clip") or f"Clip {context.get('clip_index')}"
    latest_version = context.get("latest_version") or {}
    feedback = context.get("feedback") or {}
    items = feedback.get("feedback_items", [])

    lines.append(f"### Assistant Status for {clip_name}")

    if wants_clip_media_gallery(user_message):
        media = build_clip_media_gallery(context)
        if media:
            lines.append("\n**Available Media & Reference Assets:**")
            for item in media[:10]:
                lines.append(f"- [{item.get('label')}]({item.get('url')}) ({item.get('type')})")
        else:
            lines.append("\nNo assets or media files attached to this clip.")

    if wants_prompt_display(user_message):
        lines.append(f"\n{format_latest_prompt_reply(context)}")

    if wants_clip_summary_or_analysis(user_message):
        lines.append(f"\n**Clip Analysis:**\n{summarize_clip_context(context.get('clip_context') or {})}")

    if items:
        lines.append("\n**Aligned Feedback Items:**")
        for item in items[:5]:
            lines.append(f"- #{item.get('raw_index')}: {item.get('text')}")

    if action_suggestions:
        lines.append("\n**Recommended Workflow Actions:**")
        for action in action_suggestions[:4]:
            lines.append(f"- **{action.get('label')}**: {action.get('reasoning')}")

    if len(lines) == 1:
        lines.append(f"\nReceived message for clip **{clip_name}**. Ready to analyze feedback, manage assets, or run prompt generation workflows.")

    return "\n".join(lines)


def reference_frame_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "extract_reference_frame",
            "description": "Extract a frame at a specific timeline timestamp and attach it as a visual reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Timestamp in MM:SS or HH:MM:SS format."},
                    "reason": {"type": "string", "description": "Why this frame is being extracted as a reference."},
                },
                "required": ["timestamp"],
            },
        },
    }


def _response_output_items(response: Any) -> list[Any]:
    choices = getattr(response, "choices", None)
    if choices and isinstance(choices, list) and len(choices) > 0:
        message = getattr(choices[0], "message", None)
        if message:
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", None)
            return [*tool_calls, content] if content else list(tool_calls)
    candidates = getattr(response, "candidates", None)
    if candidates and isinstance(candidates, list) and len(candidates) > 0:
        content = getattr(candidates[0], "content", None)
        if content and hasattr(content, "parts"):
            return list(content.parts)
    return []


def _output_item_value(item: Any, attr: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        msg = getattr(choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            return str(msg.content).strip()
    return ""


def _tool_call_arguments(tool_call: Any) -> dict:
    if isinstance(tool_call, dict):
        args = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments")
    else:
        func = getattr(tool_call, "function", None)
        args = getattr(func, "arguments", None) if func else getattr(tool_call, "args", None)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}


def _tool_call_id(tool_call: Any) -> str:
    return str(_output_item_value(tool_call, "id", "tool_call"))


def _extract_reference_frame_tool_calls(response: Any) -> list[dict]:
    calls = []
    for item in _response_output_items(response):
        name = _output_item_value(item, "name") or getattr(getattr(item, "function", None), "name", None)
        if name == "extract_reference_frame":
            calls.append({"id": _tool_call_id(item), "args": _tool_call_arguments(item)})
    return calls


def _run_reference_frame_tool_call(project_name: str, clip_index: int, tool_call: dict) -> dict:
    args = tool_call.get("args") or {}
    timestamp = str(args.get("timestamp") or "").strip()
    reason = str(args.get("reason") or "").strip()
    return extract_reference_frame(
        project_name=project_name,
        current_clip_index=clip_index,
        timestamp=timestamp,
        reason=reason,
        attach_to="current_feedback",
    )


def reference_frame_tool_schema() -> dict:
    return {
        "type": "function",
        "name": "extract_reference_frame",
        "description": (
            "Extract a still frame from the project timeline at an explicit timestamp and attach it "
            "as a reference to the currently open clip chat. Use only when the user clearly asks to "
            "extract, grab, capture, save, add, or attach a frame/still/reference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "Timeline timestamp such as 00:44, 1:04, or 00:01:04.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short editor-facing reason for the reference frame.",
                },
                "attach_to": {
                    "type": "string",
                    "enum": ["current_feedback"],
                    "description": "Where to attach the extracted frame.",
                },
            },
            "required": ["timestamp", "reason", "attach_to"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _response_output_items(response: Any) -> list[Any]:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    return list(output or [])


def _output_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is None and isinstance(response, dict):
        text = response.get("output_text")
    return (text or "").strip()


def _tool_call_arguments(item: Any) -> dict:
    raw_args = _output_item_value(item, "arguments", "{}")
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tool_call_id(item: Any) -> str:
    return str(_output_item_value(item, "call_id") or _output_item_value(item, "id") or "")


def _extract_reference_frame_tool_calls(response: Any) -> list[Any]:
    calls = []
    for item in _response_output_items(response):
        if _output_item_value(item, "type") == "function_call" and _output_item_value(item, "name") == "extract_reference_frame":
            calls.append(item)
    return calls


def _run_reference_frame_tool_call(project_name: str, context: dict, arguments: dict) -> dict:
    from src.reference_frames import extract_reference_frame
    return extract_reference_frame(
        project_name=project_name,
        current_clip_index=int(context.get("clip_index") or 0),
        timestamp=str(arguments.get("timestamp") or ""),
        reason=str(arguments.get("reason") or ""),
        attach_to=str(arguments.get("attach_to") or "current_feedback"),
    )


async def generate_chat_reply_with_tools(provider: str, message: str, context: dict, messages: list[dict]) -> tuple[str, list[dict]]:
    from config.settings import OPENAI_LITE_MODEL, LITE_MODEL
    from src.reference_frames import parse_explicit_reference_frame_request
    from src.clip_state import format_latest_prompt_reply, wants_prompt_display
    from src.logging_utils import log_event

    if wants_prompt_display(message):
        return format_latest_prompt_reply(context), []

    if wants_clip_summary_or_analysis(message):
        if context.get("clip_context"):
            return summarize_clip_context(context.get("clip_context")), []
        if context.get("clip_context_error"):
            return context.get("clip_context_error"), []
        return "This clip has not been analyzed yet. Ask me to analyze the clip with OpenAI, or run the workflow first.", []

    explicit_reference = parse_explicit_reference_frame_request(message)
    if explicit_reference:
        result = _run_reference_frame_tool_call(context.get("project", {}).get("name", ""), context, explicit_reference)
        return result["message"], [result]

    system_prompt = (
        "You are Loka15 Studio's clip assistant inside a video feedback and generation tool. "
        "Answer as a practical editor-facing collaborator. Use only the supplied context. "
        "You can discuss timeline, clip details, feedback, assets, prior prompt generations, quality reports, memory, workflow execution, and video-generation handoff. "
        "When discussing or revising prompts, consider learning_state.relevant_lessons, unresolved prompt feedback, eval cases, and the latest learning eval report from the supplied context. "
        "Do not claim the model has permanently learned; say the app will remember and apply approved lessons. "
        "If the user wants to save feedback, suggest or approve a lesson, revise a prompt from feedback, or run a learning eval, recommend the matching chat action instead of claiming it already happened. "
        "You can call extract_reference_frame when the user clearly asks to extract/grab/capture/save/add/attach a frame or still at an explicit timestamp as a reference for this clip. "
        "Format replies as normal Markdown. When showing prompt text, write it as plain paragraphs or bullets under a short heading; do not use ```text, ```markdown, or any Markdown code fence for prompts. "
        "Never wrap the whole reply in a Markdown code fence or indent prompt text as a code block. "
        "If the user asks to save a durable preference, include a final line starting with 'Memory:' followed by the exact note. "
        "Do not claim that you executed actions unless a tool result says the action completed."
    )
    context_text = compact_chat_context(context)
    history = [
        {"role": item.get("role"), "content": item.get("content")}
        for item in messages[-10:]
        if item.get("role") in {"user", "assistant"}
    ]

    try:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context JSON:\n{context_text}"},
                *history,
                {"role": "user", "content": message},
            ]
            response = client.responses.create(
                model=os.environ.get("OPENAI_CHAT_MODEL", OPENAI_LITE_MODEL),
                input=input_messages,
                tools=[reference_frame_tool_schema()],
            )
            tool_calls = _extract_reference_frame_tool_calls(response)
            tool_results = []
            if tool_calls:
                tool_outputs = []
                for call in tool_calls:
                    call_id = _tool_call_id(call)
                    try:
                        result = _run_reference_frame_tool_call(
                            context.get("project", {}).get("name", ""),
                            context,
                            _tool_call_arguments(call),
                        )
                    except HTTPException as exc:
                        result = {"status": "error", "message": exc.detail}
                    except Exception as exc:
                        result = {"status": "error", "message": str(exc)}
                    tool_results.append(result)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    })

                followup = client.responses.create(
                    model=os.environ.get("OPENAI_CHAT_MODEL", OPENAI_LITE_MODEL),
                    input=tool_outputs,
                    previous_response_id=getattr(response, "id", None),
                )
                reply_text = _response_text(followup)
                if reply_text:
                    return normalize_chat_reply_markdown(reply_text), tool_results
                successful = [result for result in tool_results if result.get("status") == "ok"]
                if successful:
                    return normalize_chat_reply_markdown(successful[-1].get("message", "Reference frame extracted.")), tool_results
                return normalize_chat_reply_markdown(tool_results[-1].get("message", "Reference frame extraction failed.")), tool_results
            reply_text = _response_text(response)
            return normalize_chat_reply_markdown(reply_text or fallback_chat_reply(message, context)), []

        if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            from google import genai

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            prompt = (
                f"{system_prompt}\n\nContext JSON:\n{context_text}\n\n"
                f"Recent messages:\n{json.dumps(history, ensure_ascii=False)}\n\nUser: {message}"
            )
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_CHAT_MODEL", LITE_MODEL),
                contents=prompt,
            )
            return normalize_chat_reply_markdown((response.text or "").strip()), []
    except Exception as exc:
        log_event(
            "clip_chat.llm_error",
            provider=provider,
            project=context.get("project", {}).get("name"),
            clip_index=context.get("clip_index"),
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )

    return normalize_chat_reply_markdown(fallback_chat_reply(message, context)), []


async def generate_chat_reply(provider: str, message: str, context: dict, messages: list[dict]) -> str:
    text, _tool_results = await generate_chat_reply_with_tools(provider, message, context, messages)
    return text
