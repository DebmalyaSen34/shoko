import os
import re
import json
from pathlib import Path
from typing import Any, Optional
from fastapi import HTTPException

from src.storage_paths import (
    ASSETS_DIR,
    DATA_DIR,
    chat_agent_runs_path,
    chat_memory_path,
    clip_chat_key,
    clip_selections_path,
    file_mtime_iso,
    now_iso,
    project_data_dir,
    project_events_path,
    prompt_eval_cases_path,
    prompt_feedback_path,
    prompt_lessons_path,
    read_json_file,
    stable_json_hash,
    stable_state_id,
    write_json_file,
)
from src.project_manager import (
    append_project_event,
    generated_videos_for_version,
    infer_media_type,
    media_size_for_path,
    static_url_for_path,
)
from src.chat_memory import ChatMemoryStore
from src.prompt_learning import PromptEvalCaseStore, PromptLearningStore


def memory_store(project_name: str) -> ChatMemoryStore:
    return ChatMemoryStore(chat_memory_path(project_name))


def prompt_learning_store(project_name: str) -> PromptLearningStore:
    return PromptLearningStore(prompt_lessons_path(project_name))


def prompt_eval_case_store(project_name: str) -> PromptEvalCaseStore:
    return PromptEvalCaseStore(prompt_eval_cases_path(project_name))


def load_clip_selections(project_name: str) -> dict:
    return read_json_file(clip_selections_path(project_name), {})


def save_clip_selections(project_name: str, data: dict) -> None:
    write_json_file(clip_selections_path(project_name), data)


def clip_selection_for_key(project_name: str, clip_key: str) -> dict:
    data = load_clip_selections(project_name)
    return data.get("clips", {}).get(clip_key, {})


def _asset_refs_from_ids_paths(asset_ids: list[str], asset_paths: list[str], *, source: str = "clip_selection") -> list[dict]:
    refs = []
    seen = set()
    max_count = max(len(asset_ids), len(asset_paths))
    for index in range(max_count):
        asset_id = asset_ids[index] if index < len(asset_ids) else None
        asset_path = asset_paths[index] if index < len(asset_paths) else None
        key = asset_id or asset_path
        if not key or key in seen:
            continue
        seen.add(key)
        refs.append({
            "asset_id": asset_id,
            "path": asset_path,
            "role": "selected_reference",
            "reason": "Persisted from legacy selected asset state.",
            "confidence": 0.75,
            "source": source,
        })
    return refs


def normalize_selection_asset_refs(selection: dict) -> list[dict]:
    raw_refs = selection.get("selected_assets")
    if raw_refs is None:
        raw_refs = selection.get("pinned_assets")
    if isinstance(raw_refs, list):
        refs = []
        for ref in raw_refs:
            if isinstance(ref, dict):
                asset_id = ref.get("asset_id")
                asset_path = ref.get("path") or ref.get("asset_path") or ref.get("selected_path")
                if not asset_id and not asset_path:
                    continue
                refs.append({
                    "asset_id": asset_id,
                    "path": asset_path,
                    "role": ref.get("role") or "selected_reference",
                    "reason": ref.get("reason") or "Selected for this clip.",
                    "confidence": float(ref.get("confidence") if ref.get("confidence") is not None else 0.8),
                    "source": ref.get("source") or "clip_selection",
                })
            elif isinstance(ref, str):
                refs.append({
                    "asset_id": None,
                    "path": ref,
                    "role": "selected_reference",
                    "reason": "Selected for this clip.",
                    "confidence": 0.75,
                    "source": "clip_selection",
                })
        return refs

    return _asset_refs_from_ids_paths(
        list(selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or []),
        list(selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or []),
    )


def _mirror_asset_ref_lists(selection: dict) -> None:
    refs = normalize_selection_asset_refs(selection)
    selection["selected_assets"] = refs
    selection["pinned_assets"] = refs
    selection["selected_asset_ids"] = [ref["asset_id"] for ref in refs if ref.get("asset_id")]
    selection["selected_asset_paths"] = [ref["path"] for ref in refs if ref.get("path")]
    selection["pinned_asset_ids"] = selection["selected_asset_ids"]
    selection["pinned_asset_paths"] = selection["selected_asset_paths"]


def update_clip_selection(project_name: str, clip_key: str, updates: dict) -> dict:
    selections = load_clip_selections(project_name)
    clips = selections.setdefault("clips", {})
    now = now_iso()
    current = clips.get(clip_key)
    if not isinstance(current, dict):
        current = {
            "clip_key": clip_key,
            "created_at": now,
        }
    current.update({
        key: value
        for key, value in updates.items()
        if key in {
            "active_prompt_version_id",
            "active_generated_video_id",
            "selected_assets",
            "pinned_assets",
            "selected_asset_ids",
            "selected_asset_paths",
            "pinned_asset_ids",
            "pinned_asset_paths",
        }
    })
    if "selected_asset_ids" in updates and "pinned_asset_ids" not in updates:
        current["pinned_asset_ids"] = updates["selected_asset_ids"]
    if "selected_asset_paths" in updates and "pinned_asset_paths" not in updates:
        current["pinned_asset_paths"] = updates["selected_asset_paths"]
    if any(key in updates for key in {
        "selected_assets",
        "pinned_assets",
        "selected_asset_ids",
        "selected_asset_paths",
        "pinned_asset_ids",
        "pinned_asset_paths",
    }):
        _mirror_asset_ref_lists(current)
    current["clip_key"] = clip_key
    current["updated_at"] = now
    clips[clip_key] = current
    save_clip_selections(project_name, selections)
    append_project_event(
        project_name,
        "clip_selection_updated",
        actor="chat_agent" if any(key in updates for key in {"selected_assets", "pinned_assets"}) else "backend",
        clip_key=clip_key,
        entity="clip_selection",
        entity_id=clip_key,
        payload={
            "updated_keys": sorted(updates.keys()),
            "active_prompt_version_id": current.get("active_prompt_version_id"),
            "active_generated_video_id": current.get("active_generated_video_id"),
            "selected_assets": current.get("selected_assets", []),
            "selected_asset_ids": current.get("selected_asset_ids", []),
            "selected_asset_paths": current.get("selected_asset_paths", []),
        },
    )
    return current


def _asset_identity(project_name: str, category: str, asset: dict) -> dict:
    path = str(asset.get("path") or asset.get("name") or "")
    return {
        **asset,
        "asset_id": stable_state_id("asset", project_name, category, path),
        "category": category,
    }


def _asset_lookup_by_path(assets: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for category, items in (assets or {}).items():
        for asset in items:
            enriched = _asset_identity("", category, asset)
            candidates = {
                str(asset.get("path") or ""),
                str(asset.get("name") or ""),
                os.path.basename(str(asset.get("path") or "")),
                os.path.basename(str(asset.get("name") or "")),
            }
            for candidate in candidates:
                if candidate:
                    lookup[candidate.lower()] = enriched
    return lookup


def _normalize_selected_asset(project_name: str, asset_path: str, lookup: dict[str, dict], role: str = "selected_reference") -> dict:
    path_text = str(asset_path or "")
    basename = os.path.basename(path_text)
    known = lookup.get(path_text.lower()) or lookup.get(basename.lower())
    if known:
        return {
            **known,
            "asset_id": stable_state_id("asset", project_name, known.get("category"), known.get("path") or known.get("name")),
            "role": role,
            "source": "prompt_version",
            "selected_path": path_text,
        }
    return {
        "asset_id": stable_state_id("asset", project_name, path_text),
        "name": basename or path_text,
        "path": path_text,
        "url": static_url_for_path(path_text),
        "type": infer_media_type(path_text),
        "size": media_size_for_path(path_text),
        "category": None,
        "role": role,
        "source": "prompt_version",
        "selected_path": path_text,
        "missing": not bool(static_url_for_path(path_text)),
    }


def _selected_assets_from_selection(project_name: str, selection: dict, asset_lookup: dict[str, dict]) -> list[dict]:
    structured_refs = normalize_selection_asset_refs(selection)
    pinned_assets: list[dict] = []
    seen_ids: set[str] = set()

    for ref in structured_refs:
        ref_path = ref.get("path")
        ref_id = ref.get("asset_id")
        normalized = _normalize_selected_asset(project_name, ref_path or ref_id, asset_lookup, role=ref.get("role") or "selected_reference")
        if ref_id:
            normalized["asset_id"] = ref_id
        normalized["source"] = ref.get("source") or "clip_selection"
        normalized["role"] = ref.get("role") or normalized.get("role") or "selected_reference"
        normalized["reason"] = ref.get("reason") or "Selected for this clip."
        normalized["confidence"] = ref.get("confidence", 0.8)
        normalized["structured_ref"] = ref
        if normalized["asset_id"] not in seen_ids:
            pinned_assets.append(normalized)
            seen_ids.add(normalized["asset_id"])

    return pinned_assets


def _referenced_frames_for_state(feedback: Optional[dict], latest_version: Optional[dict], project_name: str) -> list[dict]:
    referenced_frames = list((latest_version or {}).get("referenced_frames") or [])
    feedback = feedback or {}
    for item in feedback.get("feedback_items", []):
        referenced_frames.extend(item.get("referenced_frames") or [])
    referenced_frames.extend(feedback.get("referenced_frames") or [])

    normalized = []
    seen = set()
    for ref in referenced_frames:
        frame_path = str(ref.get("frame_path") or "")
        key = frame_path or json.dumps(ref, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            **ref,
            "reference_id": stable_state_id("ref", project_name, ref.get("timestamp"), frame_path, ref.get("clip_used")),
            "url": static_url_for_path(frame_path),
        })
    return normalized


def _prompt_versions_for_state(prompt: Optional[dict], project_name: str, clip_index: int) -> list[dict]:
    if not prompt:
        return []
    from src.job_manager import prompt_versions_for_record
    from src.prompt_feedback import prompt_feedback_summary_by_version
    versions = prompt_versions_for_record(prompt)
    feedback_summaries = prompt_feedback_summary_by_version(project_name, clip_index)
    prompt_id = stable_state_id(
        "prompt",
        project_name,
        prompt.get("clip_used") or prompt.get("matched_clip"),
        prompt.get("clip_occurrence"),
    )
    normalized = []
    for index, version in enumerate(versions):
        v_id = stable_state_id(
            "version",
            prompt_id,
            index,
            version.get("timestamp"),
            version.get("video_model_prompt"),
        )
        normalized.append({
            **version,
            "prompt_id": prompt_id,
            "prompt_version_id": v_id,
            "version_index": index,
            "is_latest": index == len(versions) - 1,
            "clip_index": clip_index,
            "feedback_summary": feedback_summaries.get(
                v_id,
                {
                    "prompt_version_id": v_id,
                    "status": "unreviewed",
                    "total_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "open_negative_count": 0,
                    "rejected_count": 0,
                    "latest_feedback_at": None,
                },
            ),
        })
    return normalized


def query_requests_latest_prompt(query: str = "") -> bool:
    lower = str(query or "").lower()
    has_prompt = "prompt" in lower or "prompts" in lower
    has_latest = any(token in lower for token in ["latest", "newest", "recent", "current", "last generated"])
    return has_prompt and has_latest


def has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def wants_prompt_display(text: str) -> bool:
    lower = text.lower()
    has_prompt = has_word(lower, "prompt") or has_word(lower, "prompts")
    has_display = any(has_word(lower, word) for word in ["show", "view", "display", "see", "give", "tell", "paste", "what"])
    return has_prompt and (has_display or query_requests_latest_prompt(text))


def format_latest_prompt_reply(context: dict) -> str:
    active_prompt = (context.get("clip_state") or {}).get("active_prompt") or {}
    latest_version = context.get("latest_version") or {}
    versions = active_prompt.get("versions") or []
    version_index = latest_version.get("version_index")
    if version_index is None and versions:
        version_index = len(versions) - 1
    prompt_text = str(latest_version.get("video_model_prompt") or "").strip()
    if not prompt_text:
        return "No generated prompt exists for this clip yet."
    label = f"v{int(version_index) + 1}" if version_index is not None else "latest"
    return f"Latest Prompt ({label})\n\n{prompt_text}"


def user_explicitly_requests_video_generation(text: str) -> bool:
    lower = text.lower()
    if wants_prompt_display(text):
        return False
    has_video_target = any(has_word(lower, word) for word in ["video", "clip", "render", "seedance"])
    has_generation_verb = any(has_word(lower, word) for word in ["generate", "render", "create", "make", "start"])
    return has_video_target and has_generation_verb


def build_clip_freshness(
    project_name: str,
    *,
    clip: dict,
    clip_key: str,
    feedback_state: Optional[dict],
    prompt_versions: list[dict],
    active_version: Optional[dict],
    assets: dict,
    selected_assets: list[dict],
    pinned_assets: list[dict],
    referenced_frames: list[dict],
    normalized_videos: list[dict],
    active_video: Optional[dict],
    selection: dict,
    selection_stale_reasons: list[str],
) -> dict:
    from src.prompt_feedback import prompt_feedback_summary_by_version

    source_paths = {
        "timeline": project_data_dir(project_name) / "timeline.json",
        "feedback": project_data_dir(project_name) / "feedback.json",
        "prompts": project_data_dir(project_name) / "video_prompts.json",
        "clip_selections": clip_selections_path(project_name),
        "agent_runs": chat_agent_runs_path(project_name),
        "memory": chat_memory_path(project_name),
        "prompt_feedback": prompt_feedback_path(project_name),
        "prompt_lessons": prompt_lessons_path(project_name),
        "prompt_eval_cases": prompt_eval_cases_path(project_name),
        "events": project_events_path(project_name),
    }
    fingerprints = {
        "timeline": stable_json_hash({
            "clip": clip.get("clip"),
            "start_s": clip.get("start_s"),
            "end_s": clip.get("end_s"),
            "duration_s": clip.get("duration_s"),
        }),
        "feedback": stable_json_hash(feedback_state or {}),
        "prompts": stable_json_hash({
            "prompt_version_ids": [version.get("prompt_version_id") for version in prompt_versions],
            "active_prompt_version_id": (active_version or {}).get("prompt_version_id"),
            "active_prompt_text": (active_version or {}).get("video_model_prompt"),
        }),
        "assets": stable_json_hash({
            "selected_assets": selected_assets,
            "pinned_assets": pinned_assets,
            "referenced_frames": referenced_frames,
            "available_asset_ids": sorted(
                stable_state_id("asset", project_name, category, asset.get("path") or asset.get("name"))
                for category, items in (assets or {}).items()
                for asset in items
            ),
        }),
        "videos": stable_json_hash({
            "generated_video_ids": [video.get("generated_video_id") for video in normalized_videos],
            "active_video_id": (active_video or {}).get("generated_video_id"),
        }),
        "selection": stable_json_hash({
            "active_prompt_version_id": selection.get("active_prompt_version_id"),
            "active_generated_video_id": selection.get("active_generated_video_id"),
            "selected_assets": normalize_selection_asset_refs(selection),
        }),
        "prompt_feedback": stable_json_hash(prompt_feedback_summary_by_version(project_name, active_version.get("clip_index", -1) if active_version else -1)),
        "prompt_lessons": stable_json_hash(prompt_learning_store(project_name).list_lessons(clip_key=clip_key, include_archived=True, limit=500)),
        "prompt_eval_cases": stable_json_hash(prompt_eval_case_store(project_name).list_eval_cases(clip_index=active_version.get("clip_index", -1) if active_version else -1, limit=500)),
    }
    fingerprints["state"] = stable_json_hash({
        key: value
        for key, value in fingerprints.items()
        if key != "state"
    })

    return {
        "source_files": {key: str(path) for key, path in source_paths.items()},
        "source_mtimes": {key: file_mtime_iso(path) for key, path in source_paths.items()},
        "derived_from": ["timeline", "feedback", "assets", "prompts", "clip_selections", "prompt_feedback", "prompt_lessons", "prompt_eval_cases", "memory", "agent_runs", "events"],
        "fingerprints": fingerprints,
        "state_hash": fingerprints["state"],
        "stale_reasons": selection_stale_reasons,
    }


def annotate_agent_run_freshness(run: dict, current_freshness: dict) -> dict:
    annotated = dict(run)
    previous = ((run.get("context_summary") or {}).get("freshness_snapshot") or {})
    previous_fingerprints = previous.get("fingerprints") or {}
    current_fingerprints = current_freshness.get("fingerprints") or {}
    stale_reasons = []

    for key in ["timeline", "feedback", "prompts", "assets", "videos", "selection", "prompt_feedback", "prompt_lessons", "prompt_eval_cases"]:
        if previous_fingerprints.get(key) and previous_fingerprints.get(key) != current_fingerprints.get(key):
            stale_reasons.append(f"{key}_changed")

    if previous.get("state_hash") and previous.get("state_hash") != current_freshness.get("state_hash") and not stale_reasons:
        stale_reasons.append("clip_state_changed")

    annotated["freshness"] = {
        "is_stale": bool(stale_reasons),
        "stale_reasons": stale_reasons,
        "captured_state_hash": previous.get("state_hash"),
        "current_state_hash": current_freshness.get("state_hash"),
        "captured_at": previous.get("captured_at"),
    }
    return annotated


def build_learning_state_for_clip(
    project_name: str,
    *,
    clip_index: int,
    clip_key: str,
    query: str,
    active_version: Optional[dict],
) -> dict:
    from src.prompt_feedback import prompt_feedback_items_for_clip
    learning_store = prompt_learning_store(project_name)
    base_state = learning_store.for_clip(clip_key, query=query)
    project_lessons = base_state.get("project") or []
    clip_lessons = base_state.get("clip") or []
    relevant_lessons = base_state.get("relevant") or []
    eval_cases = prompt_eval_case_store(project_name).list_eval_cases(
        clip_index=clip_index,
        limit=20,
    )
    active_version_id = (active_version or {}).get("prompt_version_id")
    feedback_items = [
        item
        for item in prompt_feedback_items_for_clip(project_name, clip_index)
        if not active_version_id or item.get("prompt_version_id") == active_version_id
    ]
    open_issue_count = sum(
        1
        for item in feedback_items
        if item.get("rating") == "negative" and item.get("status") == "open"
    )
    quality_report = (active_version or {}).get("quality_report") or {}
    return {
        **base_state,
        "feedback_count": len(feedback_items),
        "open_issue_count": open_issue_count,
        "lessons": [*project_lessons, *clip_lessons],
        "relevant_lessons": relevant_lessons,
        "eval_cases": eval_cases,
        "latest_learning_report": quality_report.get("learning_eval"),
    }


def build_clip_state(project_name: str, clip_index: int, query: str = "") -> dict:
    from src.project_manager import get_project_data
    from src.clip_chat import matching_feedback, matching_prompt, load_clip_context, recent_agent_runs_for_clip
    from src.job_manager import list_recent_project_jobs

    project_data = get_project_data(project_name)
    timeline = project_data.get("timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    clip_key = clip_chat_key(clip.get("clip", ""), clip_index)
    selection = clip_selection_for_key(project_name, clip_key)
    selection_stale_reasons = []
    feedback = matching_feedback(project_data, clip, clip_index)
    prompt = matching_prompt(project_data, clip, clip_index)
    prompt_versions = _prompt_versions_for_state(prompt, project_name, clip_index)
    default_version = prompt_versions[-1] if prompt_versions else None
    selected_prompt_version_id = selection.get("active_prompt_version_id")
    if selected_prompt_version_id and query_requests_latest_prompt(query):
        selection_stale_reasons.append("active_prompt_version_overridden_by_latest_query")
        selected_prompt_version_id = None
    selected_version = next(
        (version for version in prompt_versions if version.get("prompt_version_id") == selected_prompt_version_id),
        None,
    ) if selected_prompt_version_id else None
    if selected_prompt_version_id and not selected_version:
        selection_stale_reasons.append("active_prompt_version_missing")
    latest_version = selected_version or default_version
    assets = project_data.get("assets", {})
    asset_lookup = _asset_lookup_by_path(assets)
    selected_assets = [
        _normalize_selected_asset(project_name, asset_path, asset_lookup)
        for asset_path in (latest_version or {}).get("selected_assets", [])
    ]
    pinned_assets = _selected_assets_from_selection(project_name, selection, asset_lookup)
    referenced_frames = _referenced_frames_for_state(feedback, latest_version, project_name)
    generated_videos = generated_videos_for_version(latest_version)
    normalized_videos = [
        {
            **video,
            "generated_video_id": stable_state_id(
                "video",
                project_name,
                clip_key,
                video.get("version"),
                video.get("timestamp"),
                video.get("path") or video.get("url"),
            ),
        }
        for video in generated_videos
    ]
    default_video = normalized_videos[-1] if normalized_videos else None
    selected_video_id = selection.get("active_generated_video_id")
    selected_video = next(
        (video for video in normalized_videos if video.get("generated_video_id") == selected_video_id),
        None,
    ) if selected_video_id else None
    if selected_video_id and not selected_video:
        selection_stale_reasons.append("active_generated_video_missing")
    latest_video = selected_video or default_video
    clip_context = load_clip_context(project_name, clip, clip_index)
    store = memory_store(project_name)
    feedback_items = (feedback or {}).get("feedback_items", [])
    feedback_indexes = {item.get("raw_index") for item in feedback_items}
    recent_jobs = [
        job for job in list_recent_project_jobs(project_name)
        if job.get("clip_index") == clip_index
        or job.get("feedback_index") in feedback_indexes
    ][:10]

    feedback_state = {
        **(feedback or {}),
        "feedback_id": stable_state_id("feedback", project_name, clip_key),
        "feedback_items": [
            {
                **item,
                "feedback_item_id": stable_state_id(
                    "feedback_item",
                    project_name,
                    clip_key,
                    item.get("raw_index"),
                    item.get("timestamp"),
                    item.get("remark"),
                ),
            }
            for item in feedback_items
        ],
    } if feedback else None
    freshness = build_clip_freshness(
        project_name,
        clip=clip,
        clip_key=clip_key,
        feedback_state=feedback_state,
        prompt_versions=prompt_versions,
        active_version=latest_version,
        assets=assets,
        selected_assets=selected_assets,
        pinned_assets=pinned_assets,
        referenced_frames=referenced_frames,
        normalized_videos=normalized_videos,
        active_video=latest_video,
        selection=selection,
        selection_stale_reasons=selection_stale_reasons,
    )
    agent_runs = [
        annotate_agent_run_freshness(run, freshness)
        for run in recent_agent_runs_for_clip(project_name, clip_key)
    ]

    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "clip_key": clip_key,
        "clip_state_id": stable_state_id("clip_state", project_name, clip_key),
        "updated_at": now_iso(),
        "project": {
            "name": project_data.get("project_name", project_name),
            "sequence_name": project_data.get("sequence_name", ""),
            "total_duration_tc": project_data.get("total_duration_tc", ""),
            "total_duration_s": project_data.get("total_duration_s", 0),
            "clip_count": len(timeline),
        },
        "timeline": {
            "clip": clip,
            "adjacent_clips": {
                "previous": timeline[clip_index - 1] if clip_index > 0 else None,
                "next": timeline[clip_index + 1] if clip_index < len(timeline) - 1 else None,
            },
        },
        "feedback_state": feedback_state,
        "active_prompt": {
            "prompt": prompt,
            "prompt_id": prompt_versions[-1]["prompt_id"] if prompt_versions else None,
            "version": latest_version,
            "version_id": latest_version.get("prompt_version_id") if latest_version else None,
            "version_index": latest_version.get("version_index") if latest_version else None,
            "versions": prompt_versions,
            "prompt_ready": bool((latest_version or {}).get("video_model_prompt")),
        },
        "asset_state": {
            "available_assets": {
                category: [_asset_identity(project_name, category, asset) for asset in items]
                for category, items in assets.items()
            },
            "selected_assets": selected_assets,
            "pinned_assets": pinned_assets,
            "referenced_frames": referenced_frames,
            "missing_assets": [asset for asset in [*selected_assets, *pinned_assets] if asset.get("missing")],
        },
        "video_state": {
            "generated_videos": normalized_videos,
            "latest_video": latest_video,
            "active_video": latest_video,
            "active_video_id": latest_video.get("generated_video_id") if latest_video else None,
            "generation_attempts": (latest_version or {}).get("video_generation_attempts", []),
            "latest_attempt": (latest_version or {}).get("latest_video_generation_attempt"),
        },
        "selection_state": {
            "clip_key": clip_key,
            "active_prompt_version_id": selected_prompt_version_id,
            "resolved_prompt_version_id": latest_version.get("prompt_version_id") if latest_version else None,
            "active_generated_video_id": selected_video_id,
            "resolved_generated_video_id": latest_video.get("generated_video_id") if latest_video else None,
            "selected_assets": normalize_selection_asset_refs(selection),
            "selected_asset_ids": selection.get("selected_asset_ids") or selection.get("pinned_asset_ids") or [],
            "selected_asset_paths": selection.get("selected_asset_paths") or selection.get("pinned_asset_paths") or [],
            "pinned_assets": normalize_selection_asset_refs(selection),
            "pinned_asset_ids": selection.get("pinned_asset_ids") or selection.get("selected_asset_ids") or [],
            "pinned_asset_paths": selection.get("pinned_asset_paths") or selection.get("selected_asset_paths") or [],
            "resolved_assets": pinned_assets,
            "selection_source": "persisted" if selection else "default_latest",
            "stale_reasons": selection_stale_reasons,
            "updated_at": selection.get("updated_at"),
        },
        "analysis_state": {
            "clip_context": clip_context,
            "clip_context_ready": bool(clip_context),
        },
        "memory_state": store.for_clip(clip_key, query=query),
        "learning_state": build_learning_state_for_clip(
            project_name,
            clip_index=clip_index,
            clip_key=clip_key,
            query=query,
            active_version=latest_version,
        ),
        "agent_state": {
            "recent_runs": agent_runs,
            "pending_actions": [
                action
                for run in agent_runs
                for action in run.get("suggested_actions", [])
                if run.get("status") in {"ready", "awaiting_approval", "planned"}
            ][:8],
        },
        "job_state": {
            "recent_jobs": recent_jobs,
            "active_jobs": [job for job in recent_jobs if job.get("status") in {"queued", "running", "cancelling"}],
        },
        "freshness": freshness,
    }


def promote_latest_prompt_selection(project_name: str, clip_index: Optional[int]) -> Optional[dict]:
    if clip_index is None:
        return None
    try:
        from src.project_manager import get_project_data
        from src.clip_chat import matching_prompt

        project_data = get_project_data(project_name)
        timeline = project_data.get("timeline", [])
        if clip_index < 0 or clip_index >= len(timeline):
            return None
        clip = timeline[clip_index]
        prompt = matching_prompt(project_data, clip, clip_index)
        prompt_versions = _prompt_versions_for_state(prompt, project_name, clip_index)
        if not prompt_versions:
            return None
        latest_version = prompt_versions[-1]
        clip_key = clip_chat_key(clip.get("clip", ""), clip_index)
        return update_clip_selection(
            project_name,
            clip_key,
            {"active_prompt_version_id": latest_version.get("prompt_version_id")},
        )
    except Exception as exc:
        print(f"Warning: could not promote latest prompt selection for {project_name} clip {clip_index}: {exc}")
        return None
