import os
import json
import uuid
from typing import Any, Literal, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.storage_paths import (
    atomic_write_json_file,
    now_iso,
    project_data_dir,
    prompt_eval_cases_path,
    prompt_feedback_path,
    prompt_lessons_path,
    read_json_file,
    safe_project_name,
)

from src.project_manager import append_project_event
from src.generator.client import _default_model_for_provider, generate_structured
from src.generator.validator import (
    merge_learning_eval_report,
    run_learning_eval,
    run_quality_check,
)
from src.prompt_learning import PromptEvalCaseStore, PromptLearningStore
from src.schema import PromptRevisionRequest
from src.schemas import PromptResult

PROMPT_FEEDBACK_CATEGORIES = {
    "missed_feedback",
    "wrong_visual_detail",
    "wrong_character_or_wardrobe",
    "continuity_error",
    "bad_camera_instruction",
    "bad_audio_or_dialogue",
    "unsupported_assumption",
    "format_error",
    "too_vague",
    "too_verbose",
    "provider_incompatible",
    "other",
}
PROMPT_FEEDBACK_RATINGS = {"positive", "negative"}
PROMPT_FEEDBACK_STATUSES = {"open", "approved", "rejected", "resolved"}


class PromptFeedbackCreateRequest(BaseModel):
    clip_index: int
    clip_key: str
    prompt_id: str
    prompt_version_id: str
    rating: Literal["positive", "negative"]
    categories: list[str] = Field(default_factory=list)
    severity: int = 3
    comment: str = ""
    correction: str = ""
    remember_note: str = ""
    create_eval_case: bool = False
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptFeedbackPatchRequest(BaseModel):
    rating: Optional[Literal["positive", "negative"]] = None
    categories: Optional[list[str]] = None
    severity: Optional[int] = None
    comment: Optional[str] = None
    correction: Optional[str] = None
    remember_note: Optional[str] = None
    create_eval_case: Optional[bool] = None
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptLessonCreateRequest(BaseModel):
    scope: Literal["project", "clip"] = "project"
    clip_key: Optional[str] = None
    category: str = "other"
    lesson: str
    source_feedback_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class PromptLessonPatchRequest(BaseModel):
    scope: Optional[Literal["project", "clip"]] = None
    clip_key: Optional[str] = None
    category: Optional[str] = None
    lesson: Optional[str] = None
    source_feedback_ids: Optional[list[str]] = None
    confidence: Optional[float] = None
    positive_examples: Optional[list[str]] = None
    negative_examples: Optional[list[str]] = None
    archived: Optional[bool] = None


class PromptEvalCasePatchRequest(BaseModel):
    name: Optional[str] = None
    input: Optional[dict] = None
    expected_behavior: Optional[list[str]] = None
    failure_categories: Optional[list[str]] = None
    enabled: Optional[bool] = None


class PromptRevisionRequest(BaseModel):
    clip_index: int
    feedback_ids: list[str] = Field(default_factory=list)
    lesson_ids: list[str] = Field(default_factory=list)
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestRequest(BaseModel):
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestionResult(BaseModel):
    lesson: str
    category: str = "other"
    confidence: float = 0.75
    reasoning: str = ""


def prompt_eval_case_store(project_name: str) -> PromptEvalCaseStore:
    return PromptEvalCaseStore(project_data_dir(project_name) / "prompt_eval_cases.json")


def load_prompt_feedback(project_name: str) -> dict:
    data = read_json_file(prompt_feedback_path(project_name), {"schema_version": 1, "items": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "items": []}
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    return {"schema_version": 1, "items": [item for item in items if isinstance(item, dict)]}


def save_prompt_feedback(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("items", [])
    atomic_write_json_file(prompt_feedback_path(project_name), data)


def _clean_prompt_feedback_categories(categories: list[str]) -> list[str]:
    cleaned = []
    for category in categories or []:
        if category not in PROMPT_FEEDBACK_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid prompt feedback category: {category}")
        if category not in cleaned:
            cleaned.append(category)
    return cleaned


def _validate_prompt_feedback_payload(payload: dict) -> dict:
    rating = payload.get("rating")
    if rating not in PROMPT_FEEDBACK_RATINGS:
        raise HTTPException(status_code=400, detail="Invalid prompt feedback rating")
    status = payload.get("status")
    if status and status not in PROMPT_FEEDBACK_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid prompt feedback status")

    try:
        severity = int(payload.get("severity", 3))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Severity must be a number from 1 to 5") from exc
    if severity < 1 or severity > 5:
        raise HTTPException(status_code=400, detail="Severity must be from 1 to 5")

    categories = _clean_prompt_feedback_categories(list(payload.get("categories") or []))
    comment = str(payload.get("comment") or "").strip()
    correction = str(payload.get("correction") or "").strip()
    remember_note = str(payload.get("remember_note") or "").strip()
    if rating == "negative" and not any([comment, correction, remember_note]):
        raise HTTPException(status_code=400, detail="Negative feedback needs a comment, correction, or remember note")

    prompt_version_id = str(payload.get("prompt_version_id") or "").strip()
    prompt_id = str(payload.get("prompt_id") or "").strip()
    clip_key = str(payload.get("clip_key") or "").strip()
    if not prompt_version_id:
        raise HTTPException(status_code=400, detail="Prompt version id is required")
    if not prompt_id:
        raise HTTPException(status_code=400, detail="Prompt id is required")
    if not clip_key:
        raise HTTPException(status_code=400, detail="Clip key is required")
    clip_index = payload.get("clip_index")
    if not isinstance(clip_index, int) or clip_index < 0:
        raise HTTPException(status_code=400, detail="Valid clip index is required")

    return {
        **payload,
        "clip_index": clip_index,
        "clip_key": clip_key,
        "prompt_id": prompt_id,
        "prompt_version_id": prompt_version_id,
        "rating": rating,
        "categories": categories,
        "severity": severity,
        "comment": comment,
        "correction": correction,
        "remember_note": remember_note,
        "create_eval_case": bool(payload.get("create_eval_case", False)),
        "status": status or ("approved" if rating == "positive" else "open"),
    }


def create_prompt_feedback_item(project_name: str, payload: dict, *, actor: str = "user") -> dict:
    payload = _validate_prompt_feedback_payload(payload)
    created_at = now_iso()
    item = {
        "id": str(uuid.uuid4()),
        "project_name": project_name,
        **payload,
        "created_at": created_at,
        "updated_at": created_at,
    }
    data = load_prompt_feedback(project_name)
    data.setdefault("items", []).append(item)
    save_prompt_feedback(project_name, data)
    eval_case = None
    if item.get("create_eval_case") and item.get("rating") == "negative":
        try:
            eval_case = create_eval_case_from_feedback_item(project_name, item)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_feedback_created",
        actor=actor,
        clip_index=item.get("clip_index"),
        clip_key=item.get("clip_key"),
        entity="prompt_feedback",
        entity_id=item.get("id"),
        payload={
            "prompt_id": item.get("prompt_id"),
            "prompt_version_id": item.get("prompt_version_id"),
            "rating": item.get("rating"),
            "status": item.get("status"),
            "categories": item.get("categories", []),
        },
    )
    if eval_case:
        append_project_event(
            project_name,
            "prompt_eval_case_created",
            actor=actor,
            clip_index=item.get("clip_index"),
            clip_key=item.get("clip_key"),
            entity="prompt_eval_case",
            entity_id=eval_case.get("id"),
            payload={
                "source_feedback_id": item.get("id"),
                "failure_categories": eval_case.get("failure_categories", []),
            },
        )
    return {"item": item, "eval_case": eval_case}


def prompt_feedback_items_for_clip(project_name: str, clip_index: Optional[int] = None) -> list[dict]:
    items = load_prompt_feedback(project_name).get("items", [])
    if clip_index is None:
        return items
    return [item for item in items if item.get("clip_index") == clip_index]


def prompt_feedback_summary_by_version(project_name: str, clip_index: int) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for item in prompt_feedback_items_for_clip(project_name, clip_index):
        version_id = str(item.get("prompt_version_id") or "")
        if not version_id:
            continue
        summary = summaries.setdefault(version_id, {
            "prompt_version_id": version_id,
            "status": "unreviewed",
            "total_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "open_negative_count": 0,
            "rejected_count": 0,
            "latest_feedback_at": None,
        })
        summary["total_count"] += 1
        if item.get("rating") == "positive":
            summary["positive_count"] += 1
        if item.get("rating") == "negative":
            summary["negative_count"] += 1
            if item.get("status") == "open":
                summary["open_negative_count"] += 1
        if item.get("status") == "rejected":
            summary["rejected_count"] += 1
        updated_at = item.get("updated_at") or item.get("created_at")
        if updated_at and (not summary.get("latest_feedback_at") or updated_at > summary["latest_feedback_at"]):
            summary["latest_feedback_at"] = updated_at

    for summary in summaries.values():
        if summary["rejected_count"]:
            summary["status"] = "rejected"
        elif summary["open_negative_count"]:
            summary["status"] = "needs_revision"
        elif summary["positive_count"] and not summary["open_negative_count"]:
            summary["status"] = "approved"
        else:
            summary["status"] = "unreviewed"
    return summaries


def find_prompt_feedback_item(project_name: str, feedback_id: str) -> Optional[dict]:
    return next((item for item in load_prompt_feedback(project_name).get("items", []) if item.get("id") == feedback_id), None)


def clip_state_for_lesson_sources(project_name: str, source_feedback_ids: list[str]) -> Optional[dict]:
    from src.clip_state import build_clip_state
    for feedback_id in source_feedback_ids or []:
        item = find_prompt_feedback_item(project_name, feedback_id)
        if item and isinstance(item.get("clip_index"), int):
            return build_clip_state(project_name, item["clip_index"])
    return None


def _prompt_context_for_feedback(project_name: str, feedback_item: dict) -> dict:
    from src.clip_state import build_clip_state
    clip_index = feedback_item.get("clip_index")
    if not isinstance(clip_index, int):
        return {}
    state = build_clip_state(project_name, clip_index)
    versions = (state.get("active_prompt") or {}).get("versions") or []
    version = next(
        (candidate for candidate in versions if candidate.get("prompt_version_id") == feedback_item.get("prompt_version_id")),
        None,
    )
    return {
        "clip": ((state.get("timeline") or {}).get("clip") or {}),
        "feedback_state": state.get("feedback_state"),
        "prompt_version": version or {},
    }


def _eval_case_context_for_feedback(project_name: str, feedback_item: dict) -> dict:
    context = _prompt_context_for_feedback(project_name, feedback_item)
    prompt_version = context.get("prompt_version") or {}
    feedback_state = context.get("feedback_state") or {}
    return {
        "clip_summary": (
            prompt_version.get("clip_context_summary")
            or (((context.get("clip") or {}).get("summary")) if isinstance(context.get("clip"), dict) else "")
            or ""
        ),
        "selected_assets": prompt_version.get("selected_assets") or [],
        "feedback_items": feedback_state.get("feedback_items") or [
            {
                "remark": feedback_item.get("comment") or feedback_item.get("correction") or feedback_item.get("remember_note") or "",
                "category": "video",
            }
        ],
    }


def create_eval_case_from_feedback_item(project_name: str, feedback_item: dict) -> dict:
    context = _eval_case_context_for_feedback(project_name, feedback_item)
    return prompt_eval_case_store(project_name).add_eval_case_from_feedback(
        feedback_item,
        clip_summary=context.get("clip_summary", ""),
        selected_assets=context.get("selected_assets", []),
        feedback_items=context.get("feedback_items", []),
    )


def _client_for_prompt_provider(provider: Literal["openai", "gemini"]):
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")
    from google import genai

    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _feedback_revision_text(feedback_items: list[dict]) -> str:
    return "\n".join(
        (
            f"- rating={item.get('rating')} severity={item.get('severity')} "
            f"categories={item.get('categories') or []}\n"
            f"  comment: {item.get('comment') or ''}\n"
            f"  correction: {item.get('correction') or ''}\n"
            f"  remember: {item.get('remember_note') or ''}"
        )
        for item in feedback_items
    )


def _lesson_revision_text(lessons: list[dict]) -> str:
    return "\n".join(
        f"- [{lesson.get('category')}] {lesson.get('lesson')}"
        for lesson in lessons
        if lesson.get("lesson")
    )


def _eval_case_revision_text(eval_cases: list[dict]) -> str:
    return "\n".join(
        f"- {case.get('name')}: {'; '.join(case.get('expected_behavior') or [])}"
        for case in eval_cases
        if case.get("enabled", True)
    )


def _compact_applied_eval_cases(eval_cases: list[dict]) -> list[dict]:
    return [
        {
            "id": case.get("id"),
            "name": case.get("name"),
            "failure_categories": case.get("failure_categories", []),
            "expected_behavior": case.get("expected_behavior", []),
        }
        for case in eval_cases
    ]


def _suggest_prompt_lesson(project_name: str, feedback_item: dict, provider: Literal["openai", "gemini"]) -> dict:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    context = _prompt_context_for_feedback(project_name, feedback_item)
    prompt_version = context.get("prompt_version") or {}
    prompt_text = str(prompt_version.get("video_model_prompt") or "")[:2400]
    prompt = (
        "Create one durable lesson from this prompt feedback. The lesson must help future prompt generation avoid "
        "repeating the same kind of mistake. Avoid copying the raw complaint, avoid overgeneralizing from one clip, "
        "and write a rule that is useful across similar prompt-generation tasks.\n\n"
        f"PROJECT: {project_name}\n"
        f"FEEDBACK CATEGORY: {json.dumps(feedback_item.get('categories') or [], ensure_ascii=False)}\n"
        f"RATING: {feedback_item.get('rating')}\n"
        f"SEVERITY: {feedback_item.get('severity')}\n"
        f"COMMENT: {feedback_item.get('comment') or ''}\n"
        f"CORRECTION: {feedback_item.get('correction') or ''}\n"
        f"REMEMBER NOTE: {feedback_item.get('remember_note') or ''}\n"
        f"CLIP: {json.dumps(context.get('clip') or {}, ensure_ascii=False)}\n"
        f"PROMPT EXCERPT:\n{prompt_text}"
    )
    try:
        suggestion = generate_structured(
            provider=provider,
            client=client,
            model=_default_model_for_provider(provider),
            contents=[prompt],
            schema=PromptLessonSuggestionResult,
            system_instruction=(
                "You convert user prompt feedback into one concise, editable lesson. "
                "Return only a reusable rule, a category, confidence, and brief reasoning."
            ),
            temperature=0.1,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to suggest lesson: {exc}") from exc

    category = suggestion.get("category") or (feedback_item.get("categories") or ["other"])[0]
    if category not in PROMPT_FEEDBACK_CATEGORIES:
        category = (feedback_item.get("categories") or ["other"])[0]
    try:
        confidence = max(0.0, min(1.0, float(suggestion.get("confidence", 0.75))))
    except Exception:
        confidence = 0.75
    return {
        "lesson": str(suggestion.get("lesson") or "").strip(),
        "category": category,
        "confidence": confidence,
        "reasoning": str(suggestion.get("reasoning") or "").strip(),
        "source_feedback_id": feedback_item.get("id"),
    }


def revise_prompt_from_feedback(project_name: str, prompt_version_id: str, request: PromptRevisionRequest):
    from src.clip_state import build_clip_state, update_clip_selection
    from src.job_manager import append_prompt_version
    project_name = safe_project_name(project_name)
    state = build_clip_state(project_name, request.clip_index)
    clip_key = state.get("clip_key")
    versions = (state.get("active_prompt") or {}).get("versions") or []
    source_version = next(
        (version for version in versions if version.get("prompt_version_id") == prompt_version_id),
        None,
    )
    if not source_version:
        raise HTTPException(status_code=404, detail="Prompt version was not found for this clip.")
    source_prompt = str(source_version.get("video_model_prompt") or "").strip()
    if not source_prompt:
        raise HTTPException(status_code=400, detail="Selected prompt version has no prompt text.")

    feedback_items = [
        item
        for item in load_prompt_feedback(project_name).get("items", [])
        if item.get("clip_index") == request.clip_index
        and item.get("prompt_version_id") == prompt_version_id
    ]
    if request.feedback_ids:
        requested_feedback_ids = set(request.feedback_ids)
        selected_feedback = [item for item in feedback_items if item.get("id") in requested_feedback_ids]
        if len(selected_feedback) != len(requested_feedback_ids):
            raise HTTPException(status_code=400, detail="One or more feedback ids do not belong to this prompt version and clip.")
    else:
        selected_feedback = [
            item
            for item in feedback_items
            if item.get("rating") == "negative" and item.get("status") == "open"
        ]
    if not selected_feedback:
        raise HTTPException(status_code=400, detail="No matching prompt feedback was selected for revision.")

    lesson_store = prompt_learning_store(project_name)
    all_lessons = lesson_store.list_lessons(include_archived=False, limit=500)
    if request.lesson_ids:
        requested_lesson_ids = set(request.lesson_ids)
        selected_lessons = [lesson for lesson in all_lessons if lesson.get("id") in requested_lesson_ids]
        if len(selected_lessons) != len(requested_lesson_ids):
            raise HTTPException(status_code=400, detail="One or more lesson ids were not found.")
        for lesson in selected_lessons:
            if lesson.get("scope") == "clip" and lesson.get("clip_key") != clip_key:
                raise HTTPException(status_code=400, detail="One or more lesson ids belong to another clip.")
    else:
        query = " ".join(
            " ".join(str(item.get(key) or "") for key in ("comment", "correction", "remember_note"))
            for item in selected_feedback
        )
        selected_lessons = [
            candidate.item
            for candidate in lesson_store.retrieve_lessons(
                query=query,
                clip_key=clip_key,
                limit=6,
            )
        ]

    eval_case_candidates = prompt_eval_case_store(project_name).retrieve_eval_cases(
        query=" ".join(
            " ".join(str(item.get(key) or "") for key in ("comment", "correction", "remember_note"))
            for item in selected_feedback
        ),
        clip_key=clip_key,
        clip_index=request.clip_index,
        limit=6,
    )
    eval_cases = [candidate.item for candidate in eval_case_candidates]

    client = _client_for_prompt_provider(request.provider)
    model = _default_model_for_provider(request.provider)
    clip = (state.get("timeline") or {}).get("clip") or {}
    clip_context = (state.get("analysis_state") or {}).get("clip_context") or {}
    selected_assets = source_version.get("selected_assets") or []
    previous_quality_report = source_version.get("quality_report") or {}
    revision_prompt = (
        "Revise this existing AI video generation prompt using the supplied user feedback. "
        "Create a new improved version; do not overwrite history. Keep all correct details from the original prompt, "
        "preserve clip continuity, and change only what is necessary to satisfy feedback, lessons, and eval cases.\n\n"
        f"PROJECT: {project_name}\n"
        f"CLIP: {json.dumps(clip, ensure_ascii=False)}\n"
        f"CLIP_CONTEXT: {json.dumps(clip_context, ensure_ascii=False)}\n"
        f"SELECTED_ASSETS: {json.dumps(selected_assets, ensure_ascii=False)}\n\n"
        f"ORIGINAL_PROMPT:\n{source_prompt}\n\n"
        f"USER_FEEDBACK_TO_FIX:\n{_feedback_revision_text(selected_feedback)}\n\n"
        f"APPROVED_OR_RELEVANT_LESSONS:\n{_lesson_revision_text(selected_lessons) or 'None'}\n\n"
        f"RELEVANT_EVAL_CASE_EXPECTATIONS:\n{_eval_case_revision_text(eval_cases) or 'None'}\n\n"
        f"PREVIOUS_QUALITY_REPORT:\n{json.dumps(previous_quality_report, ensure_ascii=False)}\n\n"
        "Return a revised prompt and a short explanation of what changed. The revised prompt must remain English-only, "
        "provider-compatible, concrete, and detailed."
    )
    try:
        revision = generate_structured(
            provider=request.provider,
            client=client,
            model=model,
            contents=[revision_prompt],
            schema=PromptResult,
            system_instruction=(
                "You are a careful AI video prompt editor. You revise existing prompts using explicit user feedback, "
                "approved lessons, and eval case expectations while preserving correct continuity and visual details."
            ),
            temperature=0.15,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to revise prompt: {exc}") from exc

    revised_prompt = str(revision.get("video_model_prompt") or "").strip()
    if not revised_prompt:
        raise HTTPException(status_code=500, detail="Prompt revision returned empty prompt text.")

    quality_report = run_quality_check(
        provider=request.provider,
        client=client,
        model=model,
        prompt=revised_prompt,
        feedback_items=[
            {"remark": item.get("correction") or item.get("comment") or item.get("remember_note") or ""}
            for item in selected_feedback
        ],
        selected_assets=selected_assets,
        has_clip=bool(source_version.get("clip_frame_paths")),
    )
    learning_report = run_learning_eval(
        provider=request.provider,
        client=client,
        model=model,
        prompt=revised_prompt,
        eval_cases=eval_cases,
        lessons=selected_lessons,
    )
    quality_report = merge_learning_eval_report(quality_report, learning_report)

    new_version_payload = {
        **source_version,
        "timestamp": now_iso(),
        "provider": request.provider,
        "video_model_prompt": revised_prompt,
        "explanation": revision.get("explanation") or "Revised from prompt feedback.",
        "quality_report": quality_report,
        "revision_source_prompt_version_id": prompt_version_id,
        "revision_feedback_ids": [item.get("id") for item in selected_feedback],
        "revision_lesson_ids": [lesson.get("id") for lesson in selected_lessons],
        "applied_prompt_lessons": [
            {
                "id": lesson.get("id"),
                "scope": lesson.get("scope"),
                "category": lesson.get("category"),
                "lesson": lesson.get("lesson"),
            }
            for lesson in selected_lessons
        ],
        "applied_prompt_eval_cases": _compact_applied_eval_cases(eval_cases),
    }
    append_prompt_version(project_name, request.clip_index, new_version_payload, provider=request.provider)
    refreshed_state = build_clip_state(project_name, request.clip_index)
    latest_version = ((refreshed_state.get("active_prompt") or {}).get("versions") or [])[-1]
    update_clip_selection(project_name, clip_key, {"active_prompt_version_id": latest_version.get("prompt_version_id")})
    refreshed_state = build_clip_state(project_name, request.clip_index)
    prompt_version = (refreshed_state.get("active_prompt") or {}).get("version") or latest_version
    append_project_event(
        project_name,
        "prompt_version_revised_from_feedback",
        actor="user",
        clip_index=request.clip_index,
        clip_key=clip_key,
        entity="prompt_version",
        entity_id=prompt_version.get("prompt_version_id"),
        payload={
            "source_prompt_version_id": prompt_version_id,
            "feedback_ids": [item.get("id") for item in selected_feedback],
            "lesson_ids": [lesson.get("id") for lesson in selected_lessons],
            "eval_case_ids": [case.get("id") for case in eval_cases],
        },
    )
    return {
        "prompt_version": prompt_version,
        "quality_report": quality_report,
        "learning_report": learning_report,
        "clip_state": refreshed_state,
    }


