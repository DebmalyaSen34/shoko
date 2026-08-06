import json
from typing import Optional
from fastapi import APIRouter, HTTPException

from src.storage_paths import now_iso, safe_project_name
from src.prompt_feedback import (
    PromptEvalCasePatchRequest,
    PromptFeedbackCreateRequest,
    PromptFeedbackPatchRequest,
    PromptLessonCreateRequest,
    PromptLessonPatchRequest,
    PromptLessonSuggestRequest,
    PromptRevisionRequest,
    _compact_applied_eval_cases,
    _eval_case_revision_text,
    _feedback_revision_text,
    _lesson_revision_text,
    _validate_prompt_feedback_payload,
)
from src.schemas import PromptResult


from src.clip_state import (
    build_clip_state,
    prompt_eval_case_store,
    prompt_learning_store,
    update_clip_selection,
)
from src.generator.client import (
    _default_model_for_provider,
    generate_structured,
)
from src.generator.validator import (
    merge_learning_eval_report,
    run_learning_eval,
    run_quality_check,
)
from src.job_manager import append_prompt_version
from src.project_manager import append_project_event
from src.prompt_feedback import (
    _client_for_prompt_provider,
    clip_state_for_lesson_sources,
    create_eval_case_from_feedback_item,
    create_prompt_feedback_item,
    find_prompt_feedback_item,
    load_prompt_feedback,
    prompt_feedback_items_for_clip,
    prompt_feedback_summary_by_version,
    save_prompt_feedback,
)
router = APIRouter(tags=["feedback"])


@router.get("/api/projects/{project_name}/prompt-feedback")
def get_prompt_feedback(project_name: str, clip_index: Optional[int] = None):
    project_name = safe_project_name(project_name)
    items = prompt_feedback_items_for_clip(project_name, clip_index)
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "items": items,
        "summaries": prompt_feedback_summary_by_version(project_name, clip_index) if clip_index is not None else {},
    }


@router.get("/api/projects/{project_name}/prompt-lessons")
def get_prompt_lessons(
    project_name: str,
    clip_key: Optional[str] = None,
    category: Optional[str] = None,
    include_archived: bool = False,
):
    project_name = safe_project_name(project_name)
    lessons = prompt_learning_store(project_name).list_lessons(
        clip_key=clip_key,
        category=category,
        include_archived=include_archived,
    )
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_key": clip_key,
        "category": category,
        "lessons": lessons,
    }


@router.get("/api/projects/{project_name}/prompt-eval-cases")
def get_prompt_eval_cases(
    project_name: str,
    clip_index: Optional[int] = None,
    enabled: Optional[bool] = None,
):
    project_name = safe_project_name(project_name)
    cases = prompt_eval_case_store(project_name).list_eval_cases(
        clip_index=clip_index,
        enabled=enabled,
    )
    return {
        "schema_version": 1,
        "project_name": project_name,
        "clip_index": clip_index,
        "enabled": enabled,
        "cases": cases,
    }


@router.post("/api/projects/{project_name}/prompt-lessons")
def create_prompt_lesson(project_name: str, request: PromptLessonCreateRequest):
    project_name = safe_project_name(project_name)
    try:
        lesson = prompt_learning_store(project_name).add_lesson(
            request.lesson,
            scope=request.scope,
            clip_key=request.clip_key,
            category=request.category,
            source_feedback_ids=request.source_feedback_ids,
            confidence=request.confidence,
            positive_examples=request.positive_examples,
            negative_examples=request.negative_examples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_lesson_created",
        actor="user",
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
        "lesson": lesson,
        "clip_state": clip_state_for_lesson_sources(project_name, lesson.get("source_feedback_ids", [])),
    }


@router.patch("/api/projects/{project_name}/prompt-lessons/{lesson_id}")
def patch_prompt_lesson(project_name: str, lesson_id: str, request: PromptLessonPatchRequest):
    project_name = safe_project_name(project_name)
    try:
        lesson = prompt_learning_store(project_name).update_lesson(
            lesson_id,
            request.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prompt lesson was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_lesson_updated",
        actor="user",
        clip_key=lesson.get("clip_key"),
        entity="prompt_lesson",
        entity_id=lesson.get("id"),
        payload={
            "scope": lesson.get("scope"),
            "category": lesson.get("category"),
            "archived": lesson.get("archived"),
        },
    )
    return {
        "lesson": lesson,
        "clip_state": clip_state_for_lesson_sources(project_name, lesson.get("source_feedback_ids", [])),
    }


@router.post("/api/projects/{project_name}/prompts/{prompt_version_id}/revise-from-feedback")
def revise_prompt_from_feedback(project_name: str, prompt_version_id: str, request: PromptRevisionRequest):
    from src.prompt_feedback import revise_prompt_from_feedback as revise_prompt_from_feedback_service

    return revise_prompt_from_feedback_service(project_name, prompt_version_id, request)


@router.post("/api/projects/{project_name}/prompt-feedback")
def create_prompt_feedback(project_name: str, request: PromptFeedbackCreateRequest):
    project_name = safe_project_name(project_name)
    created = create_prompt_feedback_item(project_name, request.model_dump(), actor="user")
    item = created["item"]
    response = {
        "item": item,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }
    if created.get("eval_case"):
        response["eval_case"] = created["eval_case"]
    return response


@router.post("/api/projects/{project_name}/prompt-feedback/{feedback_id}/eval-case")
def create_prompt_eval_case_from_feedback(project_name: str, feedback_id: str):
    project_name = safe_project_name(project_name)
    item = find_prompt_feedback_item(project_name, feedback_id)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")
    if item.get("rating") != "negative":
        raise HTTPException(status_code=400, detail="Eval cases can only be created from negative feedback.")
    try:
        eval_case = create_eval_case_from_feedback_item(project_name, item)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_eval_case_created",
        actor="user",
        clip_index=item.get("clip_index"),
        clip_key=item.get("clip_key"),
        entity="prompt_eval_case",
        entity_id=eval_case.get("id"),
        payload={
            "source_feedback_id": item.get("id"),
            "failure_categories": eval_case.get("failure_categories", []),
        },
    )
    return {
        "eval_case": eval_case,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }


@router.post("/api/projects/{project_name}/prompt-feedback/{feedback_id}/suggest-lesson")
def suggest_prompt_lesson(project_name: str, feedback_id: str, request: PromptLessonSuggestRequest):
    project_name = safe_project_name(project_name)
    item = find_prompt_feedback_item(project_name, feedback_id)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")
    from src.prompt_feedback import _suggest_prompt_lesson
    suggestion = _suggest_prompt_lesson(project_name, item, request.provider)
    return {
        "project_name": project_name,
        "feedback_id": feedback_id,
        "suggestion": suggestion,
    }


@router.patch("/api/projects/{project_name}/prompt-eval-cases/{case_id}")
def patch_prompt_eval_case(project_name: str, case_id: str, request: PromptEvalCasePatchRequest):
    project_name = safe_project_name(project_name)
    try:
        eval_case = prompt_eval_case_store(project_name).update_eval_case(
            case_id,
            request.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prompt eval case was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_project_event(
        project_name,
        "prompt_eval_case_updated",
        actor="user",
        clip_index=eval_case.get("clip_index"),
        clip_key=eval_case.get("clip_key"),
        entity="prompt_eval_case",
        entity_id=eval_case.get("id"),
        payload={"enabled": eval_case.get("enabled")},
    )
    return {
        "eval_case": eval_case,
        "clip_state": build_clip_state(project_name, eval_case.get("clip_index", 0)),
    }


@router.patch("/api/projects/{project_name}/prompt-feedback/{feedback_id}")
def patch_prompt_feedback(project_name: str, feedback_id: str, request: PromptFeedbackPatchRequest):
    project_name = safe_project_name(project_name)
    data = load_prompt_feedback(project_name)
    items = data.setdefault("items", [])
    item = next((candidate for candidate in items if candidate.get("id") == feedback_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt feedback was not found")

    updates = request.model_dump(exclude_none=True)
    allowed = {
        "rating",
        "categories",
        "severity",
        "comment",
        "correction",
        "remember_note",
        "create_eval_case",
        "status",
    }
    next_item = {**item, **{key: value for key, value in updates.items() if key in allowed}}
    validated = _validate_prompt_feedback_payload(next_item)
    item.update(validated)
    item["updated_at"] = now_iso()
    save_prompt_feedback(project_name, data)
    append_project_event(
        project_name,
        "prompt_feedback_updated",
        actor="user",
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
    return {
        "item": item,
        "clip_state": build_clip_state(project_name, item["clip_index"]),
    }
