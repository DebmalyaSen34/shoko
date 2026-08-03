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

    from src.prompt_feedback import _client_for_prompt_provider
    from src.generator.client import _default_model_for_provider, generate_structured
    from src.generator.validator import merge_learning_eval_report, run_learning_eval, run_quality_check

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
