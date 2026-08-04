from typing import Literal, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.storage_paths import safe_project_name


from src.clip_chat import (
    LEARNING_CHAT_ACTION_TYPES,
    apply_agent_run_action_policy,
    build_clip_chat_context,
    create_chat_agent_run,
    dynamic_chat_suggestions,
    ensure_clip_context_for_chat,
    execute_safe_agent_run_tools,
    extract_memory_notes,
    generate_chat_reply_with_tools,
    infer_action_suggestions,
    load_chat_history,
    plan_chat_agent_run,
    save_chat_history,
    save_pending_workflow_intents,
    strip_risky_autonomous_actions,
    update_chat_agent_run,
    wants_clip_summary_or_analysis,
    wants_regenerated_clip_summary,
)
from src.clip_state import memory_store
from src.job_manager import dispatch_agent_run_jobs
from src.project_manager import (
    build_clip_media_gallery,
    make_chat_message,
    wants_clip_media_gallery,
)

router = APIRouter(tags=["chat"])


class ClipChatRequest(BaseModel):
    clip_index: int
    message: str
    provider: Literal["openai", "gemini"] = "openai"


class ClipChatActionRequest(BaseModel):
    clip_index: int
    action: dict
    message: str = ""
    provider: Literal["openai", "gemini"] = "openai"


class ClipMemoryRequest(BaseModel):
    clip_index: int
    text: str
    scope: Literal["clip", "project"] = "clip"


@router.get("/api/projects/{project_name}/chat/clip/{clip_index}")
def get_clip_chat(project_name: str, clip_index: int):
    project_name = safe_project_name(project_name)
    context = build_clip_chat_context(project_name, clip_index)
    history = load_chat_history(project_name)
    messages = history.get("clips", {}).get(context["clip_key"], [])

    if not messages:
        messages = [
            make_chat_message(
                "assistant",
                (
                    f"I am attached to clip #{clip_index + 1}. I can use timeline, feedback, assets, "
                    "saved memory, generated prompts, workflow execution, and video-generation handoff context."
                ),
                {"kind": "welcome"},
            )
        ]
        history.setdefault("clips", {})[context["clip_key"]] = messages
        save_chat_history(project_name, history)

    return {
        "project_name": project_name,
        "clip_index": clip_index,
        "clip_key": context["clip_key"],
        "messages": messages,
        "memory": context["memory"],
        "agent_runs": context["agent_runs"],
        "context": {
            "project": context["project"],
            "clip": context["clip"],
            "clip_state": context["clip_state"],
            "learning_state": context.get("learning", {}),
            "adjacent_clips": context["adjacent_clips"],
            "feedback_count": len((context.get("feedback") or {}).get("feedback_items", [])),
            "selected_asset_count": len((context.get("latest_version") or {}).get("selected_assets", []) or []),
            "prompt_ready": bool((context.get("latest_version") or {}).get("video_model_prompt")),
        },
    }


@router.post("/api/projects/{project_name}/chat/clip")
async def post_clip_chat(project_name: str, request: ClipChatRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
    history = load_chat_history(project_name)
    clip_messages = history.setdefault("clips", {}).setdefault(context["clip_key"], [])

    user_message = make_chat_message("user", message_text)
    clip_messages.append(user_message)

    analysis_requested = wants_clip_summary_or_analysis(message_text)
    regenerate_summary = wants_regenerated_clip_summary(message_text)
    if analysis_requested:
        _clip_context, analysis_error = ensure_clip_context_for_chat(
            project_name,
            context,
            request.provider,
            force=regenerate_summary,
        )
        if analysis_error:
            context["clip_context_error"] = analysis_error

    assistant_text, tool_results = await generate_chat_reply_with_tools(request.provider, message_text, context, clip_messages)
    if tool_results:
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
    memory_notes = extract_memory_notes(message_text, assistant_text)
    saved_memories = []

    if memory_notes:
        store = memory_store(project_name)
        for note in memory_notes:
            saved_memories.append(
                store.add(
                    note,
                    scope="clip",
                    clip_key=context["clip_key"],
                    source="chat",
                    tags=["chat"],
                    confidence=0.86,
                )
            )
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)

    explicit_actions = infer_action_suggestions(message_text, context)
    actions = dynamic_chat_suggestions(message_text, context, explicit_actions)
    agent_run = plan_chat_agent_run(message_text, context, explicit_actions, actions)
    actions = apply_agent_run_action_policy(actions, agent_run)
    agent_run["available_actions"] = actions
    agent_run["suggested_actions"] = actions
    agent_run["tool_results"] = tool_results
    agent_run = execute_safe_agent_run_tools(project_name, agent_run, context, message_text, saved_memories, provider=request.provider)
    mutation_results = [
        result for result in agent_run.get("tool_results", [])
        if isinstance(result, dict) and result.get("mutates_state")
    ]
    if mutation_results:
        context = build_clip_chat_context(project_name, request.clip_index, query=message_text)
        successful_mutations = [result for result in mutation_results if result.get("status") == "ok"]
        if successful_mutations:
            assistant_text = "\n".join(result.get("message", "State updated.") for result in successful_mutations)
    if any(result.get("status") == "error" for result in tool_results if isinstance(result, dict)):
        agent_run["status"] = "failed"
        agent_run["errors"] = [
            str(result.get("message") or result)
            for result in tool_results
            if isinstance(result, dict) and result.get("status") == "error"
        ]
    agent_run = create_chat_agent_run(project_name, agent_run)
    agent_run = dispatch_agent_run_jobs(project_name, agent_run, background_tasks, provider=request.provider)
    if agent_run.get("jobs"):
        actions = strip_risky_autonomous_actions(actions)
        agent_run["available_actions"] = actions
        agent_run["suggested_actions"] = actions
        agent_run = update_chat_agent_run(
            project_name,
            agent_run["id"],
            {
                "status": agent_run.get("status"),
                "plan_steps": agent_run.get("plan_steps", []),
                "tool_results": agent_run.get("tool_results", []),
                "available_actions": actions,
                "suggested_actions": actions,
                "jobs": agent_run.get("jobs", []),
            },
        )
    save_pending_workflow_intents(project_name, actions, message_text, context)
    media = build_clip_media_gallery(context) if (wants_clip_media_gallery(message_text) or tool_results or analysis_requested) else []
    assistant_message = make_chat_message(
        "assistant",
        assistant_text,
        {
            "provider": request.provider,
            "actions": actions,
            "media": media,
            "saved_memory_ids": [item["id"] for item in saved_memories],
            "tool_results": tool_results,
            "agent_run": agent_run,
            "clip_state_id": context.get("clip_state", {}).get("clip_state_id"),
            "prompt_version_id": (context.get("clip_state", {}).get("active_prompt") or {}).get("version_id"),
        },
    )
    clip_messages.append(assistant_message)
    save_chat_history(project_name, history)

    return {
        "project_name": project_name,
        "clip_index": request.clip_index,
        "clip_key": context["clip_key"],
        "messages": clip_messages,
        "assistant_message": assistant_message,
        "memory": context["memory"],
        "suggested_actions": actions,
        "agent_run": agent_run,
        "clip_state": context["clip_state"],
    }


@router.post("/api/projects/{project_name}/chat/clip/action")
def execute_clip_chat_action(project_name: str, request: ClipChatActionRequest):
    project_name = safe_project_name(project_name)
    action = dict(request.action or {})
    action_type = action.get("type")
    if action_type not in LEARNING_CHAT_ACTION_TYPES:
        raise HTTPException(status_code=400, detail="This endpoint only executes prompt learning chat actions.")

    context = build_clip_chat_context(project_name, request.clip_index, query=request.message or action.get("prompt") or action.get("label") or "")
    run = plan_chat_agent_run(
        request.message or action.get("label") or action_type,
        context,
        [action],
        [action],
    )
    for step in run.get("plan_steps", []):
        if step.get("tool") == action_type:
            step["status"] = "pending"
            step["requires_approval"] = False
    run["approval_required"] = False
    run["autonomy_level"] = "approved_action"
    run["status"] = "planned"
    run = execute_safe_agent_run_tools(
        project_name,
        run,
        context,
        request.message or action.get("prompt") or action.get("label") or "",
        [],
        provider=request.provider,
    )
    run = create_chat_agent_run(project_name, run)
    updated_context = build_clip_chat_context(project_name, request.clip_index, query=request.message or action.get("prompt") or "")
    return {
        "project_name": project_name,
        "clip_index": request.clip_index,
        "clip_key": updated_context["clip_key"],
        "action_result": (run.get("tool_results") or [{}])[-1],
        "agent_run": run,
        "clip_state": updated_context["clip_state"],
    }


@router.post("/api/projects/{project_name}/chat/memory")
def add_clip_memory(project_name: str, request: ClipMemoryRequest):
    project_name = safe_project_name(project_name)
    context = build_clip_chat_context(project_name, request.clip_index)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Memory text cannot be empty")

    item = memory_store(project_name).add(
        text,
        scope=request.scope,
        clip_key=context["clip_key"] if request.scope == "clip" else None,
        source="manual",
        tags=["manual"],
        confidence=0.9,
    )
    updated_context = build_clip_chat_context(project_name, request.clip_index)
    return {"memory": updated_context["memory"], "item": item}
