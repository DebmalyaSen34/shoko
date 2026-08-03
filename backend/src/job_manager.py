import os
import re
import sys
import json
import time
import uuid
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from fastapi import BackgroundTasks, HTTPException

import src.storage_paths as storage_paths
from src.storage_paths import (
    BACKEND_DIR,
    atomic_write_json_file,
    chat_agent_runs_path,
    chat_workflow_intents_path,
    clip_chat_key,
    file_mtime_iso,
    now_iso,
    project_assets_dir,
    project_data_dir,
    project_events_path,
    project_jobs_path,
    read_json_file,
    safe_project_name,
    write_json_file,
)
from src.logging_utils import log_event
from src.project_manager import (
    append_project_event,
    ensure_raw_feedback,
)
from src.workflows.prompt_generation import extract_last_frame, get_video_duration
from scripts.generate_seedance_video import (
    SeedanceGenerationRecoveryError,
    SupabaseAssetUrlCache,
    build_segmind_payload,
    create_seedance_task,
    save_video_bytes,
)

PROMPT_VERSION_HANDOFF_FIELDS = [
    "video_provider",
    "segmind_model",
    "segmind_payload_status",
    "segmind_payload",
    "segmind_prompt",
    "segmind_first_frame_url",
    "segmind_reference_images",
    "segmind_reference_videos",
    "segmind_reference_audios",
    "segmind_payload_error",
    "audio_reference_path",
    "trimmed_audio_path",
    "audio_trim_start_s",
    "audio_trim_end_s",
    "audio_trim_duration_s",
    "audio_trim_source",
    "audio_trim_error",
    "audio_transcript",
    "dialogue_text",
    "dialogue_language",
    "dialogue_extraction_reasoning",
    "audio_used",
    "audio_path",
    "audio_url",
    "referenced_frames",
    "referenced_frame_paths",
    "referenced_frame_labels",
    "is_dialogue_active",
    "generate_audio",
    "ratio",
    "duration",
    "matched_clip",
    "clip_occurrence",
    "clip_start_tc",
    "clip_end_tc",
    "clip_start_s",
    "clip_end_s",
    "clip_duration_s",
    "clip_context_path",
    "clip_segment_path",
    "clip_context_summary",
    "clip_context_status",
    "applied_prompt_lessons",
    "applied_prompt_eval_cases",
    "revision_source_prompt_version_id",
    "revision_feedback_ids",
    "revision_lesson_ids",
]


def load_project_jobs(project_name: str) -> dict:
    data = read_json_file(project_jobs_path(project_name), {"schema_version": 1, "jobs": []})
    if not isinstance(data, dict):
        return {"schema_version": 1, "jobs": []}
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        jobs = []
    return {"schema_version": 1, "jobs": jobs}


def save_project_jobs(project_name: str, data: dict) -> None:
    data.setdefault("schema_version", 1)
    data.setdefault("jobs", [])
    write_json_file(project_jobs_path(project_name), data)


def create_project_job(
    project_name: str,
    job_type: str,
    *,
    clip_index: Optional[int] = None,
    feedback_index: Optional[int] = None,
    provider: str = "",
    payload: Optional[dict] = None,
    agent_run_id: Optional[str] = None,
) -> dict:
    data = load_project_jobs(project_name)
    now = now_iso()
    job = {
        "id": str(uuid.uuid4()),
        "type": job_type,
        "status": "queued",
        "project_name": project_name,
        "clip_index": clip_index,
        "feedback_index": feedback_index,
        "provider": provider,
        "payload": payload or {},
        "agent_run_id": agent_run_id,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
    }
    data.setdefault("jobs", []).append(job)
    save_project_jobs(project_name, data)
    append_project_event(
        project_name,
        "job_created",
        actor="agent" if agent_run_id else "backend",
        clip_index=clip_index,
        entity="job",
        entity_id=job["id"],
        payload={
            "job_type": job_type,
            "status": job["status"],
            "feedback_index": feedback_index,
            "provider": provider,
            "agent_run_id": agent_run_id,
        },
    )
    return job


def update_project_job(project_name: str, job_id: str, updates: dict) -> dict:
    data = load_project_jobs(project_name)
    updated = {}
    previous_status = None
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            previous_status = job.get("status")
            job.update(updates)
            job["updated_at"] = now_iso()
            updated = job
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    save_project_jobs(project_name, data)
    if "status" in updates and updates.get("status") != previous_status:
        append_project_event(
            project_name,
            "job_status_changed",
            actor="backend",
            clip_index=updated.get("clip_index"),
            entity="job",
            entity_id=job_id,
            payload={
                "job_type": updated.get("type"),
                "previous_status": previous_status,
                "status": updated.get("status"),
                "feedback_index": updated.get("feedback_index"),
                "agent_run_id": updated.get("agent_run_id"),
                "error": updated.get("error"),
            },
        )
    elif "result" in updates:
        append_project_event(
            project_name,
            "job_result_updated",
            actor="backend",
            clip_index=updated.get("clip_index"),
            entity="job",
            entity_id=job_id,
            payload={
                "job_type": updated.get("type"),
                "status": updated.get("status"),
                "has_self_evaluation": bool((updated.get("result") or {}).get("self_evaluation")),
            },
        )
    return updated


def append_project_job_log(project_name: str, job_id: str, message: str) -> dict:
    data = load_project_jobs(project_name)
    updated = {}
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            job.setdefault("logs", []).append({
                "timestamp": now_iso(),
                "message": message,
            })
            job["updated_at"] = now_iso()
            updated = job
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    save_project_jobs(project_name, data)
    return updated


def get_project_job(project_name: str, job_id: str) -> dict:
    data = load_project_jobs(project_name)
    for job in data.get("jobs", []):
        if job.get("id") == job_id:
            return job
    raise HTTPException(status_code=404, detail="Job not found")


def list_recent_project_jobs(project_name: str, limit: int = 50) -> list[dict]:
    data = load_project_jobs(project_name)
    jobs = list(data.get("jobs", []))
    jobs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return jobs[:limit]


def project_job_cancel_requested(project_name: str, job_id: str) -> bool:
    try:
        return bool(get_project_job(project_name, job_id).get("cancel_requested"))
    except HTTPException:
        return False


def copy_prompt_handoff_fields(target: dict, source: dict) -> None:
    for field in PROMPT_VERSION_HANDOFF_FIELDS:
        if field in source:
            target[field] = source.get(field)


def prompt_history_entry(item: dict, entry_provider: str) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": entry_provider,
        "video_model_prompt": item.get("video_model_prompt"),
        "selected_assets": item.get("selected_assets", []),
        "explanation": item.get("explanation"),
        "quality_report": item.get("quality_report"),
        "initial_frame_image_path": item.get("initial_frame_image_path"),
        "initial_frame_prompt": item.get("initial_frame_prompt"),
        "clip_frame_paths": item.get("clip_frame_paths", []),
    }
    copy_prompt_handoff_fields(entry, item)
    return entry


def _same_prompt_clip(prompt_record: dict, matched_clip: str, clip_occurrence: Optional[int]) -> bool:
    clip_used = prompt_record.get("clip_used")
    same_clip = (
        clip_used == matched_clip
        or (clip_used and matched_clip and os.path.basename(clip_used) == os.path.basename(matched_clip))
    )
    existing_occurrence = prompt_record.get("clip_occurrence")
    same_occurrence = (
        clip_occurrence is None
        or existing_occurrence is None
        or existing_occurrence == clip_occurrence
    )
    return bool(same_clip and same_occurrence)


def append_prompt_version_to_records(
    prompts_data: list[dict],
    gen_item: dict,
    provider: str = "unknown",
) -> dict:
    matched_clip = gen_item.get("matched_clip") or gen_item.get("clip_used")
    if not matched_clip:
        raise ValueError("matched_clip or clip_used is required to append a prompt version")
    clip_occurrence = gen_item.get("clip_occurrence")

    for prompt_record in prompts_data:
        if not _same_prompt_clip(prompt_record, matched_clip, clip_occurrence):
            continue
        prompt_record["latest_error"] = None
        history = prompt_record.get("history", [])
        if not isinstance(history, list):
            history = []
        if len(history) == 0 and prompt_record.get("video_model_prompt"):
            first_entry = prompt_history_entry(prompt_record, prompt_record.get("provider", "unknown"))
            first_entry["timestamp"] = prompt_record.get("created_at", first_entry["timestamp"])
            history.append(first_entry)
        new_entry = prompt_history_entry(gen_item, provider)
        history.append(new_entry)
        prompt_record["history"] = history
        prompt_record["video_model_prompt"] = gen_item.get("video_model_prompt")
        prompt_record["selected_assets"] = gen_item.get("selected_assets", [])
        prompt_record["explanation"] = gen_item.get("explanation")
        prompt_record["status"] = "success"
        prompt_record["quality_report"] = gen_item.get("quality_report")
        prompt_record["initial_frame_image_path"] = gen_item.get("initial_frame_image_path")
        prompt_record["initial_frame_prompt"] = gen_item.get("initial_frame_prompt")
        prompt_record["clip_frame_paths"] = gen_item.get("clip_frame_paths", [])
        prompt_record["referenced_frames"] = gen_item.get("referenced_frames", [])
        prompt_record["referenced_frame_paths"] = gen_item.get("referenced_frame_paths", [])
        prompt_record["referenced_frame_labels"] = gen_item.get("referenced_frame_labels", [])
        copy_prompt_handoff_fields(prompt_record, gen_item)
        return new_entry

    new_entry = prompt_history_entry(gen_item, provider)
    prompt_record = {
        "clip_used": matched_clip,
        "matched_clip": gen_item.get("matched_clip") or matched_clip,
        "clip_occurrence": gen_item.get("clip_occurrence"),
        "category": gen_item.get("category", "video"),
        "generation_type": gen_item.get("prompt_format", gen_item.get("generation_type", "complex")),
        "video_model_prompt": gen_item.get("video_model_prompt"),
        "selected_assets": gen_item.get("selected_assets", []),
        "status": "success",
        "explanation": gen_item.get("explanation"),
        "quality_report": gen_item.get("quality_report"),
        "initial_frame_image_path": gen_item.get("initial_frame_image_path"),
        "initial_frame_prompt": gen_item.get("initial_frame_prompt"),
        "clip_frame_paths": gen_item.get("clip_frame_paths", []),
        "referenced_frames": gen_item.get("referenced_frames", []),
        "referenced_frame_paths": gen_item.get("referenced_frame_paths", []),
        "referenced_frame_labels": gen_item.get("referenced_frame_labels", []),
        "history": [new_entry],
    }
    copy_prompt_handoff_fields(prompt_record, gen_item)
    prompts_data.append(prompt_record)
    return new_entry


def append_prompt_version(project_name: str, clip_index: int, new_prompt_version: dict, provider: str = "unknown") -> dict:
    project_name = safe_project_name(project_name)
    timeline_path = project_data_dir(project_name) / "timeline.json"
    timeline_data = read_json_file(timeline_path, {})
    timeline = timeline_data.get("video_timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")
    clip = timeline[clip_index]
    prompts_path = project_data_dir(project_name) / "video_prompts.json"
    prompts_data = read_json_file(prompts_path, [])
    if not isinstance(prompts_data, list):
        prompts_data = []
    gen_item = {
        **new_prompt_version,
        "matched_clip": new_prompt_version.get("matched_clip") or new_prompt_version.get("clip_used") or clip.get("clip"),
        "clip_used": new_prompt_version.get("clip_used") or new_prompt_version.get("matched_clip") or clip.get("clip"),
        "clip_occurrence": new_prompt_version.get("clip_occurrence", clip_index),
        "category": new_prompt_version.get("category", "video"),
    }
    appended = append_prompt_version_to_records(prompts_data, gen_item, provider)
    write_json_file(prompts_path, prompts_data)
    return appended


def save_output_to_prompts(project: str, provider: str = "unknown"):
    project_dir = project_data_dir(project)
    output_json = project_dir / "output.json"
    prompts_json = project_dir / "video_prompts.json"

    if not output_json.exists():
        return

    try:
        with output_json.open("r", encoding="utf-8") as f:
            output_data = json.load(f)
    except Exception as e:
        print(f"Error loading output.json: {e}")
        return

    if not isinstance(output_data, list) or len(output_data) == 0:
        return

    prompts_data = []
    if prompts_json.exists():
        try:
            with prompts_json.open("r", encoding="utf-8") as f:
                prompts_data = json.load(f)
        except Exception as e:
            print(f"Error loading video_prompts.json: {e}")

    updated = False
    for gen_item in output_data:
        matched_clip = gen_item.get("matched_clip")
        if not matched_clip:
            continue

        prompt_text = gen_item.get("video_model_prompt", "")
        prompt_text_lower = prompt_text.lower()
        error_phrases = ["quota expired", "model usage in spike", "rate limit", "error:", "api error", "limit exceeded", "overloaded"]
        is_error = any(phrase in prompt_text_lower for phrase in error_phrases)

        found = False
        if is_error:
            for p_item in prompts_data:
                if _same_prompt_clip(p_item, matched_clip, gen_item.get("clip_occurrence")):
                    p_item["latest_error"] = prompt_text
                    updated = True
                    found = True
                    break
        else:
            append_prompt_version_to_records(prompts_data, gen_item, provider)
            updated = True
            found = True

        if not found:
            if is_error:
                prompts_data.append({
                    "clip_used": matched_clip,
                    "matched_clip": gen_item.get("matched_clip"),
                    "clip_occurrence": gen_item.get("clip_occurrence"),
                    "category": gen_item.get("category", "video"),
                    "generation_type": gen_item.get("prompt_format", "complex"),
                    "video_model_prompt": "",
                    "selected_assets": [],
                    "status": "failed",
                    "latest_error": prompt_text,
                    "history": []
                })
            updated = True

    if updated:
        try:
            with prompts_json.open("w", encoding="utf-8") as f:
                json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving video_prompts.json: {e}")


def build_workflow_subprocess_command(project: str, index: int, provider: str) -> tuple[list[str], Optional[str], Optional[str], Optional[str]]:
    safe_project_name(project)
    try:
        raw_feedback_path = ensure_raw_feedback(project)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate raw feedback mapping: {exc}") from exc

    assets_dir = project_assets_dir(project)
    if not assets_dir.exists():
        raise HTTPException(status_code=404, detail=f"Assets not found for project {project}")

    project_dir = project_data_dir(project)
    timeline_path = project_dir / "timeline.json"
    output_json = project_dir / "output.json"
    output_report = project_dir / "output_report.md"
    continuity_frame_path = None
    continuity_note = None
    continuity_error = None
    intent = workflow_intent_for_feedback(project, index)
    if intent:
        continuity_frame_path, continuity_note = prepare_continuity_reference_from_intent(project, index, intent)
        if not continuity_frame_path and intent.get("continuity_reference"):
            continuity_error = "Continuity reference requested but could not be prepared."
            mark_workflow_intent_used(project, index, {"status": "failed", "error": continuity_error})

    cmd = [
        sys.executable,
        "main.py",
        "--index", str(index),
        "--feedback-path", str(raw_feedback_path),
        "--timeline-path", str(timeline_path),
        "--assets-dir", str(assets_dir),
        "--output-json", str(output_json),
        "--output-report", str(output_report),
        "--provider", provider,
    ]
    if continuity_frame_path:
        cmd.extend(["--continuity-frame-path", continuity_frame_path])
    if continuity_note:
        cmd.extend(["--continuity-note", continuity_note])
    return cmd, continuity_frame_path, continuity_note, continuity_error


def terminal_job_status(status: str) -> bool:
    return status in {"succeeded", "failed", "cancelled"}


def agent_run_jobs_terminal(project_name: str, agent_run_id: str) -> bool:
    jobs = [
        job for job in load_project_jobs(project_name).get("jobs", [])
        if job.get("agent_run_id") == agent_run_id
    ]
    return bool(jobs) and all(terminal_job_status(str(job.get("status"))) for job in jobs)


def record_self_evaluation_after_job(project_name: str, job_id: str, clip_index: Optional[int], trigger: str) -> Optional[dict]:
    if clip_index is None:
        append_project_job_log(project_name, job_id, "[SELF_EVAL] Skipped: could not resolve clip index.")
        return None

    try:
        from src.clip_chat import evaluate_agent_state_for_clip

        evaluation = evaluate_agent_state_for_clip(project_name, int(clip_index), trigger=trigger, agent_run_id=get_project_job(project_name, job_id).get("agent_run_id"))
    except Exception as exc:
        append_project_job_log(project_name, job_id, f"[SELF_EVAL] Skipped: {exc}")
        return None
    job = get_project_job(project_name, job_id)
    result = dict(job.get("result") or {})
    result["self_evaluation"] = evaluation
    update_project_job(project_name, job_id, {"result": result})
    append_project_job_log(project_name, job_id, f"[SELF_EVAL] {evaluation['message']}")
    append_project_event(
        project_name,
        "self_evaluation_completed",
        actor="chat_agent",
        clip_index=clip_index,
        clip_key=evaluation.get("clip_key"),
        entity="job",
        entity_id=job_id,
        payload={
            "trigger": trigger,
            "verdict": evaluation.get("verdict"),
            "next_action": evaluation.get("next_action"),
            "state_hash": evaluation.get("state_hash"),
            "agent_run_id": job.get("agent_run_id"),
        },
    )

    agent_run_id = job.get("agent_run_id")
    if agent_run_id:
        current_run = next(
            (run for run in load_chat_agent_runs(project_name).get("runs", []) if run.get("id") == agent_run_id),
            None,
        )
        if current_run:
            tool_results = list(current_run.get("tool_results") or [])
            tool_results.append(evaluation)
            updates = {
                "tool_results": tool_results,
                "self_evaluation": evaluation,
            }
            if agent_run_jobs_terminal(project_name, agent_run_id):
                updates["status"] = "completed" if evaluation.get("verdict") == "passed" else "needs_attention"
            update_chat_agent_run(project_name, agent_run_id, updates)
    return evaluation


async def execute_workflow_job(project: str, job_id: str) -> None:
    job = get_project_job(project, job_id)
    feedback_index = int(job.get("feedback_index"))
    provider = job.get("provider") or "openai"
    update_project_job(project, job_id, {"status": "running", "started_at": now_iso()})
    append_project_job_log(project, job_id, f"[START] Launching workflow subprocess for feedback index {feedback_index}.")
    workflow_started = time.perf_counter()

    try:
        cmd, continuity_frame_path, _continuity_note, continuity_error = build_workflow_subprocess_command(project, feedback_index, provider)
        if continuity_frame_path:
            append_project_job_log(project, job_id, f"[CONTEXT] Prepared continuity reference: {continuity_frame_path}")
        elif continuity_error:
            append_project_job_log(project, job_id, f"[CONTEXT] {continuity_error}")
        append_project_job_log(project, job_id, f"Executing command: {' '.join(cmd)}")
        log_event("workflow.job.start", project=project, index=feedback_index, provider=provider, job_id=job_id, command=cmd)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BACKEND_DIR),
        )

        async def read_stream(stream, prefix=""):
            while True:
                if project_job_cancel_requested(project, job_id):
                    process.terminate()
                    append_project_job_log(project, job_id, "[CANCEL] Cancellation requested.")
                    return "cancelled"
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                append_project_job_log(project, job_id, f"{prefix}{decoded}")
            return None

        cancelled = await read_stream(process.stdout)
        if not cancelled:
            cancelled = await read_stream(process.stderr, "[STDERR] ")
        rc = await process.wait()
        duration_ms = round((time.perf_counter() - workflow_started) * 1000, 2)

        if cancelled or project_job_cancel_requested(project, job_id):
            update_project_job(
                project,
                job_id,
                {
                    "status": "cancelled",
                    "finished_at": now_iso(),
                    "error": "Workflow job cancelled.",
                    "result": {"exit_code": rc, "duration_ms": duration_ms},
                },
            )
            return

        if rc == 0:
            save_output_to_prompts(project, provider)
            from src.clip_chat import clip_index_for_feedback_index
            from src.clip_state import promote_latest_prompt_selection
            clip_index = clip_index_for_feedback_index(project, feedback_index)
            promoted_selection = promote_latest_prompt_selection(project, clip_index)
            update_project_job(
                project,
                job_id,
                {
                    "status": "succeeded",
                    "finished_at": now_iso(),
                    "result": {
                        "exit_code": rc,
                        "duration_ms": duration_ms,
                        "clip_index": clip_index,
                        "active_prompt_version_id": (promoted_selection or {}).get("active_prompt_version_id"),
                    },
                },
            )
            append_project_job_log(project, job_id, f"[SUCCESS] Workflow execution finished with exit code {rc}.")
            record_self_evaluation_after_job(project, job_id, clip_index, "workflow_job")
            log_event("workflow.job.finish", project=project, index=feedback_index, provider=provider, job_id=job_id, status="success", exit_code=rc, duration_ms=duration_ms)
        else:
            update_project_job(
                project,
                job_id,
                {
                    "status": "failed",
                    "finished_at": now_iso(),
                    "error": f"Workflow execution failed with exit code {rc}.",
                    "result": {"exit_code": rc, "duration_ms": duration_ms},
                },
            )
            append_project_job_log(project, job_id, f"[ERROR] Workflow execution failed with exit code {rc}.")
    except Exception as exc:
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": str(exc)})
        append_project_job_log(project, job_id, f"[ERROR] {exc}")
        log_event("workflow.job.error", project=project, index=feedback_index, provider=provider, job_id=job_id, error_type=type(exc).__name__, error=str(exc)[:500])


def execute_video_generation_job(project: str, job_id: str) -> None:
    job = get_project_job(project, job_id)
    payload = job.get("payload") or {}
    update_project_job(project, job_id, {"status": "running", "started_at": now_iso()})
    append_project_job_log(project, job_id, "[START] Video generation job started.")
    if project_job_cancel_requested(project, job_id):
        update_project_job(project, job_id, {"status": "cancelled", "finished_at": now_iso(), "error": "Video generation job cancelled before start."})
        return

    try:
        result = generate_clip_video(
            project,
            int(payload.get("clip_index") if payload.get("clip_index") is not None else job.get("clip_index")),
            payload.get("prompt_version_index"),
            resolution=payload.get("resolution", "720p"),
            generate_audio=bool(payload.get("generate_audio", False)),
            aspect_ratio=payload.get("aspect_ratio", "9:16"),
            duration=int(payload.get("duration", 5)),
        )
        update_project_job(project, job_id, {"status": "succeeded", "finished_at": now_iso(), "result": result})
        append_project_job_log(project, job_id, "[SUCCESS] Video generation finished.")
        record_self_evaluation_after_job(project, job_id, int(result.get("clip_index")), "video_generation_job")
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": detail})
        append_project_job_log(project, job_id, f"[ERROR] {detail}")
    except Exception as exc:
        update_project_job(project, job_id, {"status": "failed", "finished_at": now_iso(), "error": str(exc)})
        append_project_job_log(project, job_id, f"[ERROR] {exc}")


def dispatch_agent_run_jobs(project_name: str, agent_run: dict, background_tasks: BackgroundTasks, provider: str = "openai") -> dict:
    dispatched_jobs = []
    tool_results = list(agent_run.get("tool_results") or [])

    for step in agent_run.get("plan_steps", []):
        if step.get("status") != "dispatch_ready":
            continue
        action = step.get("action") or {}
        action_type = action.get("type")
        job = None
        if action_type == "execute_workflow" and action.get("feedback_index") is not None:
            job = create_project_job(
                project_name,
                "workflow",
                feedback_index=int(action.get("feedback_index")),
                provider=action.get("provider") or provider,
                payload=action,
                agent_run_id=agent_run.get("id"),
            )
            background_tasks.add_task(execute_workflow_job, project_name, job["id"])
        elif action_type == "generate_video":
            clip_index = int(agent_run.get("clip_index") or 0)
            payload = {
                "clip_index": clip_index,
                "prompt_version_index": action.get("prompt_version_index"),
                "provider": "segmind",
                "resolution": action.get("resolution", "720p"),
                "generate_audio": bool(action.get("generate_audio", False)),
                "aspect_ratio": action.get("aspect_ratio", "9:16"),
                "duration": int(action.get("duration", 5)),
            }
            job = create_project_job(
                project_name,
                "generate_video",
                clip_index=clip_index,
                provider="segmind",
                payload=payload,
                agent_run_id=agent_run.get("id"),
            )
            background_tasks.add_task(execute_video_generation_job, project_name, job["id"])

        if job:
            step["status"] = "queued"
            step["job_id"] = job["id"]
            dispatched_jobs.append(job)
            append_project_event(
                project_name,
                "agent_job_dispatched",
                actor="chat_agent",
                clip_index=agent_run.get("clip_index"),
                clip_key=agent_run.get("clip_key"),
                entity="job",
                entity_id=job["id"],
                payload={
                    "agent_run_id": agent_run.get("id"),
                    "job_type": job.get("type"),
                    "action_type": action_type,
                    "step_id": step.get("id"),
                    "feedback_index": job.get("feedback_index"),
                },
            )
            tool_results.append({
                "status": "queued",
                "tool": action_type,
                "job_id": job["id"],
                "message": f"Created {job['type']} job.",
            })

    if dispatched_jobs:
        agent_run["status"] = "running"
        agent_run["tool_results"] = tool_results
        agent_run["jobs"] = dispatched_jobs
    return agent_run


def workflow_intent_for_feedback(project_name: str, feedback_index: int) -> dict:
    from src.clip_chat import load_chat_workflow_intents
    intents = load_chat_workflow_intents(project_name)
    intent = (intents.get("feedback") or {}).get(str(feedback_index), {})
    if intent.get("status") not in {"pending", None}:
        return {}
    return intent


def mark_workflow_intent_used(project_name: str, feedback_index: int, updates: dict) -> None:
    from src.clip_chat import load_chat_workflow_intents, save_chat_workflow_intents
    intents = load_chat_workflow_intents(project_name)
    feedback_intents = intents.setdefault("feedback", {})
    key = str(feedback_index)
    current = feedback_intents.get(key, {})
    current.update(updates)
    current["updated_at"] = now_iso()
    feedback_intents[key] = current
    save_chat_workflow_intents(project_name, intents)


def feedback_context_for_raw_index(project_name: str, feedback_index: int) -> Optional[dict]:
    from src.project_manager import get_project_data

    project_data = get_project_data(project_name)
    for group in project_data.get("feedback", []):
        for item in group.get("feedback_items", []):
            if item.get("raw_index") == feedback_index:
                clip_index = group.get("clip_occurrence")
                if clip_index is None:
                    clip_name = group.get("clip_used")
                    for index, clip in enumerate(project_data.get("timeline", [])):
                        if clip.get("clip") == clip_name:
                            clip_index = index
                            break
                return {
                    "group": group,
                    "item": item,
                    "clip_index": clip_index,
                    "timeline": project_data.get("timeline", []),
                }
    return None


def resolve_clip_file_for_continuity(project_name: str, clip_name: str) -> Optional[Path]:
    project_assets = project_assets_dir(project_name)
    for rel_dir in ("06_clips/_final", "06_clips/_raw"):
        candidate = project_assets / rel_dir / clip_name
        if candidate.exists():
            return candidate
    return None


def prepare_continuity_reference_from_intent(project_name: str, feedback_index: int, intent: dict) -> tuple[Optional[str], Optional[str]]:
    if intent.get("continuity_reference") != "previous_clip_last_frame":
        return None, None

    feedback_context = feedback_context_for_raw_index(project_name, feedback_index)
    if not feedback_context:
        return None, "No feedback context found for continuity reference."

    clip_index = feedback_context.get("clip_index")
    timeline = feedback_context.get("timeline") or []
    if not isinstance(clip_index, int) or clip_index <= 0 or clip_index >= len(timeline):
        return None, "This clip has no previous timeline clip to use for continuity."

    previous_clip = timeline[clip_index - 1]
    previous_clip_name = previous_clip.get("clip")
    if not previous_clip_name:
        return None, "Previous clip name is missing from the timeline."

    previous_clip_path = resolve_clip_file_for_continuity(project_name, previous_clip_name)
    if not previous_clip_path:
        return None, f"Previous clip file not found for {previous_clip_name}."

    output_dir = project_data_dir(project_name) / "chat_continuity_frames" / f"feedback_{feedback_index}"
    from src.workflows.prompt_generation import extract_last_frame, get_video_duration

    duration_s = get_video_duration(str(previous_clip_path)) or float(previous_clip.get("duration_s") or 0.0) or 5.0
    frame_path = extract_last_frame(str(previous_clip_path), str(output_dir), duration_s)
    if not frame_path:
        return None, f"Could not extract the last frame from {previous_clip_name}."

    note = (
        f"Use the extracted last frame from previous clip '{previous_clip_name}' as the continuity "
        f"reference and first-frame anchor for the workflow."
    )
    mark_workflow_intent_used(
        project_name,
        feedback_index,
        {
            "status": "prepared",
            "continuity_frame_path": frame_path,
            "continuity_note": note,
            "previous_clip": previous_clip_name,
        },
    )
    return frame_path, note


def _prompt_version_signature(version: dict) -> tuple:
    return (
        str(version.get("video_model_prompt") or "").strip(),
        str(version.get("initial_frame_prompt") or "").strip(),
        str(version.get("initial_frame_image_path") or ""),
        json.dumps(version.get("selected_assets") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(version.get("referenced_frames") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(version.get("referenced_frame_paths") or [], sort_keys=True, ensure_ascii=False),
    )


def _top_level_prompt_version(prompt: dict) -> Optional[dict]:
    if not prompt.get("video_model_prompt"):
        return None
    entry = {
        key: value
        for key, value in prompt.items()
        if key not in {"history"}
    }
    if not entry.get("timestamp"):
        entry["timestamp"] = prompt.get("updated_at") or prompt.get("created_at")
    return entry


def prompt_versions_for_record(prompt: dict) -> list[dict]:
    history = prompt.get("history")
    versions = []
    if isinstance(history, list) and history:
        versions = [version for version in history if isinstance(version, dict)]
    top_level_version = _top_level_prompt_version(prompt)
    if top_level_version:
        if not versions or _prompt_version_signature(versions[-1]) != _prompt_version_signature(top_level_version):
            versions.append(top_level_version)
        else:
            versions[-1] = {**versions[-1], **top_level_version}
    return versions


def _safe_video_stem(clip_name: str) -> str:
    stem = Path(clip_name or "clip").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "clip"


def _next_generated_video_path(project_name: str, clip_name: str, version_number: int) -> Path:
    output_dir = project_assets_dir(project_name) / "generated_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{_safe_video_stem(clip_name)}_v{version_number:03d}.mp4"
    candidate = output_dir / base
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{_safe_video_stem(clip_name)}_v{version_number:03d}_{suffix}.mp4"
        suffix += 1
    return candidate


def _video_asset_url(path: Path) -> str:
    from urllib.parse import quote
    rel = path.resolve().relative_to(storage_paths.ASSETS_DIR)
    return f"/assets/{quote(rel.as_posix())}"


def append_video_generation_attempt(
    prompt_record: dict,
    selected_version: dict,
    attempt: dict,
    *,
    is_latest_version: bool,
) -> None:
    selected_version.setdefault("video_generation_attempts", []).append(attempt)
    if selected_version is not prompt_record and is_latest_version:
        prompt_record["video_generation_attempts"] = selected_version["video_generation_attempts"]
    elif selected_version is prompt_record:
        prompt_record["video_generation_attempts"] = selected_version["video_generation_attempts"]
    prompt_record["latest_video_generation_attempt"] = attempt


def generate_clip_video(
    project_name: str,
    clip_index: int,
    prompt_version_index: Optional[int] = None,
    *,
    resolution: str = "720p",
    generate_audio: bool = False,
    aspect_ratio: str = "9:16",
    duration: int = 5,
) -> dict:
    project_name = safe_project_name(project_name)
    api_key = os.environ.get("SEGMIND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SEGMIND_API_KEY is not configured.")

    timeline_data = read_json_file(project_data_dir(project_name) / "timeline.json", {})
    timeline = timeline_data.get("video_timeline", [])
    if clip_index < 0 or clip_index >= len(timeline):
        raise HTTPException(status_code=404, detail="Clip index not found")

    clip = timeline[clip_index]
    prompts_path = project_data_dir(project_name) / "video_prompts.json"
    prompts_data = read_json_file(prompts_path, [])
    if not isinstance(prompts_data, list):
        raise HTTPException(status_code=404, detail="Prompt history was not found.")

    from src.clip_chat import matching_prompt
    prompt_record = matching_prompt({"prompts": prompts_data}, clip, clip_index)
    if not prompt_record:
        raise HTTPException(status_code=404, detail="No generated prompt exists for this clip.")

    versions = prompt_versions_for_record(prompt_record)
    if not versions:
        raise HTTPException(status_code=400, detail="No usable prompt version exists for this clip.")

    selected_index = prompt_version_index if prompt_version_index is not None else len(versions) - 1
    if selected_index < 0 or selected_index >= len(versions):
        raise HTTPException(status_code=400, detail="Prompt version index is out of range.")
    selected_version = versions[selected_index]
    if not selected_version.get("video_model_prompt"):
        raise HTTPException(status_code=400, detail="Selected prompt version has no video prompt.")

    from src.project_manager import generated_videos_for_version
    existing_videos = generated_videos_for_version(selected_version)
    next_version_number = len(existing_videos) + 1
    output_path = _next_generated_video_path(project_name, clip.get("clip", "clip.mp4"), next_version_number)
    is_latest_version = selected_index == len(versions) - 1

    cache = SupabaseAssetUrlCache()
    payload = None
    try:
        payload = build_segmind_payload(
            item={**prompt_record, **selected_version, "duration": duration, "generate_audio": generate_audio},
            api_key=api_key,
            cache=cache,
            use_local_initial_frame=True,
            initial_image_url=None,
        )
        payload["resolution"] = resolution
        payload["generate_audio"] = generate_audio
        payload["aspect_ratio"] = aspect_ratio
        payload["duration"] = max(4, min(15, int(duration or 5)))
        result = create_seedance_task(api_key=api_key, payload=payload)
        video_bytes = (result.get("content") or {}).get("bytes")
        if not video_bytes:
            raise RuntimeError(f"Segmind response did not include video bytes: {result}")
        save_video_bytes(video_bytes, output_path)
    except HTTPException:
        raise
    except SeedanceGenerationRecoveryError as exc:
        failed_attempt = {
            "timestamp": now_iso(),
            "status": "recovery_failed",
            "provider": "segmind",
            "model": os.environ.get("SEEDANCE_MODEL", "seedance-2.0"),
            "request_id": exc.request_id,
            "clip_used": clip.get("clip"),
            "clip_occurrence": clip_index,
            "prompt_version_index": selected_index,
            "prompt_timestamp": selected_version.get("timestamp"),
            "error": str(exc),
            "error_status_code": exc.status_code,
            "duration": (payload or {}).get("duration"),
            "resolution": (payload or {}).get("resolution"),
            "generate_audio": (payload or {}).get("generate_audio"),
            "ratio": (payload or {}).get("aspect_ratio"),
            "recoverable": bool(exc.request_id),
        }
        append_video_generation_attempt(
            prompt_record,
            selected_version,
            failed_attempt,
            is_latest_version=is_latest_version,
        )
        write_json_file(prompts_path, prompts_data)
        log_event(
            "video_generation.recovery_failed",
            project=project_name,
            clip_index=clip_index,
            prompt_version_index=selected_index,
            request_id=exc.request_id,
            status_code=exc.status_code,
            error=str(exc)[:500],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Segmind charged/submitted the job, but the app could not retrieve the result.",
                "request_id": exc.request_id,
                "status_code": exc.status_code,
                "recoverable": bool(exc.request_id),
                "next_step": "Do not regenerate immediately. Use this request id with Segmind support or retry recovery when available.",
            },
        ) from exc
    except Exception as exc:
        log_event(
            "video_generation.error",
            project=project_name,
            clip_index=clip_index,
            prompt_version_index=selected_index,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise HTTPException(status_code=500, detail=f"Video generation failed: {exc}") from exc

    video_entry = {
        "version": next_version_number,
        "timestamp": now_iso(),
        "provider": "segmind",
        "model": os.environ.get("SEEDANCE_MODEL", "seedance-2.0"),
        "request_id": result.get("id"),
        "clip_used": clip.get("clip"),
        "clip_occurrence": clip_index,
        "prompt_version_index": selected_index,
        "prompt_timestamp": selected_version.get("timestamp"),
        "path": str(output_path.resolve()),
        "url": _video_asset_url(output_path),
        "label": f"Generated video v{next_version_number}",
        "source_output_url": (result.get("content") or {}).get("video_url"),
        "duration": payload.get("duration"),
        "resolution": payload.get("resolution"),
        "generate_audio": payload.get("generate_audio"),
        "ratio": payload.get("aspect_ratio"),
        "storage_root": str(storage_paths.ASSETS_DIR.resolve()),
    }

    successful_attempt = {
        "timestamp": video_entry["timestamp"],
        "status": "succeeded",
        "provider": video_entry["provider"],
        "model": video_entry["model"],
        "request_id": video_entry["request_id"],
        "clip_used": video_entry["clip_used"],
        "clip_occurrence": video_entry["clip_occurrence"],
        "prompt_version_index": selected_index,
        "prompt_timestamp": video_entry["prompt_timestamp"],
        "path": video_entry["path"],
        "url": video_entry["url"],
        "duration": video_entry["duration"],
        "resolution": video_entry["resolution"],
        "generate_audio": video_entry["generate_audio"],
        "ratio": video_entry["ratio"],
    }
    append_video_generation_attempt(
        prompt_record,
        selected_version,
        successful_attempt,
        is_latest_version=is_latest_version,
    )
    selected_version.setdefault("generated_videos", []).append(video_entry)
    if selected_version is not prompt_record and is_latest_version:
        prompt_record["generated_videos"] = selected_version["generated_videos"]
    elif selected_version is prompt_record:
        prompt_record["generated_videos"] = selected_version["generated_videos"]
    prompt_record["latest_generated_video"] = video_entry

    write_json_file(prompts_path, prompts_data)
    log_event(
        "video_generation.success",
        project=project_name,
        clip_index=clip_index,
        prompt_version_index=selected_index,
        generated_video=video_entry["path"],
    )
    return {
        "project_name": project_name,
        "clip_index": clip_index,
        "prompt_version_index": selected_index,
        "video": video_entry,
        "generated_videos": selected_version["generated_videos"],
    }
