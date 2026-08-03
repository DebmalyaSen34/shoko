import os
import sys
import json
import time
import asyncio
import subprocess
from typing import Literal, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.storage_paths import (
    project_assets_dir,
    project_data_dir,
    safe_project_name,
)
from src.logging_utils import log_event


from src.job_manager import (
    create_project_job,
    execute_video_generation_job,
    execute_workflow_job,
    generate_clip_video,
    get_project_job,
    list_recent_project_jobs,
    mark_workflow_intent_used,
    prepare_continuity_reference_from_intent,
    save_output_to_prompts,
    update_project_job,
    workflow_intent_for_feedback,
)
from src.project_manager import ensure_raw_feedback
from src.storage_paths import BACKEND_DIR
router = APIRouter(tags=["jobs"])


class GenerateClipVideoRequest(BaseModel):
    clip_index: int
    prompt_version_index: Optional[int] = None
    provider: str = "segmind"
    resolution: Literal["480p", "720p", "1080p", "4k"] = "720p"
    generate_audio: bool = False
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"] = "9:16"
    duration: int = 5


class WorkflowJobRequest(BaseModel):
    feedback_index: int
    provider: Literal["openai", "gemini"] = "openai"
    agent_run_id: Optional[str] = None


@router.post("/api/projects/{project_name}/generate-video")
def post_generate_clip_video(project_name: str, request: GenerateClipVideoRequest):
    return generate_clip_video(
        project_name,
        request.clip_index,
        request.prompt_version_index,
        resolution=request.resolution,
        generate_audio=request.generate_audio,
        aspect_ratio=request.aspect_ratio,
        duration=request.duration,
    )


@router.get("/api/projects/{project_name}/jobs")
def get_project_jobs(project_name: str):
    project_name = safe_project_name(project_name)
    return {"project_name": project_name, "jobs": list_recent_project_jobs(project_name)}


@router.get("/api/projects/{project_name}/jobs/{job_id}")
def get_project_job_endpoint(project_name: str, job_id: str):
    project_name = safe_project_name(project_name)
    return get_project_job(project_name, job_id)


@router.post("/api/projects/{project_name}/jobs/{job_id}/cancel")
def cancel_project_job(project_name: str, job_id: str):
    project_name = safe_project_name(project_name)
    job = get_project_job(project_name, job_id)
    if job.get("status") in {"succeeded", "failed", "cancelled"}:
        return job
    return update_project_job(project_name, job_id, {"cancel_requested": True, "status": "cancelling"})


@router.post("/api/projects/{project_name}/jobs/workflow")
def create_workflow_job(project_name: str, request: WorkflowJobRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    job = create_project_job(
        project_name,
        "workflow",
        feedback_index=request.feedback_index,
        provider=request.provider,
        payload={"feedback_index": request.feedback_index, "provider": request.provider},
        agent_run_id=request.agent_run_id,
    )
    background_tasks.add_task(execute_workflow_job, project_name, job["id"])
    return job


@router.post("/api/projects/{project_name}/jobs/generate-video")
def create_generate_video_job(project_name: str, request: GenerateClipVideoRequest, background_tasks: BackgroundTasks):
    project_name = safe_project_name(project_name)
    payload = request.model_dump()
    job = create_project_job(
        project_name,
        "generate_video",
        clip_index=request.clip_index,
        provider=request.provider,
        payload=payload,
    )
    background_tasks.add_task(execute_video_generation_job, project_name, job["id"])
    return job


@router.get("/api/run-workflow")
async def run_workflow(project: str, index: int, provider: str = "openai"):
    safe_project_name(project)
    log_event("workflow.request", project=project, index=index, provider=provider)
    try:
        raw_feedback_path = ensure_raw_feedback(project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate raw feedback mapping: {str(e)}")

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
        "--provider", provider
    ]
    if continuity_frame_path:
        cmd.extend(["--continuity-frame-path", continuity_frame_path])
    if continuity_note:
        cmd.extend(["--continuity-note", continuity_note])

    async def log_generator():
        yield f"data: [START] Launching workflow subprocess for feedback index {index}...\n\n"
        if continuity_frame_path:
            yield f"data: [CONTEXT] Prepared previous-clip last-frame continuity reference: {continuity_frame_path}\n\n"
        elif continuity_error:
            yield f"data: [CONTEXT] {continuity_error}\n\n"
        yield f"data: Executing command: {' '.join(cmd)}\n\n\n"
        workflow_started = time.perf_counter()
        log_event("workflow.subprocess.start", project=project, index=index, provider=provider, command=cmd)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BACKEND_DIR)
            )

            async def read_stream(stream, prefix=""):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    yield f"data: {prefix}{decoded}\n\n"

            async for out_line in read_stream(process.stdout):
                yield out_line

            async for err_line in read_stream(process.stderr, "[STDERR] "):
                yield err_line

            rc = await process.wait()
            if rc == 0:
                save_output_to_prompts(project, provider)
                log_event(
                    "workflow.subprocess.finish",
                    project=project,
                    index=index,
                    provider=provider,
                    status="success",
                    exit_code=rc,
                    duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                )
                yield f"\ndata: [SUCCESS] Workflow execution finished with exit code {rc}\n\n"
            else:
                log_event(
                    "workflow.subprocess.finish",
                    project=project,
                    index=index,
                    provider=provider,
                    status="failed",
                    exit_code=rc,
                    duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                )
                yield f"\ndata: [ERROR] Workflow execution failed with exit code {rc}\n\n"

        except Exception as e:
            log_event(
                "workflow.subprocess.error",
                project=project,
                index=index,
                provider=provider,
                duration_ms=round((time.perf_counter() - workflow_started) * 1000, 2),
                error_type=type(e).__name__,
                error=str(e)[:500],
            )
            yield f"data: [ERROR] Failed to run subprocess: {str(e)}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")


@router.get("/api/workflow-result")
def get_workflow_result(project: str):
    output_json = project_data_dir(project) / "output.json"
    if not output_json.exists():
        raise HTTPException(status_code=404, detail="No workflow result found. Run a feedback workflow first.")
    with output_json.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse result JSON: {str(e)}")
