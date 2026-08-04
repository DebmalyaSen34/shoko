import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.logging_utils import log_event
from src.routers.chat import router as chat_router
from src.routers.clips import router as clips_router
from src.routers.feedback import router as feedback_router
from src.routers.jobs import router as jobs_router
from src.routers.projects import router as projects_router
from src.routers.system import router as system_router
from src.storage_paths import ASSETS_DIR, DATA_DIR
from src.clip_chat import (
    build_clip_chat_context,
    compact_chat_context,
    create_chat_agent_run,
    dynamic_chat_suggestions,
    ensure_clip_context_for_chat,
    evaluate_agent_state_for_clip,
    execute_safe_agent_run_tools,
    fallback_chat_reply,
    generate_chat_reply_with_tools,
    infer_action_suggestions,
    normalize_chat_reply_markdown,
    plan_chat_agent_run,
    recent_agent_runs_for_clip,
    wants_clip_summary_or_analysis,
    wants_regenerated_clip_summary,
)
from src.clip_state import (
    build_clip_state,
    prompt_eval_case_store,
    prompt_learning_store,
    update_clip_selection,
)
from src.job_manager import (
    append_project_job_log,
    append_prompt_version,
    create_project_job,
    execute_video_generation_job,
    get_project_job,
    list_recent_project_jobs,
    prepare_continuity_reference_from_intent,
    prompt_versions_for_record,
    update_project_job,
)
from src.project_manager import (
    append_project_event,
    build_clip_media_gallery,
    ensure_raw_feedback,
    get_assets_list,
    get_project_data,
    load_project_events,
)
from src.reference_frames import extract_reference_frame, parse_explicit_reference_frame_request
from src.storage_paths import read_json_file, stable_state_id

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - compatibility symbol for tests/patching.
    OpenAI = None

app = FastAPI(title="Shoko: A Video Feedback Engine")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(projects_router)
app.include_router(clips_router)
app.include_router(feedback_router)
app.include_router(jobs_router)
app.include_router(chat_router)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_event(
            "http.error",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise

    log_event(
        "http.request",
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        user_agent=request.headers.get("user-agent", "")[:160],
    )
    return response


app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LOKA_BACKEND_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="info")
