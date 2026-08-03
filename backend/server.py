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

app = FastAPI(title="Video Project Timeline & Feedback UI")

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
