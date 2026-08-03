import os
import sys
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.config import get_runtime_config, update_runtime_secrets
from src.schema import RuntimeSecretsUpdate
from src.storage_paths import APP_VERSION, BACKEND_STARTED_AT

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "loka15-backend",
        "version": APP_VERSION,
        "started_at": BACKEND_STARTED_AT.isoformat(),
    }


@router.get("/version")
def version():
    return {
        "service": "loka15-backend",
        "version": APP_VERSION,
        "python": sys.version.split()[0],
    }


@router.post("/shutdown")
async def shutdown(request: Request, background_tasks: BackgroundTasks):
    expected_token = os.environ.get("LOKA_BACKEND_SHUTDOWN_TOKEN")
    supplied_token = request.headers.get("x-loka-shutdown-token")
    if not expected_token or supplied_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid shutdown token")

    def stop_process():
        time.sleep(0.2)
        os._exit(0)

    background_tasks.add_task(stop_process)
    return {"status": "shutting_down"}


@router.get("/api/config/runtime")
def get_runtime_config_endpoint():
    return get_runtime_config()


@router.post("/api/config/secrets")
def update_runtime_secrets_endpoint(payload: RuntimeSecretsUpdate):
    return update_runtime_secrets(payload)
