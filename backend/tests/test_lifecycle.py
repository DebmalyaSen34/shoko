import importlib
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("LOKA_STORAGE_DIR", tempfile.mkdtemp(prefix="loka-lifecycle-test-"))
server = importlib.import_module("server")


def test_health_endpoint_reports_backend_status():
    client = TestClient(server.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "loka15-backend"
    assert body["version"]
    assert body["started_at"]


def test_version_endpoint_reports_backend_version():
    client = TestClient(server.app)

    response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "loka15-backend"
    assert body["version"]
    assert body["python"]


def test_shutdown_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("LOKA_BACKEND_SHUTDOWN_TOKEN", "correct-token")
    client = TestClient(server.app)

    response = client.post("/shutdown", headers={"x-loka-shutdown-token": "wrong-token"})

    assert response.status_code == 403


def test_shutdown_rejects_when_no_token_is_configured(monkeypatch):
    monkeypatch.delenv("LOKA_BACKEND_SHUTDOWN_TOKEN", raising=False)
    client = TestClient(server.app)

    response = client.post("/shutdown")

    assert response.status_code == 403
