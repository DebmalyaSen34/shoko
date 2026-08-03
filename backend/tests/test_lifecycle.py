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
from config import settings as settings_module
from src import config as config_module


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


def test_runtime_config_reports_secret_status_without_values(tmp_path, monkeypatch):
    for key in settings_module.SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    monkeypatch.setattr(config_module, "APP_STORAGE_DIR", tmp_path / "app-storage")
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "app-storage" / "data")
    monkeypatch.setattr(config_module, "ASSETS_DIR", tmp_path / "app-storage" / "assets")
    monkeypatch.setattr(config_module, "OS_ENV_KEYS_AT_START", frozenset({"OPENAI_API_KEY"}))
    client = TestClient(server.app)

    response = client.get("/api/config/runtime")

    assert response.status_code == 200
    body = response.json()
    assert "secret-openai-key" not in str(body)
    assert body["secrets"]["keys"]["OPENAI_API_KEY"]["configured"] is True
    assert body["secrets"]["keys"]["OPENAI_API_KEY"]["locked_by_os_env"] is True


def test_update_runtime_secrets_writes_installed_app_env_file(tmp_path, monkeypatch):
    for key in settings_module.SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "APP_STORAGE_DIR", tmp_path / "app-storage")
    monkeypatch.setattr(config_module, "OS_ENV_KEYS_AT_START", frozenset())
    client = TestClient(server.app)

    response = client.post(
        "/api/config/secrets",
        json={
            "openai_api_key": "new-openai-key",
            "gemini_api_key": "",
        },
    )

    assert response.status_code == 200
    body = response.json()
    config_env_path = Path(body["config_env_path"])
    assert config_env_path == tmp_path / "app-storage" / "config" / ".env"
    assert "new-openai-key" not in str(body)
    assert os.environ["OPENAI_API_KEY"] == "new-openai-key"
    assert "OPENAI_API_KEY='new-openai-key'" in config_env_path.read_text(encoding="utf-8")


def test_update_runtime_secrets_rejects_os_controlled_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "APP_STORAGE_DIR", tmp_path / "app-storage")
    monkeypatch.setattr(config_module, "OS_ENV_KEYS_AT_START", frozenset({"OPENAI_API_KEY"}))
    client = TestClient(server.app)

    response = client.post("/api/config/secrets", json={"openai_api_key": "new-openai-key"})

    assert response.status_code == 409
