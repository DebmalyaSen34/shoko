import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("LOKA_STORAGE_DIR", tempfile.mkdtemp(prefix="loka-memory-test-"))

server = importlib.import_module("server")
from src.chat_memory import ChatMemoryStore


def test_memory_store_dedupes_and_reinforces(tmp_path):
    store = ChatMemoryStore(tmp_path / "memory.json")

    first = store.add("Keep Vir's shirt red.", scope="clip", clip_key="clip.mp4::0", source="user")
    second = store.add("keep vir's shirt red.", scope="clip", clip_key="clip.mp4::0", source="assistant")

    data = store.load()
    assert len(data["memories"]) == 1
    assert first["id"] == second["id"]
    assert data["memories"][0]["access_count"] == 1
    assert data["memories"][0]["source"] == "assistant"


def test_memory_store_retrieves_same_clip_and_project_memory(tmp_path):
    store = ChatMemoryStore(tmp_path / "memory.json")
    store.add("Use the red shirt continuity for Vir.", scope="clip", clip_key="clip-a.mp4::0")
    store.add("Client prefers soft cinematic lighting.", scope="project")
    store.add("Use blue props.", scope="clip", clip_key="clip-b.mp4::1")

    results = store.retrieve("red shirt cinematic lighting", clip_key="clip-a.mp4::0", limit=5)

    texts = [candidate.item["text"] for candidate in results]
    assert "Use the red shirt continuity for Vir." in texts
    assert "Client prefers soft cinematic lighting." in texts
    assert "Use blue props." not in texts


def test_memory_store_migrates_previous_shape(tmp_path):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps({
            "project": [{"text": "Project preference", "source": "manual"}],
            "clips": {"clip.mp4::0": [{"text": "Clip preference", "source": "chat"}]},
        }),
        encoding="utf-8",
    )

    data = ChatMemoryStore(memory_path).load()

    assert data["schema_version"] == 2
    assert {item["text"] for item in data["memories"]} == {"Project preference", "Clip preference"}


def test_chat_memory_endpoint_persists_scoped_memory(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:01",
                    "start_s": 0,
                    "end_s": 1,
                    "duration_s": 1,
                }
            ],
            "sequence_name": "Draft",
            "total_duration_tc": "00:01",
            "total_duration_s": 1,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/memory",
        json={"clip_index": 0, "scope": "clip", "text": "Keep the doorway framing."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["item"]["text"] == "Keep the doorway framing."
    assert body["memory"]["clip"][0]["text"] == "Keep the doorway framing."
