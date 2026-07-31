import importlib
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("LOKA_STORAGE_DIR", tempfile.mkdtemp(prefix="loka-prompt-feedback-test-"))
server = importlib.import_module("server")


def _seed_project(tmp_path, monkeypatch):
    data_dir = tmp_path / "storage" / "data"
    assets_dir = tmp_path / "storage" / "assets"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    (assets_dir / "project-a" / "06_clips" / "_raw").mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        """
        {
          "video_timeline": [
            {"clip": "clip001.mp4", "start_tc": "00:00", "end_tc": "00:05", "start_s": 0, "end_s": 5, "duration_s": 5}
          ],
          "summary": {},
          "sequence_name": "Test",
          "total_duration_tc": "00:05",
          "total_duration_s": 5
        }
        """,
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text("[]", encoding="utf-8")
    (project_dir / "video_prompts.json").write_text(
        """
        [
          {
            "clip_used": "clip001.mp4",
            "clip_occurrence": 0,
            "video_model_prompt": "A precise cinematic prompt preserving the clip.",
            "history": [
              {
                "timestamp": "2026-01-01T00:00:00",
                "provider": "openai",
                "video_model_prompt": "A precise cinematic prompt preserving the clip."
              }
            ]
          }
        ]
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.chdir(tmp_path)
    state = server.build_clip_state("project-a", 0)
    version = state["active_prompt"]["version"]
    return data_dir, {
        "clip_index": 0,
        "clip_key": state["clip_key"],
        "prompt_id": version["prompt_id"],
        "prompt_version_id": version["prompt_version_id"],
    }


def _payload(ids, **overrides):
    payload = {
        **ids,
        "rating": "negative",
        "categories": ["continuity_error", "missed_feedback"],
        "severity": 4,
        "comment": "The prompt changed wardrobe continuity.",
        "correction": "Preserve the same wardrobe and slower hand movement.",
        "remember_note": "Keep wardrobe continuity explicit.",
        "create_eval_case": True,
    }
    payload.update(overrides)
    return payload


def test_create_prompt_feedback_file_and_list_by_clip(tmp_path, monkeypatch):
    data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)

    response = client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids))

    assert response.status_code == 200
    assert (data_dir / "project-a" / "prompt_feedback.json").exists()
    item = response.json()["item"]
    assert item["rating"] == "negative"
    assert item["status"] == "open"

    listed = client.get("/api/projects/project-a/prompt-feedback?clip_index=0")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1


def test_prompt_feedback_validation_rejects_bad_payloads(tmp_path, monkeypatch):
    _data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)

    assert client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids, categories=["bad"])).status_code == 400
    assert client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids, severity=6)).status_code == 400
    assert client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids, prompt_version_id="")).status_code == 400
    assert client.post(
        "/api/projects/project-a/prompt-feedback",
        json=_payload(ids, rating="negative", comment="", correction="", remember_note=""),
    ).status_code == 400


def test_patch_prompt_feedback_updates_allowed_fields(tmp_path, monkeypatch):
    _data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    created = client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids)).json()["item"]

    response = client.patch(
        f"/api/projects/project-a/prompt-feedback/{created['id']}",
        json={"status": "resolved", "severity": 2, "comment": "Fixed now."},
    )

    assert response.status_code == 200
    item = response.json()["item"]
    assert item["status"] == "resolved"
    assert item["severity"] == 2
    assert item["comment"] == "Fixed now."


def test_prompt_feedback_summary_status_rules(tmp_path, monkeypatch):
    _data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)

    state = client.get("/api/projects/project-a/clips/0/state").json()
    assert state["active_prompt"]["version"]["feedback_summary"]["status"] == "unreviewed"

    client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids, rating="positive", categories=[], status="approved"))
    state = client.get("/api/projects/project-a/clips/0/state").json()
    assert state["active_prompt"]["version"]["feedback_summary"]["status"] == "approved"

    negative = client.post("/api/projects/project-a/prompt-feedback", json=_payload(ids)).json()["item"]
    state = client.get("/api/projects/project-a/clips/0/state").json()
    summary = state["active_prompt"]["version"]["feedback_summary"]
    assert summary["status"] == "needs_revision"
    assert summary["open_negative_count"] == 1

    client.patch(f"/api/projects/project-a/prompt-feedback/{negative['id']}", json={"status": "rejected"})
    state = client.get("/api/projects/project-a/clips/0/state").json()
    assert state["active_prompt"]["version"]["feedback_summary"]["status"] == "rejected"
