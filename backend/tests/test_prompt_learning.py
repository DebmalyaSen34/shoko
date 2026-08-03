import importlib
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("LOKA_STORAGE_DIR", tempfile.mkdtemp(prefix="loka-prompt-learning-test-"))
server = importlib.import_module("server")

from src.prompt_learning import PromptEvalCaseStore, PromptLearningStore


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
            "video_model_prompt": "A prompt that should preserve wardrobe continuity.",
            "history": [
              {
                "timestamp": "2026-01-01T00:00:00",
                "provider": "openai",
                "video_model_prompt": "A prompt that should preserve wardrobe continuity."
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


def _feedback_payload(ids):
    return {
        **ids,
        "rating": "negative",
        "categories": ["continuity_error"],
        "severity": 4,
        "comment": "The prompt changed wardrobe continuity.",
        "correction": "Preserve the same wardrobe and slower hand movement.",
        "remember_note": "Keep wardrobe continuity explicit.",
        "create_eval_case": False,
    }


def test_prompt_learning_store_create_patch_filter_and_retrieve(tmp_path):
    store = PromptLearningStore(tmp_path / "prompt_lessons.json")

    project_lesson = store.add_lesson(
        "Preserve wardrobe continuity when users mention continuity.",
        scope="project",
        category="continuity_error",
        confidence=0.86,
    )
    clip_lesson = store.add_lesson(
        "For this clip, keep hand movement slow and deliberate.",
        scope="clip",
        clip_key="clip001.mp4::0",
        category="missed_feedback",
        source_feedback_ids=["feedback-1"],
    )
    archived = store.add_lesson("Archived rule", scope="project", category="other")
    store.update_lesson(archived["id"], {"archived": True})

    assert (tmp_path / "prompt_lessons.json").exists()
    assert [item["id"] for item in store.list_lessons(category="continuity_error")] == [project_lesson["id"]]
    assert all(not item["archived"] for item in store.list_lessons())
    assert any(item["id"] == archived["id"] for item in store.list_lessons(include_archived=True))

    updated = store.update_lesson(clip_lesson["id"], {"confidence": 0.92, "positive_examples": ["good"], "negative_examples": ["bad"]})
    assert updated["confidence"] == 0.92
    assert updated["positive_examples"] == ["good"]

    retrieved = store.retrieve_lessons("wardrobe continuity and slow hand movement", clip_key="clip001.mp4::0", limit=2)
    assert retrieved[0].item["id"] in {project_lesson["id"], clip_lesson["id"]}
    assert retrieved[0].score >= retrieved[-1].score


def test_prompt_lessons_endpoints_save_and_preserve_source_feedback(tmp_path, monkeypatch):
    data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    feedback = client.post("/api/projects/project-a/prompt-feedback", json=_feedback_payload(ids)).json()["item"]

    response = client.post(
        "/api/projects/project-a/prompt-lessons",
        json={
            "scope": "project",
            "category": "continuity_error",
            "lesson": "Preserve wardrobe continuity explicitly.",
            "source_feedback_ids": [feedback["id"]],
            "confidence": 0.86,
            "positive_examples": [],
            "negative_examples": [],
        },
    )

    assert response.status_code == 200
    lesson = response.json()["lesson"]
    assert lesson["source_feedback_ids"] == [feedback["id"]]
    assert (data_dir / "project-a" / "prompt_lessons.json").exists()

    listed = client.get("/api/projects/project-a/prompt-lessons?category=continuity_error")
    assert listed.status_code == 200
    assert len(listed.json()["lessons"]) == 1

    patched = client.patch(f"/api/projects/project-a/prompt-lessons/{lesson['id']}", json={"archived": True})
    assert patched.status_code == 200
    assert patched.json()["lesson"]["archived"] is True


def test_suggest_prompt_lesson_requires_provider_key(tmp_path, monkeypatch):
    _data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    feedback = client.post("/api/projects/project-a/prompt-feedback", json=_feedback_payload(ids)).json()["item"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post(f"/api/projects/project-a/prompt-feedback/{feedback['id']}/suggest-lesson", json={"provider": "openai"})

    assert response.status_code == 400


def test_suggest_prompt_lesson_returns_mocked_suggestion(tmp_path, monkeypatch):
    _data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    feedback = client.post("/api/projects/project-a/prompt-feedback", json=_feedback_payload(ids)).json()["item"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        server,
        "generate_structured",
        lambda **_kwargs: {
            "lesson": "When continuity is mentioned, preserve wardrobe and body motion explicitly.",
            "category": "continuity_error",
            "confidence": 0.88,
            "reasoning": "The feedback was about wardrobe continuity.",
        },
    )

    response = client.post(f"/api/projects/project-a/prompt-feedback/{feedback['id']}/suggest-lesson", json={"provider": "openai"})

    assert response.status_code == 200
    suggestion = response.json()["suggestion"]
    assert suggestion["category"] == "continuity_error"
    assert "wardrobe" in suggestion["lesson"]


def test_prompt_eval_case_store_create_patch_filter_and_retrieve(tmp_path):
    store = PromptEvalCaseStore(tmp_path / "prompt_eval_cases.json")
    created = store.add_eval_case(
        name="Continuity wardrobe preservation",
        source_feedback_id="feedback-1",
        clip_index=4,
        clip_key="clip001.mp4::4",
        input={
            "feedback_items": [{"remark": "Preserve wardrobe and move hand slowly."}],
            "clip_summary": "Boy in blue shirt raises his hand.",
            "selected_assets": ["character.png"],
        },
        expected_behavior=[
            "Must preserve visible wardrobe.",
            "Must mention slow deliberate hand movement.",
        ],
        failure_categories=["continuity_error"],
    )

    assert (tmp_path / "prompt_eval_cases.json").exists()
    assert store.list_eval_cases(clip_index=4)[0]["id"] == created["id"]
    assert store.list_eval_cases(enabled=True)[0]["enabled"] is True

    patched = store.update_eval_case(created["id"], {"enabled": False, "name": "Archived continuity case"})
    assert patched["enabled"] is False
    assert patched["name"] == "Archived continuity case"
    assert store.list_eval_cases(enabled=True) == []

    store.update_eval_case(created["id"], {"enabled": True})
    retrieved = store.retrieve_eval_cases(
        "wardrobe slow hand movement",
        clip_key="clip001.mp4::4",
        clip_index=4,
    )
    assert retrieved[0].item["id"] == created["id"]


def test_prompt_eval_case_endpoints_auto_create_and_patch(tmp_path, monkeypatch):
    data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    payload = _feedback_payload(ids)
    payload["create_eval_case"] = True

    response = client.post("/api/projects/project-a/prompt-feedback", json=payload)

    assert response.status_code == 200
    assert (data_dir / "project-a" / "prompt_eval_cases.json").exists()
    eval_case = response.json()["eval_case"]
    assert eval_case["source_feedback_id"] == response.json()["item"]["id"]
    assert eval_case["enabled"] is True
    assert "Preserve the same wardrobe" in eval_case["expected_behavior"][0]

    listed = client.get("/api/projects/project-a/prompt-eval-cases?clip_index=0&enabled=true")
    assert listed.status_code == 200
    assert len(listed.json()["cases"]) == 1

    patched = client.patch(
        f"/api/projects/project-a/prompt-eval-cases/{eval_case['id']}",
        json={
            "enabled": False,
            "expected_behavior": ["Must preserve wardrobe continuity."],
            "failure_categories": ["continuity_error", "unsupported_assumption"],
        },
    )
    assert patched.status_code == 200
    assert patched.json()["eval_case"]["enabled"] is False
    assert patched.json()["eval_case"]["failure_categories"] == ["continuity_error", "unsupported_assumption"]


def test_prompt_eval_case_auto_create_skips_positive_feedback(tmp_path, monkeypatch):
    data_dir, ids = _seed_project(tmp_path, monkeypatch)
    client = TestClient(server.app)
    payload = _feedback_payload(ids)
    payload.update({
        "rating": "positive",
        "categories": [],
        "comment": "Works well.",
        "correction": "",
        "remember_note": "",
        "create_eval_case": True,
        "status": "approved",
    })

    response = client.post("/api/projects/project-a/prompt-feedback", json=payload)

    assert response.status_code == 200
    assert "eval_case" not in response.json()
    assert not (data_dir / "project-a" / "prompt_eval_cases.json").exists()
