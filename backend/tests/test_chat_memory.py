import importlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

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


def test_chat_infers_autonomous_previous_last_frame_workflow():
    context = {
        "feedback": {
            "feedback_items": [
                {"raw_index": 4, "category": "video", "remark": "Continue the shot."}
            ]
        },
        "latest_version": {},
    }

    actions = server.infer_action_suggestions(
        "use the last frame from the previous clip as a continuity reference and execute the workflow",
        context,
    )

    assert actions == [
        {
            "type": "execute_workflow",
            "label": "Run With Previous Last Frame",
            "feedback_index": 4,
            "continuity_reference": "previous_clip_last_frame",
            "autonomous": True,
        }
    ]


def test_dynamic_chat_suggestions_include_contextual_next_steps():
    context = {
        "feedback": {
            "feedback_items": [
                {"raw_index": 2, "category": "video", "remark": "Make the action clearer."}
            ]
        },
        "latest_version": {
            "video_model_prompt": "Generated prompt",
            "clip_frame_paths": ["/tmp/frame_001.jpg"],
        },
    }

    actions = server.dynamic_chat_suggestions("What should I do next?", context, [])

    assert {"type": "execute_workflow", "label": "Run Workflow Again", "feedback_index": 2} in actions
    assert {"type": "generate_video", "label": "Generate Video"} in actions
    assert {
        "type": "send_message",
        "label": "Show Feedback",
        "prompt": "Show the feedback for this clip.",
    } in actions


def test_dynamic_chat_suggestions_include_regenerate_summary_when_summary_shown():
    context = {
        "feedback": {"feedback_items": []},
        "latest_version": {},
        "clip_context": {"summary": "Cached visual summary."},
    }

    actions = server.dynamic_chat_suggestions("give me summary", context, [])

    assert {
        "type": "send_message",
        "label": "Regenerate Summary",
        "prompt": "Regenerate the clip summary by analyzing the clip frames and audio again.",
    } in actions


def test_agent_run_planner_structures_autonomous_workflow_plan():
    context = {
        "clip_index": 1,
        "clip_key": "current.mp4::1",
        "feedback": {
            "feedback_items": [
                {"raw_index": 4, "category": "video", "remark": "Continue the shot."}
            ]
        },
        "latest_version": {},
        "clip_context": None,
    }
    explicit_actions = server.infer_action_suggestions(
        "use the last frame from the previous clip as a continuity reference and execute the workflow",
        context,
    )
    actions = server.dynamic_chat_suggestions("execute the workflow", context, explicit_actions)

    run = server.plan_chat_agent_run(
        "use the last frame from the previous clip as a continuity reference and execute the workflow",
        context,
        explicit_actions,
        actions,
    )

    assert run["intent"] == "continuity_workflow"
    assert run["autonomy_level"] == "full_autopilot"
    assert run["approval_required"] is False
    assert run["required_actions"][0]["type"] == "execute_workflow"
    assert run["plan_steps"][0]["tool"] == "execute_workflow"


def test_agent_run_planner_marks_risky_suggestions_for_approval():
    context = {
        "clip_index": 0,
        "clip_key": "clip.mp4::0",
        "feedback": {
            "feedback_items": [
                {"raw_index": 2, "category": "video", "remark": "Make the action clearer."}
            ]
        },
        "latest_version": {},
    }
    explicit_actions = server.infer_action_suggestions("can you run the workflow?", context)
    actions = server.dynamic_chat_suggestions("can you run the workflow?", context, explicit_actions)

    run = server.plan_chat_agent_run("can you run the workflow?", context, explicit_actions, actions)

    assert run["intent"] == "workflow_execution"
    assert run["autonomy_level"] == "approval_required"
    assert run["approval_required"] is True
    assert run["status"] == "awaiting_approval"
    assert run["plan_steps"][0]["requires_approval"] is True


def test_agent_run_persistence_round_trips_project_json(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_DIR", data_dir)

    run = server.create_chat_agent_run(
        "project-a",
        {
            "goal": "Run the workflow",
            "clip_index": 0,
            "clip_key": "clip.mp4::0",
            "status": "planned",
            "intent": "workflow_execution",
            "confidence": 0.8,
            "autonomy_level": "suggest",
            "approval_required": False,
            "plan_steps": [],
            "required_actions": [],
            "available_actions": [],
            "suggested_actions": [],
            "tool_results": [],
            "errors": [],
        },
    )

    saved = json.loads((project_dir / "chat_agent_runs.json").read_text(encoding="utf-8"))
    recent = server.recent_agent_runs_for_clip("project-a", "clip.mp4::0")

    assert saved["schema_version"] == 1
    assert saved["runs"][0]["id"] == run["id"]
    assert recent[0]["goal"] == "Run the workflow"


def test_safe_agent_executor_searches_assets_and_completes_step():
    context = {
        "assets": {
            "01_characters": [
                {"name": "vir-red-shirt.png", "path": "01_characters/vir-red-shirt.png", "url": "/assets/project/vir.png", "type": "image"},
                {"name": "blue-prop.png", "path": "02_props/blue-prop.png", "url": "/assets/project/blue.png", "type": "image"},
            ]
        }
    }
    run = {
        "status": "planned",
        "tool_results": [],
        "errors": [],
        "plan_steps": [
            {
                "id": "step-1",
                "label": "Search attached clip assets and project media.",
                "status": "pending",
                "tool": "search_assets",
                "requires_approval": False,
            }
        ],
    }

    updated = server.execute_safe_agent_run_tools("project-a", run, context, "show Vir assets", [])

    assert updated["status"] == "completed"
    assert updated["plan_steps"][0]["status"] == "completed"
    assert updated["tool_results"][0]["tool"] == "search_assets"
    assert updated["tool_results"][0]["matches"][0]["name"] == "vir-red-shirt.png"


def test_safe_agent_executor_blocks_missing_prompt_evaluation():
    run = {
        "status": "planned",
        "tool_results": [],
        "errors": [],
        "plan_steps": [
            {
                "id": "step-1",
                "label": "Evaluate the latest prompt.",
                "status": "pending",
                "tool": "evaluate_prompt",
                "requires_approval": False,
            }
        ],
    }

    updated = server.execute_safe_agent_run_tools("project-a", run, {"latest_version": {}, "feedback": {"feedback_items": []}}, "evaluate prompt", [])

    assert updated["status"] == "completed"
    assert updated["plan_steps"][0]["status"] == "blocked"
    assert updated["tool_results"][0]["status"] == "missing"


def test_safe_agent_executor_records_saved_memory_result():
    run = {
        "status": "planned",
        "tool_results": [],
        "errors": [],
        "plan_steps": [
            {
                "id": "step-1",
                "label": "Persist memory.",
                "status": "pending",
                "tool": "add_memory",
                "requires_approval": False,
            }
        ],
    }

    updated = server.execute_safe_agent_run_tools(
        "project-a",
        run,
        {},
        "remember that the client likes soft light",
        [{"id": "mem-1", "text": "Client likes soft light.", "scope": "clip"}],
    )

    assert updated["status"] == "completed"
    assert updated["tool_results"][0]["memory_ids"] == ["mem-1"]


def test_chat_endpoint_persists_agent_run_and_keeps_suggestions(tmp_path, monkeypatch):
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
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make the action clearer."}
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Can you run the workflow?", "provider": "openai"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_run"]["intent"] == "workflow_execution"
    assert body["agent_run"]["approval_required"] is True
    assert body["assistant_message"]["metadata"]["agent_run"]["id"] == body["agent_run"]["id"]
    assert {"type": "execute_workflow", "label": "Run Workflow", "feedback_index": 0} in body["suggested_actions"]
    assert (project_dir / "chat_agent_runs.json").exists()


def test_chat_endpoint_executes_asset_review_agent_tool(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir-red-shirt.png"
    project_dir.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"image")
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
            ]
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Show Vir assets", "provider": "openai"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_run"]["intent"] == "asset_review"
    assert body["agent_run"]["status"] == "completed"
    assert body["agent_run"]["tool_results"][0]["tool"] == "search_assets"
    assert body["agent_run"]["tool_results"][0]["matches"][0]["name"] == "vir-red-shirt.png"


def test_clip_state_endpoint_returns_stable_prompt_asset_and_video_ids(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir-red-shirt.png"
    project_dir.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"image")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make action clearer."}
                ],
            }
        ]),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "video_model_prompt": "Vir steps into warm light.",
                "selected_assets": [str(asset_path)],
                "quality_report": {"passed": True},
                "history": [
                    {
                        "timestamp": "2026-07-28T00:00:00",
                        "video_model_prompt": "Vir steps into warm light.",
                        "selected_assets": [str(asset_path)],
                        "generated_videos": [
                            {
                                "version": 1,
                                "timestamp": "2026-07-28T00:01:00",
                                "path": str(assets_dir / "project-a" / "generated_videos" / "clip_v001.mp4"),
                                "url": "/assets/project-a/generated_videos/clip_v001.mp4",
                            }
                        ],
                    }
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.get("/api/projects/project-a/clips/0/state")

    assert response.status_code == 200
    state = response.json()
    assert state["clip_key"] == "clip.mp4::0"
    assert state["active_prompt"]["prompt_id"].startswith("prompt_")
    assert state["active_prompt"]["version_id"].startswith("version_")
    assert state["asset_state"]["selected_assets"][0]["asset_id"].startswith("asset_")
    assert state["asset_state"]["selected_assets"][0]["role"] == "selected_reference"
    assert state["video_state"]["generated_videos"][0]["generated_video_id"].startswith("video_")
    assert state["feedback_state"]["feedback_items"][0]["feedback_item_id"].startswith("feedback_item_")


def test_clip_selection_persists_active_prompt_video_and_assets(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    first_asset = assets_dir / "project-a" / "01_characters" / "vir-red-shirt.png"
    pinned_asset = assets_dir / "project-a" / "02_props" / "doorway.png"
    generated_dir = assets_dir / "project-a" / "generated_videos"
    for path in (first_asset, pinned_asset, generated_dir / "clip_v001.mp4", generated_dir / "clip_v002.mp4"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "history": [
                    {
                        "timestamp": "2026-07-28T00:00:00",
                        "video_model_prompt": "First prompt.",
                        "selected_assets": [str(first_asset)],
                        "generated_videos": [
                            {
                                "version": 1,
                                "timestamp": "2026-07-28T00:01:00",
                                "path": str(generated_dir / "clip_v001.mp4"),
                            }
                        ],
                    },
                    {
                        "timestamp": "2026-07-28T00:02:00",
                        "video_model_prompt": "Second prompt.",
                        "selected_assets": [],
                        "generated_videos": [
                            {
                                "version": 2,
                                "timestamp": "2026-07-28T00:03:00",
                                "path": str(generated_dir / "clip_v002.mp4"),
                            }
                        ],
                    },
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    default_state = client.get("/api/projects/project-a/clips/0/state").json()
    assert default_state["active_prompt"]["version_index"] == 1
    first_version_id = default_state["active_prompt"]["versions"][0]["prompt_version_id"]
    first_video_id = default_state["active_prompt"]["versions"][0]["generated_videos"][0]

    first_video_id = server.stable_state_id(
        "video",
        "project-a",
        "clip.mp4::0",
        first_video_id.get("version"),
        first_video_id.get("timestamp"),
        first_video_id.get("path") or first_video_id.get("url"),
    )
    response = client.patch(
        "/api/projects/project-a/clips/0/selection",
        json={
            "active_prompt_version_id": first_version_id,
            "active_generated_video_id": first_video_id,
            "selected_asset_paths": [str(pinned_asset)],
        },
    )

    assert response.status_code == 200
    state = response.json()["clip_state"]
    assert state["selection_state"]["selection_source"] == "persisted"
    assert state["active_prompt"]["version_id"] == first_version_id
    assert state["active_prompt"]["version_index"] == 0
    assert state["video_state"]["active_video_id"] == first_video_id
    assert state["selection_state"]["selected_asset_paths"] == [str(pinned_asset)]
    assert state["asset_state"]["pinned_assets"][0]["selected_path"] == str(pinned_asset)
    assert state["selection_state"]["selected_assets"][0]["role"] == "selected_reference"
    assert state["selection_state"]["resolved_assets"][0]["selected_path"] == str(pinned_asset)

    stored = json.loads((project_dir / "clip_selections.json").read_text(encoding="utf-8"))
    assert stored["clips"]["clip.mp4::0"]["active_prompt_version_id"] == first_version_id
    assert stored["clips"]["clip.mp4::0"]["selected_assets"][0]["path"] == str(pinned_asset)


def test_clip_selection_accepts_structured_asset_references(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"image")
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
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.patch(
        "/api/projects/project-a/clips/0/selection",
        json={
            "selected_assets": [
                {
                    "path": str(asset_path),
                    "role": "character_reference",
                    "reason": "Keep Vir's wardrobe consistent.",
                    "confidence": 0.93,
                    "source": "test",
                }
            ]
        },
    )

    assert response.status_code == 200
    state = response.json()["clip_state"]
    structured = state["selection_state"]["selected_assets"][0]
    assert structured["role"] == "character_reference"
    assert structured["reason"] == "Keep Vir's wardrobe consistent."
    assert structured["confidence"] == 0.93
    assert state["selection_state"]["selected_asset_paths"] == [str(asset_path)]
    assert state["asset_state"]["pinned_assets"][0]["role"] == "character_reference"


def test_clip_selection_marks_missing_saved_ids_stale(tmp_path, monkeypatch):
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
        }),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "video_model_prompt": "Current prompt.",
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.patch(
        "/api/projects/project-a/clips/0/selection",
        json={
            "active_prompt_version_id": "version_missing",
            "active_generated_video_id": "video_missing",
        },
    )

    assert response.status_code == 200
    state = response.json()["clip_state"]
    assert state["active_prompt"]["version"]["video_model_prompt"] == "Current prompt."
    assert "active_prompt_version_missing" in state["selection_state"]["stale_reasons"]
    assert "active_generated_video_missing" in state["freshness"]["stale_reasons"]


def test_chat_agent_mutates_active_prompt_and_asset_selection(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"image")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "history": [
                    {"timestamp": "2026-07-28T00:00:00", "video_model_prompt": "First prompt."},
                    {"timestamp": "2026-07-28T00:01:00", "video_model_prompt": "Second prompt."},
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Set active prompt version 1 and attach asset 'vir.png'"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_run"]["intent"] == "state_mutation"
    assert body["clip_state"]["active_prompt"]["version_index"] == 0
    assert body["clip_state"]["asset_state"]["pinned_assets"][0]["name"] == "vir.png"
    assert body["clip_state"]["selection_state"]["selected_assets"][0]["role"] == "selected_reference"
    assert body["clip_state"]["selection_state"]["selected_assets"][0]["source"] == "chat_agent"
    assert {result["tool"] for result in body["agent_run"]["tool_results"] if result.get("mutates_state")} == {
        "set_active_prompt_version",
        "attach_asset",
    }
    events = server.load_project_events("project-a", limit=20)
    event_types = [event["type"] for event in events]
    assert "agent_run_created" in event_types
    assert "active_prompt_version_set" in event_types
    assert "asset_attached" in event_types
    assert all(event.get("event_hash") for event in events)
    assert events[-1]["previous_event_hash"] == events[-2]["event_hash"]


def test_chat_agent_detaches_asset_and_marks_feedback_resolved(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"image")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make action clearer."}
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.update_clip_selection("project-a", "clip.mp4::0", {"selected_asset_paths": [str(asset_path)]})
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Detach asset 'vir.png' and mark feedback #0 resolved"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["clip_state"]["asset_state"]["pinned_assets"] == []
    assert body["clip_state"]["feedback_state"]["feedback_items"][0]["resolved"] is True
    feedback_data = json.loads((project_dir / "feedback.json").read_text(encoding="utf-8"))
    assert feedback_data[0]["feedback_items"][0]["status"] == "resolved"


def test_chat_agent_add_reference_frame_reuses_existing_tool_result(tmp_path, monkeypatch):
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
                    "end_tc": "00:10",
                    "start_s": 0,
                    "end_s": 10,
                    "duration_s": 10,
                }
            ],
        }),
        encoding="utf-8",
    )

    calls = {"count": 0}

    def fake_extract_reference_frame(project_name, current_clip_index, timestamp, reason="", attach_to="current_feedback"):
        calls["count"] += 1
        return {
            "status": "ok",
            "reference_frame": {
                "timestamp": timestamp,
                "reason": reason,
                "frame_path": str(project_dir / "referenced_frames" / "ref.jpg"),
            },
            "message": f"Extracted frame at {timestamp}.",
        }

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "extract_reference_frame", fake_extract_reference_frame)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Extract and attach reference frame at 00:05"},
    )

    assert response.status_code == 200
    body = response.json()
    assert calls["count"] == 1
    assert body["agent_run"]["plan_steps"][0]["tool"] == "add_reference_frame"
    assert body["agent_run"]["plan_steps"][0]["status"] == "completed"
    assert body["agent_run"]["tool_results"][-1]["tool"] == "add_reference_frame"


def test_clip_state_marks_agent_runs_stale_after_state_changes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    new_asset_path = assets_dir / "project-a" / "02_props" / "doorway.png"
    for path in (asset_path, new_asset_path, assets_dir / "project-a" / "generated_videos" / "clip_v001.mp4"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make action clearer."}
                ],
            }
        ]),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "history": [
                    {
                        "timestamp": "2026-07-28T00:00:00",
                        "video_model_prompt": "First prompt.",
                        "selected_assets": [str(asset_path)],
                        "generated_videos": [],
                    }
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)

    initial_state = server.build_clip_state("project-a", 0)
    run = server.create_chat_agent_run(
        "project-a",
        {
            "goal": "Evaluate current clip.",
            "clip_index": 0,
            "clip_key": "clip.mp4::0",
            "status": "completed",
            "intent": "prompt_evaluation",
            "confidence": 0.8,
            "autonomy_level": "suggest",
            "approval_required": False,
            "plan_steps": [],
            "required_actions": [],
            "available_actions": [],
            "suggested_actions": [],
            "tool_results": [],
            "errors": [],
            "context_summary": {
                "freshness_snapshot": {
                    **initial_state["freshness"],
                    "captured_at": "2026-07-28T00:00:00+00:00",
                }
            },
        },
    )

    server.update_clip_selection(
        "project-a",
        "clip.mp4::0",
        {
            "selected_assets": [
                {
                    "path": str(new_asset_path),
                    "role": "prop_reference",
                    "reason": "Use doorway prop.",
                    "confidence": 0.91,
                    "source": "test",
                }
            ]
        },
    )
    feedback_data = json.loads((project_dir / "feedback.json").read_text(encoding="utf-8"))
    feedback_data[0]["feedback_items"][0]["resolved"] = True
    feedback_data[0]["feedback_items"][0]["status"] = "resolved"
    (project_dir / "feedback.json").write_text(json.dumps(feedback_data), encoding="utf-8")
    prompts_data = json.loads((project_dir / "video_prompts.json").read_text(encoding="utf-8"))
    prompts_data[0]["history"].append(
        {
            "timestamp": "2026-07-28T00:01:00",
            "video_model_prompt": "Second prompt.",
            "selected_assets": [str(new_asset_path)],
            "generated_videos": [
                {
                    "version": 1,
                    "timestamp": "2026-07-28T00:02:00",
                    "path": str(assets_dir / "project-a" / "generated_videos" / "clip_v001.mp4"),
                }
            ],
        }
    )
    (project_dir / "video_prompts.json").write_text(json.dumps(prompts_data), encoding="utf-8")

    updated_state = server.build_clip_state("project-a", 0)
    stale_run = next(candidate for candidate in updated_state["agent_state"]["recent_runs"] if candidate["id"] == run["id"])

    assert stale_run["freshness"]["is_stale"] is True
    assert set(stale_run["freshness"]["stale_reasons"]) >= {
        "feedback_changed",
        "prompts_changed",
        "assets_changed",
        "videos_changed",
        "selection_changed",
    }
    assert stale_run["freshness"]["captured_state_hash"] == initial_state["freshness"]["state_hash"]
    assert stale_run["freshness"]["current_state_hash"] == updated_state["freshness"]["state_hash"]


def test_chat_snapshot_uses_same_canonical_clip_state(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    client = TestClient(server.app)

    state = client.get("/api/projects/project-a/clips/0/state").json()
    chat = client.get("/api/projects/project-a/chat/clip/0").json()

    assert chat["context"]["clip_state"]["clip_state_id"] == state["clip_state_id"]
    assert chat["context"]["clip_state"]["clip_key"] == "clip.mp4::0"
    assert chat["context"]["prompt_ready"] is False


def test_clip_chat_context_includes_compact_learning_summary(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.append_prompt_version(
        "project-a",
        0,
        {
            "video_model_prompt": "Vir keeps the red wardrobe while moving slowly.",
            "quality_report": {
                "passed": False,
                "learning_eval": {
                    "passed": False,
                    "score": 0.62,
                    "failed_cases": ["case-wardrobe"],
                    "case_results": [],
                    "suggestions": ["Preserve wardrobe continuity explicitly."],
                },
            },
        },
        provider="openai",
    )
    state = server.build_clip_state("project-a", 0)
    prompt = state["active_prompt"]
    version = prompt["version"]

    client = TestClient(server.app)
    feedback = client.post(
        "/api/projects/project-a/prompt-feedback",
        json={
            "clip_index": 0,
            "clip_key": state["clip_key"],
            "prompt_id": prompt["prompt_id"],
            "prompt_version_id": version["prompt_version_id"],
            "rating": "negative",
            "categories": ["continuity_error"],
            "severity": 4,
            "comment": "Wardrobe continuity was not explicit.",
            "correction": "Preserve the visible red wardrobe.",
            "remember_note": "Always name wardrobe continuity when users flag continuity.",
            "create_eval_case": True,
        },
    ).json()["item"]
    server.prompt_learning_store("project-a").add_lesson(
        "Preserve wardrobe continuity explicitly when continuity feedback is present.",
        scope="project",
        category="continuity_error",
        source_feedback_ids=[feedback["id"]],
    )

    context = server.build_clip_chat_context("project-a", 0, query="wardrobe continuity")
    learning = context["learning"]

    assert learning["feedback_count"] == 1
    assert learning["open_issue_count"] == 1
    assert learning["relevant_lessons"]
    assert len(learning["eval_cases"]) == 1
    assert learning["latest_learning_report"]["passed"] is False

    compact = json.loads(server.compact_chat_context(context))

    assert compact["learning_state"]["open_issue_count"] == 1
    assert compact["learning_state"]["relevant_lesson_count"] == 1
    assert compact["learning_state"]["eval_case_count"] == 1
    assert compact["learning_state"]["latest_learning_passed"] is False
    assert compact["learning_state"]["latest_learning_failed_cases"] == ["case-wardrobe"]


def test_chat_learning_actions_are_planned_with_approval(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.append_prompt_version("project-a", 0, {"video_model_prompt": "Vir moves through the doorway."}, provider="openai")

    context = server.build_clip_chat_context("project-a", 0, query="save this as feedback: wardrobe continuity is missing")
    actions = server.infer_action_suggestions("save this as feedback: wardrobe continuity is missing", context)
    run = server.plan_chat_agent_run("save this as feedback: wardrobe continuity is missing", context, actions, actions)

    assert actions[0]["type"] == "save_prompt_feedback"
    assert run["approval_required"] is True
    assert run["status"] == "awaiting_approval"
    assert run["plan_steps"][0]["tool"] == "save_prompt_feedback"
    assert run["plan_steps"][0]["status"] == "awaiting_approval"


def test_chat_learning_action_endpoint_saves_feedback_and_lesson(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.append_prompt_version("project-a", 0, {"video_model_prompt": "Vir moves through the doorway."}, provider="openai")
    state = server.build_clip_state("project-a", 0)
    version = state["active_prompt"]["version"]
    client = TestClient(server.app)

    feedback_response = client.post(
        "/api/projects/project-a/chat/clip/action",
        json={
            "clip_index": 0,
            "provider": "openai",
            "message": "The prompt misses wardrobe continuity.",
            "action": {
                "type": "save_prompt_feedback",
                "label": "Save Prompt Feedback",
                "prompt": "The prompt misses wardrobe continuity.",
                "rating": "negative",
                "categories": ["continuity_error"],
                "prompt_version_id": version["prompt_version_id"],
            },
        },
    )

    assert feedback_response.status_code == 200
    feedback_body = feedback_response.json()
    feedback_id = feedback_body["action_result"]["feedback_id"]
    assert feedback_body["action_result"]["status"] == "ok"
    assert feedback_body["clip_state"]["learning_state"]["feedback_count"] == 1

    lesson_response = client.post(
        "/api/projects/project-a/chat/clip/action",
        json={
            "clip_index": 0,
            "provider": "openai",
            "message": "Remember this: preserve wardrobe continuity explicitly.",
            "action": {
                "type": "approve_prompt_lesson",
                "label": "Approve Lesson",
                "lesson": "Preserve wardrobe continuity explicitly.",
                "source_feedback_ids": [feedback_id],
                "category": "continuity_error",
            },
        },
    )

    assert lesson_response.status_code == 200
    lesson_body = lesson_response.json()
    assert lesson_body["action_result"]["status"] == "ok"
    assert lesson_body["action_result"]["lesson"]["source_feedback_ids"] == [feedback_id]
    events = server.load_project_events("project-a", event_type="prompt_lesson_created")
    assert events[0]["actor"] == "chat_agent"


def test_chat_learning_eval_action_uses_validator(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "_client_for_prompt_provider", lambda provider: object())
    monkeypatch.setattr(server, "_default_model_for_provider", lambda provider: "mock-model")
    monkeypatch.setattr(
        server,
        "run_learning_eval",
        lambda **_kwargs: {
            "passed": False,
            "score": 0.5,
            "failed_cases": ["case-1"],
            "case_results": [],
            "suggestions": ["Preserve continuity."],
        },
    )
    server.append_prompt_version("project-a", 0, {"video_model_prompt": "Vir moves through the doorway."}, provider="openai")
    server.prompt_eval_case_store("project-a").add_eval_case(
        name="Continuity",
        source_feedback_id=None,
        clip_index=0,
        clip_key="clip.mp4::0",
        input={"feedback_items": [], "clip_summary": "", "selected_assets": []},
        expected_behavior=["Preserve continuity."],
        failure_categories=["continuity_error"],
    )
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip/action",
        json={
            "clip_index": 0,
            "provider": "openai",
            "message": "Run learning eval.",
            "action": {"type": "run_prompt_learning_eval", "label": "Run Learning Eval"},
        },
    )

    assert response.status_code == 200
    result = response.json()["action_result"]
    assert result["status"] == "ok"
    assert result["learning_eval"]["failed_cases"] == ["case-1"]


def test_chat_system_prompt_mentions_learning_without_permanent_claims():
    source = inspect.getsource(server.generate_chat_reply_with_tools)

    assert "learning_state.relevant_lessons" in source
    assert "Do not claim the model has permanently learned" in source
    assert "the app will remember and apply approved lessons" in source


def test_project_job_store_persists_and_cancels(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_DIR", data_dir)

    job = server.create_project_job(
        "project-a",
        "workflow",
        feedback_index=3,
        provider="openai",
        payload={"feedback_index": 3},
    )
    server.append_project_job_log("project-a", job["id"], "queued from test")
    cancelled = server.update_project_job("project-a", job["id"], {"cancel_requested": True, "status": "cancelling"})

    saved = json.loads((project_dir / "jobs.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert cancelled["status"] == "cancelling"
    assert server.get_project_job("project-a", job["id"])["logs"][0]["message"] == "queued from test"
    events = server.load_project_events("project-a")
    assert [event["type"] for event in events] == ["job_created", "job_status_changed"]
    assert events[1]["previous_event_hash"] == events[0]["event_hash"]


def test_project_events_endpoint_filters_append_only_log(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    first = server.append_project_event(
        "project-a",
        "asset_attached",
        actor="chat_agent",
        clip_key="clip.mp4::0",
        entity="asset",
        entity_id="asset_1",
        payload={"role": "character_reference"},
    )
    second = server.append_project_event(
        "project-a",
        "feedback_resolved",
        actor="chat_agent",
        clip_key="clip.mp4::0",
        entity="feedback_item",
        entity_id="0",
        payload={"feedback_index": 0},
    )
    client = TestClient(server.app)

    response = client.get("/api/projects/project-a/events", params={"event_type": "feedback_resolved"})

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["id"] == second["id"]
    assert second["previous_event_hash"] == first["event_hash"]


def test_video_generation_job_records_success(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_DIR", data_dir)

    job = server.create_project_job(
        "project-a",
        "generate_video",
        clip_index=0,
        provider="segmind",
        payload={"clip_index": 0, "prompt_version_index": 1, "resolution": "720p", "duration": 5},
    )
    monkeypatch.setattr(
        server,
        "generate_clip_video",
        lambda project, clip_index, prompt_version_index=None, **_kwargs: {
            "project_name": project,
            "clip_index": clip_index,
            "prompt_version_index": prompt_version_index,
            "video": {"version": 1, "path": "/tmp/video.mp4", "url": "/assets/video.mp4"},
            "generated_videos": [],
        },
    )

    server.execute_video_generation_job("project-a", job["id"])

    updated = server.get_project_job("project-a", job["id"])
    assert updated["status"] == "succeeded"
    assert updated["result"]["prompt_version_index"] == 1
    assert any(log["message"] == "[SUCCESS] Video generation finished." for log in updated["logs"])


def test_self_evaluation_reports_next_action_for_missing_video(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"image")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make Vir action clearer.", "resolved": True, "status": "resolved"}
                ],
            }
        ]),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "history": [
                    {
                        "timestamp": "2026-07-28T00:00:00",
                        "video_model_prompt": "Vir makes the action clearer.",
                        "selected_assets": [str(asset_path)],
                        "generated_videos": [],
                    }
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.update_clip_selection(
        "project-a",
        "clip.mp4::0",
        {
            "selected_assets": [
                {
                    "path": str(asset_path),
                    "role": "character_reference",
                    "reason": "Keep Vir consistent.",
                    "confidence": 0.9,
                    "source": "test",
                }
            ]
        },
    )

    evaluation = server.evaluate_agent_state_for_clip("project-a", 0)

    assert evaluation["verdict"] == "blocked"
    assert evaluation["checks"]["video_status"]["status"] == "missing"
    assert evaluation["next_action"]["type"] == "generate_video"


def test_video_generation_job_records_self_evaluation(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    asset_path = assets_dir / "project-a" / "01_characters" / "vir.png"
    video_path = assets_dir / "project-a" / "generated_videos" / "clip_v001.mp4"
    for path in (asset_path, video_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")
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
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make Vir action clearer.", "resolved": True, "status": "resolved"}
                ],
            }
        ]),
        encoding="utf-8",
    )
    (project_dir / "video_prompts.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "history": [
                    {
                        "timestamp": "2026-07-28T00:00:00",
                        "video_model_prompt": "Vir makes the action clearer.",
                        "selected_assets": [str(asset_path)],
                        "generated_videos": [],
                    }
                ],
            }
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    server.update_clip_selection(
        "project-a",
        "clip.mp4::0",
        {
            "selected_assets": [
                {
                    "path": str(asset_path),
                    "role": "character_reference",
                    "reason": "Keep Vir consistent.",
                    "confidence": 0.9,
                    "source": "test",
                }
            ]
        },
    )
    job = server.create_project_job(
        "project-a",
        "generate_video",
        clip_index=0,
        provider="segmind",
        payload={"clip_index": 0, "prompt_version_index": 0, "resolution": "720p", "duration": 5},
    )

    def fake_generate_clip_video(project, clip_index, prompt_version_index=None, **_kwargs):
        prompts_data = json.loads((project_dir / "video_prompts.json").read_text(encoding="utf-8"))
        video = {
            "version": 1,
            "timestamp": "2026-07-28T00:02:00",
            "path": str(video_path),
            "url": "/assets/project-a/generated_videos/clip_v001.mp4",
        }
        prompts_data[0]["history"][0]["generated_videos"] = [video]
        (project_dir / "video_prompts.json").write_text(json.dumps(prompts_data), encoding="utf-8")
        return {"project_name": project, "clip_index": clip_index, "prompt_version_index": prompt_version_index, "video": video}

    monkeypatch.setattr(server, "generate_clip_video", fake_generate_clip_video)

    server.execute_video_generation_job("project-a", job["id"])

    updated = server.get_project_job("project-a", job["id"])
    assert updated["status"] == "succeeded"
    assert updated["result"]["self_evaluation"]["tool"] == "self_evaluate"
    assert updated["result"]["self_evaluation"]["checks"]["video_status"]["status"] == "pass"


def test_chat_autonomous_workflow_dispatches_backend_job(tmp_path, monkeypatch):
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
            ]
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "clip.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:00", "category": "video", "remark": "Make action clearer."}
                ],
            }
        ]),
        encoding="utf-8",
    )

    async def fake_execute_workflow_job(project, job_id):
        server.append_project_job_log(project, job_id, "fake workflow dispatch")

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "execute_workflow_job", fake_execute_workflow_job)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={"clip_index": 0, "message": "Execute the workflow", "provider": "openai"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_run"]["status"] == "running"
    assert body["agent_run"]["jobs"][0]["type"] == "workflow"
    assert body["agent_run"]["plan_steps"][0]["status"] == "queued"
    assert not any(action.get("autonomous") for action in body["suggested_actions"] if action["type"] == "execute_workflow")
    jobs = server.list_recent_project_jobs("project-a")
    assert jobs[0]["agent_run_id"] == body["agent_run"]["id"]
    assert jobs[0]["logs"][0]["message"] == "fake workflow dispatch"


def test_prepare_continuity_reference_prefers_previous_final_clip(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "previous.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:01",
                    "start_s": 0,
                    "end_s": 1,
                    "duration_s": 1,
                },
                {
                    "clip": "current.mp4",
                    "start_tc": "00:01",
                    "end_tc": "00:02",
                    "start_s": 1,
                    "end_s": 2,
                    "duration_s": 1,
                },
            ]
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "current.mp4",
                "clip_occurrence": 1,
                "feedback_items": [
                    {"timestamp": "00:01", "category": "video", "remark": "Continue action."}
                ],
            }
        ]),
        encoding="utf-8",
    )
    final_dir = assets_dir / "project-a" / "06_clips" / "_final"
    raw_dir = assets_dir / "project-a" / "06_clips" / "_raw"
    final_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    final_clip = final_dir / "previous.mp4"
    raw_clip = raw_dir / "previous.mp4"
    final_clip.write_bytes(b"final-video")
    raw_clip.write_bytes(b"raw-video")

    seen = {}

    def fake_extract_last_frame(video_path, output_dir, duration_s):
        seen["video_path"] = video_path
        frame_path = Path(output_dir) / "last_frame.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"frame")
        return str(frame_path)

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "get_video_duration", lambda _path: 1.0)
    monkeypatch.setattr(server, "extract_last_frame", fake_extract_last_frame)

    frame_path, note = server.prepare_continuity_reference_from_intent(
        "project-a",
        0,
        {"continuity_reference": "previous_clip_last_frame"},
    )

    assert seen["video_path"] == str(final_clip)
    assert frame_path and frame_path.endswith("last_frame.jpg")
    assert "previous.mp4" in note


def test_build_clip_media_gallery_includes_clip_assets_and_reference_frames(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    selected_asset = assets_dir / "project-a" / "01_characters" / "vir.png"
    clip_frame = data_dir / "project-a" / "video_frames" / "frame_001.jpg"
    ref_frame = data_dir / "project-a" / "referenced_frames" / "ref.jpg"
    for path in (selected_asset, clip_frame, ref_frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)

    media = server.build_clip_media_gallery({
        "clip": {"clip_url": "/assets/project-a/06_clips/_raw/clip.mp4"},
        "latest_version": {
            "selected_assets": [str(selected_asset)],
            "clip_frame_paths": [str(clip_frame)],
            "referenced_frames": [{"frame_path": str(ref_frame), "timestamp": "00:04"}],
        },
        "feedback": {},
    })

    assert [item["source"] for item in media] == [
        "selected_assets",
        "clip_frames",
        "referenced_frames",
    ]
    assert media[0]["url"] == "/assets/project-a/01_characters/vir.png"
    assert media[1]["url"] == "/data/project-a/video_frames/frame_001.jpg"
    assert media[2]["label"] == "Referenced frame 00:04"


def test_build_clip_media_gallery_omits_analyzed_clip_context_media(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    segment = data_dir / "project-a" / "analysis" / "clip_context" / "0_clip" / "segment.mp4"
    audio = data_dir / "project-a" / "analysis" / "clip_context" / "0_clip" / "audio" / "segment_audio.mp3"
    frame = data_dir / "project-a" / "analysis" / "clip_context" / "0_clip" / "frames" / "frame_001.jpg"
    for path in (segment, audio, frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)

    media = server.build_clip_media_gallery({
        "latest_version": {},
        "feedback": {},
        "clip_context": {
            "clip_segment_path": str(segment),
            "audio_segment_path": str(audio),
            "frame_paths": [str(frame)],
        },
    })

    assert media == []


def test_chat_summarizes_saved_clip_context(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    context_dir = project_dir / "analysis" / "clip_context" / "0_clip"
    context_dir.mkdir(parents=True)
    project_dir.mkdir(exist_ok=True)
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:02",
                    "start_s": 0,
                    "end_s": 2,
                    "duration_s": 2,
                }
            ]
        }),
        encoding="utf-8",
    )
    (context_dir / "clip_context.json").write_text(
        json.dumps({
            "status": "video_and_frames",
            "summary": "Mother looks into the camera without emotion.",
            "visible_characters": ["Mother"],
            "gaze": ["toward camera"],
            "actions": ["standing still"],
            "frame_paths": [],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)

    context = server.build_clip_chat_context("project-a", 0, query="summarize this clip")
    reply = server.fallback_chat_reply("summarize this clip", context)

    assert "Mother looks into the camera without emotion." in reply
    assert "Visible characters: Mother" in reply
    assert "Gaze: toward camera" in reply


def test_normalize_chat_reply_markdown_unwraps_prompt_code_fence():
    reply = server.normalize_chat_reply_markdown(
        "Here is the latest prompt:\n\n```text\nVideo Model Prompt:\nMother smiles gently.\n```"
    )

    assert "```" not in reply
    assert "Here is the latest prompt:" in reply
    assert "Video Model Prompt:" in reply
    assert "Mother smiles gently." in reply


def test_chat_summary_request_runs_clip_context_analysis_when_missing(tmp_path, monkeypatch):
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
                    "end_tc": "00:02",
                    "start_s": 0,
                    "end_s": 2,
                    "duration_s": 2,
                }
            ]
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with (
        mock.patch("server.OpenAI", create=True) as openai_cls,
        mock.patch("server.analyze_clip_context") as analyze,
    ):
        analyze.return_value = {
            "status": "frames_only",
            "summary": "A woman stands in a close shot.",
            "visible_characters": ["Woman"],
            "frame_paths": [],
        }
        context = server.build_clip_chat_context("project-a", 0, query="summarize this clip")
        clip_context, error = server.ensure_clip_context_for_chat("project-a", context, "openai")

    assert error is None
    assert clip_context["summary"] == "A woman stands in a close shot."
    assert context["clip_context"] == clip_context
    analyze.assert_called_once()
    assert analyze.call_args.kwargs["output_base_dir"] == str(data_dir)


def test_chat_summary_uses_cached_context_unless_regeneration_is_forced(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    context_dir = project_dir / "analysis" / "clip_context" / "0_clip"
    context_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "clip.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:02",
                    "start_s": 0,
                    "end_s": 2,
                    "duration_s": 2,
                }
            ]
        }),
        encoding="utf-8",
    )
    (context_dir / "clip_context.json").write_text(
        json.dumps({"status": "frames_only", "summary": "Cached summary.", "frame_paths": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with mock.patch("server.analyze_clip_context") as analyze:
        context = server.build_clip_chat_context("project-a", 0, query="summary")
        clip_context, error = server.ensure_clip_context_for_chat("project-a", context, "openai")

    assert error is None
    assert clip_context["summary"] == "Cached summary."
    analyze.assert_not_called()

    with (
        mock.patch("server.OpenAI", create=True),
        mock.patch("server.analyze_clip_context") as analyze,
    ):
        analyze.return_value = {"status": "frames_only", "summary": "Fresh summary.", "frame_paths": []}
        context = server.build_clip_chat_context("project-a", 0, query="regenerate summary")
        clip_context, error = server.ensure_clip_context_for_chat("project-a", context, "openai", force=True)

    assert error is None
    assert clip_context["summary"] == "Fresh summary."
    analyze.assert_called_once()


def test_short_summary_request_triggers_clip_analysis():
    assert server.wants_clip_summary_or_analysis("give me summary")
    assert server.wants_clip_summary_or_analysis("summarize")
    assert server.wants_clip_summary_or_analysis("what do you see?")
    assert not server.wants_clip_summary_or_analysis("show feedback")
    assert server.wants_regenerated_clip_summary("regenerate the clip summary")
    assert server.wants_regenerated_clip_summary("reanalyze context")
    assert not server.wants_regenerated_clip_summary("give me summary")


def test_extract_reference_frame_attaches_to_current_feedback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    raw_dir = assets_dir / "project-a" / "06_clips" / "_raw"
    project_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    (raw_dir / "reaction.mp4").write_bytes(b"video")
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "setup.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:40",
                    "start_s": 0,
                    "end_s": 40,
                    "duration_s": 40,
                },
                {
                    "clip": "reaction.mp4",
                    "start_tc": "00:40",
                    "end_tc": "00:50",
                    "start_s": 40,
                    "end_s": 50,
                    "duration_s": 10,
                },
            ]
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "setup.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:02", "category": "video", "remark": "Use the later reaction."}
                ],
            }
        ]),
        encoding="utf-8",
    )

    def fake_extract(_clip_path, _offset_s, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"frame")
        return True

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "extract_frame_at_offset", fake_extract)

    result = server.extract_reference_frame(
        "project-a",
        current_clip_index=0,
        timestamp="00:44",
        reason="Vir evil-smile reaction",
    )

    assert result["status"] == "ok"
    ref = result["reference_frame"]
    assert ref["clip_used"] == "reaction.mp4"
    assert ref["offset_s"] == 4
    assert Path(ref["frame_path"]).exists()

    feedback = json.loads((project_dir / "feedback.json").read_text(encoding="utf-8"))
    group_refs = feedback[0]["referenced_frames"]
    item_refs = feedback[0]["feedback_items"][0]["referenced_frames"]
    assert len(group_refs) == 1
    assert item_refs == group_refs
    assert group_refs[0]["timestamp"] == "00:44"


def test_parse_explicit_reference_frame_request_requires_clear_attachment_intent():
    parsed = server.parse_explicit_reference_frame_request(
        "Can you extract frame from the clip belonging to 00:44 and add as a reference here"
    )

    assert parsed == {
        "timestamp": "00:44",
        "reason": "Can you extract frame from the clip belonging to 00:44 and add as a reference here",
        "attach_to": "current_feedback",
    }
    assert server.parse_explicit_reference_frame_request("What happens around 00:44?") is None


def test_clip_chat_extracts_reference_frame_and_returns_media(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    assets_dir = tmp_path / "assets"
    project_dir = data_dir / "project-a"
    raw_dir = assets_dir / "project-a" / "06_clips" / "_raw"
    project_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    (raw_dir / "reaction.mp4").write_bytes(b"video")
    (project_dir / "timeline.json").write_text(
        json.dumps({
            "video_timeline": [
                {
                    "clip": "setup.mp4",
                    "start_tc": "00:00",
                    "end_tc": "00:40",
                    "start_s": 0,
                    "end_s": 40,
                    "duration_s": 40,
                },
                {
                    "clip": "reaction.mp4",
                    "start_tc": "00:40",
                    "end_tc": "00:50",
                    "start_s": 40,
                    "end_s": 50,
                    "duration_s": 10,
                },
            ],
            "sequence_name": "Draft",
            "total_duration_tc": "00:50",
            "total_duration_s": 50,
        }),
        encoding="utf-8",
    )
    (project_dir / "feedback.json").write_text(
        json.dumps([
            {
                "clip_used": "setup.mp4",
                "clip_occurrence": 0,
                "feedback_items": [
                    {"timestamp": "00:02", "category": "video", "remark": "Use the later reaction."}
                ],
            }
        ]),
        encoding="utf-8",
    )

    def fake_extract(_clip_path, _offset_s, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"frame")
        return True

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(server, "extract_frame_at_offset", fake_extract)
    client = TestClient(server.app)

    response = client.post(
        "/api/projects/project-a/chat/clip",
        json={
            "clip_index": 0,
            "provider": "openai",
            "message": "Can you extract frame from the clip belonging to 00:44 and add as a reference here",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assistant = body["assistant_message"]
    assert "Extracted frame at 00:44" in assistant["content"]
    media = assistant["metadata"]["media"]
    assert len(media) == 1
    assert media[0]["source"] == "referenced_frames"
    assert media[0]["url"].startswith("/data/project-a/referenced_frames/")
    assert body["suggested_actions"]
