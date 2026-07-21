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
    assert {"type": "prepare_video", "label": "Generate Video"} in actions
    assert {
        "type": "send_message",
        "label": "Show Feedback",
        "prompt": "Show the feedback for this clip.",
    } in actions


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
