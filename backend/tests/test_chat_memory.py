import importlib
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
