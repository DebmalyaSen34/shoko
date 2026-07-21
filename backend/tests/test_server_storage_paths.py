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

os.environ.setdefault("LOKA_STORAGE_DIR", tempfile.mkdtemp(prefix="loka-server-test-"))
server = importlib.import_module("server")


def test_list_projects_reads_configured_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text("{}", encoding="utf-8")
    (project_dir / "feedback.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.chdir(tmp_path)

    assert server.list_projects() == ["project-a"]


def test_list_projects_includes_timeline_without_feedback(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    project_dir = data_dir / "project-draft"
    project_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        '{"video_timeline": [], "summary": {}, "sequence_name": "Draft", "total_duration_s": 0}',
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.chdir(tmp_path)

    assert server.list_projects() == ["project-draft"]


def test_get_project_data_allows_missing_feedback(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    assets_dir = tmp_path / "app-storage" / "assets"
    project_dir = data_dir / "project-draft"
    project_dir.mkdir(parents=True)
    (project_dir / "timeline.json").write_text(
        '{"video_timeline": [{"clip": "clip.mp4", "start_tc": "00:00", "end_tc": "00:01", "start_s": 0, "end_s": 1, "duration_s": 1}], "summary": {}, "sequence_name": "Draft", "total_duration_tc": "00:01", "total_duration_s": 1}',
        encoding="utf-8",
    )
    (assets_dir / "project-draft" / "06_clips" / "_raw").mkdir(parents=True)

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.chdir(tmp_path)

    data = server.get_project_data("project-draft")

    assert data["project_name"] == "project-draft"
    assert data["feedback"] == []
    assert len(data["timeline"]) == 1


def test_get_assets_list_builds_urls_from_configured_assets_dir(tmp_path, monkeypatch):
    assets_dir = tmp_path / "app-storage" / "assets"
    project_assets_dir = assets_dir / "project-a"
    character_dir = project_assets_dir / "01_characters"
    character_dir.mkdir(parents=True)
    (character_dir / "hero.png").write_bytes(b"image")

    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.chdir(tmp_path)

    assets = server.get_assets_list(project_assets_dir, "project-a")

    assert assets["01_characters"][0]["url"] == "/assets/project-a/01_characters/hero.png"


def test_create_project_creates_required_asset_tree(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    assets_dir = tmp_path / "app-storage" / "assets"
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.chdir(tmp_path)
    client = TestClient(server.app)

    response = client.post("/api/projects", data={"project_name": "project-new"})

    assert response.status_code == 200
    expected_dirs = [
        "00_style",
        "01_characters",
        "02_props",
        "03_locations",
        "04_audio",
        "05_references",
        "06_clips/_final",
        "06_clips/_raw",
    ]
    for rel_path in expected_dirs:
        assert (assets_dir / "project-new" / rel_path).is_dir()
    assert (data_dir / "project-new").is_dir()


def test_import_premiere_package_extracts_timeline(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    assets_dir = tmp_path / "app-storage" / "assets"
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.chdir(tmp_path)
    client = TestClient(server.app)

    with (
        mock.patch("server.setup_project_workspace") as setup,
        mock.patch("server.extract_timeline_from_project") as extract,
    ):
        setup.return_value = (
            str(assets_dir / "project-new"),
            str(assets_dir / "project-new" / "edit.prproj"),
        )
        extract.return_value = str(data_dir / "project-new" / "timeline.json")

        response = client.post(
            "/api/projects/project-new/premiere-package",
            files={"package": ("project.zip", b"zip-bytes", "application/zip")},
        )

    assert response.status_code == 200
    assert response.json()["project_name"] == "project-new"
    setup.assert_called_once()
    extract.assert_called_once_with(
        prproj_path=str(assets_dir / "project-new" / "edit.prproj"),
        project_name="project-new",
        output_base_dir=str(data_dir),
    )


def test_upload_feedback_parses_and_aligns(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    assets_dir = tmp_path / "app-storage" / "assets"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    
    # 1. Create dummy timeline file
    timeline_path = project_dir / "timeline.json"
    timeline_path.write_text("{}", encoding="utf-8")
    
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "ASSETS_DIR", assets_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    client = TestClient(server.app)

    with (
        mock.patch("server.parse_and_align_feedback") as parse_mock,
        mock.patch("openai.OpenAI") as openai_mock,
    ):
        parse_mock.return_value = str(project_dir / "feedback.json")
        
        response = client.post(
            "/api/projects/project-a/feedback",
            files={"file": ("feedback.txt", b"some-feedback-text", "text/plain")},
        )
        
    assert response.status_code == 200
    assert response.json()["project_name"] == "project-a"
    parse_mock.assert_called_once()


def test_ensure_raw_feedback_preserves_group_context(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "feedback.json").write_text(
        json.dumps(
            [
                {
                    "clip_used": "output (23).mp4",
                    "clip_occurrence": 7,
                    "feedback_items": [
                        {
                            "timestamp": "00:45",
                            "category": "video",
                            "remark": "Show Vir grasping his mother's hand.",
                        },
                        {
                            "timestamp": "00:46",
                            "category": "video",
                            "remark": "Have mother look down.",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.chdir(tmp_path)

    raw_path = Path(server.ensure_raw_feedback("project-a"))

    raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_items[0]["group_id"] == "output (23).mp4::7"
    assert raw_items[0]["clip_used"] == "output (23).mp4"
    assert raw_items[0]["clip_occurrence"] == 7
    assert raw_items[0]["sibling_raw_indexes"] == [1]
    assert raw_items[1]["sibling_raw_indexes"] == [0]


def test_add_manual_feedback(tmp_path, monkeypatch):
    data_dir = tmp_path / "app-storage" / "data"
    project_dir = data_dir / "project-a"
    project_dir.mkdir(parents=True)
    
    # 1. Start with no feedback file
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.chdir(tmp_path)
    client = TestClient(server.app)

    # 2. Add manual feedback item
    payload = {
        "clip_used": "clip1.mp4",
        "category": "video",
        "remark": "This is a manually added feedback remark.",
        "timestamp": "00:12"
    }
    response = client.post(
        "/api/projects/project-a/feedback/item",
        json=payload
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # 3. Read back feedback.json and verify contents
    feedback_path = project_dir / "feedback.json"
    assert feedback_path.exists()
    
    import json
    with feedback_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data) == 1
    assert data[0]["clip_used"] == "clip1.mp4"
    assert len(data[0]["feedback_items"]) == 1
    assert data[0]["feedback_items"][0]["remark"] == "This is a manually added feedback remark."
    assert data[0]["feedback_items"][0]["category"] == "video"
    assert data[0]["feedback_items"][0]["timestamp"] == "00:12"

