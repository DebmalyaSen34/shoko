import importlib
import os
import sys
import tempfile
from pathlib import Path


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
