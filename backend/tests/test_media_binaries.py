import os
import sys

from src import utils
from src.generator.media import _extract_video_frames


def test_resolve_media_binary_env_override_wins(monkeypatch):
    monkeypatch.setenv("LOKA_FFMPEG_PATH", r"C:\tools\ffmpeg.exe")

    assert utils.resolve_media_binary("ffmpeg") == r"C:\tools\ffmpeg.exe"


def test_resolve_media_binary_uses_pyinstaller_bin_exe(monkeypatch, tmp_path):
    bundled_bin = tmp_path / "bin"
    bundled_bin.mkdir()
    ffmpeg = bundled_bin / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    monkeypatch.delenv("LOKA_FFMPEG_PATH", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(utils.sys, "platform", "win32")

    assert utils.resolve_media_binary("ffmpeg") == str(ffmpeg)


def test_resolve_media_binary_falls_back_to_path_name(monkeypatch):
    monkeypatch.delenv("LOKA_FFMPEG_PATH", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(utils.sys, "platform", "linux")
    monkeypatch.setattr(utils.sys, "frozen", False, raising=False)
    monkeypatch.setattr(utils, "app_resource_path", lambda *parts: os.path.join("/missing", *parts))

    assert utils.resolve_media_binary("ffmpeg") == "ffmpeg"


def test_resolve_media_binary_ffprobe_uses_matching_env_override(monkeypatch):
    monkeypatch.setenv("LOKA_FFPROBE_PATH", "/opt/media/ffprobe")

    assert utils.resolve_media_binary("ffprobe") == "/opt/media/ffprobe"


def test_extract_video_frames_uses_resolved_media_binary(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("LOKA_FFMPEG_PATH", "/opt/media/ffmpeg")

    def fake_run(command, **_kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("src.generator.media.subprocess.run", fake_run)

    _extract_video_frames("clip.mp4", str(tmp_path), duration_s=1.0)

    assert calls
    assert all(command[0] == "/opt/media/ffmpeg" for command in calls)
