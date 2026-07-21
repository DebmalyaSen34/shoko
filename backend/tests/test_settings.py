import os
from pathlib import Path
from unittest import mock

from config.settings import configured_app_storage_dir, default_app_storage_dir, load_runtime_env


def test_default_app_storage_dir_uses_macos_application_support():
    with mock.patch("config.settings.sys.platform", "darwin"), mock.patch.object(Path, "home", return_value=Path("/Users/alex")):
        assert default_app_storage_dir() == "/Users/alex/Library/Application Support/Loka15 Studio"


def test_default_app_storage_dir_uses_windows_local_app_data():
    with (
        mock.patch("config.settings.sys.platform", "win32"),
        mock.patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\Alex\AppData\Local"}, clear=True),
    ):
        assert default_app_storage_dir() == str(Path(r"C:\Users\Alex\AppData\Local") / "Loka15 Studio")


def test_default_app_storage_dir_falls_back_to_windows_app_data():
    with (
        mock.patch("config.settings.sys.platform", "win32"),
        mock.patch.dict(os.environ, {"APPDATA": r"C:\Users\Alex\AppData\Roaming"}, clear=True),
    ):
        assert default_app_storage_dir() == str(Path(r"C:\Users\Alex\AppData\Roaming") / "Loka15 Studio")


def test_default_app_storage_dir_uses_xdg_on_linux():
    with (
        mock.patch("config.settings.sys.platform", "linux"),
        mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/home/alex/.local/state"}, clear=True),
    ):
        assert default_app_storage_dir() == "/home/alex/.local/state/Loka15 Studio"


def test_configured_app_storage_dir_honors_os_override():
    with mock.patch.dict(os.environ, {"LOKA_STORAGE_DIR": "/tmp/loka-custom"}, clear=True):
        assert configured_app_storage_dir() == "/tmp/loka-custom"


def test_load_runtime_env_keeps_os_env_over_env_files(tmp_path, monkeypatch):
    app_env = tmp_path / "app.env"
    dev_env = tmp_path / "dev.env"
    app_env.write_text("OPENAI_API_KEY=from-app\nGEMINI_API_KEY=from-app\n", encoding="utf-8")
    dev_env.write_text("OPENAI_API_KEY=from-dev\nSEGMIND_API_KEY=from-dev\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-os")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SEGMIND_API_KEY", raising=False)

    loaded_paths = load_runtime_env([app_env, dev_env])

    assert loaded_paths == [str(app_env), str(dev_env)]
    assert os.environ["OPENAI_API_KEY"] == "from-os"
    assert os.environ["GEMINI_API_KEY"] == "from-app"
    assert os.environ["SEGMIND_API_KEY"] == "from-dev"
