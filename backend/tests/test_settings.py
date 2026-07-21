import os
from pathlib import Path
from unittest import mock

from config.settings import default_app_storage_dir


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
