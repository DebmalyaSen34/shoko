import os
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

from dotenv import load_dotenv

#TODO: remove gemini models and keep only OPENAI
LITE_MODEL = "gemini-2.5-flash"
REASONING_MODEL = "gemini-2.5-flash"

OPENAI_LITE_MODEL = "gpt-5.6-luna"
OPENAI_REASONING_MODEL = "gpt-5.6-luna"

OPENAI_IMAGE_MODEL = "gpt-image-1-mini"

DEFAULT_FEEDBACK_JSON_PATH = "data/feedback/feedback.json"
DEFAULT_TIMELINE_PATH = "data/timeline/output.json"
DEFAULT_ASSETS_DIR = "assets"
DEFAULT_OUTPUT_JSON = "data/output/generated_prompts/generated_prompts_openai_0.json"
DEFAULT_OUTPUT_REPORT = "data/output/result_docs/workflow_report_openai_0.md"

APP_NAME = "Shoko"
SECRET_ENV_KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "SEGMIND_API_KEY")
RUNTIME_ENV_KEYS = (
    "LOKA_STORAGE_DIR",
    "LOKA_APP_VERSION",
    "LOKA_BACKEND_PORT",
    "LOKA_BACKEND_SHUTDOWN_TOKEN",
    "LOKA_ASSET_CACHE_BACKEND",
    "LOKA_ENABLE_REFERENCE_AUDIO_UPLOAD",
)
OS_ENV_KEYS_AT_START = frozenset(
    key for key in (*SECRET_ENV_KEYS, *RUNTIME_ENV_KEYS, "LOKA_ENV_FILE") if key in os.environ
)


def default_app_storage_dir(app_name: str = APP_NAME) -> str:
    """Return the per-user app data directory for the current OS."""
    if sys.platform == "darwin":
        home = PurePosixPath(str(Path.home()).replace("\\", "/"))
        return str(home / "Library" / "Application Support" / app_name)
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return str(Path(base) / app_name)
        home = Path.home()
        return str(home / "AppData" / "Local" / app_name)

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return str(PurePosixPath(xdg_data_home.replace("\\", "/")) / app_name)
    home = PurePosixPath(str(Path.home()).replace("\\", "/"))
    return str(home / ".local" / "share" / app_name)


def configured_app_storage_dir(app_name: str = APP_NAME) -> str:
    """Return the app storage root, honoring only the real OS env override."""
    configured_dir = os.environ.get("LOKA_STORAGE_DIR")
    if configured_dir:
        return os.path.expanduser(configured_dir)
    return default_app_storage_dir(app_name)


def backend_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def app_config_dir(storage_dir: str | Path | None = None) -> Path:
    root = Path(storage_dir or configured_app_storage_dir()).expanduser()
    return root / "config"


def app_config_env_path(storage_dir: str | Path | None = None) -> Path:
    return app_config_dir(storage_dir) / ".env"


def local_dev_env_path() -> Path:
    return backend_root_dir() / ".env"


def env_load_paths() -> list[Path]:
    paths = [app_config_env_path(), local_dev_env_path()]
    extra_env_path = os.environ.get("LOKA_ENV_FILE")
    if extra_env_path:
        paths.insert(0, Path(extra_env_path).expanduser())
    return paths


def load_runtime_env(paths: Iterable[Path] | None = None) -> list[str]:
    """
    Load runtime env files without overriding already exported OS variables.

    Precedence:
    1. OS environment variables
    2. Explicit LOKA_ENV_FILE, when provided
    3. Installed app config: {app_storage}/config/.env
    4. Local development: backend/.env
    5. Code defaults
    """
    loaded_paths: list[str] = []
    for env_path in paths or env_load_paths():
        if env_path.exists():
            load_dotenv(env_path, override=False)
            loaded_paths.append(str(env_path))
    return loaded_paths


LOADED_ENV_FILES = load_runtime_env()
if "LOKA_STORAGE_DIR" not in OS_ENV_KEYS_AT_START:
    os.environ.pop("LOKA_STORAGE_DIR", None)
LOKA_STORAGE_DIR = configured_app_storage_dir()
