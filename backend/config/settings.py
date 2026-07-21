import os
import sys
from pathlib import Path

LITE_MODEL = "gemini-2.5-flash"
REASONING_MODEL = "gemini-2.5-flash"
OPENAI_LITE_MODEL = "gpt-5.4-mini"
OPENAI_REASONING_MODEL = "gpt-5.4-mini"
OPENAI_IMAGE_MODEL = "gpt-image-1-mini"

DEFAULT_FEEDBACK_JSON_PATH = "data/feedback/feedback.json"
DEFAULT_TIMELINE_PATH = "data/timeline/output.json"
DEFAULT_ASSETS_DIR = "assets"
DEFAULT_OUTPUT_JSON = "data/output/generated_prompts/generated_prompts_openai_0.json"
DEFAULT_OUTPUT_REPORT = "data/output/result_docs/workflow_report_openai_0.md"

APP_NAME = "Loka15 Studio"


def default_app_storage_dir(app_name: str = APP_NAME) -> str:
    """Return the per-user app data directory for the current OS."""
    home = Path.home()
    if sys.platform == "darwin":
        return str(home / "Library" / "Application Support" / app_name)
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return str(Path(base) / app_name)
        return str(home / "AppData" / "Local" / app_name)

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return str(Path(xdg_data_home) / app_name)
    return str(home / ".local" / "share" / app_name)


LOKA_STORAGE_DIR = default_app_storage_dir()
