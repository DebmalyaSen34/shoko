import os
import json
import sys
from pathlib import PureWindowsPath
from config.premiere_pro_conf import TICKS_PER_SEC
from typing import Optional


def app_resource_path(*parts: str) -> str:
    """Resolve a backend resource from source or a PyInstaller bundle."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return os.path.join(bundle_root, *parts)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    return os.path.join(project_root, *parts)

def parse_timestamp_to_seconds(ts_str: Optional[str]) -> Optional[float]:
    """Convert MM:SS, H:MM:SS, or raw seconds to seconds."""
    if not ts_str:
        return None

    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts_str)
    except (TypeError, ValueError):
        return None

def ticks_to_tc(ticks_str, fps=25):
    """Convert Premiere ticks string to HH:MM:SS:FF timecode."""
    if not ticks_str:
        return "00:00:00:00"
    total = int(ticks_str) / TICKS_PER_SEC
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    f = round((total - int(total)) * fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def ticks_to_s(ticks_str):
    """Convert Premiere ticks string to seconds (float)."""
    if not ticks_str:
        return 0.0
    return round(int(ticks_str) / TICKS_PER_SEC, 3)


def win_basename(path):
    """Extract filename from a Windows path string."""
    if not path:
        return ""
    try:
        return PureWindowsPath(path).name
    except Exception:
        return os.path.basename(path.replace("\\", "/"))


def load_prompt_templates() -> dict[str, str]:
    """Load system prompts from config/prompt_templates.json."""
    templates_path = app_resource_path("config", "prompt_templates.json")
    try:
        with open(templates_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load prompt templates from {templates_path}: {e}")
        return {}


def get_prompt_template(key: str, default: str) -> str:
    """Retrieve a specific prompt template by key, falling back to a default value."""
    templates = load_prompt_templates()
    return templates.get(key, default)
