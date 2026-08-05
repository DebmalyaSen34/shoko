import os
import json
import subprocess
import sys
from pathlib import Path
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


MEDIA_BINARY_ENV_VARS = {
    "ffmpeg": "LOKA_FFMPEG_PATH",
    "ffprobe": "LOKA_FFPROBE_PATH",
}


def resolve_media_binary(binary: str) -> str:
    """Resolve bundled media tools, falling back to PATH for development."""
    normalized = binary.lower().removesuffix(".exe")
    env_var = MEDIA_BINARY_ENV_VARS.get(normalized)
    if not env_var:
        raise ValueError(f"Unsupported media binary: {binary}")

    override = os.environ.get(env_var, "").strip()
    if override:
        return override

    names = [f"{normalized}.exe", normalized] if sys.platform.startswith("win") else [normalized, f"{normalized}.exe"]
    roots = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root))
    roots.append(Path(app_resource_path()))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)

    seen = set()
    for root in roots:
        for name in names:
            for candidate in (root / "bin" / name, root / name):
                candidate_key = str(candidate)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                if candidate.exists():
                    return candidate_key

    return f"{normalized}.exe" if sys.platform.startswith("win") else normalized


def media_binary_status() -> dict[str, dict[str, object]]:
    status = {}
    for binary in MEDIA_BINARY_ENV_VARS:
        resolved = resolve_media_binary(binary)
        details = {
            "path": resolved,
            "exists": os.path.isabs(resolved) and os.path.exists(resolved),
            "available": False,
            "version": "",
            "error": "",
        }
        try:
            completed = subprocess.run(
                [resolved, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=8,
            )
            details["available"] = completed.returncode == 0
            output = (completed.stdout or completed.stderr or "").strip()
            details["version"] = output.splitlines()[0] if output else ""
            if completed.returncode != 0:
                details["error"] = (completed.stderr or completed.stdout or "").strip()[:500]
        except Exception as exc:
            details["error"] = str(exc)
        status[binary] = details
    return status

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
