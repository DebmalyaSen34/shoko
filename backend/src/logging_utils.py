from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


MAX_TEXT_CHARS = int(os.environ.get("BACKEND_LOG_TEXT_LIMIT", "320"))
MAX_LIST_ITEMS = int(os.environ.get("BACKEND_LOG_LIST_LIMIT", "8"))


def _log_path() -> Path:
    return Path(os.environ.get("BACKEND_LOG_FILE", "data/logs/backend.jsonl"))


def get_backend_logger() -> logging.Logger:
    logger = logging.getLogger("loka.backend")
    if logger.handlers:
        return logger

    logger.setLevel(os.environ.get("BACKEND_LOG_LEVEL", "INFO").upper())
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=int(os.environ.get("BACKEND_LOG_MAX_BYTES", str(5 * 1024 * 1024))),
        backupCount=int(os.environ.get("BACKEND_LOG_BACKUPS", "5")),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(event: str, **fields: Any) -> None:
    if os.environ.get("BACKEND_LOG_ENABLED", "1").lower() in {"0", "false", "no"}:
        return
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("BACKEND_LOG_DURING_TESTS"):
        return

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    try:
        get_backend_logger().info(json.dumps(entry, ensure_ascii=False, default=str))
    except Exception as exc:
        print(f"Warning: Failed to write backend log event '{event}': {exc}")


def summarize_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if value.startswith("data:"):
            return {
                "kind": "data_url",
                "chars": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
            }
        if len(value) > MAX_TEXT_CHARS:
            return {
                "kind": "text",
                "chars": len(value),
                "preview": value[:MAX_TEXT_CHARS],
            }
        return value

    if depth >= 3:
        return {"kind": type(value).__name__}

    if isinstance(value, list):
        return {
            "kind": "list",
            "count": len(value),
            "items": [summarize_value(item, depth=depth + 1) for item in value[:MAX_LIST_ITEMS]],
        }

    if isinstance(value, dict):
        return {
            key: summarize_value(val, depth=depth + 1)
            for key, val in list(value.items())[:MAX_LIST_ITEMS]
        }

    return str(value)


def summarize_contents(contents: list[Any]) -> dict[str, Any]:
    text_chars = 0
    image_refs = 0
    file_refs = 0
    other_blocks = 0
    previews: list[Any] = []

    for item in contents:
        if isinstance(item, str):
            text_chars += len(item)
            previews.append(summarize_value(item))
        elif isinstance(item, dict):
            block_type = item.get("type")
            if block_type == "input_image":
                image_refs += 1
            elif block_type == "input_file":
                file_refs += 1
            elif block_type == "input_text":
                text = item.get("text", "")
                text_chars += len(text) if isinstance(text, str) else 0
            else:
                other_blocks += 1
            previews.append(summarize_value(item))
        else:
            other_blocks += 1
            previews.append(str(type(item).__name__))

    return {
        "blocks": len(contents),
        "text_chars": text_chars,
        "image_refs": image_refs,
        "file_refs": file_refs,
        "other_blocks": other_blocks,
        "preview": previews[:MAX_LIST_ITEMS],
    }
