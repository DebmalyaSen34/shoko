import os
from typing import Any

from fastapi import HTTPException
from dotenv import set_key, unset_key

from config.settings import (
    LOADED_ENV_FILES,
    OS_ENV_KEYS_AT_START,
    SECRET_ENV_KEYS,
    app_config_env_path,
)
from src.schema import RuntimeSecretsUpdate
from src.storage_paths import APP_STORAGE_DIR, ASSETS_DIR, DATA_DIR


def env_secret_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    config_env = app_config_env_path(APP_STORAGE_DIR)
    for key in SECRET_ENV_KEYS:
        configured = bool(os.environ.get(key))
        locked_by_os_env = key in OS_ENV_KEYS_AT_START
        if locked_by_os_env:
            source = "os_environment"
        elif configured:
            source = "app_or_dev_env_file"
        else:
            source = "missing"
        status[key] = {
            "configured": configured,
            "source": source,
            "locked_by_os_env": locked_by_os_env,
            "can_update": not locked_by_os_env,
        }
    return {
        "keys": status,
        "config_env_path": str(config_env),
    }


def get_runtime_config():
    return {
        "app_storage_dir": str(APP_STORAGE_DIR),
        "data_dir": str(DATA_DIR),
        "assets_dir": str(ASSETS_DIR),
        "loaded_env_files": LOADED_ENV_FILES,
        "secrets": env_secret_status(),
    }


def update_runtime_secrets(payload: RuntimeSecretsUpdate):
    updates = {
        "OPENAI_API_KEY": payload.openai_api_key,
        "GEMINI_API_KEY": payload.gemini_api_key,
        "SEGMIND_API_KEY": payload.segmind_api_key,
    }
    requested_updates = {key: value for key, value in updates.items() if value is not None}
    locked_keys = sorted(key for key in requested_updates if key in OS_ENV_KEYS_AT_START)
    if locked_keys:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot update keys controlled by OS environment: {', '.join(locked_keys)}",
        )

    config_env = app_config_env_path(APP_STORAGE_DIR)
    config_env.parent.mkdir(parents=True, exist_ok=True)
    if not config_env.exists():
        config_env.touch(mode=0o600)

    for key, value in requested_updates.items():
        normalized_value = value.strip()
        if normalized_value:
            set_key(str(config_env), key, normalized_value, quote_mode="always")
            os.environ[key] = normalized_value
        else:
            unset_key(str(config_env), key)
            os.environ.pop(key, None)

    try:
        config_env.chmod(0o600)
    except OSError:
        pass

    return {
        "status": "ok",
        "config_env_path": str(config_env),
        "secrets": env_secret_status(),
    }
