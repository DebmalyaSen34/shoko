import os
import json
import time
from typing import Literal, List, Dict, Any
from openai import OpenAI
from google.genai import types

from config.settings import LITE_MODEL, OPENAI_LITE_MODEL
from src.logging_utils import log_event, summarize_contents, summarize_value

Provider = Literal["gemini", "openai"]

def _detect_provider(client) -> Provider:
    return "openai" if isinstance(client, OpenAI) else "gemini"


def _default_model_for_provider(provider: Provider) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", OPENAI_LITE_MODEL)
    return LITE_MODEL


def _openai_input_from_contents(contents: List[Any]) -> List[Dict[str, Any]]:
    content_blocks = []
    for item in contents:
        if isinstance(item, str):
            content_blocks.append({"type": "input_text", "text": item})
        elif isinstance(item, dict) and item.get("type") in {"input_text", "input_image", "input_file"}:
            content_blocks.append(item)
        else:
            raise TypeError(f"Unsupported OpenAI content item: {type(item).__name__}")

    return [{"role": "user", "content": content_blocks}]


def _schema_name(schema: Any) -> str:
    return getattr(schema, "__name__", type(schema).__name__)


def _log_structured_request(
    *,
    provider: str,
    model: str,
    contents: List[Any],
    system_instruction: str,
    schema: Any,
    temperature: float,
) -> float:
    started = time.perf_counter()
    log_event(
        "llm.request",
        provider=provider,
        model=model,
        schema=_schema_name(schema),
        temperature=temperature,
        system_instruction_chars=len(system_instruction or ""),
        contents=summarize_contents(contents),
    )
    return started


def _log_structured_response(
    *,
    provider: str,
    model: str,
    schema: Any,
    response_dict: Dict[str, Any],
    started: float,
) -> None:
    log_event(
        "llm.response",
        provider=provider,
        model=model,
        schema=_schema_name(schema),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        response_keys=sorted(response_dict.keys()) if isinstance(response_dict, dict) else [],
        response=summarize_value(response_dict),
    )


def generate_structured(
    *,
    provider: Provider,
    client,
    model: str,
    contents,
    schema,
    system_instruction: str,
    temperature: float = 0.2
):
    started = _log_structured_request(
        provider=provider,
        model=model,
        contents=contents,
        system_instruction=system_instruction,
        schema=schema,
        temperature=temperature,
    )
    if provider == "gemini":
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    system_instruction=system_instruction,
                ),
            )

            response_dict = json.loads(response.text)
            _log_structured_response(
                provider=provider,
                model=model,
                schema=schema,
                response_dict=response_dict,
                started=started,
            )
            return response_dict
        except Exception as exc:
            log_event(
                "llm.error",
                provider=provider,
                model=model,
                schema=_schema_name(schema),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            raise
    elif provider == "openai":
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": "high"},
                input=_openai_input_from_contents(contents),
                text_format=schema,
                # temperature=temperature,
                instructions=system_instruction
            )

            response_dict = response.output_parsed.model_dump()
            _log_structured_response(
                provider=provider,
                model=model,
                schema=schema,
                response_dict=response_dict,
                started=started,
            )
            return response_dict
        except Exception as exc:
            log_event(
                "llm.error",
                provider=provider,
                model=model,
                schema=_schema_name(schema),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            raise
