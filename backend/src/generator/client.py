import os
import json
from typing import Literal, List, Dict, Any
from openai import OpenAI
from google.genai import types

from config.settings import LITE_MODEL, OPENAI_LITE_MODEL

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


def _log_api_response(
    provider: str,
    model: str,
    contents: List[Any],
    system_instruction: str,
    response_dict: Dict[str, Any]
) -> None:
    # Ensure data/output directory exists
    log_dir = "data/output"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "api_responses.jsonl")
    
    # Serialize contents safely, truncating huge media blocks
    serialized_contents = []
    for item in contents:
        if isinstance(item, str):
            serialized_contents.append(item)
        elif isinstance(item, dict):
            cleaned_item = item.copy()
            # Truncate potentially large base64 image strings in logs
            for key in ["image_bytes", "data"]:
                if key in cleaned_item and isinstance(cleaned_item[key], str) and len(cleaned_item[key]) > 100:
                    cleaned_item[key] = cleaned_item[key][:50] + "... [TRUNCATED]"
            serialized_contents.append(cleaned_item)
        else:
            serialized_contents.append(str(item))

    log_entry = {
        "provider": provider,
        "model": model,
        "system_instruction": system_instruction,
        "contents": serialized_contents,
        "response": response_dict
    }
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: Failed to write to API log file: {exc}")


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
    if provider == "gemini":
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
        _log_api_response(provider, model, contents, system_instruction, response_dict)
        return response_dict
    elif provider == "openai":
        response = client.responses.parse(
            model=model,
            reasoning={"effort": "high"},
            input=_openai_input_from_contents(contents),
            text_format=schema,
            # temperature=temperature,
            instructions=system_instruction
        )

        response_dict = response.output_parsed.model_dump()
        _log_api_response(provider, model, contents, system_instruction, response_dict)
        return response_dict
