import os
import time
import base64
import mimetypes
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from google import genai

from .client import Provider, _detect_provider

IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}

@dataclass
class OpenAIFileReference:
    content_block: Dict[str, Any]
    file_id: Optional[str] = None


def _openai_image_data_url(filepath: str) -> str:
    mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    with open(filepath, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _openai_file_reference(client: OpenAI, filepath: str) -> OpenAIFileReference:
    extension = os.path.splitext(filepath)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return OpenAIFileReference(
            content_block={
                "type": "input_image",
                "image_url": _openai_image_data_url(filepath),
                "detail": "auto",
            }
        )

    with open(filepath, "rb") as file:
        uploaded = client.files.create(file=file, purpose="user_data")

    return OpenAIFileReference(
        content_block={
            "type": "input_file",
            "file_id": uploaded.id,
        },
        file_id=uploaded.id,
    )


def _prepare_media_reference(client, filepath: str, provider: Provider):
    if provider == "openai":
        return upload_file_and_wait(client, filepath, provider=provider)
    return upload_file_and_wait(client, filepath)


def _append_media_reference(contents: List[Any], ref: Any) -> None:
    if isinstance(ref, OpenAIFileReference):
        contents.append(ref.content_block)
    else:
        contents.append(ref)


def upload_file_and_wait(
    client: genai.Client | OpenAI,
    filepath: str,
    provider: Optional[Provider] = None,
):
    """Upload or adapt a local media file for the selected model provider.

    Args:
        client: The model provider client instance used to interact with the model.
        filepath (str): The local path of the file to be uploaded.
        provider: Provider override. Defaults to inferring from the client type.

    Raises:
        RuntimeError: If the file processing fails or if it times out waiting for activation.
        RuntimeError: If the file does not exist locally.

    Returns:
        genai.File: The reference to the uploaded file.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found locally! Skipping upload.")
        return None

    provider = provider or _detect_provider(client)
    if provider == "openai":
        print(f"Preparing {filepath} for OpenAI Responses input...")
        return _openai_file_reference(client, filepath)

    print(f"Uploading {filepath}...")
    file_ref = client.files.upload(file=filepath)
    print(f"Uploaded as {file_ref.name}. Waiting for activation...")
    
    start_time = time.time()
    while True:
        file_ref = client.files.get(name=file_ref.name)
        state_str = str(file_ref.state)
        if "ACTIVE" in state_str:
            print(f"File {filepath} is now ACTIVE.")
            break
        elif "FAILED" in state_str:
            raise RuntimeError(f"File processing failed for {filepath}")
            
        if time.time() - start_time > 180: # 3 minutes timeout
            raise RuntimeError(f"Timeout waiting for file activation: {filepath}")
            
        time.sleep(2)
    return file_ref
