from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


from byteplussdkarkruntime import Ark


load_dotenv()

DEFAULT_PROMPTS_PATH = Path("data/output/generated_prompts_openai_0.json")
DEFAULT_OUTPUT_DIR = Path("data/output/seedance_videos")
DEFAULT_MODEL = "dreamina-seedance-2-0-260128"
DEFAULT_RATIO = "9:16"
MAX_DURATION_SECONDS = 5
MIN_DURATION_SECONDS = 2


def clamp_duration(duration: int | float | str | None, model: str | None = None) -> int:
    """Seedance accepts integer seconds; keep tests at 9:16 and max 5s. Enforce a minimum of 4s for Seedance 2.0."""
    try:
        value = int(round(float(duration)))
    except (TypeError, ValueError):
        value = MAX_DURATION_SECONDS
    
    min_dur = MIN_DURATION_SECONDS
    if model and ("seedance-2-0" in model or "seedance-2.0" in model):
        min_dur = 4
        
    return max(min_dur, min(MAX_DURATION_SECONDS, value))


def load_prompt_item(prompts_path: str | os.PathLike[str], index: int) -> dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as file:
        items = json.load(file)

    video_items = [
        item
        for item in items
        if item.get("category") == "video"
        and item.get("status") in {None, "success", "warning"}
        and item.get("video_model_prompt")
    ]
    if not video_items:
        raise ValueError(f"No successful video prompts found in {prompts_path}")
    if index < 0 or index >= len(video_items):
        raise IndexError(f"Prompt index {index} out of range for {len(video_items)} video item(s)")
    return video_items[index]


def _file_data_url(path: str | os.PathLike[str]) -> str:
    filepath = Path(path)
    mime_type = mimetypes.guess_type(filepath.name)[0] or "application/octet-stream"
    with open(filepath, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _get_asset_type(path: str) -> str:
    path_lower = path.lower()
    if "character" in path_lower or "01_" in path_lower:
        return "character sheet"
    elif "location" in path_lower or "02_" in path_lower:
        return "location reference"
    elif "prop" in path_lower or "03_" in path_lower:
        return "prop reference"
    elif "style" in path_lower or "04_" in path_lower:
        return "style reference"
    return "visual reference"


def is_url(path: Any) -> bool:
    """Check if the given path is a web URL or a data URL."""
    if not isinstance(path, str):
        return False
    return path.startswith(("http://", "https://", "data:"))


def build_seedance_content(
    item: dict[str, Any],
    *,
    use_local_initial_frame: bool,
    initial_image_url: str | None,
    only_reference_video: bool = False,
    reference_video_url: str | None = None,
) -> list[dict[str, Any]]:
    """Generate the content list for a Seedance video task from a prompt item.

    Args:
        item (dict[str, Any]): A prompt item dictionary containing video_model_prompt and optional initial_frame_image_path.
        use_local_initial_frame (bool): Whether to use the local initial frame image path as a Base64 data URL instead of an external URL.
        initial_image_url (str | None): An optional external URL for the initial image to include in the content. If provided, it will be used instead of the local initial frame image.
        only_reference_video (bool): If True, only the raw video clip is included as a video reference in the content list.
        reference_video_url (str | None): An optional external web URL for the reference video.

    Raises:
        ValueError: If the prompt item does not include initial_frame_image_path when use_local_initial_frame is True.
        FileNotFoundError: If the initial frame image path does not exist when use_local_initial_frame is True.

    Returns:
        list[dict[str, Any]]: The generated content list for the Seedance video task.
    """

    if only_reference_video:
        video_url = reference_video_url or item.get("video_clip_url") or item.get("matched_clip_url")
        if not video_url:
            raise ValueError(
                "Reference video mode requires a web URL (http/https). "
                "Please specify it via --reference-video-url or ensure your prompt JSON has 'video_clip_url' or 'matched_clip_url'."
            )
        
        replacements = {}
        # Replace image handles with their semantic descriptors since no images are uploaded
        asset_images_path = item.get("selected_assets", [])
        for idx, path in enumerate(asset_images_path):
            handle = f"@image{idx + 1}"
            asset_type = _get_asset_type(path)
            replacements[handle] = f"the {asset_type}"
            
        replacements["@video1"] = "[Video 1]"
        
        prompt_text = item["video_model_prompt"].strip()
        for handle, ref in replacements.items():
            prompt_text = prompt_text.replace(handle, ref)
            
        return [
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "video_url",
                "video_url": {"url": video_url},
                "role": "reference_video",
            }
        ]

    images_to_append = []
    replacements = {}

    has_first_frame = False
    image_url = initial_image_url
    if use_local_initial_frame or not image_url:
        image_path = item.get("initial_frame_image_path")
        if image_path:
            if is_url(image_path):
                image_url = image_path
            elif os.path.exists(image_path):
                image_url = _file_data_url(image_path)
            elif use_local_initial_frame:
                raise FileNotFoundError(f"Initial frame image not found: {image_path}")

    if image_url:
        images_to_append.append({
            "url": image_url,
            "role": "first_frame"
        })
        has_first_frame = True

    # Get asset images and map their indices (1-based index in the images list).
    # The Seedance API accepts "reference_image" as the role for non-frame images.
    asset_images_path = item.get("selected_assets", [])
    for idx, path in enumerate(asset_images_path):
        handle = f"@image{idx + 1}"
        asset_type = _get_asset_type(path)
        if has_first_frame:
            # When a first_frame is provided, skip embedding reference images
            # (the API may not support mixing first_frame + reference_image).
            replacements[handle] = f"the {asset_type}"
        else:
            url_val = None
            if is_url(path):
                url_val = path
            elif os.path.exists(path):
                url_val = _file_data_url(path)

            if url_val:
                images_to_append.append({
                    "url": url_val,
                    "role": "reference_image"
                })
                img_idx = len(images_to_append)
                replacements[handle] = f"{asset_type} [Image {img_idx}]"
            else:
                replacements[handle] = f"the {asset_type}"

    # Get frames from initial video clip and map their indices.
    initial_video_clip_path = item.get("clip_frame_paths", [])
    if has_first_frame:
        replacements["@video1"] = "the original clip"
    else:
        clip_indices = []
        for path in initial_video_clip_path:
            url_val = None
            if is_url(path):
                url_val = path
            elif os.path.exists(path):
                url_val = _file_data_url(path)

            if url_val:
                images_to_append.append({
                    "url": url_val,
                    "role": "reference_image"
                })
                img_idx = len(images_to_append)
                clip_indices.append(img_idx)

        if clip_indices:
            clip_refs = ", ".join(f"[Image {i}]" for i in clip_indices)
            replacements["@video1"] = clip_refs
        else:
            replacements["@video1"] = "the original clip"
    # Map referenced frame paths (e.g. from other clips/timestamps) to reference images
    referenced_frame_paths = item.get("referenced_frame_paths", [])
    referenced_frame_labels = item.get("referenced_frame_labels", [])
    for path, label in zip(referenced_frame_paths, referenced_frame_labels):
        url_val = None
        if is_url(path):
            url_val = path
        elif os.path.exists(path):
            url_val = _file_data_url(path)

        if url_val:
            images_to_append.append({
                "url": url_val,
                "role": "reference_image"
            })
            img_idx = len(images_to_append)
            replacements[label] = f"[Image {img_idx}]"
        else:
            replacements[label] = label

    # Apply semantic mappings to prompt text
    prompt_text = item["video_model_prompt"].strip()
    for handle, ref in replacements.items():
        prompt_text = prompt_text.replace(handle, ref)

    content = [
        {
            "type": "text",
            "text": prompt_text,
        }
    ]

    for img in images_to_append:
        item_data = {
            "type": "image_url",
            "image_url": {"url": img["url"]},
            "role": img["role"],
        }
        content.append(item_data)

    # Append reference audio if present and generate_audio is True
    audio_url = item.get("audio_url")
    if audio_url:
        content.append({
            "type": "audio_url",
            "audio_url": {"url": audio_url},
            "role": "reference_audio"
        })

    return content

def _response_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _video_url_from_task(task: Any) -> str | None:
    content = _response_value(task, "content")
    if not content:
        return None
    return _response_value(content, "video_url")


def create_seedance_task(
    *,
    api_key: str,
    model: str,
    content: list[dict[str, Any]],
    ratio: str,
    duration: int,
    generate_audio: bool,
    watermark: bool,
) -> Any:
    client = Ark(api_key=api_key)
    return client.content_generation.tasks.create(
        model=model,
        content=content,
        generate_audio=generate_audio,
        ratio=ratio,
        duration=duration,
        watermark=watermark,
    )


def poll_seedance_task(
    *,
    api_key: str,
    task_id: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> Any:
    client = Ark(api_key=api_key)
    deadline = time.monotonic() + timeout_seconds
    while True:
        task = client.content_generation.tasks.get(task_id=task_id)
        status = _response_value(task, "status")
        print(f"Current status: {status}")

        if status == "succeeded":
            return task
        if status == "failed":
            error = _response_value(task, "error", "unknown error")
            raise RuntimeError(f"Seedance task failed: {error}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Seedance task {task_id}")
        time.sleep(poll_interval_seconds)


def download_video(video_url: str, output_path: str | os.PathLike[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(video_url) as response, open(path, "wb") as file:
        file.write(response.read())
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Seedance video task from generated prompt JSON."
    )
    parser.add_argument("--prompts-json", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--index", type=int, default=0, help="Zero-based successful video item index.")
    parser.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ratio", default=DEFAULT_RATIO)
    parser.add_argument("--duration", type=int, default=MAX_DURATION_SECONDS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--initial-image-url", default=None)
    parser.add_argument(
        "--use-local-initial-frame",
        action="store_true",
        help="Send initial_frame_image_path as a Base64 data URL.",
    )
    parser.add_argument(
        "--only-reference-video",
        action="store_true",
        help="Only include the original video clip as a reference video in the request content.",
    )
    parser.add_argument(
        "--reference-video-url",
        default=None,
        help="The external web URL for the reference video clip (HTTP/HTTPS).",
    )
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Seedance create-task payload without calling the API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    item = load_prompt_item(args.prompts_json, args.index)
    duration = clamp_duration(args.duration, args.model)
    content = build_seedance_content(
        item,
        use_local_initial_frame=args.use_local_initial_frame,
        initial_image_url=args.initial_image_url,
        only_reference_video=args.only_reference_video,
        reference_video_url=args.reference_video_url,
    )

    payload = {
        "model": args.model,
        "content": content,
        "generate_audio": args.generate_audio,
        "ratio": args.ratio,
        "duration": duration,
        "watermark": args.watermark,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Error: ARK_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    create_result = create_seedance_task(
        api_key=api_key,
        model=args.model,
        content=content,
        ratio=args.ratio,
        duration=duration,
        generate_audio=args.generate_audio,
        watermark=args.watermark,
    )
    task_id = _response_value(create_result, "id")
    if not task_id:
        raise RuntimeError(f"Create task response did not include an id: {create_result}")
    print(f"Created Seedance task: {task_id}")

    task = poll_seedance_task(
        api_key=api_key,
        task_id=task_id,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    video_url = _video_url_from_task(task)
    if not video_url:
        raise RuntimeError(f"Succeeded task did not include content.video_url: {task}")

    output_name = args.output_name or f"{task_id}.mp4"
    output_path = download_video(video_url, Path(args.output_dir) / output_name)
    print(f"Downloaded video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
