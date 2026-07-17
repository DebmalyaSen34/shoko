from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_PROMPTS_PATH = Path("data/output/generated_prompts_openai_0.json")
DEFAULT_OUTPUT_DIR = Path("data/output/seedance_videos")
DEFAULT_MODEL = "seedance-2.0"
DEFAULT_RATIO = "9:16"
DEFAULT_RESOLUTION = "720p"
DEFAULT_BITRATE_MODE = "standard"
SEGMIND_GENERATION_URL = "https://api.segmind.com/v1/seedance-2.0"
SEGMIND_UPLOAD_URL = "https://workflows-api.segmind.com/upload-asset"
MAX_DURATION_SECONDS = 15
MIN_DURATION_SECONDS = 4


def clamp_duration(duration: int | float | str | None, model: str | None = None) -> int:
    """Segmind Seedance 2.0 accepts integer durations from 4s through 15s."""
    try:
        value = int(round(float(duration)))
    except (TypeError, ValueError):
        value = 5
    return max(MIN_DURATION_SECONDS, min(MAX_DURATION_SECONDS, value))


def load_prompt_item(prompts_path: str | os.PathLike[str], index: int) -> dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as file:
        items = json.load(file)

    if isinstance(items, dict):
        items = [items]

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


def is_url(path: Any) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def is_data_url(path: Any) -> bool:
    return isinstance(path, str) and path.startswith("data:")


def _mime_type_for_path(path: str | os.PathLike[str]) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _file_data_url(path: str | os.PathLike[str]) -> str:
    filepath = Path(path)
    encoded = base64.b64encode(filepath.read_bytes()).decode("ascii")
    return f"data:{_mime_type_for_path(filepath.name)};base64,{encoded}"


def _data_url_bytes(data_url: str) -> tuple[bytes, str]:
    header, encoded = data_url.split(",", 1)
    mime_type = "application/octet-stream"
    if header.startswith("data:") and ";base64" in header:
        mime_type = header[len("data:"):header.index(";base64")]
    return base64.b64decode(encoded), mime_type


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_asset_type(path: str) -> str:
    path_lower = path.lower()
    if "character" in path_lower or "01_" in path_lower:
        return "character sheet"
    if "location" in path_lower or "03_" in path_lower:
        return "location reference"
    if "prop" in path_lower or "02_" in path_lower:
        return "prop reference"
    if "style" in path_lower or "00_" in path_lower:
        return "style reference"
    return "visual reference"


@dataclass(frozen=True)
class CachedAsset:
    public_url: str
    provider: str
    media_type: str
    source_hash: str | None = None


class SupabaseAssetUrlCache:
    """Small REST-only cache for generated public media URLs.

    Required env vars:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY

    Optional env vars:
    - SUPABASE_ASSET_URL_TABLE, default media_asset_urls
    - SUPABASE_AUDIO_BUCKET, default seedance-audio
    """

    def __init__(
        self,
        *,
        supabase_url: str | None = None,
        api_key: str | None = None,
        table_name: str | None = None,
        audio_bucket: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.supabase_url = (supabase_url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("SUPABASE_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or ""
        )
        self.table_name = table_name or os.environ.get("SUPABASE_ASSET_URL_TABLE", "media_asset_urls")
        self.audio_bucket = audio_bucket or os.environ.get("SUPABASE_AUDIO_BUCKET", "seedance-audio")
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.supabase_url and self.api_key)

    def _headers(self, *, content_type: str = "application/json") -> dict[str, str]:
        headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def get(self, *, source_hash: str, provider: str, media_type: str) -> CachedAsset | None:
        if not self.enabled:
            return None

        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        params = {
            "select": "public_url,provider,media_type,source_hash",
            "source_hash": f"eq.{source_hash}",
            "provider": f"eq.{provider}",
            "media_type": f"eq.{media_type}",
            "limit": "1",
        }
        response = self.session.get(url, headers=self._headers(), params=params, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        return CachedAsset(
            public_url=row["public_url"],
            provider=row.get("provider", provider),
            media_type=row.get("media_type", media_type),
            source_hash=row.get("source_hash", source_hash),
        )

    def upsert(
        self,
        *,
        source_hash: str,
        provider: str,
        media_type: str,
        public_url: str,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        row = {
            "source_hash": source_hash,
            "provider": provider,
            "media_type": media_type,
            "public_url": public_url,
            "source_path": source_path,
            "metadata": metadata or {},
        }
        headers = self._headers()
        headers["Prefer"] = "resolution=merge-duplicates"
        response = self.session.post(
            url,
            headers=headers,
            params={"on_conflict": "source_hash,provider,media_type"},
            json=row,
            timeout=30,
        )
        response.raise_for_status()

    def upload_audio(
        self,
        *,
        data: bytes,
        source_hash: str,
        filename: str,
        content_type: str,
    ) -> str:
        if not self.enabled:
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "to upload audio references for Segmind."
            )

        object_name = f"{source_hash}/{Path(filename).name}"
        object_path = quote(object_name, safe="/")
        upload_url = f"{self.supabase_url}/storage/v1/object/{self.audio_bucket}/{object_path}"
        headers = self._headers(content_type=content_type)
        headers["x-upsert"] = "true"
        response = self.session.post(upload_url, headers=headers, data=data, timeout=120)
        response.raise_for_status()
        return f"{self.supabase_url}/storage/v1/object/public/{self.audio_bucket}/{object_path}"


def upload_to_segmind(
    file_path: str | os.PathLike[str],
    *,
    api_key: str,
    session: requests.Session | None = None,
) -> str:
    data_url = _file_data_url(file_path)
    return upload_data_url_to_segmind(data_url, api_key=api_key, session=session)


def upload_data_url_to_segmind(
    data_url: str,
    *,
    api_key: str,
    session: requests.Session | None = None,
) -> str:
    http = session or requests.Session()
    response = http.post(
        SEGMIND_UPLOAD_URL,
        json={"data_urls": [data_url]},
        headers={
            "x-api-key": api_key,
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    file_urls = result.get("file_urls") or []
    if not file_urls:
        raise RuntimeError(f"Segmind upload response did not include file_urls: {result}")
    return file_urls[0]


def _cached_segmind_image_url(
    image_ref: str,
    *,
    api_key: str,
    cache: SupabaseAssetUrlCache | None,
    session: requests.Session | None = None,
) -> str | None:
    if is_url(image_ref):
        return image_ref

    source_hash = None
    data_url = None
    if is_data_url(image_ref):
        data, _mime_type = _data_url_bytes(image_ref)
        source_hash = _sha256_bytes(data)
        data_url = image_ref
        source_path = None
    elif os.path.exists(image_ref):
        source_hash = _sha256_file(image_ref)
        data_url = _file_data_url(image_ref)
        source_path = image_ref
    else:
        return None

    cached = cache.get(source_hash=source_hash, provider="segmind", media_type="image") if cache else None
    if cached:
        return cached.public_url

    public_url = upload_data_url_to_segmind(data_url, api_key=api_key, session=session)
    if cache:
        cache.upsert(
            source_hash=source_hash,
            provider="segmind",
            media_type="image",
            public_url=public_url,
            source_path=source_path,
        )
    return public_url


def _cached_supabase_audio_url(
    audio_ref: str,
    *,
    cache: SupabaseAssetUrlCache | None,
) -> str | None:
    if not audio_ref:
        return None
    if is_url(audio_ref):
        return audio_ref
    if cache is None:
        return None

    source_path = None
    if is_data_url(audio_ref):
        data, content_type = _data_url_bytes(audio_ref)
        source_hash = _sha256_bytes(data)
        extension = mimetypes.guess_extension(content_type) or ".mp3"
        filename = f"audio{extension}"
    elif os.path.exists(audio_ref):
        path = Path(audio_ref)
        data = path.read_bytes()
        source_hash = _sha256_bytes(data)
        content_type = _mime_type_for_path(path.name)
        filename = path.name
        source_path = str(path)
    else:
        return None

    cached = cache.get(source_hash=source_hash, provider="supabase", media_type="audio")
    if cached:
        return cached.public_url

    public_url = cache.upload_audio(
        data=data,
        source_hash=source_hash,
        filename=filename,
        content_type=content_type,
    )
    cache.upsert(
        source_hash=source_hash,
        provider="supabase",
        media_type="audio",
        public_url=public_url,
        source_path=source_path,
    )
    return public_url


def _replace_handles_for_segmind(
    prompt_text: str,
    *,
    image_labels: dict[str, str],
    video_label: str | None,
    referenced_frame_labels: dict[str, str] | None = None,
) -> str:
    replacements = dict(image_labels)
    if video_label:
        replacements["@video1"] = video_label
    if referenced_frame_labels:
        replacements.update(referenced_frame_labels)
    for handle, replacement in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        prompt_text = prompt_text.replace(handle, replacement)
    return prompt_text


def build_seedance_content(
    item: dict[str, Any],
    *,
    use_local_initial_frame: bool,
    initial_image_url: str | None,
    only_reference_video: bool = False,
    reference_video_url: str | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper returning a provider-neutral reference summary.

    Segmind generation now uses build_segmind_payload. This function remains for
    tests and dry debugging of prompt handle mapping.
    """
    payload = build_segmind_payload(
        item=item,
        api_key="dry-run",
        cache=None,
        upload_assets=False,
        use_local_initial_frame=use_local_initial_frame,
        initial_image_url=initial_image_url,
        only_reference_video=only_reference_video,
        reference_video_url=reference_video_url,
    )
    content = [{"type": "text", "text": payload["prompt"]}]
    if payload.get("first_frame_url"):
        content.append({"type": "image_url", "image_url": {"url": payload["first_frame_url"]}, "role": "first_frame"})
    for image_url in payload.get("reference_images", []):
        content.append({"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"})
    for video_url in payload.get("reference_videos", []):
        content.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})
    for audio_url in payload.get("reference_audios", []):
        content.append({"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"})
    return content


def build_segmind_payload(
    *,
    item: dict[str, Any],
    api_key: str,
    cache: SupabaseAssetUrlCache | None,
    upload_assets: bool = True,
    use_local_initial_frame: bool = True,
    initial_image_url: str | None = None,
    only_reference_video: bool = False,
    reference_video_url: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    prompt_text = item["video_model_prompt"].strip()

    reference_images: list[str] = []
    reference_videos: list[str] = []
    reference_audios: list[str] = []
    first_frame_url = initial_image_url or item.get("first_frame_url")

    if only_reference_video:
        video_url = reference_video_url or item.get("video_clip_url") or item.get("matched_clip_url")
        if not video_url:
            raise ValueError(
                "Reference video mode requires a web URL via --reference-video-url, "
                "video_clip_url, or matched_clip_url."
            )
        reference_videos.append(video_url)
        image_labels = {
            f"@image{idx + 1}": f"the { _get_asset_type(path) }"
            for idx, path in enumerate(item.get("selected_assets", []))
        }
        prompt_text = _replace_handles_for_segmind(
            prompt_text,
            image_labels=image_labels,
            video_label="video 1",
        )
    else:
        image_labels: dict[str, str] = {}

        if use_local_initial_frame or not first_frame_url:
            image_path = item.get("initial_frame_image_path")
            if image_path:
                if is_url(image_path):
                    first_frame_url = image_path
                elif upload_assets:
                    first_frame_url = _cached_segmind_image_url(
                        image_path,
                        api_key=api_key,
                        cache=cache,
                        session=session,
                    )
                elif is_data_url(image_path) or os.path.exists(image_path):
                    first_frame_url = image_path

        selected_assets = item.get("selected_assets", [])
        for idx, path in enumerate(selected_assets):
            handle = f"@image{idx + 1}"
            image_url = (
                _cached_segmind_image_url(path, api_key=api_key, cache=cache, session=session)
                if upload_assets
                else path
            )
            if image_url:
                reference_images.append(image_url)
                image_labels[handle] = f"image {len(reference_images)}"
            else:
                image_labels[handle] = f"the {_get_asset_type(path)}"

        clip_image_indices = []
        for path in item.get("clip_frame_paths", []):
            image_url = (
                _cached_segmind_image_url(path, api_key=api_key, cache=cache, session=session)
                if upload_assets
                else path
            )
            if image_url:
                reference_images.append(image_url)
                clip_image_indices.append(len(reference_images))
        video_label = (
            ", ".join(f"image {index}" for index in clip_image_indices)
            if clip_image_indices
            else "the original clip"
        )

        referenced_frame_labels = {}
        for path, label in zip(item.get("referenced_frame_paths", []), item.get("referenced_frame_labels", [])):
            image_url = (
                _cached_segmind_image_url(path, api_key=api_key, cache=cache, session=session)
                if upload_assets
                else path
            )
            if image_url:
                reference_images.append(image_url)
                referenced_frame_labels[label] = f"image {len(reference_images)}"
            else:
                referenced_frame_labels[label] = label

        prompt_text = _replace_handles_for_segmind(
            prompt_text,
            image_labels=image_labels,
            video_label=video_label,
            referenced_frame_labels=referenced_frame_labels,
        )

    audio_ref = item.get("audio_path") or item.get("audio_url")
    audio_url = _cached_supabase_audio_url(audio_ref, cache=cache) if audio_ref else None
    if audio_url:
        reference_audios.append(audio_url)

    payload = {
        "prompt": prompt_text,
        "duration": clamp_duration(item.get("duration", 5), DEFAULT_MODEL),
        "resolution": item.get("resolution", DEFAULT_RESOLUTION),
        "aspect_ratio": item.get("ratio") or item.get("aspect_ratio") or DEFAULT_RATIO,
        "generate_audio": bool(item.get("generate_audio", False)),
        "skip_moderation": bool(item.get("skip_moderation", True)),
        "bitrate_mode": item.get("bitrate_mode", DEFAULT_BITRATE_MODE),
    }

    if first_frame_url:
        payload["first_frame_url"] = first_frame_url
    if reference_images:
        payload["reference_images"] = reference_images
    if reference_videos:
        payload["reference_videos"] = reference_videos
    if reference_audios:
        payload["reference_audios"] = reference_audios
    if item.get("seed") is not None:
        payload["seed"] = item["seed"]
    if item.get("last_frame_url"):
        payload["last_frame_url"] = item["last_frame_url"]
    if item.get("return_last_frame") is not None:
        payload["return_last_frame"] = bool(item["return_last_frame"])

    return payload


def create_seedance_task(
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    model: str | None = None,
    content: list[dict[str, Any]] | None = None,
    ratio: str | None = None,
    duration: int | None = None,
    generate_audio: bool | None = None,
    watermark: bool | None = None,
    output_path: str | os.PathLike[str] | None = None,
    session: requests.Session | None = None,
) -> Any:
    """Submit a synchronous Segmind request.

    The legacy keyword arguments are accepted to keep older tests and call sites
    from failing loudly, but new code should pass a complete Segmind payload.
    """
    if payload is None:
        prompt = ""
        reference_images = []
        reference_videos = []
        reference_audios = []
        for block in content or []:
            if block.get("type") == "text":
                prompt = block.get("text", prompt)
            elif block.get("type") == "image_url":
                reference_images.append(block.get("image_url", {}).get("url"))
            elif block.get("type") == "video_url":
                reference_videos.append(block.get("video_url", {}).get("url"))
            elif block.get("type") == "audio_url":
                reference_audios.append(block.get("audio_url", {}).get("url"))
        payload = {
            "prompt": prompt,
            "duration": clamp_duration(duration, model),
            "resolution": DEFAULT_RESOLUTION,
            "aspect_ratio": ratio or DEFAULT_RATIO,
            "generate_audio": bool(generate_audio),
            "skip_moderation": True,
            "bitrate_mode": DEFAULT_BITRATE_MODE,
        }
        if reference_images:
            payload["reference_images"] = [url for url in reference_images if url]
        if reference_videos:
            payload["reference_videos"] = [url for url in reference_videos if url]
        if reference_audios:
            payload["reference_audios"] = [url for url in reference_audios if url]

    http = session or requests.Session()
    response = http.post(
        SEGMIND_GENERATION_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=1800,
    )
    response.raise_for_status()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)

    return {
        "id": response.headers.get("x-request-id") or f"segmind-{int(time.time())}",
        "status": "succeeded",
        "content": {
            "bytes": response.content,
            "content_type": response.headers.get("content-type", "video/mp4"),
        },
        "payload": payload,
    }


def poll_seedance_task(
    *,
    api_key: str,
    task_id: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> Any:
    return {"id": task_id, "status": "succeeded", "content": {}}


def _response_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _video_url_from_task(task: Any) -> str | None:
    content = _response_value(task, "content")
    if not content:
        return None
    return _response_value(content, "video_url")


def download_video(video_url: str, output_path: str | os.PathLike[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(video_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return path


def save_video_bytes(video_bytes: bytes, output_path: str | os.PathLike[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(video_bytes)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Seedance 2.0 video through Segmind.")
    parser.add_argument("--prompts-json", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--index", type=int, default=0, help="Zero-based successful video item index.")
    parser.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ratio", default=DEFAULT_RATIO)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--initial-image-url", default=None)
    parser.add_argument("--use-local-initial-frame", action="store_true")
    parser.add_argument("--only-reference-video", action="store_true")
    parser.add_argument("--reference-video-url", default=None)
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    item = load_prompt_item(args.prompts_json, args.index)
    if args.duration is not None:
        item["duration"] = args.duration
    item["ratio"] = args.ratio
    item["resolution"] = args.resolution
    if args.generate_audio:
        item["generate_audio"] = True

    api_key = os.environ.get("SEGMIND_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: SEGMIND_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    cache = SupabaseAssetUrlCache()
    payload = build_segmind_payload(
        item=item,
        api_key=api_key or "dry-run",
        cache=cache,
        upload_assets=not args.dry_run,
        use_local_initial_frame=args.use_local_initial_frame,
        initial_image_url=args.initial_image_url,
        only_reference_video=args.only_reference_video,
        reference_video_url=args.reference_video_url,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    output_name = args.output_name or f"{Path(item.get('clip_used') or 'seedance-output').stem}.mp4"
    output_path = Path(args.output_dir) / output_name
    result = create_seedance_task(api_key=api_key, payload=payload)
    save_video_bytes(result["content"]["bytes"], output_path)
    print(f"Downloaded video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
