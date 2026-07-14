#!/usr/bin/env python3
import os
import sys
import argparse
import base64
import mimetypes
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from byteplussdkarkruntime import Ark

load_dotenv()


def _file_data_url(path: str | os.PathLike[str]) -> str:
    """Encode local file to Base64 data URL format."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    mime_type = mimetypes.guess_type(filepath.name)[0]
    if not mime_type:
        # Fallback based on common extensions
        ext = filepath.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "application/octet-stream"

    with open(filepath, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_value(obj: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve value from a dict or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def download_image(url: str, output_path: Path) -> None:
    """Download an image from a web URL and save it locally."""
    print(f"Downloading generated image from: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as file:
            file.write(response.read())
        print(f"Successfully downloaded image to: {output_path}")
    except urllib.error.URLError as e:
        print(f"Error downloading image: {e}", file=sys.stderr)
        raise


def save_b64_image(b64_string: str, output_path: Path) -> None:
    """Decode a Base64 string and save it as an image file."""
    print("Decoding Base64 image data...")
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    try:
        data = base64.b64decode(b64_string)
        with open(output_path, "wb") as file:
            file.write(data)
        print(f"Successfully saved decoded image to: {output_path}")
    except Exception as e:
        print(f"Error saving Base64 image: {e}", file=sys.stderr)
        raise


def get_size_for_model(model: str) -> str:
    """Determine size parameter to enforce a 9:16 aspect ratio based on targeted model."""
    model_lower = model.lower()
    if "seededit-3-0-i2i" in model_lower:
        # i2i only supports adaptive mode currently
        return "adaptive"
    elif "seededit-3-0-t2i" in model_lower:
        return "720x1280"
    else:
        # Default to 9:16 2K resolution for Seedream models (1600x2848)
        return "1600x2848"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a copy of an input image using the Seedream Image Generation API."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("output_image", type=str, help="Path to save the generated copy.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SEEDREAM_MODEL", "seedream-4-5-251128"),
        help="Model ID or Endpoint ID to use for generation (default: seedream-4-5-251128).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="recreate this image exactly, one to one copy, preserving all visual elements, style, colors, composition, and subject matter without any changes",
        help="Text prompt for the generation model.",
    )
    parser.add_argument(
        "--response-format",
        choices=["url", "b64_json"],
        default="url",
        help="Format in which the generated image is returned (default: url).",
    )
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Include watermark in the output image.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ARK_API_KEY"),
        help="Ark API key (overrides ARK_API_KEY environment variable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("Error: ARK_API_KEY is not set. Please define it in your environment or pass via --api-key.", file=sys.stderr)
        return 1

    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Encode image to base64 URL format
        image_data_url = _file_data_url(input_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Initializing Ark client with model: {args.model}...")
    client = Ark(api_key=args.api_key)

    size = get_size_for_model(args.model)

    try:
        print(f"Submitting image copy task to ModelArk (size: {size})...")
        response = client.images.generate(
            model=args.model,
            prompt=args.prompt,
            image=image_data_url,
            response_format=args.response_format,
            watermark=args.watermark,
            sequential_image_generation="disabled",
            size=size
        )
    except Exception as e:
        print(f"API Error during image generation: {e}", file=sys.stderr)
        return 1

    # Extract image details from response
    data_list = _response_value(response, "data")
    if not data_list or len(data_list) == 0:
        error_msg = _response_value(response, "error")
        print(f"Error: No image returned in response. Details: {error_msg}", file=sys.stderr)
        return 1

    img_info = data_list[0]

    # Handle output based on selected format
    try:
        if args.response_format == "url":
            img_url = _response_value(img_info, "url")
            if not img_url:
                print("Error: Expected image URL but none was found in response.", file=sys.stderr)
                return 1
            download_image(img_url, output_path)
        else:
            b64_json = _response_value(img_info, "b64_json")
            if not b64_json:
                print("Error: Expected Base64 data but none was found in response.", file=sys.stderr)
                return 1
            save_b64_image(b64_json, output_path)
    except Exception as e:
        print(f"Failed to save generated image: {e}", file=sys.stderr)
        return 1

    print("Task completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
