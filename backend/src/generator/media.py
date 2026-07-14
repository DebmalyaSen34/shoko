import math
import os
import subprocess
from typing import List

def _aspect_ratio(frame_size: str) -> str:
    try:
        width_text, height_text = frame_size.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
        divisor = math.gcd(width, height)
        if width <= 0 or height <= 0 or divisor == 0:
            return "unknown"
        return f"{width // divisor}:{height // divisor}"
    except (ValueError, ZeroDivisionError):
        return "unknown"


def _image_size_for_aspect_ratio(aspect_ratio: str) -> str:
    if aspect_ratio in {"9:16", "3:4"}:
        return "1024x1536"
    if aspect_ratio in {"16:9", "4:3"}:
        return "1536x1024"
    return "1024x1024"


def _extract_video_frames(
    filepath: str,
    output_dir: str,
    duration_s: float,
    frame_count: int = 3,
) -> List[str]:
    """Extract evenly spaced still frames from a video clip with ffmpeg."""
    if frame_count <= 0:
        return []

    os.makedirs(output_dir, exist_ok=True)
    duration_s = max(float(duration_s or 0.0), 0.0)
    if duration_s > 0:
        offsets = [
            max(duration_s * (index + 1) / (frame_count + 1), 0.0)
            for index in range(frame_count)
        ]
    else:
        offsets = [0.0 for _ in range(frame_count)]

    frame_paths = []
    for index, offset in enumerate(offsets, start=1):
        frame_path = os.path.join(output_dir, f"frame_{index:03d}.jpg")
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{offset:.3f}",
            "-i",
            filepath,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            frame_path,
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed extracting frame {index} from {filepath}: "
                f"{completed.stderr.strip()}"
            )
        frame_paths.append(frame_path)

    return frame_paths
