import math
import os
import subprocess
from typing import List

from src.utils import resolve_media_binary

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


def _adaptive_frame_count(duration_s: float) -> int:
    """Choose a bounded frame count that scales with clip duration."""
    duration = max(float(duration_s or 0.0), 0.0)
    if duration <= 0.0:
        return 1
    if duration <= 2.0:
        return 3
    if duration <= 5.0:
        return 5
    if duration <= 10.0:
        return 7
    return 9


def _frame_offsets_for_duration(duration_s: float, frame_count: int | None = None) -> List[float]:
    """Return chronological, full-clip frame offsets using safe seek points."""
    duration = max(float(duration_s or 0.0), 0.0)
    count = frame_count if frame_count is not None else _adaptive_frame_count(duration)
    count = max(int(count or 0), 0)
    if count <= 0:
        return []
    if duration <= 0.0:
        return [0.0]

    if duration >= 0.2:
        start = 0.05
        end = max(start, duration - 0.1)
    else:
        start = max(0.0, duration * 0.25)
        end = max(start, duration * 0.75)

    if count == 1 or end == start:
        raw_offsets = [start]
    else:
        step = (end - start) / (count - 1)
        raw_offsets = [start + (step * index) for index in range(count)]

    offsets = []
    seen = set()
    for offset in raw_offsets:
        safe_offset = max(0.0, min(offset, duration))
        rounded = round(safe_offset, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        offsets.append(rounded)
    return offsets


def _extract_video_frames(
    filepath: str,
    output_dir: str,
    duration_s: float,
    frame_count: int | None = None,
) -> List[str]:
    """Extract adaptive, full-clip still frames from a video clip with ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    offsets = _frame_offsets_for_duration(duration_s, frame_count)

    frame_paths = []
    for index, offset in enumerate(offsets, start=1):
        frame_path = os.path.join(output_dir, f"frame_{index:03d}.jpg")
        command = [
            resolve_media_binary("ffmpeg"),
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
