from collections import OrderedDict
from typing import Optional

from .utils import parse_timestamp_to_seconds

def find_matching_clip_occurrence(
    video_timeline: list,
    ts_sec: Optional[float],
    tolerance: float = 0.5,
) -> tuple[Optional[int], Optional[dict]]:
    """Return the single timeline occurrence that contains or is nearest to a timestamp.

    Args:
        video_timeline (list): A list of clip dictionaries, each containing 'start_s' and 'end_s' keys.
        ts_sec (Optional[float]): The timestamp in seconds to match against the timeline.
        tolerance (float, optional): The maximum allowed difference between the timestamp and clip boundaries. Defaults to 0.5.

    Returns:
        tuple[Optional[int], Optional[dict]]: The index and dictionary of the matching clip, or (None, None) if no match is found.
    """
    if ts_sec is None:
        return None, None

    is_whole_second = float(ts_sec).is_integer()
    if is_whole_second:
        for index in range(len(video_timeline) - 1):
            current_clip = video_timeline[index]
            next_clip = video_timeline[index + 1]
            current_tail_s = current_clip["end_s"] - ts_sec
            next_start_delta_s = next_clip["start_s"] - ts_sec
            if (
                current_clip["start_s"] <= ts_sec < current_clip["end_s"]
                and 0 < current_tail_s <= tolerance
                and 0 < next_start_delta_s <= tolerance
            ):
                return index + 1, next_clip

    for index, clip in enumerate(video_timeline):
        if clip["start_s"] <= ts_sec < clip["end_s"]:
            return index, clip

    if video_timeline and ts_sec == video_timeline[-1]["end_s"]:
        return len(video_timeline) - 1, video_timeline[-1]

    candidates = []
    for index, clip in enumerate(video_timeline):
        distance = min(
            abs(clip["start_s"] - ts_sec),
            abs(clip["end_s"] - ts_sec),
        )
        candidates.append((distance, index, clip))

    if candidates:
        distance, index, clip = min(candidates, key=lambda item: (item[0], item[1]))
        if distance <= tolerance:
            return index, clip

    return None, None


def _feedback_lanes(category: str) -> tuple[str, ...]:
    if category == "both":
        return "video", "audio"
    if category == "audio":
        return ("audio",)
    return ("video",)


def cluster_feedback_by_clip(feedback_list: list, video_timeline: list) -> list:
    """Group feedback by timeline occurrence and audio/video generation lane.

    Args:
        feedback_list (list): A list of feedback dictionaries, each containing 'timestamp', 'remark', and 'category' keys.
        video_timeline (list): A list of clip dictionaries, each containing 'start_s' and 'end_s' keys.

    Returns:
        list: A list of clustered feedback items.
    """
    clusters = OrderedDict()

    for item in feedback_list:
        timestamp = item.get("timestamp")
        occurrence, clip = find_matching_clip_occurrence(
            video_timeline,
            parse_timestamp_to_seconds(timestamp),
        )

        if timestamp is None:
            remark = item.get("remark", "").lower()
            if ("freeze" in remark or "end" in remark) and video_timeline:
                occurrence = len(video_timeline) - 1
                clip = video_timeline[occurrence]

        feedback_item = {
            "timestamp": timestamp,
            "remark": item.get("remark", ""),
        }
        if item.get("referenced_frames"):
            feedback_item["referenced_frames"] = item.get("referenced_frames")
        if item.get("complex_reference_plan"):
            feedback_item["complex_reference_plan"] = item.get("complex_reference_plan")
        for lane in _feedback_lanes(item.get("category", "video")):
            key = occurrence, lane
            if key not in clusters:
                clusters[key] = {
                    "clip_occurrence": occurrence,
                    "category": lane,
                    "matched_clip": clip,
                    "feedback_items": [],
                }
            clusters[key]["feedback_items"].append(feedback_item.copy())

    return list(clusters.values())


def format_feedback_instruction(feedback_items: list) -> str:
    """Format ordered feedback items as one model instruction."""
    return "\n".join(
        f"- [{item.get('timestamp') or 'No Timestamp'}] {item.get('remark', '')}"
        for item in feedback_items
    )
