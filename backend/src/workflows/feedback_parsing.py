import os
import sys
# Add project root to sys.path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import csv
import json
import zipfile
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from dotenv import load_dotenv

from src.generator.client import generate_structured
from src.clustering import find_matching_clip_occurrence
from src.utils import parse_timestamp_to_seconds
from config.settings import OPENAI_REASONING_MODEL

load_dotenv()

# 1. Pydantic Models for Structured Output
class CleanRemark(BaseModel):
    timestamp: Optional[str] = Field(
        None, 
        description="The timestamp of the remark in MM:SS or H:MM:SS format, or null if no timestamp is present."
    )
    category: Literal["audio", "video", "both"] = Field(
        ...,
        description="Category of the remark: 'audio' (voice, SFX, dubbing, dialogue delivery), 'video' (visual adjustments, camera movements, expressions, duration of shots), or 'both'."
    )
    remark: str = Field(
        ...,
        description="The text content of the feedback remark."
    )

class CleanFeedbackList(BaseModel):
    remarks: List[CleanRemark]


# 2. File Format Readers
def read_txt_file(file_path: str) -> str:
    """Read plain text feedback file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_csv_file(file_path: str) -> str:
    """Read CSV feedback file and format it as a tabular text block."""
    lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if any(cell.strip() for cell in row):
                lines.append("\t".join(row))
    return "\n".join(lines)

def read_xlsx_file(file_path: str) -> str:
    """Read Excel (.xlsx) file and format it as a tabular text block without external dependencies."""
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            # Parse shared strings
            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                with z.open("xl/sharedStrings.xml") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    # Find si tags with namespace
                    ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    for si in root.findall(".//ns:si", ns):
                        text_parts = [t.text for t in si.findall(".//ns:t", ns) if t.text]
                        shared_strings.append("".join(text_parts))

            # Parse sheet1
            if "xl/worksheets/sheet1.xml" not in z.namelist():
                return ""

            with z.open("xl/worksheets/sheet1.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

                rows_data = []
                for row in root.findall(".//ns:row", ns):
                    cells_in_row = {}
                    for cell in row.findall("ns:c", ns):
                        r_attr = cell.get("r") or "" # e.g. "A1"
                        t_attr = cell.get("t") # e.g. "s"
                        val_el = cell.find("ns:v", ns)
                        val = val_el.text if val_el is not None else ""

                        if t_attr == "s" and val.isdigit():
                            idx = int(val)
                            if 0 <= idx < len(shared_strings):
                                val = shared_strings[idx]

                        col_letter = "".join([char for char in r_attr if char.isalpha()])
                        if col_letter:
                            cells_in_row[col_letter] = val

                    rows_data.append(cells_in_row)

                # Formulate text representation
                all_cols = set()
                for r in rows_data:
                    all_cols.update(r.keys())

                sorted_cols = sorted(list(all_cols), key=lambda x: (len(x), x))

                lines = []
                for r in rows_data:
                    row_vals = [r.get(col, "") for col in sorted_cols]
                    if any(row_val.strip() for row_val in row_vals if row_val):
                        lines.append("\t".join(row_vals))

                return "\n".join(lines)
    except Exception as e:
        return f"Error parsing XLSX: {e}"

def extract_file_content(file_path: str) -> str:
    """Detects file extension and extracts content as text."""
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()

    if ext_lower == ".txt":
        return read_txt_file(file_path)
    elif ext_lower == ".csv":
        return read_csv_file(file_path)
    elif ext_lower in (".xlsx", ".xls"):
        return read_xlsx_file(file_path)
    else:
        return read_txt_file(file_path)


# 3. Main Workflow Function
def parse_and_align_feedback(
    feedback_file_path: str,
    timeline_json_path: str,
    project_name: str,
    openai_client,
    openai_model: str = OPENAI_REASONING_MODEL,
    output_base_dir: str = "data"
) -> str:
    """
    Parses a feedback file using OpenAI and aligns the remarks to the sequence timeline.
    Groups all remarks (both video and audio) primarily by matching video clip.
    Saves the aligned and grouped results in data/{project_name}/feedback.json.

    Args:
        feedback_file_path: Path to the raw feedback file (.txt, .csv, .xlsx).
        timeline_json_path: Path to the parsed timeline.json.
        project_name: Name of the project.
        openai_client: Authenticated OpenAI client.
        openai_model: OpenAI model to use for structured outputs.
        output_base_dir: Base directory where output feedbacks are saved.

    Returns:
        The absolute path to the saved feedback.json file.
    """
    # 1. Validation
    abs_feedback_path = os.path.abspath(feedback_file_path)
    abs_timeline_path = os.path.abspath(timeline_json_path)

    if not os.path.exists(abs_feedback_path):
        raise FileNotFoundError(f"Feedback file not found at: {abs_feedback_path}")
    if not os.path.exists(abs_timeline_path):
        raise FileNotFoundError(f"Timeline JSON file not found at: {abs_timeline_path}")

    # 2. Read input file
    raw_content = extract_file_content(abs_feedback_path)

    # 3. Load timeline data
    with open(abs_timeline_path, 'r', encoding='utf-8') as f:
        timeline_data = json.load(f)

    video_timeline = timeline_data.get("video_timeline", [])
    audio_timeline_data = timeline_data.get("audio_timeline", {})
    dedicated_audio = audio_timeline_data.get("dedicated_audio_tracks", [])

    # 4. Use OpenAI to structure the raw content
    system_instruction = (
        "You are an expert video post-production supervisor.\n"
        "Analyze the following feedback transcript line-by-line.\n"
        "For each distinct remark, extract the timestamp if present, "
        "and categorize it into 'audio', 'video', or 'both'.\n\n"
        "Classification Rules:\n"
        "- 'audio': Dubbing issues, pronunciation errors, sound effects (sfx), dialogues, background noise/music.\n"
        "- 'video': Camera angles, visual reaction changes, speed/duration adjustments, facial expressions, physical action/motion.\n"
        "- 'both': If the feedback involves a combination of both audio and video changes at that timestamp."
    )

    prompt = f"Feedback Transcript:\n{raw_content}"

    # Call OpenAI via custom generate_structured wrapper
    structured_response = generate_structured(
        provider="openai",
        client=openai_client,
        model=openai_model,
        contents=[prompt],
        schema=CleanFeedbackList,
        system_instruction=system_instruction
    )

    raw_remarks = structured_response.get("remarks", [])

    # 5. Segment-Based Math Alignment and Grouping (Option A)
    # Group key: (video_clip_name, video_occurrence_index) or (None, audio_clip_name)
    segments = {}

    for item in raw_remarks:
        timestamp = item.get("timestamp")
        category = item.get("category", "video")
        remark = item.get("remark", "")

        ts_sec = parse_timestamp_to_seconds(timestamp)

        # Handling special cases like "freeze" or "end" if timestamp is missing
        if ts_sec is None and timestamp is None:
            remark_lower = remark.lower()
            if ("freeze" in remark_lower or "end" in remark_lower) and video_timeline:
                ts_sec = video_timeline[-1]["end_s"]

        # Match to video clip first
        v_idx, v_clip = find_matching_clip_occurrence(video_timeline, ts_sec)

        # Match to audio track
        a_idx, a_clip = find_matching_clip_occurrence(dedicated_audio, ts_sec)
        audio_name = a_clip["clip"] if a_clip else None

        if v_clip:
            clip_name = v_clip["clip"]
            prev_clip = video_timeline[v_idx - 1]["clip"] if v_idx > 0 else None
            key = (clip_name, v_idx)
            if key not in segments:
                segments[key] = {
                    "clip_used": clip_name,
                    "clip_occurrence": v_idx,
                    "previous_clip": prev_clip,
                    "clip_start_tc": v_clip["start_tc"],
                    "clip_end_tc": v_clip["end_tc"],
                    "clip_start_s": v_clip["start_s"],
                    "clip_end_s": v_clip["end_s"],
                    "audio_used": audio_name,
                    "feedback_items": []
                }
            # Overwrite or append audio if it is discovered for this segment
            if audio_name and not segments[key]["audio_used"]:
                segments[key]["audio_used"] = audio_name

            segments[key]["feedback_items"].append({
                "timestamp": timestamp,
                "category": category,
                "remark": remark
            })
        else:
            # If no video clip is matched (e.g. timestamp is out of range or empty),
            # check if there is an audio track and group by audio track, or fallback to general.
            if audio_name:
                key = (None, audio_name)
                if key not in segments:
                    segments[key] = {
                        "clip_used": None,
                        "previous_clip": None,
                        "audio_used": audio_name,
                        "audio_start_tc": a_clip["start_tc"] if a_clip else None,
                        "audio_end_tc": a_clip["end_tc"] if a_clip else None,
                        "audio_start_s": a_clip["start_s"] if a_clip else None,
                        "audio_end_s": a_clip["end_s"] if a_clip else None,
                        "feedback_items": []
                    }
                segments[key]["feedback_items"].append({
                    "timestamp": timestamp,
                    "category": category,
                    "remark": remark
                })
            else:
                # Fallback: unmatched segment
                key = (None, "unmatched")
                if key not in segments:
                    segments[key] = {
                        "clip_used": None,
                        "previous_clip": None,
                        "audio_used": None,
                        "feedback_items": []
                    }
                segments[key]["feedback_items"].append({
                    "timestamp": timestamp,
                    "category": category,
                    "remark": remark
                })

    # 6. Save output
    abs_output_base = os.path.abspath(output_base_dir)
    dest_dir = os.path.join(abs_output_base, project_name)
    os.makedirs(dest_dir, exist_ok=True)

    output_file_path = os.path.join(dest_dir, "feedback.json")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(segments.values()), f, ensure_ascii=False, indent=2)

    return os.path.abspath(output_file_path)

if __name__ == "__main__":

    feedback_file = "data/feedback/feedback.txt"
    timeline_file = "data/project-red-and-green-ep5/timeline.json"
    project = "project-red-and-green-ep5"

    from openai import OpenAI
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    output_path = parse_and_align_feedback(
        feedback_file_path=feedback_file,
        timeline_json_path=timeline_file,
        project_name=project,
        openai_client=openai_client,
        openai_model=OPENAI_REASONING_MODEL,
        output_base_dir="data"
    )

    print(f"Feedback aligned and saved to: {output_path}")
