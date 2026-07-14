import os
import sys
# Add project root to sys.path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from dotenv import load_dotenv
load_dotenv()

from src.generator.client import generate_structured
from config.settings import OPENAI_REASONING_MODEL

# 1. Pydantic Models for Planning Structured Output
class SegmentPlanItem(BaseModel):
    clip_used: Optional[str] = Field(None, description="The clip name being targeted.")
    previous_clip: Optional[str] = Field(None, description="The previous clip name on the timeline.")
    clip_start_tc: Optional[str] = None
    clip_end_tc: Optional[str] = None
    clip_start_s: Optional[float] = None
    clip_end_s: Optional[float] = None
    generation_type: Literal["complex", "simple", "none"] = Field(
        ...,
        description=(
            "Classification category:\n"
            "- 'complex': Requires character design sheets or location sheets to generate new video (e.g. character action performance changes, new actions/interactions, or object interactions).\n"
            "- 'simple': Requires simple post-processing or minor transitions/zooms only, without character/setting swaps or character sheet references.\n"
            "- 'none': Audio-only changes, or no visual changes needed (e.g. SFX changes, dubbing updates)."
        )
    )
    classification_reasoning: str = Field(..., description="Reasoning for selecting simple, complex, or none.")
    audio_used: Optional[str] = Field(None, description="The audio file name mapped to this segment from feedback.json.")
    audio_path: Optional[str] = Field(None, description="The resolved local absolute path to the audio file.")
    is_dialogue_active: bool = Field(False, description="True if dialogue/speech is active during this segment.")
    characters_present: List[str] = Field(default_factory=list, description="List of character names mentioned or present in the segment (e.g., 'Vir', 'Rian', 'Mother').")
    location: Optional[str] = Field(None, description="Location mentioned or present in the segment (e.g. 'hall', 'living room').")
    requires_previous_clip_continuity: bool = Field(..., description="True if this segment visually depends on the previous shot's actions or setting for continuity.")
    previous_clip_dependency_reason: Optional[str] = Field(None, description="Reason for previous clip continuity dependency if required.")
    remarks_to_process: List[str] = Field(..., description="List of visual remarks to be processed for this clip.")

class GenerationPlan(BaseModel):
    plans: List[SegmentPlanItem]


# 2. Planning Function
def plan_generation_workflow(
    feedback_json_path: str,
    project_name: str,
    openai_client,
    openai_model: str = OPENAI_REASONING_MODEL,
    output_base_dir: str = "data"
) -> str:
    """
    Parses feedback.json and plans the video generation workflow using OpenAI.
    Categorizes clips, identifies context needs, and sorts the plan chronologically.

    Args:
        feedback_json_path: Path to the grouped feedback.json file.
        project_name: Name of the project.
        openai_client: Authenticated OpenAI client.
        openai_model: OpenAI model to use.
        output_base_dir: Base directory where output plans are saved. Defaults to "data".

    Returns:
        The absolute path to the generated generation_plan.json file.
    """
    abs_feedback_path = os.path.abspath(feedback_json_path)
    if not os.path.exists(abs_feedback_path):
        raise FileNotFoundError(f"Feedback JSON not found at: {abs_feedback_path}")

    # Load feedback items
    with open(abs_feedback_path, 'r', encoding='utf-8') as f:
        feedback_data = json.load(f)

    # 1. System Prompt for Planner
    system_instruction = (
        "You are an expert AI video production coordinator.\n"
        "Your task is to analyze client feedback segments and plan the video regeneration workflow.\n"
        "For each timeline segment in the input JSON, classify it into one of these generation types:\n"
        "- 'complex': If the feedback involves character performance adjustments, new physical actions (e.g., 'smashing a vase', 'grasping hand'), or character/setting swaps that require character reference sheets or style sheets.\n"
        "- 'simple': If the feedback involves simple visual edits, post-processing camera moves (e.g., 'zoom in reaction', 'crop from floor'), simple timing, or transitions, without requiring character/location sheet assets.\n"
        "- 'none': If the segment has no visual edits (e.g., only audio/dubbing remarks are present, or there are no visual changes requested).\n\n"
        "Identify context requirements such as characters mentioned, locations mentioned, and whether there is visual continuity dependency on the previous clip.\n"
        "Also, detect whether there is active dialogue or spoken speech in this segment based on: (1) if audio_used contains 'dub', 'dialogue', 'vo', or 'voice'; (2) if the feedback remarks mention speech pronunciation, dialog line changes, or quotes; or (3) if the audio_used is present. If active speech/dialogue is detected, set is_dialogue_active to true."
    )

    prompt = f"Grouped Feedback Segments JSON:\n{json.dumps(feedback_data, indent=2)}"

    # 2. Call OpenAI to get structured planning output
    structured_response = generate_structured(
        provider="openai",
        client=openai_client,
        model=openai_model,
        contents=[prompt],
        schema=GenerationPlan,
        system_instruction=system_instruction
    )

    plans_list = structured_response.get("plans", [])

    # 3. Post-Processing: Sort plans chronologically by clip_start_s
    # If clip_start_s is None (e.g., unmatched segments), we place them at the end.
    def get_start_s(item):
        val = item.get("clip_start_s")
        if val is None:
            return float('inf')
        return val

    sorted_plans = sorted(plans_list, key=get_start_s)

    # 4. Resolve local audio paths and referenced frames if present
    # Load feedback.json to map referenced_frames to plan items
    ref_frames_map = {}
    if os.path.exists(abs_feedback_path):
        try:
            with open(abs_feedback_path, 'r', encoding='utf-8') as f:
                feedback_json_data = json.load(f)
            for seg in feedback_json_data:
                clip_used = seg.get("clip_used")
                start_s = seg.get("clip_start_s")
                ref_frames = seg.get("referenced_frames")
                if ref_frames:
                    unique_refs = []
                    for ref in ref_frames:
                        if ref.get("frame_path") not in [r.get("frame_path") for r in unique_refs]:
                            unique_refs.append(ref)
                    ref_frames_map[(clip_used, start_s)] = unique_refs
        except Exception as exc:
            print(f"Warning: Failed to load feedback.json for mapping referenced frames: {exc}")

    assets_audio_dir = os.path.join(os.getcwd(), "assets", project_name, "04_audio")
    for item in sorted_plans:
        # Resolve referenced frames
        clip_used = item.get("clip_used")
        start_s = item.get("clip_start_s")
        item["referenced_frames"] = ref_frames_map.get((clip_used, start_s), [])

        audio_name = item.get("audio_used")
        if audio_name:
            # Check if file exists under the standard project audio path
            local_audio_path = os.path.join(assets_audio_dir, audio_name)
            if os.path.exists(local_audio_path):
                item["audio_path"] = os.path.abspath(local_audio_path)
            else:
                # Also check relative to root or workspace just in case
                fallback_path = os.path.abspath(os.path.join("assets", project_name, "04_audio", audio_name))
                if os.path.exists(fallback_path):
                    item["audio_path"] = fallback_path

    # 5. Save the plan
    abs_output_base = os.path.abspath(output_base_dir)
    dest_dir = os.path.join(abs_output_base, project_name)
    os.makedirs(dest_dir, exist_ok=True)

    output_plan_path = os.path.join(dest_dir, "generation_plan.json")
    with open(output_plan_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_plans, f, ensure_ascii=False, indent=2)

    return os.path.abspath(output_plan_path)

if __name__ == "__main__":

    feedback_path = "data/project-red-and-green/feedback.json"
    project = "project-red-and-green"
    
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    plan_path = plan_generation_workflow(
        feedback_json_path=feedback_path,
        project_name=project,
        openai_client=openai_client,
        openai_model=OPENAI_REASONING_MODEL,
        output_base_dir="data"
    )
    print(f"Generation plan saved at: {plan_path}")
