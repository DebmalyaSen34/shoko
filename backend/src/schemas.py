from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# Pydantic structure for feedback data
class Remark(BaseModel):
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

class FeedbackList(BaseModel):
    remarks: List[Remark]


class SelectedAssets(BaseModel):
    relevant_assets: List[str] = Field(
        description="List of relative file paths from the provided asset list that are relevant as visual/context references (like character sheets or locations) for the feedback. Do not include raw clip files or audio files."
    )
    reasoning: str = Field(
        description="Brief explanation of why these assets are selected based on characters or locations mentioned."
    )

class PromptResult(BaseModel):
    video_model_prompt: str = Field(
        description="The detailed AI video generation prompt incorporating feedback and preproduction assets."
    )
    explanation: str = Field(
        description="Brief explanation of how the feedback was addressed and which assets were used."
    )


class BatchPromptItem(BaseModel):
    cluster_id: int = Field(
        description="Stable identifier of the input clip cluster."
    )
    selected_assets: List[str] = Field(
        description="Reference asset paths used to generate this clip prompt."
    )
    prompt_format: Literal["plain_text", "json", "production_prompt"] = Field(
        description="Seedance output format selected using the supplied skill rules."
    )
    reference_legend: str = Field(
        description="English upload legend mapping selected images and the original clip to reference labels."
    )
    english_prompt: str = Field(
        description="Detailed English-only Seedance 2.0 prompt with no timeline prompting."
    )
    explanation: str = Field(
        description="How the feedback and selected references were applied."
    )


class BatchPromptResult(BaseModel):
    results: List[BatchPromptItem]


class SelectedAssetItem(BaseModel):
    cluster_id: int = Field(
        description="Stable identifier of the input clip cluster."
    )
    selected_assets: List[str] = Field(
        description="Reference asset paths selected for this cluster. Use only paths from the supplied candidate list."
    )
    reasoning: str = Field(
        description="Brief explanation of why these assets are needed."
    )


class BatchSelectedAssetsResult(BaseModel):
    results: List[SelectedAssetItem]


class InitialFramePromptItem(BaseModel):
    cluster_id: int = Field(
        description="Stable identifier of the input clip cluster."
    )
    initial_frame_prompt: str = Field(
        description="Detailed English prompt for generating the first frame image for this clip."
    )
    explanation: str = Field(
        description="How the initial frame prompt preserves clip continuity and feedback intent."
    )


class BatchInitialFramePromptResult(BaseModel):
    results: List[InitialFramePromptItem]

class AudioInstructionResult(BaseModel):
    audio_instruction: str = Field(
        description="A structured instruction for the voiceover artist, sound editor, or SFX generator."
    )
    explanation: str = Field(
        description="Brief explanation of the audio adjustment."
    )


class QualityCheckResult(BaseModel):
    passed: bool = Field(
        description="True if the prompt passes all quality checks, False otherwise."
    )
    feedback_adherence: str = Field(
        description="Evaluation of whether all feedback items are fully addressed in the prompt."
    )
    clothing_consistency: str = Field(
        description="Evaluation of whether clothing/wardrobe is explicitly described and consistent with the video clip."
    )
    forbidden_terms_found: List[str] = Field(
        description="List of forbidden, negative, or vague terms found in the prompt (e.g., 'fast', 'no flickering', etc.)."
    )
    vague_feedback_resolution: str = Field(
        description="Evaluation of whether vague feedback was successfully clarified/expanded in the prompt."
    )
    suggestions: List[str] = Field(
        description="Specific actionable suggestions to improve the prompt quality."
    )


class RefinedPromptResult(BaseModel):
    english_prompt: str = Field(
        description="The improved and corrected detailed English Seedance 2.0 prompt."
    )
    refinement_explanation: str = Field(
        description="Explanation of how the quality suggestions were addressed."
    )
