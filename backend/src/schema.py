from pydantic import BaseModel, Field
from typing import Optional, Literal

class RuntimeSecretsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    segmind_api_key: Optional[str] = None

class ManualFeedbackRequest(BaseModel):
    clip_used: str
    category: str
    remark: str
    timestamp: Optional[str] = None


class ClipChatRequest(BaseModel):
    clip_index: int
    message: str
    provider: Literal["openai", "gemini"] = "openai"


class ClipChatActionRequest(BaseModel):
    clip_index: int
    action: dict
    message: str = ""
    provider: Literal["openai", "gemini"] = "openai"


class ClipMemoryRequest(BaseModel):
    clip_index: int
    text: str
    scope: Literal["clip", "project"] = "clip"


class GenerateClipVideoRequest(BaseModel):
    clip_index: int
    prompt_version_index: Optional[int] = None
    provider: str = "segmind"
    resolution: Literal["480p", "720p", "1080p", "4k"] = "720p"
    generate_audio: bool = False
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"] = "9:16"
    duration: int = 5


class WorkflowJobRequest(BaseModel):
    feedback_index: int
    provider: Literal["openai", "gemini"] = "openai"
    agent_run_id: Optional[str] = None

class PromptFeedbackCreateRequest(BaseModel):
    clip_index: int
    clip_key: str
    prompt_id: str
    prompt_version_id: str
    rating: Literal["positive", "negative"]
    categories: list[str] = Field(default_factory=list)
    severity: int = 3
    comment: str = ""
    correction: str = ""
    remember_note: str = ""
    create_eval_case: bool = False
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptFeedbackPatchRequest(BaseModel):
    rating: Optional[Literal["positive", "negative"]] = None
    categories: Optional[list[str]] = None
    severity: Optional[int] = None
    comment: Optional[str] = None
    correction: Optional[str] = None
    remember_note: Optional[str] = None
    create_eval_case: Optional[bool] = None
    status: Optional[Literal["open", "approved", "rejected", "resolved"]] = None


class PromptLessonCreateRequest(BaseModel):
    scope: Literal["project", "clip"] = "project"
    clip_key: Optional[str] = None
    category: str = "other"
    lesson: str
    source_feedback_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class PromptLessonPatchRequest(BaseModel):
    scope: Optional[Literal["project", "clip"]] = None
    clip_key: Optional[str] = None
    category: Optional[str] = None
    lesson: Optional[str] = None
    source_feedback_ids: Optional[list[str]] = None
    confidence: Optional[float] = None
    positive_examples: Optional[list[str]] = None
    negative_examples: Optional[list[str]] = None
    archived: Optional[bool] = None


class PromptEvalCasePatchRequest(BaseModel):
    name: Optional[str] = None
    input: Optional[dict] = None
    expected_behavior: Optional[list[str]] = None
    failure_categories: Optional[list[str]] = None
    enabled: Optional[bool] = None


class PromptRevisionRequest(BaseModel):
    clip_index: int
    feedback_ids: list[str] = Field(default_factory=list)
    lesson_ids: list[str] = Field(default_factory=list)
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestRequest(BaseModel):
    provider: Literal["openai", "gemini"] = "openai"


class PromptLessonSuggestionResult(BaseModel):
    lesson: str
    category: str = "other"
    confidence: float = 0.75
    reasoning: str = ""


class ClipSelectionUpdate(BaseModel):
    active_prompt_version_id: Optional[str] = None
    active_generated_video_id: Optional[str] = None
    selected_assets: Optional[list[dict]] = None
    pinned_assets: Optional[list[dict]] = None
    selected_asset_ids: Optional[list[str]] = None
    selected_asset_paths: Optional[list[str]] = None
    pinned_asset_ids: Optional[list[str]] = None
    pinned_asset_paths: Optional[list[str]] = None