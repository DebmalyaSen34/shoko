export type Provider = "gemini" | "openai";

export type AssetFile = {
  asset_id?: string;
  category?: string | null;
  name: string;
  path: string;
  url: string;
  type: "image" | "video" | "audio" | "other";
  size: string;
  role?: string;
  source?: string;
  selected_path?: string;
  reason?: string;
  confidence?: number;
  missing?: boolean;
};

export type FeedbackItem = {
  feedback_item_id?: string;
  timestamp?: string;
  category: string;
  remark: string;
  raw_index: number;
};

export type FeedbackGroup = {
  feedback_id?: string;
  clip_used: string;
  category?: "audio" | "video" | "both" | string;
  clip_occurrence?: number | null;
  feedback_items: FeedbackItem[];
};

export type TimelineClip = {
  clip: string;
  start_tc: string;
  end_tc: string;
  start_s: number;
  end_s: number;
  duration_s: number;
  clip_url?: string | null;
};

export type TimelineFilmstripFrame = {
  offset_s: number;
  url: string;
};

export type TimelineAudioSegment = {
  audio_index?: number;
  clip: string;
  start_tc: string;
  end_tc: string;
  start_s: number;
  end_s: number;
  duration_s: number;
  audio_url?: string | null;
  audio_path?: string | null;
};

export type TimelineWaveformSegment = TimelineAudioSegment & {
  peaks: number[];
  error?: string;
};

export type QualityReport = {
  passed?: boolean;
  feedback_adherence?: string;
  clothing_consistency?: string;
  forbidden_terms_found?: string[];
  suggestions?: string[];
};

export type PromptFeedbackCategory =
  | "missed_feedback"
  | "wrong_visual_detail"
  | "wrong_character_or_wardrobe"
  | "continuity_error"
  | "bad_camera_instruction"
  | "bad_audio_or_dialogue"
  | "unsupported_assumption"
  | "format_error"
  | "too_vague"
  | "too_verbose"
  | "provider_incompatible"
  | "other";

export type PromptFeedbackRating = "positive" | "negative";
export type PromptFeedbackStatus = "open" | "approved" | "rejected" | "resolved";
export type PromptFeedbackDerivedStatus = "unreviewed" | "approved" | "needs_revision" | "rejected";

export type PromptFeedbackSummary = {
  prompt_version_id: string;
  status: PromptFeedbackDerivedStatus;
  total_count: number;
  positive_count: number;
  negative_count: number;
  open_negative_count: number;
  rejected_count: number;
  latest_feedback_at?: string | null;
};

export type PromptFeedbackItem = {
  id: string;
  project_name: string;
  clip_index: number;
  clip_key: string;
  prompt_id: string;
  prompt_version_id: string;
  rating: PromptFeedbackRating;
  categories: PromptFeedbackCategory[];
  severity: number;
  comment: string;
  correction: string;
  remember_note: string;
  create_eval_case: boolean;
  status: PromptFeedbackStatus;
  created_at: string;
  updated_at?: string;
};

export type PromptFeedbackPayload = {
  clip_index: number;
  clip_key: string;
  prompt_id: string;
  prompt_version_id: string;
  rating: PromptFeedbackRating;
  categories: PromptFeedbackCategory[];
  severity: number;
  comment: string;
  correction: string;
  remember_note: string;
  create_eval_case: boolean;
  status?: PromptFeedbackStatus;
};

export type PromptFeedbackTarget = {
  clip_index: number;
  clip_key: string;
  prompt_id: string;
  prompt_version_id: string;
  prompt_version_index?: number | null;
  prompt_label: string;
  prompt_text: string;
  feedback_summary?: PromptFeedbackSummary;
};

export type PromptLessonScope = "project" | "clip";

export type PromptLesson = {
  id: string;
  scope: PromptLessonScope;
  clip_key?: string | null;
  category: string;
  lesson: string;
  source_feedback_ids: string[];
  confidence: number;
  positive_examples: string[];
  negative_examples: string[];
  created_at: string;
  updated_at?: string;
  last_accessed_at?: string | null;
  access_count?: number;
  archived: boolean;
  relevance_score?: number;
  relevance_reasons?: string[];
};

export type PromptLessonPayload = {
  scope: PromptLessonScope;
  clip_key?: string | null;
  category: string;
  lesson: string;
  source_feedback_ids: string[];
  confidence: number;
  positive_examples: string[];
  negative_examples: string[];
};

export type PromptLessonSuggestion = {
  lesson: string;
  category: string;
  confidence: number;
  reasoning: string;
  source_feedback_id?: string;
};

export type LearningState = {
  project: PromptLesson[];
  clip: PromptLesson[];
  relevant: PromptLesson[];
};

export type PromptVersion = {
  prompt_id?: string;
  prompt_version_id?: string;
  version_index?: number;
  is_latest?: boolean;
  clip_index?: number;
  timestamp?: string;
  provider?: string;
  video_provider?: string;
  segmind_model?: string;
  segmind_payload_status?: "ready" | "skipped" | "failed" | string;
  segmind_payload?: Record<string, unknown>;
  segmind_prompt?: string;
  segmind_first_frame_url?: string | null;
  segmind_reference_images?: string[];
  segmind_reference_videos?: string[];
  segmind_reference_audios?: string[];
  segmind_payload_error?: string | null;
  video_model_prompt?: string;
  selected_assets?: string[];
  explanation?: string;
  quality_report?: QualityReport;
  initial_frame_image_path?: string;
  initial_frame_prompt?: string;
  clip_frame_paths?: string[];
  referenced_frames?: ReferencedFrameState[];
  referenced_frame_paths?: string[];
  referenced_frame_labels?: string[];
  audio_used?: string | null;
  audio_path?: string | null;
  audio_url?: string | null;
  audio_reference_path?: string | null;
  trimmed_audio_path?: string | null;
  audio_trim_start_s?: number | null;
  audio_trim_end_s?: number | null;
  audio_trim_duration_s?: number | null;
  audio_trim_source?: string | null;
  audio_trim_error?: string | null;
  audio_transcript?: string;
  dialogue_text?: string;
  dialogue_language?: string;
  dialogue_extraction_reasoning?: string;
  is_dialogue_active?: boolean;
  generate_audio?: boolean;
  ratio?: string;
  duration?: number;
  generated_videos?: GeneratedVideo[];
  video_generation_attempts?: VideoGenerationAttempt[];
  latest_video_generation_attempt?: VideoGenerationAttempt;
  latest_generated_video?: GeneratedVideo;
  feedback_summary?: PromptFeedbackSummary;
};

export type PromptRecord = PromptVersion & {
  clip_used?: string;
  clip_occurrence?: number | null;
  matched_clip?: string | null;
  latest_error?: string | null;
  history?: PromptVersion[];
};

export type GeneratedVideo = {
  generated_video_id?: string;
  version: number;
  timestamp?: string;
  provider?: string;
  model?: string;
  request_id?: string | null;
  clip_used?: string | null;
  clip_occurrence?: number | null;
  prompt_version_index?: number;
  prompt_timestamp?: string;
  path: string;
  url: string;
  label?: string;
  source_output_url?: string | null;
  duration?: number;
  resolution?: VideoResolution;
  generate_audio?: boolean;
  ratio?: string;
  storage_root?: string;
};

export type VideoGenerationAttempt = {
  timestamp?: string;
  status: "succeeded" | "recovery_failed" | string;
  provider?: string;
  model?: string;
  request_id?: string | null;
  clip_used?: string | null;
  clip_occurrence?: number | null;
  prompt_version_index?: number;
  prompt_timestamp?: string;
  path?: string;
  url?: string;
  error?: string;
  error_status_code?: number | null;
  recoverable?: boolean;
  duration?: number;
  resolution?: VideoResolution;
  generate_audio?: boolean;
  ratio?: string;
};

export type VideoResolution = "480p" | "720p" | "1080p" | "4k";
export type VideoAspectRatio = "16:9" | "9:16" | "1:1" | "4:3" | "3:4" | "21:9" | "adaptive";

export type GenerateVideoOptions = {
  resolution: VideoResolution;
  generate_audio: boolean;
  aspect_ratio: VideoAspectRatio;
  duration: number;
};

export type ProjectData = {
  project_name: string;
  timeline: TimelineClip[];
  audio_timeline?: {
    dedicated_audio_tracks?: TimelineAudioSegment[];
    embedded_video_audio?: TimelineAudioSegment[];
  };
  feedback: FeedbackGroup[];
  assets: Record<string, AssetFile[]>;
  prompts: PromptRecord[];
  sequence_name: string;
  total_duration_tc: string;
  total_duration_s: number;
  fps?: number;
  frame_size?: string;
};

export type ProjectImportResult = {
  project_name: string;
  timeline_path?: string;
};

export type PreviewState = {
  file: AssetFile;
} | null;

export type ResultState = {
  prompt: string;
  initialPrompt?: string;
  explanation: string;
  initialImage?: string;
  assets: string[];
  quality?: QualityReport;
  clipFrames?: string[];
  audioTrim?: {
    start?: number | null;
    end?: number | null;
    duration?: number | null;
    source?: string | null;
    path?: string | null;
    error?: string | null;
  };
  generatedVideos?: GeneratedVideo[];
} | null;

export type Toast = {
  id: number;
  message: string;
  type: "info" | "success" | "error";
};

export type RuntimeSecretKeyStatus = {
  configured: boolean;
  source: "os_environment" | "app_or_dev_env_file" | "missing" | string;
  locked_by_os_env: boolean;
  can_update: boolean;
};

export type RuntimeConfig = {
  app_storage_dir: string;
  data_dir: string;
  assets_dir: string;
  loaded_env_files: string[];
  secrets: {
    config_env_path: string;
    keys: Record<"OPENAI_API_KEY" | "GEMINI_API_KEY" | "SEGMIND_API_KEY", RuntimeSecretKeyStatus>;
  };
};

export type ActiveClipChat = {
  clipIndex: number;
} | null;

export type ClipChatAction = {
  type:
    | "execute_workflow"
    | "prepare_video"
    | "generate_video"
    | "send_message"
    | "set_active_prompt_version"
    | "attach_asset"
    | "detach_asset"
    | "mark_feedback_resolved"
    | "add_reference_frame";
  label: string;
  feedback_index?: number;
  autonomous?: boolean;
  continuity_reference?: "previous_clip_last_frame" | string;
  prompt?: string;
  prompt_version_id?: string | null;
  active_generated_video_id?: string | null;
  asset_id?: string | null;
  asset_path?: string | null;
  role?: string;
  reason?: string;
  confidence?: number;
  timestamp?: string;
  attach_to?: string;
};

export type ClipAgentRunStep = {
  id: string;
  label: string;
  status: "pending" | "completed" | "awaiting_approval" | "dispatch_ready" | "blocked" | "failed" | string;
  tool?: string;
  requires_approval?: boolean;
  action?: ClipChatAction | null;
};

export type ClipAgentRun = {
  id: string;
  goal: string;
  clip_index: number;
  clip_key: string;
  status: "planned" | "ready" | "awaiting_approval" | "failed" | "completed" | string;
  intent: string;
  confidence: number;
  autonomy_level: "manual" | "suggest" | "approval_required" | "full_autopilot" | string;
  approval_required: boolean;
  plan_steps: ClipAgentRunStep[];
  required_actions: ClipChatAction[];
  available_actions: ClipChatAction[];
  suggested_actions: ClipChatAction[];
  tool_results?: Record<string, unknown>[];
  errors?: string[];
  self_evaluation?: Record<string, unknown>;
  freshness?: {
    is_stale: boolean;
    stale_reasons: string[];
    captured_state_hash?: string | null;
    current_state_hash?: string | null;
    captured_at?: string | null;
  };
  created_at: string;
  updated_at: string;
};

export type ProjectJob = {
  id: string;
  type: "workflow" | "generate_video" | string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "cancelling" | string;
  project_name: string;
  clip_index?: number | null;
  feedback_index?: number | null;
  provider?: string;
  payload?: Record<string, unknown>;
  agent_run_id?: string | null;
  logs?: { timestamp: string; message: string }[];
  result?: Record<string, unknown> | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  cancel_requested?: boolean;
};

export type ProjectEvent = {
  schema_version: number;
  id: string;
  sequence: number;
  type: string;
  actor: string;
  project_name: string;
  clip_index?: number | null;
  clip_key?: string | null;
  entity?: string | null;
  entity_id?: string | null;
  payload: Record<string, unknown>;
  previous_event_hash?: string | null;
  event_hash: string;
  created_at: string;
};

export type ReferencedFrameState = {
  reference_id: string;
  timestamp?: string | null;
  reason?: string;
  frame_path?: string;
  url?: string | null;
  clip_used?: string | null;
  offset_s?: number | null;
  usage?: string;
};

export type ClipAssetReference = {
  asset_id?: string | null;
  path?: string | null;
  role: string;
  reason: string;
  confidence: number;
  source: string;
};

export type ClipSelectionState = {
  clip_key: string;
  active_prompt_version_id?: string | null;
  resolved_prompt_version_id?: string | null;
  active_generated_video_id?: string | null;
  resolved_generated_video_id?: string | null;
  selected_assets: ClipAssetReference[];
  selected_asset_ids: string[];
  selected_asset_paths: string[];
  pinned_assets: ClipAssetReference[];
  pinned_asset_ids: string[];
  pinned_asset_paths: string[];
  resolved_assets: AssetFile[];
  selection_source: "persisted" | "default_latest" | string;
  stale_reasons: string[];
  updated_at?: string | null;
};

export type ClipState = {
  schema_version: number;
  project_name: string;
  clip_index: number;
  clip_key: string;
  clip_state_id: string;
  updated_at: string;
  project: {
    name: string;
    sequence_name?: string;
    clip_count: number;
    total_duration_tc?: string;
    total_duration_s?: number;
  };
  timeline: {
    clip: TimelineClip;
    adjacent_clips: {
      previous?: TimelineClip | null;
      next?: TimelineClip | null;
    };
  };
  feedback_state?: FeedbackGroup | null;
  active_prompt: {
    prompt?: PromptRecord | null;
    prompt_id?: string | null;
    version?: PromptVersion | null;
    version_id?: string | null;
    version_index?: number | null;
    versions: PromptVersion[];
    prompt_ready: boolean;
  };
  asset_state: {
    available_assets: Record<string, AssetFile[]>;
    selected_assets: AssetFile[];
    pinned_assets: AssetFile[];
    referenced_frames: ReferencedFrameState[];
    missing_assets: AssetFile[];
  };
  video_state: {
    generated_videos: GeneratedVideo[];
    latest_video?: GeneratedVideo | null;
    active_video?: GeneratedVideo | null;
    active_video_id?: string | null;
    generation_attempts: VideoGenerationAttempt[];
    latest_attempt?: VideoGenerationAttempt | null;
  };
  selection_state: ClipSelectionState;
  analysis_state: {
    clip_context?: Record<string, unknown> | null;
    clip_context_ready: boolean;
  };
  memory_state: ClipChatMemory;
  learning_state: LearningState;
  agent_state: {
    recent_runs: ClipAgentRun[];
    pending_actions: ClipChatAction[];
  };
  job_state: {
    recent_jobs: ProjectJob[];
    active_jobs: ProjectJob[];
  };
  freshness: {
    source_files: Record<string, string>;
    source_mtimes?: Record<string, string | null>;
    derived_from: string[];
    fingerprints?: Record<string, string>;
    state_hash?: string;
    stale_reasons: string[];
  };
};

export type ClipChatMedia = {
  type: "image" | "video" | "audio" | "other";
  label: string;
  source: string;
  name: string;
  path?: string | null;
  url: string;
  size?: string;
  thumbnail_url?: string | null;
};

export type ClipChatMessage = {
  id: string;
  role: "assistant" | "user" | "tool";
  content: string;
  created_at: string;
  metadata?: {
    provider?: Provider;
    actions?: ClipChatAction[];
    media?: ClipChatMedia[];
    saved_memory_ids?: string[];
    tool_results?: Record<string, unknown>[];
    agent_run?: ClipAgentRun;
    clip_state_id?: string;
    prompt_version_id?: string | null;
    kind?: string;
  };
};

export type ClipMemoryItem = {
  id: string;
  scope: "clip" | "project";
  clip_key?: string | null;
  text: string;
  created_at: string;
  updated_at?: string;
  last_accessed_at?: string | null;
  access_count?: number;
  confidence?: number;
  tags?: string[];
  relevance_score?: number;
  relevance_reasons?: string[];
  source: string;
};

export type ClipChatMemory = {
  project: ClipMemoryItem[];
  clip: ClipMemoryItem[];
  relevant: ClipMemoryItem[];
};

export type ClipChatSnapshot = {
  project_name: string;
  clip_index: number;
  clip_key: string;
  messages: ClipChatMessage[];
  memory: ClipChatMemory;
  agent_runs: ClipAgentRun[];
  context: {
    project: {
      name: string;
      sequence_name?: string;
      clip_count: number;
      total_duration_tc?: string;
      total_duration_s?: number;
    };
    clip: TimelineClip;
    clip_state: ClipState;
    adjacent_clips: {
      previous?: TimelineClip | null;
      next?: TimelineClip | null;
    };
    feedback_count: number;
    selected_asset_count: number;
    prompt_ready: boolean;
  };
};

export type ClipChatResponse = {
  project_name: string;
  clip_index: number;
  clip_key: string;
  messages: ClipChatMessage[];
  assistant_message: ClipChatMessage;
  memory: ClipChatMemory;
  suggested_actions: ClipChatAction[];
  agent_run: ClipAgentRun;
  clip_state: ClipState;
};
