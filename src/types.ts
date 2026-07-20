export type Provider = "gemini" | "openai";

export type AssetFile = {
  name: string;
  path: string;
  url: string;
  type: "image" | "video" | "audio" | "other";
  size: string;
};

export type FeedbackItem = {
  timestamp?: string;
  category: string;
  remark: string;
  raw_index: number;
};

export type FeedbackGroup = {
  clip_used: string;
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

export type QualityReport = {
  passed?: boolean;
  feedback_adherence?: string;
  clothing_consistency?: string;
  forbidden_terms_found?: string[];
  suggestions?: string[];
};

export type PromptVersion = {
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
  is_dialogue_active?: boolean;
  generate_audio?: boolean;
  ratio?: string;
  duration?: number;
};

export type PromptRecord = PromptVersion & {
  clip_used?: string;
  clip_occurrence?: number | null;
  matched_clip?: string | null;
  latest_error?: string | null;
  history?: PromptVersion[];
};

export type ProjectData = {
  project_name: string;
  timeline: TimelineClip[];
  feedback: FeedbackGroup[];
  assets: Record<string, AssetFile[]>;
  prompts: PromptRecord[];
  sequence_name: string;
  total_duration_tc: string;
  total_duration_s: number;
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
} | null;

export type Toast = {
  id: number;
  message: string;
  type: "info" | "success" | "error";
};

export type ActiveClipChat = {
  clipIndex: number;
} | null;

export type ClipChatAction = {
  type: "execute_workflow" | "prepare_video";
  label: string;
  feedback_index?: number;
};

export type ClipChatMessage = {
  id: string;
  role: "assistant" | "user" | "tool";
  content: string;
  created_at: string;
  metadata?: {
    provider?: Provider;
    actions?: ClipChatAction[];
    saved_memory_ids?: string[];
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
  context: {
    project: {
      name: string;
      sequence_name?: string;
      clip_count: number;
      total_duration_tc?: string;
      total_duration_s?: number;
    };
    clip: TimelineClip;
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
};
