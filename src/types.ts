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
  video_model_prompt?: string;
  selected_assets?: string[];
  explanation?: string;
  quality_report?: QualityReport;
  initial_frame_image_path?: string;
  initial_frame_prompt?: string;
};

export type PromptRecord = PromptVersion & {
  clip_used?: string;
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
} | null;

export type Toast = {
  id: number;
  message: string;
  type: "info" | "success" | "error";
};
