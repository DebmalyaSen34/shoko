import { invoke } from "@tauri-apps/api/core";
import type { ClipState, PromptEvalCase, PromptFeedbackItem, PromptFeedbackPayload, PromptLesson, PromptLessonPayload, PromptLessonSuggestion, PromptRevisionPayload, PromptRevisionResponse, Provider } from "../types";

export let API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function isTauriApp() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function initializeApiBase() {
  if (!isTauriApp()) return API_BASE;

  try {
    const backendBaseUrl = await invoke<string>("backend_base_url");
    API_BASE = backendBaseUrl.replace(/\/$/, "");
  } catch (error) {
    console.error("Failed to initialize Tauri-managed backend", error);
  }

  return API_BASE;
}

export function getApiBase() {
  return API_BASE;
}

export function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

export function staticUrl(path?: string | null) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;

  let cleanPath = path.replace(/\\/g, "/");

  for (const marker of ["/assets/", "assets/", "/data/", "data/"]) {
    const idx = cleanPath.indexOf(marker);
    if (idx !== -1) {
      const prefix = marker.includes("assets") ? "/assets/" : "/data/";
      const start = marker.startsWith("/") ? idx + marker.length : idx + marker.length;
      cleanPath = `${prefix}${cleanPath.slice(start)}`;
      break;
    }
  }

  if (!cleanPath.startsWith("/")) {
    cleanPath = `/${cleanPath}`;
  }
  return apiUrl(cleanPath);
}

export async function listPromptFeedback(projectName: string, clipIndex?: number) {
  const params = typeof clipIndex === "number" ? `?clip_index=${encodeURIComponent(String(clipIndex))}` : "";
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-feedback${params}`));
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{
    schema_version: number;
    project_name: string;
    clip_index?: number | null;
    items: PromptFeedbackItem[];
    summaries: Record<string, unknown>;
  }>;
}

export async function createPromptFeedback(projectName: string, payload: PromptFeedbackPayload) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-feedback`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ item: PromptFeedbackItem; clip_state: ClipState; eval_case?: PromptEvalCase }>;
}

export async function updatePromptFeedback(projectName: string, feedbackId: string, patch: Partial<PromptFeedbackPayload> & { status?: PromptFeedbackItem["status"] }) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-feedback/${encodeURIComponent(feedbackId)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ item: PromptFeedbackItem; clip_state: ClipState }>;
}

export async function listPromptLessons(projectName: string, options: { clipKey?: string; category?: string; includeArchived?: boolean } = {}) {
  const params = new URLSearchParams();
  if (options.clipKey) params.set("clip_key", options.clipKey);
  if (options.category) params.set("category", options.category);
  if (options.includeArchived) params.set("include_archived", "true");
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-lessons${suffix}`));
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{
    schema_version: number;
    project_name: string;
    clip_key?: string | null;
    category?: string | null;
    lessons: PromptLesson[];
  }>;
}

export async function listPromptEvalCases(projectName: string, options: { clipIndex?: number; enabled?: boolean } = {}) {
  const params = new URLSearchParams();
  if (typeof options.clipIndex === "number") params.set("clip_index", String(options.clipIndex));
  if (typeof options.enabled === "boolean") params.set("enabled", String(options.enabled));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-eval-cases${suffix}`));
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{
    schema_version: number;
    project_name: string;
    clip_index?: number | null;
    enabled?: boolean | null;
    cases: PromptEvalCase[];
  }>;
}

export async function createPromptEvalCaseFromFeedback(projectName: string, feedbackId: string) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-feedback/${encodeURIComponent(feedbackId)}/eval-case`), {
    method: "POST",
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ eval_case: PromptEvalCase; clip_state: ClipState }>;
}

export async function updatePromptEvalCase(projectName: string, caseId: string, patch: Partial<PromptEvalCase>) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-eval-cases/${encodeURIComponent(caseId)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ eval_case: PromptEvalCase; clip_state: ClipState }>;
}

export async function revisePromptFromFeedback(projectName: string, promptVersionId: string, payload: PromptRevisionPayload) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompts/${encodeURIComponent(promptVersionId)}/revise-from-feedback`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<PromptRevisionResponse>;
}

export async function createPromptLesson(projectName: string, payload: PromptLessonPayload) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-lessons`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ lesson: PromptLesson; clip_state?: ClipState | null }>;
}

export async function updatePromptLesson(projectName: string, lessonId: string, patch: Partial<PromptLessonPayload> & { archived?: boolean }) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-lessons/${encodeURIComponent(lessonId)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ lesson: PromptLesson; clip_state?: ClipState | null }>;
}

export async function suggestPromptLesson(projectName: string, feedbackId: string, provider: Provider) {
  const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/prompt-feedback/${encodeURIComponent(feedbackId)}/suggest-lesson`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{
    project_name: string;
    feedback_id: string;
    suggestion: PromptLessonSuggestion;
  }>;
}
