import { invoke } from "@tauri-apps/api/core";
import type { ClipState, PromptFeedbackItem, PromptFeedbackPayload } from "../types";

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
  return response.json() as Promise<{ item: PromptFeedbackItem; clip_state: ClipState }>;
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
