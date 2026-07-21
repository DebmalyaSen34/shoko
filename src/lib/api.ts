import { invoke } from "@tauri-apps/api/core";

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
  const workspaceMarker = "developement/loka/";
  const idx = cleanPath.indexOf(workspaceMarker);
  if (idx !== -1) {
    cleanPath = cleanPath.substring(idx + workspaceMarker.length);
  }
  if (!cleanPath.startsWith("/")) {
    cleanPath = `/${cleanPath}`;
  }
  return apiUrl(cleanPath);
}
