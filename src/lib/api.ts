export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

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
