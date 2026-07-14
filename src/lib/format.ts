import type { PromptRecord, PromptVersion } from "../types";

export function formatProjectName(name: string) {
  return name.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function basename(path: string) {
  return path.split(/[/\\]/).pop() || path;
}

export function clipBasename(path?: string) {
  return basename((path || "").replace(/\\/g, "/"));
}

export function getVersions(prompt?: PromptRecord | null) {
  if (!prompt) return [];
  if (prompt.history?.length) return prompt.history;
  if (prompt.video_model_prompt) return [prompt];
  return [];
}

export function versionLabel(version: PromptVersion, index: number) {
  const date = version.timestamp ? new Date(version.timestamp) : new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `Version ${index + 1} (${(version.provider || "AI").toUpperCase()} - ${time})`;
}
