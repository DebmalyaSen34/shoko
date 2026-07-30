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
  const versions = prompt.history?.length ? [...prompt.history] : [];
  if (prompt.video_model_prompt) {
    const topLevelVersion = { ...prompt };
    delete topLevelVersion.history;
    const lastVersion = versions[versions.length - 1];
    if (!lastVersion || promptVersionSignature(lastVersion) !== promptVersionSignature(topLevelVersion)) {
      versions.push(topLevelVersion);
    } else {
      versions[versions.length - 1] = { ...lastVersion, ...topLevelVersion };
    }
  }
  return versions;
}

function promptVersionSignature(version: PromptVersion) {
  return JSON.stringify({
    prompt: (version.video_model_prompt || "").trim(),
    initialPrompt: (version.initial_frame_prompt || "").trim(),
    initialImage: version.initial_frame_image_path || "",
    assets: version.selected_assets || [],
    referencedFrames: version.referenced_frames || [],
    referencedFramePaths: version.referenced_frame_paths || [],
  });
}

export function versionLabel(version: PromptVersion, index: number) {
  const date = version.timestamp ? new Date(version.timestamp) : new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `Version ${index + 1} (${(version.provider || "AI").toUpperCase()} - ${time})`;
}

export function formatSeconds(seconds?: number | null): string {
  if (typeof seconds !== "number" || isNaN(seconds)) return "--";
  if (seconds < 60) {
    return `${Number(seconds.toFixed(1))}s`;
  }
  const mins = Math.floor(seconds / 60);
  const remSecs = Number((seconds % 60).toFixed(1));
  return `${mins}m ${remSecs}s`;
}

export function formatTimecode(tc?: string | null): string {
  if (!tc) return "No Timecode";
  const trimmed = tc.trim();
  if (!trimmed) return "No Timecode";

  if (trimmed.includes(" - ")) {
    return trimmed.split(" - ").map((part) => formatTimecode(part.trim())).join(" – ");
  }
  if (trimmed.includes("->")) {
    return trimmed.split("->").map((part) => formatTimecode(part.trim())).join(" – ");
  }

  const parts = trimmed.split(":");
  if (parts.length >= 4) {
    const h = parseInt(parts[0], 10) || 0;
    const m = parts[1] || "00";
    const s = parts[2] || "00";
    if (h > 0) return `${h}:${m}:${s}`;
    return `${m}:${s}`;
  } else if (parts.length === 3) {
    const h = parseInt(parts[0], 10) || 0;
    const m = parts[1] || "00";
    const s = parts[2] ? parts[2].split(".")[0] : "00";
    if (h > 0) return `${h}:${m}:${s}`;
    return `${m}:${s}`;
  } else if (parts.length === 2) {
    return trimmed;
  }

  const num = parseFloat(trimmed);
  if (!isNaN(num) && /^\d+(\.\d+)?$/.test(trimmed)) {
    return formatSeconds(num);
  }

  return trimmed;
}
