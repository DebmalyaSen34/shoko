import type { AssetFile } from "../types";

export function Icon({ name, className = "" }: { name: string; className?: string }) {
  const paths: Record<string, string> = {
    board: "M4 5h16v12H4z M8 21h8 M12 17v4",
    box: "M4 8l8-4 8 4-8 4-8-4z M4 8v8l8 4 8-4V8 M12 12v8",
    search: "M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14z M20 20l-4-4",
    timeline: "M4 6h5 M15 6h5 M9 6a3 3 0 1 0 6 0 3 3 0 0 0-6 0z M4 18h5 M15 18h5 M9 18a3 3 0 1 0 6 0 3 3 0 0 0-6 0z",
    clock: "M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20z M12 6v6l4 2",
    chevronLeft: "M15 18l-6-6 6-6",
    chevronRight: "M9 18l6-6-6-6",
    chevronDown: "M6 9l6 6 6-6",
    plus: "M12 5v14 M5 12h14",
    minus: "M5 12h14",
    expand: "M8 3H3v5 M16 3h5v5 M3 16v5h5 M21 16v5h-5 M3 3l6 6 M21 3l-6 6 M3 21l6-6 M21 21l-6-6",
    play: "M8 5v14l11-7-11-7z",
    image: "M4 5h16v14H4z M8 13l3-3 3 4 2-2 4 5 M8 8h.01",
    video: "M4 6h11v12H4z M15 10l5-3v10l-5-3z",
    audio: "M9 18V6l10-2v12 M9 18a3 3 0 1 1-2-2.83 M19 16a3 3 0 1 1-2-2.83",
    file: "M6 3h8l4 4v14H6z M14 3v5h5",
    comments: "M4 5h16v10H8l-4 4z",
    chat: "M5 5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H9l-5 4v-4H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z M8 9h8 M8 13h5",
    memory: "M12 3a4 4 0 0 0-4 4v1H7a3 3 0 0 0 0 6h1v1a4 4 0 0 0 8 0v-1h1a3 3 0 0 0 0-6h-1V7a4 4 0 0 0-4-4z M8 8h8 M8 14h8",
    send: "M22 2L11 13 M22 2l-7 20-4-9-9-4 20-7z",
    warning: "M12 3l10 18H2z M12 9v5 M12 17h.01",
    settings: "M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z M19.4 15a1.7 1.7 0 0 0 .34 1.88l.04.05a2 2 0 0 1-2.83 2.83l-.05-.04A1.7 1.7 0 0 0 15 19.4a1.7 1.7 0 0 0-1 1.55V21a2 2 0 0 1-4 0v-.05A1.7 1.7 0 0 0 9 19.4a1.7 1.7 0 0 0-1.88.34l-.05.04a2 2 0 0 1-2.83-2.83l.04-.05A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.55-1H3a2 2 0 0 1 0-4h.05A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.34-1.88l-.04-.05a2 2 0 0 1 2.83-2.83l.05.04A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.55V3a2 2 0 0 1 4 0v.05a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.88-.34l.05-.04a2 2 0 0 1 2.83 2.83l-.04.05A1.7 1.7 0 0 0 19.4 9a1.7 1.7 0 0 0 1.55 1H21a2 2 0 0 1 0 4h-.05A1.7 1.7 0 0 0 19.4 15z",
    key: "M21 2l-2 2 M15 8l-8 8 M7 16l-2 2v3h3l2-2 M14 9a4 4 0 1 1 1 1",
    lock: "M7 11V8a5 5 0 0 1 10 0v3 M6 11h12v10H6z",
    close: "M6 6l12 12 M18 6L6 18",
    copy: "M8 8h11v11H8z M5 16H4V4h12v1",
    check: "M20 6L9 17l-5-5",
    refresh: "M20 12a8 8 0 0 1-14 5 M4 12a8 8 0 0 1 14-5 M18 3v4h-4 M6 21v-4h4",
    magic: "M5 19L19 5 M14 5h5v5 M4 5l1 2 2 1-2 1-1 2-1-2-2-1 2-1z",
    link: "M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1 M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1",
  };

  return (
    <svg aria-hidden="true" className={`icon ${className}`} viewBox="0 0 24 24">
      <path d={paths[name] || paths.file} />
    </svg>
  );
}

export function categoryIcon(category: string) {
  const name = category.toLowerCase();
  if (name.includes("audio")) return "audio";
  if (name.includes("reference") || name.includes("style")) return "image";
  if (name.includes("clip")) return "video";
  return "box";
}

export function fileIcon(type: AssetFile["type"]) {
  if (type === "image") return "image";
  if (type === "video") return "video";
  if (type === "audio") return "audio";
  return "file";
}
