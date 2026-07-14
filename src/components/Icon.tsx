import type { AssetFile } from "../types";

export function Icon({ name }: { name: string }) {
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
    warning: "M12 3l10 18H2z M12 9v5 M12 17h.01",
    close: "M6 6l12 12 M18 6L6 18",
    copy: "M8 8h11v11H8z M5 16H4V4h12v1",
    check: "M20 6L9 17l-5-5",
    refresh: "M20 12a8 8 0 0 1-14 5 M4 12a8 8 0 0 1 14-5 M18 3v4h-4 M6 21v-4h4",
    magic: "M5 19L19 5 M14 5h5v5 M4 5l1 2 2 1-2 1-1 2-1-2-2-1 2-1z",
    link: "M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1 M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1",
  };

  return (
    <svg aria-hidden="true" className="icon" viewBox="0 0 24 24">
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
