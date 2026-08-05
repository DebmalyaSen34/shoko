import type { CSSProperties } from "react";
import { Icon } from "./Icon";

type FormattedPromptProps = {
  text?: string;
  className?: string;
  style?: CSSProperties;
};

const CANONICAL_PROMPT_HEADERS = [
  "REFERENCE IMAGE MAP",
  "STYLE & LOOK",
  "REFERENCE USE",
  "SUBJECTS & CONTINUITY",
  "FEEDBACK DIRECTIVES",
  "SCENE ACTION",
  "CAMERA & MOTION",
  "AUDIO",
  "QUALITY & CONSTRAINTS",
];

const LEGACY_PROMPT_HEADERS = [
  "Style & Mood",
  "Narrative Summary",
  "Dynamic Description",
  "Static Description",
  "Quality",
  "Prompt",
  "Explanation",
  "Initial Frame Prompt",
  "Video Model Prompt",
  "Dialogue & Audio",
  "Audio Sync",
  "Characters & Wardrobe",
  "Camera Movement",
  "Transition",
];

const PROMPT_HEADERS = [...CANONICAL_PROMPT_HEADERS, ...LEGACY_PROMPT_HEADERS];

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function displayHeader(header: string) {
  const canonical = CANONICAL_PROMPT_HEADERS.find((candidate) => candidate === header.toUpperCase());
  if (canonical) {
    return canonical
      .toLowerCase()
      .split(" ")
      .map((word) => (word === "&" ? word : word.charAt(0).toUpperCase() + word.slice(1)))
      .join(" ");
  }

  return header
    .split(/\s+/)
    .map((word) => {
      if (word.toLowerCase() === "&") return "&";
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(" ");
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    const bold = part.match(/^\*\*([^*]+)\*\*$/);
    if (bold) {
      return <strong key={index}>{bold[1]}</strong>;
    }
    return part;
  });
}

export function FormattedPrompt({ text, className, style }: FormattedPromptProps) {
  if (!text) return null;

  const headerPattern = PROMPT_HEADERS.map(escapeRegex).join("|");
  const headerRegex = new RegExp(`(?:^|\\n)(?:\\d+\\.\\s*)?\\*?\\*?(${headerPattern})\\*?\\*?:`, "gi");

  const matches: { index: number; header: string; raw: string }[] = [];
  let match;
  headerRegex.lastIndex = 0;
  while ((match = headerRegex.exec(text)) !== null) {
    const lineBreakOffset = match[0].startsWith("\n") ? 1 : 0;
    matches.push({
      index: match.index + lineBreakOffset,
      header: match[1], // The matched section title (e.g., "Style & Mood")
      raw: match[0].slice(lineBreakOffset),    // The raw matched string (e.g., "**Style & Mood**:")
    });
  }

  // Fallback if no structured section headers are found
  if (matches.length === 0) {
    return (
      <div className={`formatted-prompt-fallback ${className || ""}`} style={style}>
        {text.split("\n").map((para, i) => {
          const trimmed = para.trim();
          return trimmed ? (
            <p key={i} className="prompt-paragraph">
              {renderInlineMarkdown(trimmed)}
            </p>
          ) : null;
        })}
      </div>
    );
  }

  const sections: { header: string; content: string }[] = [];
  const firstMatchIdx = matches[0].index;
  if (firstMatchIdx > 0) {
    const intro = text.substring(0, firstMatchIdx).trim();
    if (intro) {
      sections.push({ header: "", content: intro });
    }
  }

  for (let i = 0; i < matches.length; i++) {
    const current = matches[i];
    const next = matches[i + 1];
    const startIdx = current.index + current.raw.length;
    const endIdx = next ? next.index : text.length;
    const content = text.substring(startIdx, endIdx).trim();

    sections.push({
      header: displayHeader(current.header),
      content,
    });
  }

  return (
    <div className={`formatted-prompt ${className || ""}`} style={style}>
      {sections.map((section, idx) => {
        if (!section.header) {
          return (
            <p key={idx} className="prompt-intro-paragraph">
              {section.content}
            </p>
          );
        }

        return (
          <details key={idx} className="prompt-section" open>
            <summary className="prompt-section-header">
              <span className="header-text">{section.header}</span>
              <span className="details-toggle-icon">
                <Icon name="chevron-down" />
              </span>
            </summary>
            <div className="prompt-section-content">
              {section.content.split("\n").map((para, pIdx) => {
                const trimmed = para.trim();
                return trimmed ? (
                  <p key={pIdx} className="prompt-paragraph">
                    {renderInlineMarkdown(trimmed)}
                  </p>
                ) : null;
              })}
            </div>
          </details>
        );
      })}
    </div>
  );
}
