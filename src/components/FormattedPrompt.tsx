import type { CSSProperties } from "react";
import { Icon } from "./Icon";

type FormattedPromptProps = {
  text?: string;
  className?: string;
  style?: CSSProperties;
};

export function FormattedPrompt({ text, className, style }: FormattedPromptProps) {
  if (!text) return null;

  // Regex to match section headers (e.g., "Style & Mood:", "1. **Style & Mood:**", etc.)
  const headerRegex = /(?:\d+\.\s*)?\*?\*?(Style\s*&\s*Mood|Narrative\s*Summary|Dynamic\s*Description|Static\s*Description|Quality|Prompt|Explanation|Initial\s*Frame\s*Prompt|Video\s*Model\s*Prompt|Dialogue\s*&\s*Audio|Audio\s*Sync|Characters\s*&\s*Wardrobe|Camera\s*Movement|Transition)\*?\*?:/gi;

  const matches: { index: number; header: string; raw: string }[] = [];
  let match;
  headerRegex.lastIndex = 0;
  while ((match = headerRegex.exec(text)) !== null) {
    matches.push({
      index: match.index,
      header: match[1], // The matched section title (e.g., "Style & Mood")
      raw: match[0],    // The raw matched string (e.g., "**Style & Mood**:")
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
              {trimmed}
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

    // Normalize header name to Title Case
    const normalizedHeader = current.header
      .split(/\s+/)
      .map((word) => {
        if (word.toLowerCase() === "&") return "&";
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      })
      .join(" ");

    sections.push({
      header: normalizedHeader,
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
                // Clean up any remaining markdown bold markers from the content paragraph
                const cleanedPara = trimmed.replace(/\*\*([^*]+)\*\*/g, "$1");
                return cleanedPara ? (
                  <p key={pIdx} className="prompt-paragraph">
                    {cleanedPara}
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
