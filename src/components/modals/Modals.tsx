import type { PreviewState, ResultState } from "../../types";
import type { ReactNode } from "react";
import { staticUrl } from "../../lib/api";
import { basename } from "../../lib/format";
import { Icon } from "../Icon";
import { FormattedPrompt } from "../FormattedPrompt";

function getAssetUrl(path: string) {
  if (!path) return "";
  let cleanPath = path.replace(/\\/g, "/");
  const assetsIdx = cleanPath.indexOf("/assets/");
  if (assetsIdx !== -1) {
    cleanPath = cleanPath.substring(assetsIdx);
  } else {
    const assetsIdx2 = cleanPath.indexOf("assets/");
    if (assetsIdx2 !== -1) {
      cleanPath = "/" + cleanPath.substring(assetsIdx2);
    }
  }
  return staticUrl(cleanPath);
}

function getDataUrl(path: string) {
  if (!path) return "";
  let cleanPath = path.replace(/\\/g, "/");
  const dataIdx = cleanPath.indexOf("/data/");
  if (dataIdx !== -1) {
    cleanPath = cleanPath.substring(dataIdx);
  } else {
    const dataIdx2 = cleanPath.indexOf("data/");
    if (dataIdx2 !== -1) {
      cleanPath = "/" + cleanPath.substring(dataIdx2);
    }
  }
  return staticUrl(cleanPath);
}

function getReferenceUrl(path: string) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.replace(/\\/g, "/");
  if (normalized.includes("/assets/")) return getAssetUrl(normalized);
  return getDataUrl(normalized);
}

export function PreviewModal({ preview, onClose }: { preview: NonNullable<PreviewState>; onClose: () => void }) {
  const { file } = preview;

  return (
    <div className="modal active">
      <div className="modal-content glass-card">
        <div className="modal-header">
          <h3>{file.name}</h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body">
          {file.type === "image" && <img src={staticUrl(file.url)} alt={file.name} />}
          {file.type === "video" && <video src={staticUrl(file.url)} controls autoPlay playsInline preload="auto" />}
          {file.type === "audio" && (
            <div className="audio-preview">
              <Icon name="audio" />
              <audio src={staticUrl(file.url)} controls autoPlay />
            </div>
          )}
          {file.type === "other" && (
            <div className="empty-inline">
              <Icon name="file" />
              <p>No viewer available for this file extension.</p>
              <a href={staticUrl(file.url)} download className="premium-btn">
                Download Asset
              </a>
            </div>
          )}
        </div>
        <div className="modal-footer">
          Path: {file.path} | Size: {file.size}
        </div>
      </div>
    </div>
  );
}

export function ErrorModal({ errorLog, onClose }: { errorLog: string; onClose: () => void }) {
  return (
    <div className="modal active">
      <div className="modal-content glass-card error-modal-content">
        <div className="modal-header error-header">
          <h3>
            <Icon name="warning" /> Workflow Execution Failed
          </h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body error-body">
          <p>The pipeline run encountered an error. See the console logs below:</p>
          <pre className="error-log-area">{errorLog}</pre>
        </div>
        <div className="modal-footer modal-actions">
          <button className="premium-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export function ResultModal({ result, onClose, onCopy }: { result: NonNullable<ResultState>; onClose: () => void; onCopy: () => void }) {
  const hasAudioReference = Boolean(result.audioTrim?.path || result.audioTrim?.error);

  return (
    <div className="modal active">
      <div className="modal-content glass-card result-modal-content" style={{ maxWidth: "1000px", width: "94%" }}>
        <div className="modal-header">
          <h3>
            <Icon name="timeline" /> Workflow Result Verification
          </h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body result-modal-body" style={{ maxHeight: "75vh", overflowY: "auto" }}>
          <div className="result-content-layout">
            <div className="result-col" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
              <div className="result-section" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                <label className="result-label">
                  <Icon name="file" /> Generated Video Model Prompt
                </label>
                <div className="prompt-box-wrapper" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                  <FormattedPrompt
                    className="monospace-box"
                    style={{ flex: 1, maxHeight: "none", height: result.initialPrompt ? "220px" : "400px", fontSize: "15px", lineHeight: "1.7" }}
                    text={result.prompt}
                  />
                  <button className="icon-btn copy-btn" title="Copy Prompt" onClick={onCopy}>
                    <Icon name="copy" />
                  </button>
                </div>
              </div>
              {result.initialPrompt && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="image" /> Initial Frame Prompt
                  </label>
                  <pre className="monospace-box" style={{ height: "110px", maxHeight: "110px", fontSize: "14px", lineHeight: "1.6" }}>{result.initialPrompt}</pre>
                </div>
              )}
              {result.initialImage && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="image" /> Generated Initial Frame Image
                  </label>
                  <div className="initial-image-preview-wrapper" style={{ maxHeight: "200px", overflow: "hidden", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
                    <img src={result.initialImage} alt="Initial Frame" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                  </div>
                </div>
              )}
            </div>

            <div className="result-col">
              <div className="result-section reference-panel">
                <label className="result-label">
                  <Icon name="box" /> References
                </label>
                {result.assets.length === 0 && !result.clipFrames?.length && !hasAudioReference ? (
                  <div className="muted-small">No references were attached to this prompt.</div>
                ) : (
                  <>
                    {result.assets.length > 0 && (
                      <ReferenceGroup title="Context Assets" icon="box">
                        <div className="reference-grid">
                          {result.assets.map((asset) => (
                            <ReferenceImageCard
                              key={asset}
                              title={basename(asset)}
                              badge="Asset"
                              badgeClass="uploaded"
                              url={getAssetUrl(asset)}
                              fallbackIcon="file"
                            />
                          ))}
                        </div>
                      </ReferenceGroup>
                    )}

                    {result.clipFrames && result.clipFrames.length > 0 && (
                      <ReferenceGroup title="Clip Frames" icon="image">
                        <div className="reference-grid">
                          {result.clipFrames.map((framePath, idx) => (
                            <ReferenceImageCard
                              key={framePath}
                              title={`Frame ${idx + 1}`}
                              badge="Frame"
                              badgeClass="extracted"
                              url={getDataUrl(framePath)}
                              fallbackIcon="video"
                            />
                          ))}
                        </div>
                      </ReferenceGroup>
                    )}

                    {hasAudioReference && (
                      <ReferenceGroup title="Audio Reference" icon="audio">
                        <div className="audio-reference-card">
                          <div className="audio-reference-icon">
                            <Icon name="audio" />
                          </div>
                          <div className="audio-reference-content">
                            <div className="audio-reference-title">
                              {result.audioTrim?.path ? basename(result.audioTrim.path) : "Audio trim unavailable"}
                            </div>
                            {result.audioTrim?.path && (
                              <div className="audio-reference-meta">
                                {formatSeconds(result.audioTrim.start)} → {formatSeconds(result.audioTrim.end)}
                                {typeof result.audioTrim.duration === "number" ? ` · ${result.audioTrim.duration.toFixed(3)}s` : ""}
                              </div>
                            )}
                            {result.audioTrim?.path && (
                              <audio controls src={getReferenceUrl(result.audioTrim.path)} />
                            )}
                            {result.audioTrim?.error && (
                              <div className="audio-reference-error">{result.audioTrim.error}</div>
                            )}
                          </div>
                        </div>
                      </ReferenceGroup>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="modal-footer modal-actions">
          <button className="premium-btn" onClick={onClose}>
            <Icon name="check" /> Approve & Proceed
          </button>
        </div>
      </div>
    </div>
  );
}

function ReferenceGroup({ title, icon, children }: { title: string; icon: "audio" | "box" | "image"; children: ReactNode }) {
  return (
    <section className="reference-group">
      <div className="reference-group-title">
        <Icon name={icon} /> {title}
      </div>
      {children}
    </section>
  );
}

function ReferenceImageCard({
  title,
  badge,
  badgeClass,
  url,
  fallbackIcon,
}: {
  title: string;
  badge: string;
  badgeClass: "uploaded" | "extracted";
  url: string;
  fallbackIcon: "file" | "video";
}) {
  const isImg = /\.(png|jpe?g|webp|gif)(?:[?#].*)?$/i.test(url);
  return (
    <div className={`reference-card ${badgeClass === "uploaded" ? "uploaded-asset-card" : "extracted-frame-card"}`}>
      <div className="reference-thumb">
        <div className={`asset-card-badge ${badgeClass}`}>
          <Icon name={badgeClass === "uploaded" ? "box" : "video"} /> {badge}
        </div>
        {isImg ? (
          <img src={url} alt={title} />
        ) : (
          <div className="reference-fallback">
            <Icon name={fallbackIcon} />
          </div>
        )}
      </div>
      <div className="reference-name" title={title}>{title}</div>
    </div>
  );
}

function formatSeconds(value?: number | null) {
  return typeof value === "number" ? `${value.toFixed(3)}s` : "--";
}
