import type { PreviewState, ResultState } from "../../types";
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
          {file.type === "video" && <video src={staticUrl(file.url)} controls autoPlay />}
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
              <div className="result-section">
                <label className="result-label">
                  <Icon name="box" /> Context Assets Used
                </label>
                <div style={{ width: "100%" }}>
                  {result.assets.length === 0 ? (
                    <div className="muted-small">No reference assets were selected for this prompt.</div>
                  ) : (
                    <div className="assets-grid-layout" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "16px" }}>
                      {result.assets.map((asset) => {
                        const assetUrl = getAssetUrl(asset);
                        const isImg = /\.(png|jpe?g|webp|gif)$/i.test(asset);
                        return (
                          <div className="asset-grid-card uploaded-asset-card" key={asset} style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-color)", borderRadius: "8px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
                            <div className="asset-card-thumb" style={{ height: "120px", background: "#050505", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", position: "relative", padding: "8px" }}>
                              <div className="asset-card-badge uploaded">
                                <Icon name="box" /> Uploaded Reference
                              </div>
                              {isImg ? (
                                <img src={assetUrl} alt={basename(asset)} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "4px" }} />
                              ) : (
                                <Icon name="file" />
                              )}
                            </div>
                            <div className="asset-card-name" style={{ padding: "8px 10px", fontSize: "11px", color: "var(--text-secondary)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", borderTop: "1px solid var(--border-color)" }} title={basename(asset)}>
                              {basename(asset)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {result.clipFrames && result.clipFrames.length > 0 && (
                <div className="result-section" style={{ marginTop: "24px" }}>
                  <label className="result-label">
                    <Icon name="image" /> Extracted Clip Frames
                  </label>
                  <div style={{ width: "100%" }}>
                    <div className="assets-grid-layout" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "16px" }}>
                      {result.clipFrames.map((framePath, idx) => {
                        const frameUrl = getDataUrl(framePath);
                        return (
                          <div className="asset-grid-card extracted-frame-card" key={framePath} style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-color)", borderRadius: "8px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
                            <div className="asset-card-thumb" style={{ height: "120px", background: "#050505", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", position: "relative", padding: "8px" }}>
                              <div className="asset-card-badge extracted">
                                <Icon name="video" /> Video Frame
                              </div>
                              <img src={frameUrl} alt={`Frame ${idx + 1}`} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "4px" }} />
                            </div>
                            <div className="asset-card-name" style={{ padding: "8px 10px", fontSize: "11px", color: "var(--text-secondary)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", borderTop: "1px solid var(--border-color)" }} title={`Frame ${idx + 1}`}>
                              Frame {idx + 1}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
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
