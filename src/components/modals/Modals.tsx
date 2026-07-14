import type { PreviewState, ResultState } from "../../types";
import { staticUrl } from "../../lib/api";
import { basename } from "../../lib/format";
import { Icon } from "../Icon";

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
  const passed = result.quality?.passed !== false;
  const forbidden = result.quality?.forbidden_terms_found || [];

  return (
    <div className="modal active">
      <div className="modal-content glass-card result-modal-content">
        <div className="modal-header">
          <h3>
            <Icon name="timeline" /> Workflow Result Verification
          </h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body result-modal-body">
          <div className="result-content-layout">
            <div className="result-col">
              <div className="result-section">
                <label className="result-label">
                  <Icon name="file" /> Generated Video Model Prompt
                </label>
                <div className="prompt-box-wrapper">
                  <pre className="monospace-box">{result.prompt}</pre>
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
                  <pre className="monospace-box">{result.initialPrompt}</pre>
                </div>
              )}
              <div className="result-section">
                <label className="result-label">
                  <Icon name="magic" /> Generation Explanation
                </label>
                <p className="text-description">{result.explanation}</p>
              </div>
            </div>

            <div className="result-col">
              {result.initialImage && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="image" /> Generated Initial Frame Image
                  </label>
                  <div className="initial-image-preview-wrapper">
                    <img src={result.initialImage} alt="Initial Frame" />
                  </div>
                </div>
              )}
              <div className="result-section">
                <label className="result-label">
                  <Icon name="box" /> Context Assets Used
                </label>
                <div className="mini-assets-list">
                  {result.assets.length === 0 ? (
                    <div className="muted-small">No reference assets were selected for this prompt.</div>
                  ) : (
                    result.assets.map((asset) => (
                      <div className="mini-asset-item" key={asset}>
                        <Icon name="link" />
                        <span title={asset}>{basename(asset)}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
              {result.quality && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="check" /> Quality Check Report
                  </label>
                  <div className="quality-report-card">
                    <div className="quality-status">
                      Status: <span className={`tag ${passed ? "tag-audio" : "tag-both"}`}>{passed ? "PASSED" : "WARNING"}</span>
                    </div>
                    <div className="quality-details-table">
                      <QualityRow label="Feedback Adherence:" value={result.quality.feedback_adherence || "Passed"} />
                      <QualityRow label="Clothing Consistency:" value={result.quality.clothing_consistency || "Passed"} />
                      <QualityRow label="Forbidden Terms:" value={forbidden.length ? forbidden.join(", ") : "None"} danger={forbidden.length > 0} />
                    </div>
                    {Boolean(result.quality.suggestions?.length) && (
                      <div className="quality-suggestions-list">
                        {result.quality.suggestions?.map((suggestion) => (
                          <div key={suggestion}>
                            <Icon name="warning" /> {suggestion}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="modal-footer modal-actions">
          <button className="premium-btn secondary" onClick={onClose}>
            Close Verification
          </button>
          <button className="premium-btn" onClick={onClose}>
            <Icon name="check" /> Approve & Proceed
          </button>
        </div>
      </div>
    </div>
  );
}

function QualityRow({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="q-row">
      <span>{label}</span>
      <strong className={danger ? "danger" : ""}>{value}</strong>
    </div>
  );
}
