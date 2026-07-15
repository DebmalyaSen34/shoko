import { useState } from "react";
import type { FormEvent } from "react";
import { apiUrl } from "../../lib/api";
import { Icon } from "../Icon";

type AddManualFeedbackModalProps = {
  projectName: string;
  clipName: string;
  onClose: () => void;
  onAdded: () => void;
};

export function AddManualFeedbackModal({
  projectName,
  clipName,
  onClose,
  onAdded,
}: AddManualFeedbackModalProps) {
  const [remark, setRemark] = useState("");
  const [category, setCategory] = useState("video");
  const [timestamp, setTimestamp] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const canSubmit = remark.trim().length > 0 && !submitting;

  async function readError(response: Response) {
    try {
      const data = (await response.json()) as { detail?: string };
      return data.detail || response.statusText;
    } catch {
      return response.statusText;
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (remark.trim().length === 0) {
      setError("Feedback remark is required.");
      return;
    }

    setSubmitting(true);
    setError("");
    setStatus("Saving manual feedback remark...");

    try {
      const response = await fetch(
        apiUrl(`/api/projects/${encodeURIComponent(projectName)}/feedback/item`),
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            clip_used: clipName,
            category,
            remark: remark.trim(),
            timestamp: timestamp.trim() || null,
          }),
        },
      );

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      setStatus("Feedback added successfully.");
      onAdded();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to add feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal active">
      <form className="modal-content glass-card new-project-modal" onSubmit={handleSubmit}>
        <div className="modal-header">
          <h3>
            <Icon name="comments" /> Add Clip Feedback
          </h3>
          <button className="close-btn" type="button" onClick={onClose} disabled={submitting}>
            <Icon name="close" />
          </button>
        </div>

        <div className="modal-body new-project-body">
          <div className="project-form-grid">
            <label className="project-field full-width">
              <span>Target Clip</span>
              <input value={clipName} disabled type="text" style={{ opacity: 0.7 }} />
            </label>

            <label className="project-field full-width">
              <span>Feedback Category</span>
              <select
                className="premium-select"
                style={{ width: "100%", padding: "8px 12px" }}
                value={category}
                disabled={submitting}
                onChange={(event) => setCategory(event.target.value)}
              >
                <option value="video">Video (Visual reaction, camera, expressions, movement)</option>
                <option value="audio">Audio (SFX, dialogue, dubbing, background music)</option>
                <option value="both">Both Video and Audio changes</option>
              </select>
            </label>

            <label className="project-field full-width">
              <span>Timestamp (Optional)</span>
              <input
                value={timestamp}
                disabled={submitting}
                placeholder="e.g. 00:05, 01:14"
                onChange={(event) => setTimestamp(event.target.value)}
              />
            </label>

            <label className="project-field full-width">
              <span>Feedback Remark</span>
              <textarea
                value={remark}
                disabled={submitting}
                rows={4}
                placeholder="Describe the feedback instruction for the editor or video AI..."
                onChange={(event) => setRemark(event.target.value)}
                style={{
                  width: "100%",
                  background: "var(--bg-tertiary)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "6px",
                  padding: "8px 12px",
                  color: "var(--text-primary)",
                  outline: "none",
                  resize: "vertical",
                }}
              />
            </label>

            {status && <div className="project-status full-width">{status}</div>}
            {error && (
              <div className="error-warning-box full-width">
                <Icon name="warning" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer modal-actions">
          <button className="premium-btn secondary" type="button" onClick={onClose} disabled={submitting}>
            Cancel
          </button>
          <button className="premium-btn" type="submit" disabled={!canSubmit}>
            {submitting ? <Icon name="refresh" /> : <Icon name="check" />}
            Save Feedback
          </button>
        </div>
      </form>
    </div>
  );
}
