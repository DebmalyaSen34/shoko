import { useState } from "react";
import type { FormEvent } from "react";
import { apiUrl } from "../../lib/api";
import { Icon } from "../Icon";

type UploadFeedbackModalProps = {
  projectName: string;
  onClose: () => void;
  onUploaded: () => void;
};

export function UploadFeedbackModal({ projectName, onClose, onUploaded }: UploadFeedbackModalProps) {
  const [feedbackFile, setFeedbackFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const canSubmit = Boolean(feedbackFile) && !submitting;

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
    if (!feedbackFile) {
      setError("Please select a feedback file first.");
      return;
    }

    setSubmitting(true);
    setError("");
    setStatus("Uploading and parsing client feedback...");

    try {
      const uploadData = new FormData();
      uploadData.append("file", feedbackFile);

      const response = await fetch(
        apiUrl(`/api/projects/${encodeURIComponent(projectName)}/feedback`),
        {
          method: "POST",
          body: uploadData,
        },
      );

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      setStatus("Feedback parsed and aligned successfully.");
      onUploaded();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to upload feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal active">
      <form className="modal-content glass-card new-project-modal" onSubmit={handleSubmit}>
        <div className="modal-header">
          <h3>
            <Icon name="comments" /> Upload Feedback for "{projectName}"
          </h3>
          <button className="close-btn" type="button" onClick={onClose} disabled={submitting}>
            <Icon name="close" />
          </button>
        </div>

        <div className="modal-body new-project-body">
          <div className="project-form-grid">
            <section className="project-upload-section full-width required-package">
              <div className="project-section-heading">
                <Icon name="file" />
                <div>
                  <strong>Client Feedback File</strong>
                  <span>Supports .txt, .csv, .xlsx, .xls formats</span>
                </div>
              </div>
              <label className="package-picker">
                <input
                  type="file"
                  accept=".txt,.csv,.xlsx,.xls"
                  disabled={submitting}
                  onChange={(event) => setFeedbackFile(event.target.files?.[0] || null)}
                />
                <span>{feedbackFile ? feedbackFile.name : "Choose feedback file"}</span>
              </label>
            </section>

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
            Upload & Parse
          </button>
        </div>
      </form>
    </div>
  );
}
