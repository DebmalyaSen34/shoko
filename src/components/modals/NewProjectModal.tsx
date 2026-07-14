import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { apiUrl } from "../../lib/api";
import type { ProjectImportResult } from "../../types";
import { Icon } from "../Icon";

type AssetCategory = {
  key: string;
  label: string;
  directory: string;
  accept?: string;
};

const ASSET_CATEGORIES: AssetCategory[] = [
  { key: "style", label: "Style", directory: "00_style", accept: "image/*,.pdf" },
  { key: "characters", label: "Characters", directory: "01_characters", accept: "image/*,.pdf" },
  { key: "locations", label: "Locations", directory: "03_locations", accept: "image/*,.pdf,video/*" },
  { key: "props", label: "Props", directory: "02_props", accept: "image/*,.pdf" },
  { key: "audio", label: "Audio", directory: "04_audio", accept: "audio/*" },
  { key: "references", label: "References", directory: "05_references", accept: "image/*,video/*,.pdf" },
  { key: "clips", label: "Clips", directory: "06_clips/_final", accept: "video/*" },
];

type NewProjectModalProps = {
  onClose: () => void;
  onCreated: (projectName: string) => void;
};

export function NewProjectModal({ onClose, onCreated }: NewProjectModalProps) {
  const [projectName, setProjectName] = useState("");
  const [assetFiles, setAssetFiles] = useState<Record<string, File[]>>({});
  const [packageFile, setPackageFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const cleanName = projectName.trim();
  const assetCount = useMemo(
    () => Object.values(assetFiles).reduce((sum, files) => sum + files.length, 0),
    [assetFiles],
  );
  const canSubmit = cleanName.length > 0 && Boolean(packageFile) && !submitting;

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
    if (!packageFile) {
      setError("Premiere Pro ZIP is required.");
      return;
    }

    setSubmitting(true);
    setError("");

    try {
      setStatus("Creating project workspace...");
      const createData = new FormData();
      createData.append("project_name", cleanName);
      const createResponse = await fetch(apiUrl("/api/projects"), {
        method: "POST",
        body: createData,
      });
      if (!createResponse.ok && createResponse.status !== 409) {
        throw new Error(await readError(createResponse));
      }

      for (const category of ASSET_CATEGORIES) {
        const files = assetFiles[category.key] || [];
        if (files.length === 0) continue;

        setStatus(`Uploading ${category.label.toLowerCase()} assets...`);
        const uploadData = new FormData();
        files.forEach((file) => uploadData.append("files", file));
        const uploadResponse = await fetch(
          apiUrl(`/api/projects/${encodeURIComponent(cleanName)}/assets/${category.key}`),
          {
            method: "POST",
            body: uploadData,
          },
        );
        if (!uploadResponse.ok) throw new Error(await readError(uploadResponse));
      }

      setStatus("Importing Premiere package and extracting timeline...");
      const packageData = new FormData();
      packageData.append("package", packageFile);
      const packageResponse = await fetch(
        apiUrl(`/api/projects/${encodeURIComponent(cleanName)}/premiere-package`),
        {
          method: "POST",
          body: packageData,
        },
      );
      if (!packageResponse.ok) throw new Error(await readError(packageResponse));
      const result = (await packageResponse.json()) as ProjectImportResult;

      setStatus("Project imported.");
      onCreated(result.project_name);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to create project.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal active">
      <form className="modal-content glass-card new-project-modal" onSubmit={handleSubmit}>
        <div className="modal-header">
          <h3>
            <Icon name="plus" /> New Project
          </h3>
          <button className="close-btn" type="button" onClick={onClose} disabled={submitting}>
            <Icon name="close" />
          </button>
        </div>

        <div className="modal-body new-project-body">
          <div className="project-form-grid">
            <label className="project-field full-width">
              <span>Project Name</span>
              <input
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="project-name"
                disabled={submitting}
              />
            </label>

            <section className="project-upload-section full-width">
              <div className="project-section-heading">
                <Icon name="box" />
                <div>
                  <strong>Optional Assets</strong>
                  <span>{assetCount} file{assetCount === 1 ? "" : "s"} selected</span>
                </div>
              </div>

              <div className="asset-upload-grid">
                {ASSET_CATEGORIES.map((category) => {
                  const files = assetFiles[category.key] || [];
                  return (
                    <label className="upload-tile" key={category.key}>
                      <span className="upload-title">{category.label}</span>
                      <span className="upload-dir">{category.directory}</span>
                      <input
                        type="file"
                        multiple
                        accept={category.accept}
                        disabled={submitting}
                        onChange={(event) => {
                          const selected = Array.from(event.target.files || []);
                          setAssetFiles((current) => ({ ...current, [category.key]: selected }));
                        }}
                      />
                      <span className="upload-count">{files.length ? `${files.length} selected` : "Skippable"}</span>
                    </label>
                  );
                })}
              </div>
            </section>

            <section className="project-upload-section full-width required-package">
              <div className="project-section-heading">
                <Icon name="file" />
                <div>
                  <strong>Premiere Pro ZIP</strong>
                  <span>Required for timeline extraction</span>
                </div>
              </div>
              <label className="package-picker">
                <input
                  type="file"
                  accept=".zip,application/zip"
                  disabled={submitting}
                  onChange={(event) => setPackageFile(event.target.files?.[0] || null)}
                />
                <span>{packageFile ? packageFile.name : "Choose .zip package"}</span>
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
            Import Project
          </button>
        </div>
      </form>
    </div>
  );
}
