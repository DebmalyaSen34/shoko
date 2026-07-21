import { useEffect, useMemo, useState } from "react";
import { apiUrl } from "../../lib/api";
import type { RuntimeConfig } from "../../types";
import { Icon } from "../Icon";

type SecretKey = "OPENAI_API_KEY" | "GEMINI_API_KEY" | "SEGMIND_API_KEY";

type SettingsModalProps = {
  onClose: () => void;
  onSaved: () => void;
};

const secretRows: { key: SecretKey; label: string; field: "openai_api_key" | "gemini_api_key" | "segmind_api_key" }[] = [
  { key: "OPENAI_API_KEY", label: "OpenAI API Key", field: "openai_api_key" },
  { key: "GEMINI_API_KEY", label: "Gemini API Key", field: "gemini_api_key" },
  { key: "SEGMIND_API_KEY", label: "Segmind API Key", field: "segmind_api_key" },
];

function sourceLabel(source?: string) {
  if (source === "os_environment") return "OS environment";
  if (source === "app_or_dev_env_file") return "App config";
  return "Missing";
}

export function SettingsModal({ onClose, onSaved }: SettingsModalProps) {
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [values, setValues] = useState<Record<SecretKey, string>>({
    OPENAI_API_KEY: "",
    GEMINI_API_KEY: "",
    SEGMIND_API_KEY: "",
  });
  const [clearKeys, setClearKeys] = useState<Record<SecretKey, boolean>>({
    OPENAI_API_KEY: false,
    GEMINI_API_KEY: false,
    SEGMIND_API_KEY: false,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const dirty = useMemo(
    () => secretRows.some((row) => values[row.key].trim() || clearKeys[row.key]),
    [values, clearKeys],
  );

  async function loadConfig() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(apiUrl("/api/config/runtime"));
      if (!response.ok) throw new Error("Failed to load runtime configuration");
      setConfig((await response.json()) as RuntimeConfig);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load runtime configuration");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadConfig();
  }, []);

  async function saveSettings() {
    setSaving(true);
    setError("");
    try {
      const payload: Record<string, string> = {};
      for (const row of secretRows) {
        if (clearKeys[row.key]) {
          payload[row.field] = "";
        } else if (values[row.key].trim()) {
          payload[row.field] = values[row.key].trim();
        }
      }

      const response = await fetch(apiUrl("/api/config/secrets"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Failed to save settings");
      }

      setValues({ OPENAI_API_KEY: "", GEMINI_API_KEY: "", SEGMIND_API_KEY: "" });
      setClearKeys({ OPENAI_API_KEY: false, GEMINI_API_KEY: false, SEGMIND_API_KEY: false });
      await response.json().catch(() => ({}));
      await loadConfig();
      onSaved();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not save settings");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal active">
      <div className="modal-content glass-card settings-modal">
        <div className="modal-header">
          <h3>
            <Icon name="settings" /> Settings
          </h3>
          <button className="close-btn" type="button" onClick={onClose} disabled={saving}>
            <Icon name="close" />
          </button>
        </div>

        <div className="modal-body settings-body">
          {loading ? (
            <div className="project-status">Loading configuration...</div>
          ) : (
            <>
              <section className="settings-section">
                <div className="project-section-heading">
                  <Icon name="key" />
                  <div>
                    <strong>Provider API Keys</strong>
                    <span>Saved values are hidden after storage and are only used by the local backend.</span>
                  </div>
                </div>

                <div className="settings-secret-list">
                  {secretRows.map((row) => {
                    const status = config?.secrets.keys[row.key];
                    const locked = Boolean(status?.locked_by_os_env);
                    const canUpdate = Boolean(status?.can_update);
                    return (
                      <div className="settings-secret-row" key={row.key}>
                        <div className="settings-secret-meta">
                          <div className="settings-secret-title">
                            <span>{row.label}</span>
                            <span className={`settings-status ${status?.configured ? "configured" : "missing"}`}>
                              {status?.configured ? "Configured" : "Missing"}
                            </span>
                          </div>
                          <div className="settings-secret-source">
                            {locked && <Icon name="lock" />}
                            {sourceLabel(status?.source)}
                          </div>
                        </div>

                        <div className="settings-secret-controls">
                          <input
                            type="password"
                            placeholder={locked ? "Controlled by OS environment" : "Paste a new key"}
                            value={values[row.key]}
                            disabled={!canUpdate || clearKeys[row.key] || saving}
                            onChange={(event) =>
                              setValues((current) => ({
                                ...current,
                                [row.key]: event.target.value,
                              }))
                            }
                          />
                          <label className="settings-clear-toggle">
                            <input
                              type="checkbox"
                              disabled={!canUpdate || !status?.configured || saving}
                              checked={clearKeys[row.key]}
                              onChange={(event) =>
                                setClearKeys((current) => ({
                                  ...current,
                                  [row.key]: event.target.checked,
                                }))
                              }
                            />
                            <span>Clear</span>
                          </label>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>

              {config && (
                <section className="settings-section">
                  <div className="project-section-heading">
                    <Icon name="file" />
                    <div>
                      <strong>Local Storage</strong>
                      <span>Runtime data and user config are stored outside the app bundle.</span>
                    </div>
                  </div>
                  <div className="settings-path-grid">
                    <PathRow label="App storage" value={config.app_storage_dir} />
                    <PathRow label="Secrets file" value={config.secrets.config_env_path} />
                    <PathRow label="Data" value={config.data_dir} />
                    <PathRow label="Assets" value={config.assets_dir} />
                  </div>
                </section>
              )}

              {error && <div className="project-status settings-error">{error}</div>}
            </>
          )}
        </div>

        <div className="modal-footer modal-actions">
          <button className="premium-btn secondary" type="button" onClick={onClose} disabled={saving}>
            Close
          </button>
          <button className="premium-btn" type="button" onClick={saveSettings} disabled={saving || !dirty}>
            <Icon name="check" /> {saving ? "Saving..." : "Save Settings"}
          </button>
        </div>
      </div>
    </div>
  );
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="settings-path-row">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}
