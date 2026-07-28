import { formatProjectName } from "../../lib/format";
import { Icon } from "../Icon";

type TopBarProps = {
  activeProject: string;
  projects: string[];
  onExport: () => void;
  onOpenSettings: () => void;
  onRunWorkflow: () => void;
  onProjectChange: (project: string) => void;
};

export function TopBar({
  activeProject,
  projects,
  onExport,
  onOpenSettings,
  onRunWorkflow,
  onProjectChange,
}: TopBarProps) {
  return (
    <header className="shoko-topbar">
      <div className="shoko-brand">
        <div className="shoko-logo-tile">
          <span>S</span>
        </div>
        <strong>Shoko</strong>
      </div>

      <div className="shoko-project-switcher">
        <label>
          <span className="sr-only">Active Project</span>
          <select
            value={activeProject}
            onChange={(event) => onProjectChange(event.target.value)}
            disabled={projects.length === 0}
          >
            {projects.length === 0 ? (
              <option>No projects found</option>
            ) : (
              projects.map((project) => (
                <option key={project} value={project}>
                  {formatProjectName(project)}
                </option>
              ))
            )}
          </select>
        </label>
        <span className="project-chevron" aria-hidden="true">
          <Icon name="chevronDown" />
        </span>
        <span className="saved-state">
          <span />
          Saved
        </span>
      </div>

      <div className="shoko-topbar-actions">
        <button className="shoko-primary-button" type="button" onClick={onRunWorkflow}>
          <Icon name="play" /> Run Workflow
        </button>
        <button className="shoko-ghost-button" type="button" title="Export" onClick={onExport}>
          <Icon name="export" /> Export
        </button>
        <button className="shoko-icon-button" title="Settings" type="button" onClick={onOpenSettings}>
          <Icon name="settings" />
        </button>
      </div>
    </header>
  );
}
