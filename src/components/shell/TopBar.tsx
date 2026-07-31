import { formatProjectName } from "../../lib/format";
import { Icon } from "../Icon";

type TopBarProps = {
  activeProject: string;
  projects: string[];
  onOpenSettings: () => void;
  onRunWorkflow: () => void;
  onProjectChange: (project: string) => void;
  onGoHome?: () => void;
};

export function TopBar({
  activeProject,
  projects,
  onOpenSettings,
  onRunWorkflow,
  onProjectChange,
  onGoHome,
}: TopBarProps) {
  return (
    <header className="shoko-topbar">
      <div
        className={`shoko-brand ${onGoHome ? "clickable" : ""}`}
        onClick={onGoHome}
        role={onGoHome ? "button" : undefined}
        tabIndex={onGoHome ? 0 : undefined}
        onKeyDown={(e) => {
          if (onGoHome && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onGoHome();
          }
        }}
        title={onGoHome ? "Go to Home" : undefined}
      >
        <Icon name="home" className="shoko-home-icon" />
        <strong>Shoko</strong>
      </div>

      <div className="shoko-project-switcher">
        <div className={`shoko-project-select-box ${projects.length === 0 ? "disabled" : ""}`}>
          <Icon name="folder" className="project-select-folder-icon" />
          <select
            aria-label="Active Project"
            value={activeProject}
            onChange={(event) => onProjectChange(event.target.value)}
            disabled={projects.length === 0}
          >
            {projects.length === 0 ? (
              <option value="">No projects found</option>
            ) : (
              projects.map((project) => (
                <option key={project} value={project}>
                  {formatProjectName(project)}
                </option>
              ))
            )}
          </select>
          <Icon name="chevronDown" className="project-select-chevron" />
        </div>
      </div>

      <div className="shoko-topbar-actions">
        <button className="shoko-primary-button" type="button" onClick={onRunWorkflow}>
          <Icon name="play" /> Run Workflow
        </button>
        <button className="shoko-icon-button" title="Settings" type="button" onClick={onOpenSettings}>
          <Icon name="settings" />
        </button>
      </div>
    </header>
  );
}
