import type { Provider } from "../types";
import { formatProjectName } from "../lib/format";
import { Icon } from "./Icon";

type AppHeaderProps = {
  activeProject: string;
  projects: string[];
  provider: Provider;
  onProjectChange: (project: string) => void;
  onProviderChange: (provider: Provider) => void;
};

export function AppHeader({ activeProject, projects, provider, onProjectChange, onProviderChange }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="header-logo">
        <div className="logo-icon">
          <Icon name="board" />
        </div>
        <div className="logo-text">
          <h1>StudioTimeline</h1>
          <span>Feedback & Workflow Engine</span>
        </div>
      </div>

      <div className="header-controls">
        <label className="project-selector-wrapper">
          <span>AI Provider</span>
          <select className="premium-select provider-select" value={provider} onChange={(event) => onProviderChange(event.target.value as Provider)}>
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
          </select>
        </label>
        <label className="project-selector-wrapper">
          <span>Active Project</span>
          <select
            className="premium-select"
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
      </div>
    </header>
  );
}
