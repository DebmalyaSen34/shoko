import type { Provider } from "../types";
import { formatProjectName } from "../lib/format";
import { Icon } from "./Icon";

type AppHeaderProps = {
  activeProject: string;
  projects: string[];
  provider: Provider;
  onNewProject: () => void;
  onOpenSettings: () => void;
  onProjectChange: (project: string) => void;
  onProviderChange: (provider: Provider) => void;
};

export function AppHeader({ activeProject, projects, provider, onNewProject, onOpenSettings, onProjectChange, onProviderChange }: AppHeaderProps) {
  return (
    <header className="h-header-height bg-bg-secondary border-b border-border-color flex justify-between items-center px-6 z-[100] flex-none">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-9 h-9 bg-bg-tertiary border border-border-color rounded-md flex items-center justify-center text-text-primary text-lg">
          <Icon name="board" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-base font-bold leading-[1.1] text-text-primary m-0">StudioTimeline</h1>
          <span className="text-[10px] text-text-muted font-bold uppercase tracking-[0.5px]">Feedback & Workflow Engine</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button className="icon-btn" title="Settings" type="button" onClick={onOpenSettings}>
          <Icon name="settings" />
        </button>
        <button
          className="h-[34px] px-3.5 py-2 text-xs font-bold inline-flex items-center justify-center gap-2 rounded-md bg-gradient-to-br from-accent-primary to-[#4e4e5d] text-white shadow-[0_4px_12px_var(--accent-primary-glow)] hover:from-accent-primary-hover hover:to-[#5c5c6c] transition-all whitespace-nowrap cursor-pointer border-0"
          onClick={onNewProject}
        >
          <Icon name="plus" /> New Project
        </button>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] text-text-muted font-bold uppercase tracking-[0.5px]">AI Provider</span>
          <select
            className="premium-select min-width-[140px] bg-[#18181f] text-text-primary border border-border-color pl-3 pr-7 py-1.5 rounded-md text-xs font-medium outline-none appearance-none hover:border-accent-primary-hover focus:border-accent-primary-hover transition-all"
            value={provider}
            onChange={(event) => onProviderChange(event.target.value as Provider)}
          >
            <option value="openai">OpenAI</option>
            <option value="gemini">Gemini</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] text-text-muted font-bold uppercase tracking-[0.5px]">Active Project</span>
          <select
            className="premium-select min-width-[180px] bg-[#18181f] text-text-primary border border-border-color pl-3 pr-7 py-1.5 rounded-md text-xs font-medium outline-none appearance-none hover:border-accent-primary-hover focus:border-accent-primary-hover transition-all"
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
