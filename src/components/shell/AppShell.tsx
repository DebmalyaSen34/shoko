import type { ReactNode } from "react";
import type { Provider } from "../../types";
import { TopBar } from "./TopBar";
import { LeftRail, type RailItemId } from "./LeftRail";

type AppShellProps = {
  activeProject: string;
  activeRailItem: RailItemId;
  children: ReactNode;
  clipCount: number;
  projectLabel: string;
  projects: string[];
  provider: Provider;
  workflowDrawerOpen: boolean;
  onOpenSettings: () => void;
  onProjectChange: (project: string) => void;
  onRailSelect: (item: RailItemId) => void;
  onRunWorkflow: () => void;
  onToggleWorkflowDrawer: () => void;
};

export function AppShell({
  activeProject,
  activeRailItem,
  children,
  clipCount,
  projectLabel,
  projects,
  provider,
  workflowDrawerOpen,
  onOpenSettings,
  onProjectChange,
  onRailSelect,
  onRunWorkflow,
  onToggleWorkflowDrawer,
}: AppShellProps) {
  const providerLabel = provider === "openai" ? "OpenAI" : "Gemini";

  return (
    <div className="shoko-shell">
      <TopBar
        activeProject={activeProject}
        projects={projects}
        onOpenSettings={onOpenSettings}
        onRunWorkflow={onRunWorkflow}
        onProjectChange={onProjectChange}
      />

      <div className="shoko-desktop-frame">
        <LeftRail
          activeItem={activeRailItem}
          collapsedPanelOpen={workflowDrawerOpen}
          collapsedPanelOpenLabel={workflowDrawerOpen ? "Collapse project workflow" : "Open project workflow"}
          onCollapsedPanelOpen={activeRailItem === "timeline" ? onToggleWorkflowDrawer : undefined}
          projectContextActive={activeRailItem === "timeline"}
          onSelect={onRailSelect}
        />

        <main className={`shoko-workspace ${activeRailItem === "timeline" ? "timeline-active" : ""} ${workflowDrawerOpen ? "workflow-drawer-open" : ""}`}>
          <div className="shoko-workspace-inner">{children}</div>

          {activeRailItem !== "timeline" && (
            <footer className="shoko-statusbar">
              <div>
                <span className="status-dot" /> Synced to {providerLabel} - {projectLabel}
              </div>
              <div>
                Showing clips {clipCount ? 1 : 0} - {clipCount} of {clipCount} scroll horizontally
              </div>
            </footer>
          )}
        </main>
      </div>
    </div>
  );
}
