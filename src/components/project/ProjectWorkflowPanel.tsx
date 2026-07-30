import type { ReactNode } from "react";
import type { ProjectData } from "../../types";
import { formatProjectName } from "../../lib/format";
import { Icon } from "../Icon";
import { WorkflowRunningLabel } from "../WorkflowRunningLabel";

type ProjectWorkflowPanelProps = {
  activeProject: string;
  assetTotal: number;
  className?: string;
  duration: string;
  loading: boolean;
  loadError: string;
  projectData: ProjectData | null;
  runningIndexes: Set<number>;
  onCollapse?: () => void;
  onImportProject: () => void;
  onRunAssignment: () => void;
  onUploadFeedback: () => void;
  onViewResults: () => void;
};

export function ProjectWorkflowPanel({
  activeProject,
  assetTotal,
  className = "",
  duration,
  loading,
  loadError,
  projectData,
  runningIndexes,
  onCollapse,
  onImportProject,
  onRunAssignment,
  onUploadFeedback,
  onViewResults,
}: ProjectWorkflowPanelProps) {
  const feedbackCount = projectData?.feedback.reduce((count, group) => count + group.feedback_items.length, 0) || 0;
  const promptCount = projectData?.prompts.length || 0;
  const hasProject = Boolean(projectData);
  const assignmentRunning = runningIndexes.size > 0;

  return (
    <aside className={`project-workflow-column ${className}`} aria-label="Project workflow">
      {onCollapse && (
        <div className="workflow-drawer-topline">
          <span>Project Workflow</span>
          <button type="button" title="Collapse workflow" onClick={onCollapse}>
            <Icon name="chevronLeft" />
          </button>
        </div>
      )}
      <WorkflowStep index={1} title="Import Project">
        <div className="workflow-dropzone">
          <Icon name="folder" />
          <strong>{hasProject ? formatProjectName(activeProject) : "Upload Premiere Pro Project (.zip)"}</strong>
          <span>{hasProject ? `${projectData?.timeline.length || 0} clips loaded` : "Contains .prproj and all media"}</span>
          <button className="shoko-primary-button" type="button" onClick={onImportProject}>
            {hasProject ? "Switch Project" : "Upload .zip"}
          </button>
        </div>
        <WorkflowChip
          icon="file"
          title={hasProject ? `${formatProjectName(activeProject)}.zip` : "No project imported"}
          meta={hasProject ? "Uploaded" : loading ? "Loading projects..." : "Waiting for project"}
          status={hasProject ? "complete" : loadError ? "error" : "pending"}
        />
      </WorkflowStep>

      <div className="workflow-info-card">
        <h3>Project Info</h3>
        <InfoRow label="Resolution" value="1920 x 1080" />
        <InfoRow label="Frame Rate" value="23.976 fps" />
        <InfoRow label="Duration" value={duration} />
        <InfoRow label="Total Clips" value={String(projectData?.timeline.length || 0)} />
        <InfoRow label="Assets" value={String(assetTotal)} />
      </div>

      <WorkflowStep index={2} title="Upload Feedback">
        <div className="workflow-upload-card">
          <Icon name="comments" />
          <div>
            <strong>Upload feedback file</strong>
            <span>JSON, CSV or TXT</span>
          </div>
          <button className="shoko-ghost-button" type="button" onClick={onUploadFeedback} disabled={!activeProject}>
            Upload File
          </button>
        </div>
        <WorkflowChip
          icon="file"
          title={feedbackCount ? "feedback.json" : "No feedback file"}
          meta={feedbackCount ? `${feedbackCount} feedbacks loaded` : "Waiting for upload"}
          status={feedbackCount ? "complete" : "pending"}
        />
      </WorkflowStep>

      <WorkflowStep index={3} title="AI Assignment">
        <div className="workflow-assignment-card">
          <p>AI will match feedbacks to the right clips.</p>
          <div className={`assignment-status ${promptCount ? "complete" : assignmentRunning ? "running" : "pending"}`}>
            <Icon name={promptCount ? "check" : assignmentRunning ? "refresh" : "clock"} />
            <span>{promptCount ? "Completed" : assignmentRunning ? "Running" : "Ready"}</span>
          </div>
          <button className={`shoko-primary-button ${assignmentRunning ? "workflow-running-button" : ""}`} type="button" onClick={onRunAssignment} disabled={!hasProject || assignmentRunning}>
            {assignmentRunning ? <WorkflowRunningLabel /> : promptCount ? "Run Again" : "Run Assignment"}
          </button>
          <button className="shoko-ghost-button" type="button" onClick={onViewResults} disabled={!hasProject}>
            View Results
          </button>
        </div>
      </WorkflowStep>
    </aside>
  );
}

export function ProjectHomePanel({
  activeProject,
  assetTotal,
  duration,
  loading,
  projectData,
  projects,
  onImportProject,
  onOpenTimeline,
  onProjectChange,
}: {
  activeProject: string;
  assetTotal: number;
  duration: string;
  loading: boolean;
  projectData: ProjectData | null;
  projects: string[];
  onImportProject: () => void;
  onOpenTimeline: () => void;
  onProjectChange: (project: string) => void;
}) {
  const feedbackCount = projectData?.feedback.reduce((count, group) => count + group.feedback_items.length, 0) || 0;
  const promptCount = projectData?.prompts.length || 0;

  return (
    <section className="project-home-panel" aria-label="Project workspace">
      <div className="project-home-header">
        <div>
          <span className="project-home-kicker">Home</span>
          <h2>{projectData?.sequence_name || activeProject || "No project selected"}</h2>
          <p>
            Choose a project, import new work, and keep an eye on the current project before moving into the timeline.
          </p>
        </div>
        <div className="project-home-header-actions">
          <button className="shoko-primary-button" type="button" onClick={onImportProject}>
            <Icon name="plus" /> New Project
          </button>
          <button className="shoko-ghost-button" type="button" onClick={onOpenTimeline} disabled={!projectData}>
            <Icon name="timeline" /> Open Timeline
          </button>
        </div>
      </div>

      <div className="project-home-content-grid">
        <section className="project-list-panel" aria-label="All projects">
          <div className="project-list-header">
            <div>
              <h3>Projects</h3>
              <p>{projects.length ? `${projects.length} project${projects.length === 1 ? "" : "s"} available` : "No projects imported yet"}</p>
            </div>
            <button className="shoko-icon-button" type="button" title="New Project" onClick={onImportProject}>
              <Icon name="plus" />
            </button>
          </div>
          <div className="project-list">
            {projects.length ? projects.map((project) => (
              <button
                className={`project-list-item ${project === activeProject ? "active" : ""}`}
                type="button"
                key={project}
                onClick={() => onProjectChange(project)}
              >
                <Icon name="folder" />
                <span>{formatProjectName(project)}</span>
                {project === activeProject && <Icon name="check" />}
              </button>
            )) : (
              <div className="project-list-empty">
                <Icon name="folder" />
                <span>Import a project to begin.</span>
              </div>
            )}
          </div>
        </section>

        <section className="current-project-panel" aria-label="Current project">
          <div>
            <h3>Current Project</h3>
            <p>
              {loading
                ? "Loading project state..."
                : projectData
                  ? `${formatProjectName(activeProject)} has ${feedbackCount} feedback items and ${promptCount} generated prompt records.`
                  : "Select or import a project to begin the Shoko workflow."}
            </p>
          </div>

          <div className="project-home-stats">
            <StatBlock label="Duration" value={duration} />
            <StatBlock label="Timeline Clips" value={String(projectData?.timeline.length || 0)} />
            <StatBlock label="Feedback Items" value={String(feedbackCount)} />
            <StatBlock label="Assets" value={String(assetTotal)} />
          </div>

          <div className="readiness-list">
            <ReadinessItem ready={Boolean(projectData)} label="Project imported" />
            <ReadinessItem ready={assetTotal > 0} label="Assets available" />
            <ReadinessItem ready={feedbackCount > 0} label="Feedback loaded" />
            <ReadinessItem ready={promptCount > 0} label="Prompt workflow run" />
          </div>
        </section>
      </div>
    </section>
  );
}

function WorkflowStep({ index, title, children }: { index: number; title: string; children: ReactNode }) {
  return (
    <section className="workflow-step">
      <div className="workflow-step-title">
        <span>{index}.</span>
        <h3>{title}</h3>
      </div>
      {children}
    </section>
  );
}

function WorkflowChip({
  icon,
  title,
  meta,
  status,
}: {
  icon: string;
  title: string;
  meta: string;
  status: "complete" | "pending" | "error";
}) {
  return (
    <div className={`workflow-chip ${status}`}>
      <Icon name={icon} />
      <div>
        <strong>{title}</strong>
        <span>{meta}</span>
      </div>
      <Icon name={status === "complete" ? "check" : status === "error" ? "warning" : "clock"} />
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="workflow-info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ReadinessItem({ ready, label }: { ready: boolean; label: string }) {
  return (
    <div className={`readiness-item ${ready ? "ready" : ""}`}>
      <Icon name={ready ? "check" : "clock"} />
      <span>{label}</span>
    </div>
  );
}

function StatBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="project-stat-block">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
