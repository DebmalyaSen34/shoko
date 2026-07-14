import { useRef } from "react";
import type { CSSProperties, Dispatch, RefObject, SetStateAction } from "react";
import type { FeedbackGroup, ProjectData, PromptRecord, PromptVersion, TimelineClip } from "../../types";
import { basename, getVersions, versionLabel } from "../../lib/format";
import { staticUrl } from "../../lib/api";
import { EmptyState } from "../EmptyState";
import { Icon } from "../Icon";

type TimelinePanelProps = {
  duration: string;
  error: string;
  loading: boolean;
  projectData: ProjectData | null;
  refEl: RefObject<HTMLDivElement | null>;
  runningIndexes: Set<number>;
  selectedVersions: Record<string, number>;
  setSelectedVersions: Dispatch<SetStateAction<Record<string, number>>>;
  timelineStyle: CSSProperties;
  zoom: number;
  zoomClass: string;
  executeWorkflow: (feedbackIndex: number) => void;
  findPrompt: (clipName: string) => PromptRecord | null;
  openResultFromVersion: (version: PromptVersion) => void;
  setZoom: Dispatch<SetStateAction<number>>;
};

export function TimelinePanel({
  duration,
  error,
  loading,
  projectData,
  refEl,
  runningIndexes,
  selectedVersions,
  setSelectedVersions,
  timelineStyle,
  zoom,
  zoomClass,
  executeWorkflow,
  findPrompt,
  openResultFromVersion,
  setZoom,
}: TimelinePanelProps) {
  return (
    <section className="center-panel">
      <div className="panel-header sticky-header">
        <h2>
          <Icon name="timeline" /> Chronological Timeline
        </h2>
        <div className="timeline-stats">
          <span className="stat-badge">{projectData?.sequence_name || "--"}</span>
          <span className="stat-badge">
            <Icon name="clock" /> Duration: {duration}
          </span>
          <div className="zoom-controls">
            <button className="icon-btn" title="Zoom Out" onClick={() => setZoom((value) => Math.max(0.5, Number((value - 0.1).toFixed(2))))}>
              <Icon name="minus" />
            </button>
            <input
              className="premium-slider"
              type="range"
              min="0.5"
              max="1.5"
              step="0.05"
              value={zoom}
              onChange={(event) => setZoom(Number(event.target.value))}
            />
            <button className="icon-btn" title="Zoom In" onClick={() => setZoom((value) => Math.min(1.5, Number((value + 0.1).toFixed(2))))}>
              <Icon name="plus" />
            </button>
            <button className="icon-btn" title="Reset" onClick={() => setZoom(1)}>
              <Icon name="expand" />
            </button>
          </div>
        </div>
      </div>

      <Timeline
        refEl={refEl}
        className={zoomClass}
        style={timelineStyle}
        loading={loading}
        error={error}
        projectData={projectData}
        findPrompt={findPrompt}
        selectedVersions={selectedVersions}
        setSelectedVersions={setSelectedVersions}
        runningIndexes={runningIndexes}
        executeWorkflow={executeWorkflow}
        openResultFromVersion={openResultFromVersion}
      />
    </section>
  );
}

function Timeline({
  refEl,
  className,
  style,
  loading,
  error,
  projectData,
  findPrompt,
  selectedVersions,
  setSelectedVersions,
  runningIndexes,
  executeWorkflow,
  openResultFromVersion,
}: {
  refEl: RefObject<HTMLDivElement | null>;
  className: string;
  style: CSSProperties;
  loading: boolean;
  error: string;
  projectData: ProjectData | null;
  findPrompt: (clipName: string) => PromptRecord | null;
  selectedVersions: Record<string, number>;
  setSelectedVersions: Dispatch<SetStateAction<Record<string, number>>>;
  runningIndexes: Set<number>;
  executeWorkflow: (feedbackIndex: number) => void;
  openResultFromVersion: (version: PromptVersion) => void;
}) {
  const drag = useRef({ down: false, startX: 0, scrollLeft: 0 });

  if (loading) {
    return (
      <div className="timeline-flow">
        <EmptyState icon="refresh" title="Loading Timeline..." text="Fetching media clips, coordinates, and feedback." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="timeline-flow">
        <EmptyState icon="warning" title="Error Loading Project" text={error} />
      </div>
    );
  }

  if (!projectData) {
    return (
      <div className="timeline-flow">
        <EmptyState icon="video" title="No Project Selected" text="Choose a project from the header to load the timeline and assets." />
      </div>
    );
  }

  if (projectData.timeline.length === 0) {
    return (
      <div className="timeline-flow">
        <EmptyState icon="video" title="Timeline Empty" text="This project's timeline has no video clip records." />
      </div>
    );
  }

  return (
    <div
      ref={refEl}
      className={`timeline-flow ${className}`}
      style={style}
      onMouseDown={(event) => {
        const target = event.target as HTMLElement;
        if (target.closest("button, select, input, video, audio, .feedback-item-card")) return;
        if (!refEl.current) return;
        drag.current = {
          down: true,
          startX: event.pageX - refEl.current.offsetLeft,
          scrollLeft: refEl.current.scrollLeft,
        };
        refEl.current.classList.add("grabbing");
      }}
      onMouseUp={() => {
        drag.current.down = false;
        refEl.current?.classList.remove("grabbing");
      }}
      onMouseLeave={() => {
        drag.current.down = false;
        refEl.current?.classList.remove("grabbing");
      }}
      onMouseMove={(event) => {
        if (!drag.current.down || !refEl.current) return;
        event.preventDefault();
        const x = event.pageX - refEl.current.offsetLeft;
        refEl.current.scrollLeft = drag.current.scrollLeft - (x - drag.current.startX) * 1.5;
      }}
    >
      {projectData.timeline.map((clip, index) => {
        const feedback = projectData.feedback.find((item) => item.clip_used === clip.clip);
        const prompt = findPrompt(clip.clip);
        const versions = getVersions(prompt);
        const selectedVersion = selectedVersions[clip.clip] ?? Math.max(versions.length - 1, 0);
        const version = versions[selectedVersion] || prompt || undefined;

        return (
          <div className="timeline-node" key={`${clip.clip}-${index}`}>
            <TimelineCard
              clip={clip}
              index={index}
              feedback={feedback}
              prompt={prompt}
              versions={versions}
              selectedVersion={selectedVersion}
              setSelectedVersion={(versionIndex) =>
                setSelectedVersions((current) => ({
                  ...current,
                  [clip.clip]: versionIndex,
                }))
              }
              version={version}
              runningIndexes={runningIndexes}
              executeWorkflow={executeWorkflow}
              openResultFromVersion={openResultFromVersion}
            />
            {index < projectData.timeline.length - 1 && <TimelineConnector />}
          </div>
        );
      })}
    </div>
  );
}

function TimelineCard({
  clip,
  index,
  feedback,
  prompt,
  versions,
  selectedVersion,
  setSelectedVersion,
  version,
  runningIndexes,
  executeWorkflow,
  openResultFromVersion,
}: {
  clip: TimelineClip;
  index: number;
  feedback?: FeedbackGroup;
  prompt: PromptRecord | null;
  versions: PromptVersion[];
  selectedVersion: number;
  setSelectedVersion: (versionIndex: number) => void;
  version?: PromptVersion;
  runningIndexes: Set<number>;
  executeWorkflow: (feedbackIndex: number) => void;
  openResultFromVersion: (version: PromptVersion) => void;
}) {
  const hasFeedback = Boolean(feedback?.feedback_items?.length);
  const latestError = prompt?.latest_error;

  return (
    <article className={`timeline-card ${hasFeedback ? "has-feedback" : ""}`}>
      {clip.clip_url && (
        <div
          className="clip-thumbnail-wrapper"
          onMouseEnter={(event) => {
            const video = event.currentTarget.querySelector("video");
            void video?.play();
          }}
          onMouseLeave={(event) => {
            const video = event.currentTarget.querySelector("video");
            video?.pause();
          }}
        >
          <video src={`${staticUrl(clip.clip_url)}#t=0.5`} preload="metadata" muted playsInline />
          <div className="thumbnail-hover-overlay">
            <Icon name="play" />
          </div>
        </div>
      )}

      <div className="card-header">
        <div className="clip-title-info">
          <div className="clip-title">{clip.clip}</div>
          <div className="clip-meta-subtitle">Sequence Position: #{index + 1}</div>
        </div>
        <div className="duration-badge">{clip.duration_s.toFixed(2)}s</div>
      </div>

      <div className="card-details">
        <Detail label="Start Timecode" value={clip.start_tc} />
        <Detail label="End Timecode" value={clip.end_tc} />
        <Detail label="Track Bounds" value={`${clip.start_s.toFixed(2)}s - ${clip.end_s.toFixed(2)}s`} />
      </div>

      {hasFeedback ? (
        <div className="feedback-container">
          <div className="feedback-box-header">
            <Icon name="comments" /> Clip Feedback ({feedback?.feedback_items.length})
          </div>
          <div className="feedback-list">
            {feedback?.feedback_items.map((item) => (
              <div className="feedback-item-card" key={item.raw_index}>
                <div className="feedback-item-header">
                  <span className={`tag tag-${item.category}`}>{item.category}</span>
                  <span className="feedback-timestamp">{item.timestamp || "No Timecode"}</span>
                </div>
                <div className="feedback-remark">{item.remark}</div>

                {version?.video_model_prompt ? (
                  <GeneratedPlan
                    versions={versions}
                    selectedVersion={selectedVersion}
                    onSelectVersion={setSelectedVersion}
                    version={version}
                    latestError={latestError}
                    running={runningIndexes.has(item.raw_index)}
                    onRun={() => executeWorkflow(item.raw_index)}
                    onDetails={() => openResultFromVersion(version)}
                  />
                ) : (
                  <>
                    {latestError && <ErrorWarning error={latestError} />}
                    <div className="workflow-btn-wrapper">
                      {runningIndexes.has(item.raw_index) ? (
                        <div className="inline-runner-status">
                          <Icon name="refresh" /> Running workflow...
                        </div>
                      ) : (
                        <button className="premium-btn" onClick={() => executeWorkflow(item.raw_index)}>
                          <Icon name="play" /> Execute Feedback Workflow
                        </button>
                      )}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="feedback-container no-feedback">
          <Icon name="check" /> No pending feedback on this clip
        </div>
      )}
    </article>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-item">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value}</span>
    </div>
  );
}

function GeneratedPlan({
  versions,
  selectedVersion,
  onSelectVersion,
  version,
  latestError,
  running,
  onRun,
  onDetails,
}: {
  versions: PromptVersion[];
  selectedVersion: number;
  onSelectVersion: (versionIndex: number) => void;
  version: PromptVersion;
  latestError?: string | null;
  running: boolean;
  onRun: () => void;
  onDetails: () => void;
}) {
  return (
    <div className="generated-plan-box">
      {versions.length > 1 && (
        <label className="version-selector-wrapper">
          <span>History:</span>
          <select value={selectedVersion} onChange={(event) => onSelectVersion(Number(event.target.value))}>
            {versions.map((item, index) => (
              <option key={`${item.timestamp || "version"}-${index}`} value={index}>
                {versionLabel(item, index)}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="generated-plan-label">
        <Icon name="magic" /> Generated Prompt Plan
      </div>
      <pre className="generated-prompt-preview">{version.video_model_prompt}</pre>
      {Boolean(version.selected_assets?.length) && (
        <div className="asset-tags">
          {version.selected_assets?.map((asset) => (
            <span className="tag tag-characters" key={asset} title={asset}>
              <Icon name="box" /> {basename(asset)}
            </span>
          ))}
        </div>
      )}
      {latestError && <ErrorWarning error={latestError} />}
      <div className="generated-actions">
        {running ? (
          <div className="inline-runner-status">
            <Icon name="refresh" /> Running workflow...
          </div>
        ) : (
          <button className="premium-btn secondary" onClick={onRun}>
            <Icon name="refresh" /> Generate Again
          </button>
        )}
        <button className="premium-btn" onClick={onDetails}>
          <Icon name="search" /> View Details
        </button>
      </div>
    </div>
  );
}

function ErrorWarning({ error }: { error: string }) {
  return (
    <div className="error-warning-box">
      <Icon name="warning" />
      <div>
        <strong>Latest Run Failed</strong>
        <pre>{error}</pre>
      </div>
    </div>
  );
}

function TimelineConnector() {
  return (
    <div className="timeline-connector" aria-hidden="true">
      <svg className="timeline-string" width="178" height="76" viewBox="0 0 178 76" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path className="string-shadow" d="M 0 38 C 35 14 52 58 80 38 C 109 17 132 59 178 38" />
        <path className="string-main" d="M 0 38 C 35 14 52 58 80 38 C 109 17 132 59 178 38" />
        <path className="string-thread string-thread-one" d="M 0 42 C 34 21 53 63 80 42 C 108 22 132 64 178 42" />
        <path className="string-thread string-thread-two" d="M 0 34 C 36 9 53 53 80 34 C 109 12 132 54 178 34" />
        <path className="string-glint" d="M 61 52 C 69 51 74 44 80 38 C 89 30 96 25 105 25" />
        <circle className="string-anchor string-anchor-start" cx="0" cy="38" r="7" />
        <circle className="string-anchor string-anchor-end" cx="178" cy="38" r="7" />
        <circle className="string-pin string-pin-start" cx="0" cy="38" r="2.5" />
        <circle className="string-pin string-pin-end" cx="178" cy="38" r="2.5" />
      </svg>
    </div>
  );
}
