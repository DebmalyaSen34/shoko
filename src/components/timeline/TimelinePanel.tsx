import { useRef, useState } from "react";
import type { CSSProperties, Dispatch, RefObject, SetStateAction } from "react";
import type { FeedbackGroup, GeneratedVideo, PreviewState, ProjectData, PromptRecord, PromptVersion, TimelineClip } from "../../types";
import { basename, formatSeconds, formatTimecode, getVersions, versionLabel } from "../../lib/format";
import { staticUrl } from "../../lib/api";
import { EmptyState } from "../EmptyState";
import { Icon } from "../Icon";
import { FormattedPrompt } from "../FormattedPrompt";

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
  findPrompt: (clipName: string, clipOccurrence?: number) => PromptRecord | null;
  openResultFromVersion: (version: PromptVersion) => void;
  generatingVideoKeys: Set<string>;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  setZoom: Dispatch<SetStateAction<number>>;
  onUploadFeedback: () => void;
  onAddManualFeedback: (clipName: string) => void;
  onOpenClipChat: (clipIndex: number) => void;
  onPreview: (preview: PreviewState) => void;
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
  generatingVideoKeys,
  onGenerateVideo,
  setZoom,
  onUploadFeedback,
  onAddManualFeedback,
  onOpenClipChat,
  onPreview,
}: TimelinePanelProps) {
  return (
    <section className="center-panel">
      <div className="panel-header sticky-header">
        <h2>
          <Icon name="timeline" /> Chronological Timeline
        </h2>
        <div className="timeline-stats">
          {projectData && (
            <button className="premium-btn" style={{ marginRight: "16px", padding: "6px 12px", fontSize: "12px", height: "30px" }} onClick={onUploadFeedback}>
              <Icon name="comments" /> Upload Feedback
            </button>
          )}
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

      {projectData && projectData.feedback.length === 0 && (
        <div className="no-feedback-banner" style={{ padding: "12px 24px", background: "rgba(192, 192, 200, 0.05)", borderBottom: "1px solid var(--border-color)", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", color: "var(--text-secondary)" }}>
            <Icon name="comments" />
            <span>No feedback has been uploaded for this project yet. Upload a feedback file to align and generate video prompts.</span>
          </div>
          <button className="premium-btn" onClick={onUploadFeedback}>
            <Icon name="plus" /> Upload Feedback File
          </button>
        </div>
      )}

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
        generatingVideoKeys={generatingVideoKeys}
        onGenerateVideo={onGenerateVideo}
        onAddManualFeedback={onAddManualFeedback}
        onOpenClipChat={onOpenClipChat}
        onPreview={onPreview}
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
  generatingVideoKeys,
  onGenerateVideo,
  onAddManualFeedback,
  onOpenClipChat,
  onPreview,
}: {
  refEl: RefObject<HTMLDivElement | null>;
  className: string;
  style: CSSProperties;
  loading: boolean;
  error: string;
  projectData: ProjectData | null;
  findPrompt: (clipName: string, clipOccurrence?: number) => PromptRecord | null;
  selectedVersions: Record<string, number>;
  setSelectedVersions: Dispatch<SetStateAction<Record<string, number>>>;
  runningIndexes: Set<number>;
  executeWorkflow: (feedbackIndex: number) => void;
  openResultFromVersion: (version: PromptVersion) => void;
  generatingVideoKeys: Set<string>;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onAddManualFeedback: (clipName: string) => void;
  onOpenClipChat: (clipIndex: number) => void;
  onPreview: (preview: PreviewState) => void;
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
        const feedback = projectData.feedback.find(
          (item) =>
            item.clip_used === clip.clip &&
            (typeof item.clip_occurrence !== "number" || item.clip_occurrence === index),
        );
        const prompt = findPrompt(clip.clip, index);
        const versions = getVersions(prompt);
        const clipKey = `${clip.clip}::${index}`;
        const selectedVersion = selectedVersions[clipKey] ?? Math.max(versions.length - 1, 0);
        const version = versions[selectedVersion] || prompt || undefined;
        const generatingVideo = generatingVideoKeys.has(`${index}:${selectedVersion}`);

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
                  [clipKey]: versionIndex,
                }))
              }
              version={version}
              runningIndexes={runningIndexes}
              generatingVideo={generatingVideo}
              executeWorkflow={executeWorkflow}
              onGenerateVideo={() => onGenerateVideo(index, selectedVersion)}
              openResultFromVersion={openResultFromVersion}
              onAddManualFeedback={onAddManualFeedback}
              onOpenClipChat={onOpenClipChat}
              onPreview={onPreview}
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
  generatingVideo,
  executeWorkflow,
  onGenerateVideo,
  openResultFromVersion,
  onAddManualFeedback,
  onOpenClipChat,
  onPreview,
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
  generatingVideo: boolean;
  executeWorkflow: (feedbackIndex: number) => void;
  onGenerateVideo: () => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  openResultFromVersion: (version: PromptVersion) => void;
  onAddManualFeedback: (clipName: string) => void;
  onOpenClipChat: (clipIndex: number) => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const hasFeedback = Boolean(feedback?.feedback_items?.length);
  const latestError = prompt?.latest_error;
  const isVisualFeedback = (category?: string) => {
    const normalized = (category || "video").toLowerCase();
    return normalized === "video" || normalized === "both";
  };

  return (
    <article className={`timeline-card ${hasFeedback ? "has-feedback" : ""}`}>
      <div className="card-horizontal-layout">
        {clip.clip_url && (
          <div className="clip-thumbnail-side">
            <div
              className="clip-thumbnail-wrapper side"
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
          </div>
        )}

        <div className="card-content-side">
          <div className="card-header">
            <div className="clip-title-info">
              <div className="clip-title">{clip.clip}</div>
              <div className="clip-meta-subtitle">Sequence Position: #{index + 1}</div>
            </div>
            <div className="duration-badge">{formatSeconds(clip.duration_s)}</div>
          </div>

          <div className="clip-card-actions">
            <button className="premium-btn secondary" type="button" onClick={() => onOpenClipChat(index)}>
              <Icon name="chat" /> Chat
            </button>
          </div>

          <div className="card-details">
            <Detail label="Timecode" value={`${formatTimecode(clip.start_tc)} – ${formatTimecode(clip.end_tc)}`} />
            <Detail label="Bounds" value={`${formatSeconds(clip.start_s)} – ${formatSeconds(clip.end_s)}`} />
          </div>

          {hasFeedback ? (
            <div className="feedback-container">
              <div className="feedback-box-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <Icon name="comments" /> Clip Feedback ({feedback?.feedback_items.length})
                </span>
                <button
                  className="icon-btn"
                  title="Add Clip Feedback"
                  type="button"
                  onClick={() => onAddManualFeedback(clip.clip)}
                  style={{
                    padding: "4px",
                    border: "0",
                    background: "transparent",
                    cursor: "pointer",
                    color: "var(--text-secondary)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                  }}
                >
                  <Icon name="plus" />
                </button>
              </div>
              <div className="feedback-list">
                {feedback?.feedback_items.map((item) => (
                  <div className="feedback-item-card" key={item.raw_index}>
                    <div className="feedback-item-header">
                      <span className={`tag tag-${item.category}`}>{item.category}</span>
                      <span className="feedback-timestamp">{formatTimecode(item.timestamp)}</span>
                    </div>
                    <div className="feedback-remark">{item.remark}</div>

                    {isVisualFeedback(item.category) && version?.video_model_prompt ? (
                      <GeneratedPlan
                        versions={versions}
                        selectedVersion={selectedVersion}
                        onSelectVersion={setSelectedVersion}
                        version={version}
                        latestError={latestError}
                        running={runningIndexes.has(item.raw_index)}
                        generatingVideo={generatingVideo}
                        onRun={() => executeWorkflow(item.raw_index)}
                        onGenerateVideo={onGenerateVideo}
                        onDetails={() => openResultFromVersion(version)}
                        onPreview={onPreview}
                      />
                    ) : !isVisualFeedback(item.category) ? (
                      <div className="workflow-btn-wrapper">
                        <div className="muted-small">
                          Audio feedback is tracked on this clip, but this generated result is a video prompt. Run visual feedback to generate video prompt details.
                        </div>
                      </div>
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
            <div className="feedback-container no-feedback" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <Icon name="check" /> No pending feedback on this clip
              </div>
              <button
                className="premium-btn secondary"
                type="button"
                style={{ fontSize: "11px", padding: "4px 8px" }}
                onClick={() => onAddManualFeedback(clip.clip)}
              >
                <Icon name="plus" /> Add Feedback
              </button>
            </div>
          )}
        </div>
      </div>
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
  generatingVideo,
  onRun,
  onGenerateVideo,
  onDetails,
  onPreview,
}: {
  versions: PromptVersion[];
  selectedVersion: number;
  onSelectVersion: (versionIndex: number) => void;
  version: PromptVersion;
  latestError?: string | null;
  running: boolean;
  generatingVideo: boolean;
  onRun: () => void;
  onGenerateVideo: () => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onDetails: () => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const videos = version.generated_videos || [];
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(Math.max(videos.length - 1, 0));
  const selectedVideo = videos[Math.min(selectedVideoIndex, Math.max(videos.length - 1, 0))];

  return (
    <div className="generated-plan-box">
      <button
        type="button"
        className="prompt-toggle-btn"
        onClick={() => setExpanded(!expanded)}
      >
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Icon name="magic" />
          {expanded ? "Hide generated prompt ▴" : "Show generated prompt ▾"}
        </span>
      </button>

      {expanded && (
        <>
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
          
          <FormattedPrompt
            className="generated-prompt-preview"
            text={version.video_model_prompt}
          />

          {Boolean(version.selected_assets?.length) && (
            <div className="asset-tags" style={{ marginTop: "10px" }}>
              {version.selected_assets?.map((asset) => (
                <span className="tag tag-characters" key={asset} title={asset}>
                  <Icon name="box" /> {basename(asset)}
                </span>
              ))}
            </div>
          )}
        </>
      )}

      {videos.length > 0 && (
        <div className="generated-video-box">
          <div className="generated-video-header">
            <span>
              <Icon name="video" /> Generated Clip
            </span>
            {videos.length > 1 && (
              <select value={selectedVideoIndex} onChange={(event) => setSelectedVideoIndex(Number(event.target.value))}>
                {videos.map((video, index) => (
                  <option key={`${video.timestamp || "video"}-${index}`} value={index}>
                    Video v{video.version || index + 1}
                  </option>
                ))}
              </select>
            )}
          </div>
          {selectedVideo && (
            <>
              <video src={staticUrl(selectedVideo.url || selectedVideo.path)} controls playsInline preload="metadata" />
              <div className="generated-video-footer">
                <div>
                  <strong>{selectedVideo.label || `Generated video v${selectedVideo.version}`}</strong>
                  <span>
                    {selectedVideo.resolution || "720p"} · {selectedVideo.ratio || "9:16"} · {selectedVideo.duration || 5}s
                  </span>
                </div>
                <button
                  className="premium-btn secondary"
                  type="button"
                  onClick={() =>
                    onPreview({
                      file: {
                        name: basename(selectedVideo.path || selectedVideo.url),
                        path: selectedVideo.path,
                        url: selectedVideo.url || selectedVideo.path,
                        type: "video",
                        size: "",
                      },
                    })
                  }
                >
                  <Icon name="expand" /> View Larger
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {latestError && <ErrorWarning error={latestError} />}
      
      <div className="generated-actions" style={{ marginTop: "8px" }}>
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
        <button className="premium-btn" disabled={generatingVideo} onClick={() => void onGenerateVideo().catch(() => undefined)}>
          <Icon name="video" /> {generatingVideo ? "Generating..." : videos.length ? "Generate New Video" : "Generate Video"}
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
        <line x1="8" y1="38" x2="170" y2="38" stroke="var(--border-color)" strokeWidth="2" strokeDasharray="5 5" />
        <circle cx="4" cy="38" r="4" fill="var(--border-color)" />
        <circle cx="174" cy="38" r="4" fill="var(--border-color)" />
      </svg>
    </div>
  );
}
