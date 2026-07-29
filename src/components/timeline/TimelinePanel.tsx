import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, RefObject, SetStateAction } from "react";
import type { FeedbackGroup, GeneratedVideo, PreviewState, ProjectData, PromptRecord, PromptVersion, TimelineClip } from "../../types";
import { basename, formatSeconds, formatTimecode, getVersions, versionLabel } from "../../lib/format";
import { staticUrl } from "../../lib/api";
import { EmptyState } from "../EmptyState";
import { Icon } from "../Icon";
import { ClipInspectorPanel } from "./ClipInspectorPanel";
import { ClipPreviewPanel } from "./ClipPreviewPanel";

export type TimelinePanelProps = {
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

type ClipLayout = {
  clip: TimelineClip;
  index: number;
  left: number;
  width: number;
  scaledWidth: number;
  compact: boolean;
};

type TimelineMarker = {
  feedback: FeedbackGroup;
  item: FeedbackGroup["feedback_items"][number];
  left: number;
  clipIndex: number;
};

const TRACK_COLORS = ["#7c4dff", "#d99a38", "#bf3d76", "#2d6ad5", "#2f8a5b", "#6341d4"];
const WAVEFORM_PATTERN = [16, 29, 42, 55, 26, 39, 52, 23, 36, 49, 20, 33, 46, 17, 30, 43, 56, 27, 40, 53];
const REFERENCE_SELECTED_CLIP_INDEX = 2;
const TIMELINE_GUTTER = 56;
const TIMELINE_END_PADDING = 48;
const BASE_PX_PER_SECOND = 18;
const MIN_SEGMENT_WIDTH = 28;

function preferredClipIndex(clipCount: number) {
  return Math.min(REFERENCE_SELECTED_CLIP_INDEX, Math.max(clipCount - 1, 0));
}

export function TimelinePanel({
  duration,
  error,
  loading,
  projectData,
  refEl,
  runningIndexes,
  selectedVersions,
  setSelectedVersions,
  zoom,
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
    <section className="compact-timeline-panel">
      <div className="compact-timeline-header">
        <div className="timeline-title-group">
          <h2>Timeline</h2>
          <span>Duration {duration}</span>
        </div>
        <div className="compact-timeline-actions timeline-zoom-actions">
          <button className="timeline-zoom-icon" title="Zoom Out" type="button" onClick={() => setZoom((value) => Math.max(0.5, Number((value - 0.1).toFixed(2))))}>
            <Icon name="zoomOut" />
          </button>
          <button className="timeline-zoom-icon" title="Zoom In" type="button" onClick={() => setZoom((value) => Math.min(1.5, Number((value + 0.1).toFixed(2))))}>
            <Icon name="zoomIn" />
          </button>
          <div className="timeline-zoom-group">
            <input
              className="premium-slider"
              type="range"
              min="0.5"
              max="1.5"
              step="0.05"
              value={zoom}
              aria-label="Timeline zoom"
              onChange={(event) => setZoom(Number(event.target.value))}
            />
          </div>
          <button className="timeline-fit-button" type="button" onClick={() => setZoom(1)}>
            Fit
          </button>
        </div>
      </div>

      {projectData && projectData.feedback.length === 0 && (
        <div className="compact-no-feedback">
          <div>
            <Icon name="comments" />
            <span>No feedback has been uploaded for this project yet.</span>
          </div>
          <button className="shoko-ghost-button compact-action" type="button" onClick={onUploadFeedback}>
            Upload Feedback File
          </button>
        </div>
      )}

      <CompactTimeline
        error={error}
        loading={loading}
        projectData={projectData}
        refEl={refEl}
        runningIndexes={runningIndexes}
        selectedVersions={selectedVersions}
        setSelectedVersions={setSelectedVersions}
        zoom={zoom}
        executeWorkflow={executeWorkflow}
        findPrompt={findPrompt}
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

function CompactTimeline({
  error,
  loading,
  projectData,
  refEl,
  runningIndexes,
  selectedVersions,
  setSelectedVersions,
  zoom,
  executeWorkflow,
  findPrompt,
  openResultFromVersion,
  generatingVideoKeys,
  onGenerateVideo,
  onAddManualFeedback,
  onOpenClipChat,
  onPreview,
}: {
  error: string;
  loading: boolean;
  projectData: ProjectData | null;
  refEl: RefObject<HTMLDivElement | null>;
  runningIndexes: Set<number>;
  selectedVersions: Record<string, number>;
  setSelectedVersions: Dispatch<SetStateAction<Record<string, number>>>;
  zoom: number;
  executeWorkflow: (feedbackIndex: number) => void;
  findPrompt: (clipName: string, clipOccurrence?: number) => PromptRecord | null;
  openResultFromVersion: (version: PromptVersion) => void;
  generatingVideoKeys: Set<string>;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onAddManualFeedback: (clipName: string) => void;
  onOpenClipChat: (clipIndex: number) => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const drag = useRef({ down: false, startX: 0, scrollLeft: 0 });
  const [selectedClipIndex, setSelectedClipIndex] = useState(REFERENCE_SELECTED_CLIP_INDEX);

  const { layouts, contentWidth, pxPerSecond, totalSeconds } = useMemo(() => {
    const clips = projectData?.timeline || [];
    const sequenceSeconds = Math.max(projectData?.total_duration_s || 0, ...clips.map((clip) => clip.end_s || 0));
    const nextPxPerSecond = BASE_PX_PER_SECOND * zoom;
    const nextLayouts: ClipLayout[] = clips.map((clip, index) => {
      const duration = Math.max(clip.duration_s || clip.end_s - clip.start_s || 0, 0.1);
      const scaledWidth = duration * nextPxPerSecond;
      const width = Math.max(scaledWidth, MIN_SEGMENT_WIDTH);
      return {
        clip,
        index,
        left: TIMELINE_GUTTER + Math.max(0, clip.start_s || 0) * nextPxPerSecond,
        width,
        scaledWidth,
        compact: width < 84,
      };
    });
    return {
      layouts: nextLayouts,
      contentWidth: Math.max(1320, TIMELINE_GUTTER + sequenceSeconds * nextPxPerSecond + TIMELINE_END_PADDING),
      pxPerSecond: nextPxPerSecond,
      totalSeconds: sequenceSeconds,
    };
  }, [projectData, zoom]);

  useEffect(() => {
    const clipCount = projectData?.timeline.length || 0;
    if (clipCount === 0) return;
    setSelectedClipIndex(preferredClipIndex(clipCount));
  }, [projectData?.project_name]);

  useEffect(() => {
    const clipCount = projectData?.timeline.length || 0;
    if (clipCount === 0) return;
    setSelectedClipIndex((current) => Math.min(Math.max(current, 0), clipCount - 1));
  }, [projectData?.timeline.length]);

  if (loading) {
    return (
      <div className="compact-timeline-empty">
        <LoadingPulse label="Loading timeline..." description="Fetching media clips, coordinates, and feedback." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="compact-timeline-empty">
        <EmptyState icon="warning" title="Error Loading Project" text={error} />
      </div>
    );
  }

  if (!projectData) {
    return (
      <div className="compact-timeline-empty">
        <EmptyState icon="video" title="No Project Selected" text="Choose a project from the header to load the timeline." />
      </div>
    );
  }

  if (projectData.timeline.length === 0) {
    return (
      <div className="compact-timeline-empty">
        <EmptyState icon="video" title="Timeline Empty" text="This project's timeline has no video clip records." />
      </div>
    );
  }

  const clampedSelectedClipIndex = Math.min(Math.max(selectedClipIndex, 0), layouts.length - 1);
  const selectedLayout = layouts.find((layout) => layout.index === clampedSelectedClipIndex) || layouts[0];
  const selectedClip = selectedLayout.clip;
  const selectedFeedback = findFeedback(projectData, selectedClip, clampedSelectedClipIndex);
  const selectedPrompt = findPrompt(selectedClip.clip, clampedSelectedClipIndex);
  const selectedVersionsForClip = getVersions(selectedPrompt);
  const clipKey = `${selectedClip.clip}::${clampedSelectedClipIndex}`;
  const selectedVersionIndex = selectedVersions[clipKey] ?? Math.max(selectedVersionsForClip.length - 1, 0);
  const selectedVersion = selectedVersionsForClip[selectedVersionIndex] || selectedPrompt || undefined;
  const playheadLeft = TIMELINE_GUTTER + Math.max(0, selectedClip.start_s || 0) * pxPerSecond;
  const markers = buildTimelineMarkers(projectData, layouts, pxPerSecond);

  return (
    <div className="compact-timeline-workspace">
      <div
        ref={refEl}
        className="compact-timeline-scroll"
        onMouseDown={(event) => {
          const target = event.target as HTMLElement;
          if (target.closest("button, select, input, video")) return;
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
          refEl.current.scrollLeft = drag.current.scrollLeft - (x - drag.current.startX) * 1.25;
        }}
      >
        <div className="compact-timeline-stage" style={{ width: contentWidth }}>
          <TimelineRuler width={contentWidth} totalSeconds={totalSeconds} pxPerSecond={pxPerSecond} />
          <div
            className="timeline-selected-range"
            style={{
              left: selectedLayout.left,
              width: Math.max(selectedLayout.scaledWidth, 2),
            }}
          />
          <div className="timeline-playhead" style={{ left: playheadLeft }}>
            <span>{formatTimecode(selectedClip.start_tc)}</span>
            <i />
          </div>

          <div className="compact-clip-row">
            <div className="timeline-lane-label">
              <span>V1</span>
            </div>
            {layouts.map((layout) => {
              return (
                <CompactClipCard
                  key={`${layout.clip.clip}-${layout.index}`}
                  layout={layout}
                  selected={clampedSelectedClipIndex === layout.index}
                  onSelect={() => setSelectedClipIndex(layout.index)}
                  onOpenChat={() => onOpenClipChat(layout.index)}
                  onPreview={onPreview}
                />
              );
            })}
          </div>

          <FeedbackMarkers markers={markers} onSelectClip={setSelectedClipIndex} />
          <VideoTrack layouts={layouts} />
          <MockWaveform width={contentWidth} pxPerSecond={pxPerSecond} totalSeconds={totalSeconds} />
        </div>
      </div>

      <SelectedClipDock
        projectData={projectData}
        clip={selectedClip}
        clipIndex={clampedSelectedClipIndex}
        feedback={selectedFeedback}
        versions={selectedVersionsForClip}
        selectedVersion={selectedVersion}
        selectedVersionIndex={selectedVersionIndex}
        setSelectedVersion={(versionIndex) =>
          setSelectedVersions((current) => ({
            ...current,
            [clipKey]: versionIndex,
          }))
        }
        runningIndexes={runningIndexes}
        generatingVideo={generatingVideoKeys.has(`${clampedSelectedClipIndex}:${selectedVersionIndex}`)}
        executeWorkflow={executeWorkflow}
        onAddManualFeedback={onAddManualFeedback}
        onGenerateVideo={() => onGenerateVideo(clampedSelectedClipIndex, selectedVersionIndex)}
        onOpenChat={() => onOpenClipChat(clampedSelectedClipIndex)}
        onOpenDetails={openResultFromVersion}
        onPreview={onPreview}
      />
    </div>
  );
}

function TimelineRuler({ width, totalSeconds, pxPerSecond }: { width: number; totalSeconds: number; pxPerSecond: number }) {
  const majorStepSeconds = rulerStepForScale(pxPerSecond, totalSeconds);
  const labelSeconds = new Set<number>();
  for (let second = 0; second <= totalSeconds; second += majorStepSeconds) {
    labelSeconds.add(Number(second.toFixed(3)));
  }
  const lastMajor = Math.floor(totalSeconds / majorStepSeconds) * majorStepSeconds;
  if (totalSeconds - lastMajor >= Math.min(majorStepSeconds * 0.55, 8)) {
    labelSeconds.add(Math.max(0, totalSeconds));
  }
  const ticks = Array.from(labelSeconds).map((second) => {
    return {
      left: TIMELINE_GUTTER + second * pxPerSecond,
      label: secondsToTimecode(second),
    };
  });
  const minorTickWidth = Math.max(8, pxPerSecond * Math.max(1, majorStepSeconds / 5));

  return (
    <div className="timeline-ruler" style={{ width }}>
      {ticks.map((tick) => (
        <div className="ruler-tick" style={{ left: tick.left }} key={tick.label}>
          <span>{tick.label}</span>
          <i />
        </div>
      ))}
      <div className="minor-ticks" style={{ left: TIMELINE_GUTTER, right: TIMELINE_END_PADDING, backgroundSize: `${minorTickWidth}px 12px` }} />
    </div>
  );
}

function CompactClipCard({
  layout,
  selected,
  onSelect,
  onOpenChat,
  onPreview,
}: {
  layout: ClipLayout;
  selected: boolean;
  onSelect: () => void;
  onOpenChat: () => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const clip = layout.clip;
  const canShowThumbnail = layout.width >= 112;
  const canShowName = layout.width >= 84;
  const canShowDuration = layout.width >= 104;
  const canShowCompactIndex = layout.width >= 18;

  return (
    <article className={`compact-clip-card ${selected ? "selected" : ""} ${layout.compact ? "compact" : ""}`} style={{ left: layout.left, width: layout.width }}>
      <button className="clip-thumb-button" type="button" aria-label={`Select ${basename(clip.clip)}`} onClick={onSelect}>
        {canShowThumbnail && (
          <div className="compact-thumb">
            {clip.clip_url ? <video src={`${staticUrl(clip.clip_url)}#t=0.5`} preload="metadata" muted playsInline /> : <Icon name="video" />}
            <span className="compact-play-mark">
              <Icon name="play" />
            </span>
          </div>
        )}
        {canShowName && (
          <div className="compact-clip-meta">
            <strong title={clip.clip}>{basename(clip.clip)}</strong>
            {canShowDuration && <span>{formatSeconds(clip.duration_s)}</span>}
          </div>
        )}
        {!canShowName && canShowCompactIndex && <span className="compact-clip-index">{layout.index + 1}</span>}
      </button>
      {!layout.compact && <div className="compact-clip-actions">
        <button type="button" title="Chat" onClick={onOpenChat}>
          <Icon name="chat" />
        </button>
        {clip.clip_url && (
          <button
            type="button"
            title="Preview"
            onClick={() =>
              onPreview({
                file: {
                  name: basename(clip.clip),
                  path: clip.clip_url || "",
                  url: clip.clip_url || "",
                  type: "video",
                  size: "",
                },
              })
            }
          >
            <Icon name="expand" />
          </button>
        )}
      </div>}
    </article>
  );
}

function VideoTrack({ layouts }: { layouts: ClipLayout[] }) {
  return (
    <div className="compact-video-track">
      <div className="track-label" aria-label="Video track">
        <Icon name="video" />
      </div>
      <div className="track-content">
        {layouts.map((layout) => {
          return (
            <div
              className="track-block"
              key={`${layout.clip.clip}-track`}
              style={{
                left: layout.left,
                width: Math.max(layout.scaledWidth, 2),
                background: TRACK_COLORS[layout.index % TRACK_COLORS.length],
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function FeedbackMarkers({ markers, onSelectClip }: { markers: TimelineMarker[]; onSelectClip: (clipIndex: number) => void }) {
  if (markers.length === 0) return null;

  return (
    <div className="timeline-feedback-markers" aria-label="Feedback markers">
      {markers.map((marker) => (
        <button
          className="timeline-feedback-marker"
          key={`${marker.feedback.clip_used}-${marker.item.raw_index}`}
          style={{ left: marker.left }}
          type="button"
          title={`${formatTimecode(marker.item.timestamp)}: ${marker.item.remark}`}
          onClick={() => onSelectClip(marker.clipIndex)}
        >
          <Icon name="comments" />
        </button>
      ))}
    </div>
  );
}

function MockWaveform({ width, pxPerSecond, totalSeconds }: { width: number; pxPerSecond: number; totalSeconds: number }) {
  const timelineWidth = Math.max(0, totalSeconds * pxPerSecond);
  const count = Math.max(80, Math.floor(timelineWidth / 3));
  const bars = Array.from({ length: count }, (_item, index) => WAVEFORM_PATTERN[index % WAVEFORM_PATTERN.length]);

  return (
    <div className="compact-audio-track">
      <div className="track-label" aria-label="Audio track">
        <Icon name="audio" />
      </div>
      <div className="waveform" aria-hidden="true" style={{ width, paddingLeft: TIMELINE_GUTTER, paddingRight: TIMELINE_END_PADDING }}>
        {bars.map((height, index) => (
          <i key={index} style={{ height: Math.max(8, Math.round(height * 0.52)) }} />
        ))}
      </div>
    </div>
  );
}

function SelectedClipDock({
  projectData,
  clip,
  clipIndex,
  feedback,
  versions,
  selectedVersion,
  selectedVersionIndex,
  setSelectedVersion,
  runningIndexes,
  generatingVideo,
  executeWorkflow,
  onAddManualFeedback,
  onGenerateVideo,
  onOpenChat,
  onOpenDetails,
  onPreview,
}: {
  projectData: ProjectData;
  clip: TimelineClip;
  clipIndex: number;
  feedback?: FeedbackGroup;
  versions: PromptVersion[];
  selectedVersion?: PromptVersion;
  selectedVersionIndex: number;
  setSelectedVersion: (versionIndex: number) => void;
  runningIndexes: Set<number>;
  generatingVideo: boolean;
  executeWorkflow: (feedbackIndex: number) => void;
  onAddManualFeedback: (clipName: string) => void;
  onGenerateVideo: () => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onOpenChat: () => void;
  onOpenDetails: (version: PromptVersion) => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const feedbackItems = feedback?.feedback_items || [];
  const runnableFeedback = feedbackItems[0];
  const running = feedbackItems.some((item) => runningIndexes.has(item.raw_index));
  const [activeTab, setActiveTab] = useState<"feedback" | "details" | "assets">("assets");

  return (
    <aside className="lower-context-panels">
      <ClipPreviewPanel clip={clip} onPreview={onPreview} />
      <ClipInspectorPanel
        activeTab={activeTab}
        clip={clip}
        clipIndex={clipIndex}
        feedbackItems={feedbackItems}
        projectData={projectData}
        selectedVersion={selectedVersion}
        setActiveTab={setActiveTab}
        onAddManualFeedback={onAddManualFeedback}
        onPreview={onPreview}
      />
      <TimelineClipChatPanel
        clip={clip}
        feedbackItems={feedbackItems}
        runnableFeedback={runnableFeedback}
        running={running}
        generatingVideo={generatingVideo}
        hasPrompt={Boolean(selectedVersion?.video_model_prompt)}
        versions={versions}
        selectedVersion={selectedVersion}
        selectedVersionIndex={selectedVersionIndex}
        setSelectedVersion={setSelectedVersion}
        executeWorkflow={executeWorkflow}
        onGenerateVideo={onGenerateVideo}
        onOpenChat={onOpenChat}
        onOpenDetails={onOpenDetails}
      />
    </aside>
  );
}

function TimelineClipChatPanel({
  clip,
  feedbackItems,
  runnableFeedback,
  running,
  generatingVideo,
  hasPrompt,
  versions,
  selectedVersion,
  selectedVersionIndex,
  setSelectedVersion,
  executeWorkflow,
  onGenerateVideo,
  onOpenChat,
  onOpenDetails,
}: {
  clip: TimelineClip;
  feedbackItems: FeedbackGroup["feedback_items"];
  runnableFeedback?: FeedbackGroup["feedback_items"][number];
  running: boolean;
  generatingVideo: boolean;
  hasPrompt: boolean;
  versions: PromptVersion[];
  selectedVersion?: PromptVersion;
  selectedVersionIndex: number;
  setSelectedVersion: (versionIndex: number) => void;
  executeWorkflow: (feedbackIndex: number) => void;
  onGenerateVideo: () => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onOpenChat: () => void;
  onOpenDetails: (version: PromptVersion) => void;
}) {
  const promptSummary = feedbackItems[0]?.remark || "Ask anything about this clip.";

  return (
    <section className="timeline-chat-panel">
      <div className="timeline-chat-header">
        <div>
          <h3>Chat with AI</h3>
          <span>{basename(clip.clip)}</span>
        </div>
        <button type="button" title="Open full chat" onClick={onOpenChat}>
          ...
        </button>
      </div>

      <div className="timeline-chat-log">
        <div className="chat-bubble user">
          <p>Make this shot more dramatic.</p>
          <time>10:30 AM</time>
        </div>
        <div className="chat-bubble assistant">
          <p>I can help improve this clip.</p>
          <p>Here is what I recommend:</p>
          <ul>
            <li>Add depth and contrast</li>
            <li>Enhance lighting and visual rhythm</li>
            <li>Use the clip feedback as the primary constraint</li>
          </ul>
          <p className="chat-context-line">{promptSummary}</p>
          <time>10:31 AM</time>
        </div>
      </div>

      <div className="timeline-chat-actions">
        {versions.length > 1 && (
          <select value={selectedVersionIndex} onChange={(event) => setSelectedVersion(Number(event.target.value))}>
            {versions.map((version, index) => (
              <option value={index} key={`${version.timestamp || "version"}-${index}`}>
                {versionLabel(version, index)}
              </option>
            ))}
          </select>
        )}
        {hasPrompt && selectedVersion ? (
          <>
            <button className="shoko-ghost-button compact-action" type="button" onClick={() => onOpenDetails(selectedVersion)}>
              Show examples
            </button>
            <button className="shoko-primary-button compact-action" type="button" disabled={generatingVideo} onClick={() => void onGenerateVideo().catch(() => undefined)}>
              {generatingVideo ? "Generating..." : "Yes, generate"}
            </button>
          </>
        ) : (
          <button
            className="shoko-primary-button compact-action"
            type="button"
            disabled={!runnableFeedback || running}
            onClick={() => runnableFeedback && executeWorkflow(runnableFeedback.raw_index)}
          >
            {running ? "Running..." : "Run Workflow"}
          </button>
        )}
      </div>

      <form
        className="timeline-chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          onOpenChat();
        }}
      >
        <input placeholder="Ask anything about this clip..." aria-label="Ask anything about this clip" />
        <button type="submit" title="Open chat">
          <Icon name="send" />
        </button>
      </form>
    </section>
  );
}

function findFeedback(projectData: ProjectData, clip: TimelineClip, index: number) {
  return projectData.feedback.find(
    (item) =>
      item.clip_used === clip.clip &&
      (typeof item.clip_occurrence !== "number" || item.clip_occurrence === index),
  );
}

function buildTimelineMarkers(projectData: ProjectData, layouts: ClipLayout[], pxPerSecond: number): TimelineMarker[] {
  return projectData.feedback.flatMap((feedback) => {
    const clipIndex = findFeedbackClipIndex(projectData, feedback);
    const layout = layouts.find((item) => item.index === clipIndex);
    if (!layout) return [];

    return feedback.feedback_items.map((item) => {
      const timestampSeconds = parseTimestampSeconds(item.timestamp, projectData.fps || 25);
      const markerSeconds =
        typeof timestampSeconds === "number"
          ? Math.min(Math.max(timestampSeconds, layout.clip.start_s), layout.clip.end_s)
          : layout.clip.start_s;

      return {
        feedback,
        item,
        clipIndex: layout.index,
        left: TIMELINE_GUTTER + markerSeconds * pxPerSecond,
      };
    });
  });
}

function findFeedbackClipIndex(projectData: ProjectData, feedback: FeedbackGroup) {
  const occurrence = typeof feedback.clip_occurrence === "number" ? feedback.clip_occurrence : undefined;
  const exactIndex = projectData.timeline.findIndex((clip, index) => clip.clip === feedback.clip_used && (occurrence === undefined || occurrence === index));
  if (exactIndex !== -1) return exactIndex;
  return Math.max(0, projectData.timeline.findIndex((clip) => clip.clip === feedback.clip_used));
}

function parseTimestampSeconds(timestamp?: string | null, fps = 25) {
  if (!timestamp) return undefined;
  const trimmed = timestamp.trim();
  if (!trimmed) return undefined;
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);

  const parts = trimmed.split(":").map((part) => Number(part));
  if (parts.some((part) => Number.isNaN(part))) return undefined;
  if (parts.length >= 4) {
    const [hours, minutes, seconds, frames] = parts;
    return hours * 3600 + minutes * 60 + seconds + frames / fps;
  }
  if (parts.length === 3) {
    const [hours, minutes, seconds] = parts;
    return hours * 3600 + minutes * 60 + seconds;
  }
  if (parts.length === 2) {
    const [minutes, seconds] = parts;
    return minutes * 60 + seconds;
  }
  return undefined;
}

function rulerStepForScale(pxPerSecond: number, totalSeconds: number) {
  const targetLabelSpacing = 150;
  const candidates = totalSeconds <= 90 ? [1, 2, 5, 10, 15, 30] : [5, 10, 15, 30, 60, 120, 300, 600];
  return candidates.find((step) => step * pxPerSecond >= targetLabelSpacing) || candidates[candidates.length - 1];
}

function secondsToTimecode(seconds?: number | null) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}:00`;
}

function LoadingPulse({ label, description }: { label: string; description: string }) {
  return (
    <div className="loading-pulse" role="status" aria-live="polite">
      <span className="loader-orbit large" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <div>
        <strong>{label}</strong>
        <p>{description}</p>
      </div>
    </div>
  );
}
