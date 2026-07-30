import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, MouseEvent, RefObject, SetStateAction } from "react";
import type { ClipState, FeedbackGroup, GeneratedVideo, PreviewState, ProjectData, PromptRecord, PromptVersion, Provider, TimelineClip, TimelineFilmstripFrame, TimelineWaveformSegment } from "../../types";
import { basename, formatSeconds, formatTimecode, getVersions } from "../../lib/format";
import { apiUrl, staticUrl } from "../../lib/api";
import { EmptyState } from "../EmptyState";
import { Icon } from "../Icon";
import { ClipChatPanel } from "../chat/ClipChatPanel";
import { ClipInspectorPanel, type ClipInspectorTab } from "./ClipInspectorPanel";
import { ClipPreviewPanel } from "./ClipPreviewPanel";

export type TimelinePanelProps = {
  duration: string;
  error: string;
  loading: boolean;
  projectData: ProjectData | null;
  provider: Provider;
  refEl: RefObject<HTMLDivElement | null>;
  runningIndexes: Set<number>;
  zoom: number;
  executeWorkflow: (feedbackIndex: number) => Promise<void> | void;
  findPrompt: (clipName: string, clipOccurrence?: number) => PromptRecord | null;
  openResultFromVersion: (version: PromptVersion) => void;
  generatingVideoKeys: Set<string>;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  setZoom: Dispatch<SetStateAction<number>>;
  onUploadFeedback: () => void;
  onAddManualFeedback: (clipName: string) => void;
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

type FilmstripLoadState = {
  status: "loading" | "ready" | "failed";
  frames: TimelineFilmstripFrame[];
};

type WaveformLoadState = {
  status: "idle" | "loading" | "ready" | "failed";
  segments: TimelineWaveformSegment[];
};

const REFERENCE_SELECTED_CLIP_INDEX = 2;
const TIMELINE_GUTTER = 56;
const TIMELINE_END_PADDING = 48;
const BASE_PX_PER_SECOND = 18;
const MIN_SEGMENT_WIDTH = 28;
const MIN_TIMELINE_ZOOM = 0.5;
const DEFAULT_TIMELINE_ZOOM = 2;
const MAX_TIMELINE_ZOOM = 2.5;
const MIN_FILMSTRIP_WIDTH = 64;

function preferredClipIndex(clipCount: number) {
  return Math.min(REFERENCE_SELECTED_CLIP_INDEX, Math.max(clipCount - 1, 0));
}

export function TimelinePanel({
  duration,
  error,
  loading,
  projectData,
  provider,
  refEl,
  runningIndexes,
  zoom,
  executeWorkflow,
  findPrompt,
  openResultFromVersion,
  generatingVideoKeys,
  onGenerateVideo,
  setZoom,
  onUploadFeedback,
  onAddManualFeedback,
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
          <button className="timeline-zoom-icon" title="Zoom Out" type="button" onClick={() => setZoom((value) => Math.max(MIN_TIMELINE_ZOOM, Number((value - 0.1).toFixed(2))))}>
            <Icon name="zoomOut" />
          </button>
          <button className="timeline-zoom-icon" title="Zoom In" type="button" onClick={() => setZoom((value) => Math.min(MAX_TIMELINE_ZOOM, Number((value + 0.1).toFixed(2))))}>
            <Icon name="zoomIn" />
          </button>
          <div className="timeline-zoom-group">
            <input
              className="premium-slider"
              type="range"
              min={MIN_TIMELINE_ZOOM}
              max={MAX_TIMELINE_ZOOM}
              step="0.05"
              value={zoom}
              aria-label="Timeline zoom"
              onChange={(event) => setZoom(Number(event.target.value))}
            />
          </div>
          <button className="timeline-fit-button" type="button" onClick={() => setZoom(DEFAULT_TIMELINE_ZOOM)}>
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
        provider={provider}
        refEl={refEl}
        runningIndexes={runningIndexes}
        zoom={zoom}
        executeWorkflow={executeWorkflow}
        findPrompt={findPrompt}
        openResultFromVersion={openResultFromVersion}
        generatingVideoKeys={generatingVideoKeys}
        onGenerateVideo={onGenerateVideo}
        onAddManualFeedback={onAddManualFeedback}
        onPreview={onPreview}
      />
    </section>
  );
}

function CompactTimeline({
  error,
  loading,
  projectData,
  provider,
  refEl,
  runningIndexes,
  zoom,
  executeWorkflow,
  findPrompt,
  openResultFromVersion,
  generatingVideoKeys,
  onGenerateVideo,
  onAddManualFeedback,
  onPreview,
}: {
  error: string;
  loading: boolean;
  projectData: ProjectData | null;
  provider: Provider;
  refEl: RefObject<HTMLDivElement | null>;
  runningIndexes: Set<number>;
  zoom: number;
  executeWorkflow: (feedbackIndex: number) => Promise<void> | void;
  findPrompt: (clipName: string, clipOccurrence?: number) => PromptRecord | null;
  openResultFromVersion: (version: PromptVersion) => void;
  generatingVideoKeys: Set<string>;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onAddManualFeedback: (clipName: string) => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const drag = useRef({ down: false, mode: "pan" as "pan" | "scrub", startX: 0, scrollLeft: 0 });
  const requestedFilmstrips = useRef<Set<string>>(new Set());
  const [selectedClipIndex, setSelectedClipIndex] = useState(REFERENCE_SELECTED_CLIP_INDEX);
  const [selectedClipState, setSelectedClipState] = useState<ClipState | null>(null);
  const [playheadSeconds, setPlayheadSeconds] = useState(0);
  const [filmstrips, setFilmstrips] = useState<Record<number, FilmstripLoadState>>({});
  const [waveform, setWaveform] = useState<WaveformLoadState>({ status: "idle", segments: [] });
  const promptStateKey = useMemo(
    () => (projectData?.prompts || [])
      .map((prompt) => {
        const versions = getVersions(prompt);
        const latest = versions[versions.length - 1];
        return `${prompt.clip_used || ""}:${prompt.clip_occurrence ?? ""}:${versions.length}:${latest?.prompt_version_id || latest?.timestamp || ""}`;
      })
      .join("|"),
    [projectData?.prompts],
  );

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
    const nextIndex = preferredClipIndex(clipCount);
    setSelectedClipIndex(nextIndex);
    setPlayheadSeconds(projectData?.timeline[nextIndex]?.start_s || 0);
  }, [projectData?.project_name]);

  useEffect(() => {
    const clipCount = projectData?.timeline.length || 0;
    if (clipCount === 0) return;
    setSelectedClipIndex((current) => Math.min(Math.max(current, 0), clipCount - 1));
  }, [projectData?.timeline.length]);

  useEffect(() => {
    requestedFilmstrips.current.clear();
    setFilmstrips({});
    setWaveform({ status: "idle", segments: [] });
    setSelectedClipState(null);
  }, [projectData?.project_name]);

  useEffect(() => {
    if (!projectData?.project_name || projectData.timeline.length === 0) return;
    const controller = new AbortController();
    const clipIndex = Math.min(Math.max(selectedClipIndex, 0), projectData.timeline.length - 1);
    fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/clips/${clipIndex}/state`), {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load clip state");
        return response.json() as Promise<ClipState>;
      })
      .then((state) => setSelectedClipState(state))
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setSelectedClipState(null);
      });

    return () => controller.abort();
  }, [projectData?.project_name, projectData?.timeline.length, selectedClipIndex, promptStateKey]);

  useEffect(() => {
    if (!projectData?.project_name) return;
    const controller = new AbortController();
    setWaveform({ status: "loading", segments: [] });
    fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/audio-waveform?bins=600&mode=master`), {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load audio waveform");
        return response.json() as Promise<{ segments?: TimelineWaveformSegment[] }>;
      })
      .then((data) => {
        setWaveform({ status: "ready", segments: data.segments || [] });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setWaveform({ status: "failed", segments: [] });
      });

    return () => controller.abort();
  }, [projectData?.project_name]);

  useEffect(() => {
    if (!projectData?.project_name || layouts.length === 0) return;
    const projectName = projectData.project_name;
    const controllers: AbortController[] = [];

    layouts.forEach((layout) => {
      const requestKey = `${projectName}:${layout.index}`;
      if (requestedFilmstrips.current.has(requestKey) || !layout.clip.clip_url) return;
      requestedFilmstrips.current.add(requestKey);
      const controller = new AbortController();
      controllers.push(controller);
      setFilmstrips((current) => ({
        ...current,
        [layout.index]: { status: "loading", frames: [] },
      }));
      fetch(apiUrl(`/api/projects/${encodeURIComponent(projectName)}/clips/${layout.index}/filmstrip?frames=4`), {
        signal: controller.signal,
      })
        .then((response) => {
          if (!response.ok) throw new Error("Failed to load filmstrip");
          return response.json() as Promise<{ frames?: TimelineFilmstripFrame[] }>;
        })
        .then((data) => {
          setFilmstrips((current) => ({
            ...current,
            [layout.index]: { status: "ready", frames: data.frames || [] },
          }));
        })
        .catch((error) => {
          if (error instanceof DOMException && error.name === "AbortError") return;
          setFilmstrips((current) => ({
            ...current,
            [layout.index]: { status: "failed", frames: [] },
          }));
        });
    });

    return () => controllers.forEach((controller) => controller.abort());
  }, [layouts, projectData?.project_name]);

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
  const backendPromptState = selectedClipState?.clip_index === clampedSelectedClipIndex ? selectedClipState.active_prompt : null;
  const selectedPrompt = backendPromptState?.prompt || findPrompt(selectedClip.clip, clampedSelectedClipIndex);
  const selectedVersionsForClip = backendPromptState?.versions?.length ? backendPromptState.versions : getVersions(selectedPrompt);
  const selectedVersionIndex = backendPromptState?.version_index ?? Math.max(selectedVersionsForClip.length - 1, 0);
  const selectedVersion = backendPromptState?.version || selectedVersionsForClip[selectedVersionIndex] || selectedPrompt || undefined;
  const clampedPlayheadSeconds = Math.min(Math.max(playheadSeconds, 0), totalSeconds);
  const playheadLeft = TIMELINE_GUTTER + clampedPlayheadSeconds * pxPerSecond;
  const markers = buildTimelineMarkers(projectData, layouts, pxPerSecond);
  const playheadOffsetForSelectedClip =
    clampedPlayheadSeconds >= selectedClip.start_s && clampedPlayheadSeconds <= selectedClip.end_s
      ? Math.max(0, clampedPlayheadSeconds - selectedClip.start_s)
      : undefined;

  const clipIndexForSeconds = (seconds: number) => {
    const exact = layouts.find((layout) => seconds >= layout.clip.start_s && seconds < layout.clip.end_s);
    if (exact) return exact.index;
    if (layouts.length > 0 && seconds === layouts[layouts.length - 1].clip.end_s) return layouts[layouts.length - 1].index;
    let nearest = layouts[0];
    for (const layout of layouts) {
      const currentDistance = Math.min(Math.abs(seconds - layout.clip.start_s), Math.abs(seconds - layout.clip.end_s));
      const nearestDistance = Math.min(Math.abs(seconds - nearest.clip.start_s), Math.abs(seconds - nearest.clip.end_s));
      if (currentDistance < nearestDistance) nearest = layout;
    }
    return nearest.index;
  };

  const setTimelinePlayhead = (seconds: number) => {
    const nextSeconds = Math.min(Math.max(seconds, 0), totalSeconds);
    setPlayheadSeconds(nextSeconds);
    setSelectedClipIndex(clipIndexForSeconds(nextSeconds));
  };

  const secondsFromPointer = (clientX: number) => {
    const stage = refEl.current?.querySelector<HTMLElement>(".compact-timeline-stage");
    if (!stage) return clampedPlayheadSeconds;
    const rect = stage.getBoundingClientRect();
    return (clientX - rect.left - TIMELINE_GUTTER) / pxPerSecond;
  };

  const scrubFromPointer = (clientX: number) => {
    setTimelinePlayhead(secondsFromPointer(clientX));
  };

  const seekClipFromClick = (event: MouseEvent<HTMLElement>, layout: ClipLayout) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    setTimelinePlayhead(layout.clip.start_s + ratio * Math.max(layout.clip.duration_s, 0.1));
  };

  const seekMarker = (marker: TimelineMarker) => {
    const markerSeconds = (marker.left - TIMELINE_GUTTER) / pxPerSecond;
    setTimelinePlayhead(markerSeconds);
  };

  return (
    <div className="compact-timeline-workspace">
      <div
        ref={refEl}
        className="compact-timeline-scroll"
        onPointerDown={(event) => {
          const target = event.target as HTMLElement;
          if (target.closest(".timeline-playhead-grab")) {
            if (!refEl.current) return;
            drag.current = { down: true, mode: "scrub", startX: event.pageX - refEl.current.offsetLeft, scrollLeft: refEl.current.scrollLeft };
            refEl.current.classList.add("scrubbing");
            scrubFromPointer(event.clientX);
            event.currentTarget.setPointerCapture(event.pointerId);
            return;
          }
          if (target.closest("button, select, input, video")) return;
          if (!refEl.current) return;
          const scrubTarget = target.closest(".timeline-ruler, .compact-audio-track, .timeline-playhead-grab");
          if (scrubTarget) {
            drag.current = { down: true, mode: "scrub", startX: event.pageX - refEl.current.offsetLeft, scrollLeft: refEl.current.scrollLeft };
            refEl.current.classList.add("scrubbing");
            scrubFromPointer(event.clientX);
            event.currentTarget.setPointerCapture(event.pointerId);
            return;
          }
          drag.current = {
            down: true,
            mode: "pan",
            startX: event.pageX - refEl.current.offsetLeft,
            scrollLeft: refEl.current.scrollLeft,
          };
          refEl.current.classList.add("grabbing");
        }}
        onPointerUp={(event) => {
          drag.current.down = false;
          refEl.current?.classList.remove("grabbing");
          refEl.current?.classList.remove("scrubbing");
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onPointerLeave={() => {
          drag.current.down = false;
          refEl.current?.classList.remove("grabbing");
          refEl.current?.classList.remove("scrubbing");
        }}
        onPointerMove={(event) => {
          if (!drag.current.down || !refEl.current) return;
          event.preventDefault();
          if (drag.current.mode === "scrub") {
            scrubFromPointer(event.clientX);
            return;
          }
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
            <span>{secondsToTimecode(clampedPlayheadSeconds)}</span>
            <button className="timeline-playhead-grab" type="button" aria-label="Drag playhead" />
            <i />
          </div>

          <div className="compact-clip-row">
            <div className="timeline-lane-label" aria-label="Video track">
              <Icon name="video" />
            </div>
            {layouts.map((layout) => {
              return (
                <CompactClipCard
                  key={`${layout.clip.clip}-${layout.index}`}
                  layout={layout}
                  selected={clampedSelectedClipIndex === layout.index}
                  filmstrip={filmstrips[layout.index]}
                  onSeek={(event) => seekClipFromClick(event, layout)}
                />
              );
            })}
            <FeedbackMarkers markers={markers} onSeek={seekMarker} />
          </div>

          <AudioWaveformTrack waveform={waveform} width={contentWidth} pxPerSecond={pxPerSecond} />
        </div>
      </div>

      <SelectedClipDock
        projectData={projectData}
        provider={provider}
        clipIndex={clampedSelectedClipIndex}
        clipState={selectedClipState?.clip_index === clampedSelectedClipIndex ? selectedClipState : null}
        clip={selectedClip}
        feedback={selectedFeedback}
        prompt={selectedPrompt}
        versions={selectedVersionsForClip}
        selectedVersion={selectedVersion}
        selectedVersionIndex={selectedVersionIndex}
        runningIndexes={runningIndexes}
        externalGeneratingVideo={generatingVideoKeys.has(`${clampedSelectedClipIndex}:${selectedVersionIndex}`)}
        executeWorkflow={executeWorkflow}
        onAddManualFeedback={onAddManualFeedback}
        onGenerateVideo={onGenerateVideo}
        onOpenDetails={openResultFromVersion}
        onPreview={onPreview}
        playheadOffsetSeconds={playheadOffsetForSelectedClip}
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
  filmstrip,
  onSeek,
}: {
  layout: ClipLayout;
  selected: boolean;
  filmstrip?: FilmstripLoadState;
  onSeek: (event: MouseEvent<HTMLElement>) => void;
}) {
  const clip = layout.clip;
  const canShowThumbnail = layout.width >= MIN_FILMSTRIP_WIDTH;
  const canShowName = layout.width >= 84;
  const canShowDuration = layout.width >= 104;
  const canShowCompactIndex = layout.width >= 18;

  return (
    <article className={`compact-clip-card ${selected ? "selected" : ""} ${layout.compact ? "compact" : ""}`} style={{ left: layout.left, width: layout.width }}>
      <button className="clip-thumb-button" type="button" aria-label={`Seek ${basename(clip.clip)}`} onClick={onSeek}>
        {canShowThumbnail && (
          <TimelineFilmstrip filmstrip={filmstrip} clip={clip} />
        )}
        {canShowName && (
          <div className="compact-clip-meta">
            <strong title={clip.clip}>{basename(clip.clip)}</strong>
            {canShowDuration && <span>{formatSeconds(clip.duration_s)}</span>}
          </div>
        )}
        {!canShowName && canShowCompactIndex && <span className="compact-clip-index">{layout.index + 1}</span>}
      </button>
    </article>
  );
}

function TimelineFilmstrip({ filmstrip, clip }: { filmstrip?: FilmstripLoadState; clip: TimelineClip }) {
  if (filmstrip?.status === "ready" && filmstrip.frames.length > 0) {
    return (
      <div className="compact-thumb filmstrip-thumb">
        {filmstrip.frames.map((frame, index) => (
          <img src={staticUrl(frame.url)} alt="" aria-hidden="true" key={`${frame.url}-${index}`} />
        ))}
      </div>
    );
  }

  if (filmstrip?.status === "loading") {
    return (
      <div className="compact-thumb filmstrip-loading" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
    );
  }

  return (
    <div className="compact-thumb filmstrip-fallback">
      {clip.clip_url ? <video src={`${staticUrl(clip.clip_url)}#t=0.5`} preload="metadata" muted playsInline /> : <Icon name="video" />}
    </div>
  );
}

function FeedbackMarkers({ markers, onSeek }: { markers: TimelineMarker[]; onSeek: (marker: TimelineMarker) => void }) {
  if (markers.length === 0) return null;

  return (
    <div className="timeline-feedback-markers" aria-label="Feedback markers">
      {markers.map((marker) => (
        <button
          className={`timeline-feedback-marker ${feedbackMarkerClass(marker)}`}
          key={`${marker.feedback.clip_used}-${marker.item.raw_index}`}
          style={{ left: marker.left }}
          type="button"
          title={`${marker.item.category || marker.feedback.category || "video"} review - ${formatTimecode(marker.item.timestamp)}: ${marker.item.remark}`}
          onClick={() => onSeek(marker)}
        >
          <Icon name="comments" />
        </button>
      ))}
    </div>
  );
}

function AudioWaveformTrack({ waveform, width, pxPerSecond }: { waveform: WaveformLoadState; width: number; pxPerSecond: number }) {
  const placeholderPeaks = [0.18, 0.34, 0.22, 0.42, 0.26, 0.3, 0.16, 0.38];

  return (
    <div className="compact-audio-track">
      <div className="track-label" aria-label="Audio track">
        <Icon name="audio" />
      </div>
      <div className={`waveform real-waveform ${waveform.status}`} aria-hidden="true" style={{ width }}>
        {waveform.status === "loading" && <span className="waveform-loading" style={{ left: TIMELINE_GUTTER, right: TIMELINE_END_PADDING }} />}
        {waveform.status !== "loading" && waveform.segments.length === 0 && <span className="waveform-empty-lane" style={{ left: TIMELINE_GUTTER, right: TIMELINE_END_PADDING }} />}
        {waveform.segments.map((segment) => {
          const left = TIMELINE_GUTTER + Math.max(0, segment.start_s || 0) * pxPerSecond;
          const segmentWidth = Math.max(2, Math.max(0, (segment.end_s || 0) - (segment.start_s || 0)) * pxPerSecond);
          const displayStep = Math.max(1, Math.ceil(segment.peaks.length / Math.max(80, Math.floor(segmentWidth / 3))));
          const peaks = segment.peaks.length ? segment.peaks.filter((_peak, index) => index % displayStep === 0) : placeholderPeaks;
          return (
            <div className={`waveform-segment ${segment.peaks.length ? "" : "empty"}`} key={`${segment.clip}-${segment.audio_index}-${segment.start_s}`} style={{ left, width: segmentWidth }} title={segment.clip}>
              {peaks.map((peak, index) => (
                <i key={`${segment.audio_index}-${index}`} style={{ height: `${Math.max(8, Math.round(peak * 34))}px` }} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function feedbackMarkerClass(marker: TimelineMarker) {
  const category = `${marker.item.category || marker.feedback.category || "video"}`.toLowerCase();
  if (category.includes("both")) return "both-review";
  if (category.includes("audio")) return "audio-review";
  return "video-review";
}

function SelectedClipDock({
  projectData,
  provider,
  clipIndex,
  clipState,
  clip,
  feedback,
  prompt,
  versions,
  selectedVersion,
  selectedVersionIndex,
  runningIndexes,
  externalGeneratingVideo,
  executeWorkflow,
  onAddManualFeedback,
  onGenerateVideo,
  onOpenDetails,
  onPreview,
  playheadOffsetSeconds,
}: {
  projectData: ProjectData;
  provider: Provider;
  clipIndex: number;
  clipState?: ClipState | null;
  clip: TimelineClip;
  feedback?: FeedbackGroup;
  prompt: PromptRecord | null;
  versions: PromptVersion[];
  selectedVersion?: PromptVersion;
  selectedVersionIndex: number;
  runningIndexes: Set<number>;
  externalGeneratingVideo: boolean;
  executeWorkflow: (feedbackIndex: number) => Promise<void> | void;
  onAddManualFeedback: (clipName: string) => void;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onOpenDetails: (version: PromptVersion) => void;
  onPreview: (preview: PreviewState) => void;
  playheadOffsetSeconds?: number;
}) {
  const feedbackItems = feedback?.feedback_items || [];
  const [activeTab, setActiveTab] = useState<ClipInspectorTab>("assets");

  return (
    <aside className="lower-context-panels">
      <ClipPreviewPanel clip={clip} playheadOffsetSeconds={playheadOffsetSeconds} />
      <ClipInspectorPanel
        activeTab={activeTab}
        clip={clip}
        feedbackItems={feedbackItems}
        projectData={projectData}
        clipState={clipState}
        versions={versions}
        selectedVersion={selectedVersion}
        selectedVersionIndex={selectedVersionIndex}
        setActiveTab={setActiveTab}
        onAddManualFeedback={onAddManualFeedback}
        onPreview={onPreview}
      />
      <ClipChatPanel
        variant="dock"
        clipIndex={clipIndex}
        feedback={feedback}
        prompt={prompt}
        projectData={projectData}
        provider={provider}
        runningIndexes={runningIndexes}
        externalGeneratingVideo={externalGeneratingVideo}
        onExecuteWorkflow={executeWorkflow}
        onOpenPromptDetails={onOpenDetails}
        onGenerateVideo={onGenerateVideo}
        onPreview={onPreview}
      />
    </aside>
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
