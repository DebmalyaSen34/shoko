import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import type { TimelineClip } from "../../types";
import { staticUrl } from "../../lib/api";
import { basename, formatSeconds, formatTimecode } from "../../lib/format";
import { Icon } from "../Icon";

type ClipPreviewPanelProps = {
  clip: TimelineClip;
  playheadOffsetSeconds?: number;
};

export function ClipPreviewPanel({ clip, playheadOffsetSeconds }: ClipPreviewPanelProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastSyncedPlayhead = useRef<number | undefined>(undefined);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(clip.duration_s || 0);
  const [loadFailed, setLoadFailed] = useState(false);
  const clipSrc = clip.clip_url ? staticUrl(clip.clip_url) : "";
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.currentTime = 0;
      video.load();
    }
    setCurrentTime(0);
    setDuration(clip.duration_s || 0);
    setLoadFailed(false);
    lastSyncedPlayhead.current = undefined;
  }, [clip.clip_url, clip.duration_s]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || playheadOffsetSeconds === undefined || !Number.isFinite(playheadOffsetSeconds)) return;
    const availableDuration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : duration || clip.duration_s || 0;
    const nextTime = Math.max(0, Math.min(playheadOffsetSeconds, availableDuration));
    if (lastSyncedPlayhead.current === nextTime || Math.abs(video.currentTime - nextTime) < 0.08) return;
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
    lastSyncedPlayhead.current = nextTime;
  }, [clip.duration_s, duration, playheadOffsetSeconds]);

  const seekToPoint = (event: MouseEvent<HTMLButtonElement>) => {
    const video = videoRef.current;
    if (!video) return;
    const seekDuration = video.duration || duration;
    if (seekDuration <= 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    video.currentTime = ratio * seekDuration;
    setCurrentTime(video.currentTime);
  };

  return (
    <section className="clip-preview-panel">
      <h3>Clip: {basename(clip.clip)}</h3>
      <div className="clip-preview-media">
        {clip.clip_url ? (
          <video
            ref={videoRef}
            src={clipSrc}
            controls
            controlsList="nodownload"
            preload="metadata"
            muted
            playsInline
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || clip.duration_s || 0)}
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
            onError={() => setLoadFailed(true)}
          />
        ) : (
          <Icon name="video" />
        )}
        {loadFailed && <div className="clip-preview-error">Preview unavailable</div>}
      </div>
      <div className="clip-preview-controls">
        <span className="clip-preview-time">{formatSeconds(currentTime)} / {formatSeconds(duration || clip.duration_s)}</span>
        <button className="preview-progress" type="button" aria-label="Seek clip" disabled={!clip.clip_url} onClick={seekToPoint}>
          <i style={{ width: `${progress * 100}%` }} />
          <b style={{ left: `calc(${progress * 100}% - 6px)` }} />
        </button>
      </div>
      <div className="clip-preview-stats">
        <Detail label="Duration" value={formatSeconds(clip.duration_s)} />
        <Detail label="In Point" value={formatTimecode(clip.start_tc)} />
        <Detail label="Out Point" value={formatTimecode(clip.end_tc)} />
      </div>
    </section>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="compact-detail">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
