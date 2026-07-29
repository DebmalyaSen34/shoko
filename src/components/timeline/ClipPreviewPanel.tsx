import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import type { PreviewState, TimelineClip } from "../../types";
import { staticUrl } from "../../lib/api";
import { basename, formatSeconds, formatTimecode } from "../../lib/format";
import { Icon } from "../Icon";

type ClipPreviewPanelProps = {
  clip: TimelineClip;
  onPreview: (preview: PreviewState) => void;
  playheadOffsetSeconds?: number;
};

export function ClipPreviewPanel({ clip, onPreview, playheadOffsetSeconds }: ClipPreviewPanelProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastSyncedPlayhead = useRef<number | undefined>(undefined);
  const [playing, setPlaying] = useState(false);
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
    setPlaying(false);
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

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      try {
        await video.play();
      } catch {
        setPlaying(false);
      }
    } else {
      video.pause();
    }
  };

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

  const enterFullscreen = async () => {
    const video = videoRef.current;
    if (!video || !clip.clip_url) return;
    if (video.requestFullscreen) {
      await video.requestFullscreen();
      return;
    }
    onPreview({
      file: {
        name: basename(clip.clip),
        path: clip.clip_url,
        url: clip.clip_url,
        type: "video",
        size: "",
      },
    });
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
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onError={() => {
              setLoadFailed(true);
              setPlaying(false);
            }}
          />
        ) : (
          <Icon name="video" />
        )}
        {loadFailed && (
          <button
            className="clip-preview-error"
            type="button"
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
            Open video preview
          </button>
        )}
      </div>
      <div className="clip-preview-controls">
        <button type="button" title={playing ? "Pause clip" : "Play clip"} disabled={!clip.clip_url} onClick={togglePlayback}>
          <Icon name={playing ? "pause" : "play"} />
        </button>
        <span>{formatSeconds(currentTime)} / {formatSeconds(duration || clip.duration_s)}</span>
        <button className="preview-progress" type="button" aria-label="Seek clip" disabled={!clip.clip_url} onClick={seekToPoint}>
          <i style={{ width: `${progress * 100}%` }} />
          <b style={{ left: `calc(${progress * 100}% - 4px)` }} />
        </button>
        <button type="button" title="Fullscreen" disabled={!clip.clip_url} onClick={enterFullscreen}>
          <Icon name="expand" />
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
