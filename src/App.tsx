import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type Provider = "gemini" | "openai";

type AssetFile = {
  name: string;
  path: string;
  url: string;
  type: "image" | "video" | "audio" | "other";
  size: string;
};

type FeedbackItem = {
  timestamp?: string;
  category: string;
  remark: string;
  raw_index: number;
};

type FeedbackGroup = {
  clip_used: string;
  feedback_items: FeedbackItem[];
};

type TimelineClip = {
  clip: string;
  start_tc: string;
  end_tc: string;
  start_s: number;
  end_s: number;
  duration_s: number;
  clip_url?: string | null;
};

type QualityReport = {
  passed?: boolean;
  feedback_adherence?: string;
  clothing_consistency?: string;
  forbidden_terms_found?: string[];
  suggestions?: string[];
};

type PromptVersion = {
  timestamp?: string;
  provider?: string;
  video_model_prompt?: string;
  selected_assets?: string[];
  explanation?: string;
  quality_report?: QualityReport;
  initial_frame_image_path?: string;
  initial_frame_prompt?: string;
};

type PromptRecord = PromptVersion & {
  clip_used?: string;
  latest_error?: string | null;
  history?: PromptVersion[];
};

type ProjectData = {
  project_name: string;
  timeline: TimelineClip[];
  feedback: FeedbackGroup[];
  assets: Record<string, AssetFile[]>;
  prompts: PromptRecord[];
  sequence_name: string;
  total_duration_tc: string;
  total_duration_s: number;
};

type PreviewState = {
  file: AssetFile;
} | null;

type ResultState = {
  prompt: string;
  initialPrompt?: string;
  explanation: string;
  initialImage?: string;
  assets: string[];
  quality?: QualityReport;
} | null;

type Toast = {
  id: number;
  message: string;
  type: "info" | "success" | "error";
};

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function staticUrl(path?: string | null) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;

  let cleanPath = path.replace(/\\/g, "/");
  const workspaceMarker = "developement/loka/";
  const idx = cleanPath.indexOf(workspaceMarker);
  if (idx !== -1) {
    cleanPath = cleanPath.substring(idx + workspaceMarker.length);
  }
  if (!cleanPath.startsWith("/")) {
    cleanPath = `/${cleanPath}`;
  }
  return apiUrl(cleanPath);
}

function formatProjectName(name: string) {
  return name.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function basename(path: string) {
  return path.split(/[/\\]/).pop() || path;
}

function clipBasename(path?: string) {
  return basename((path || "").replace(/\\/g, "/"));
}

function Icon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    board: "M4 5h16v12H4z M8 21h8 M12 17v4",
    box: "M4 8l8-4 8 4-8 4-8-4z M4 8v8l8 4 8-4V8 M12 12v8",
    search: "M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14z M20 20l-4-4",
    timeline: "M4 6h5 M15 6h5 M9 6a3 3 0 1 0 6 0 3 3 0 0 0-6 0z M4 18h5 M15 18h5 M9 18a3 3 0 1 0 6 0 3 3 0 0 0-6 0z",
    clock: "M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20z M12 6v6l4 2",
    chevronLeft: "M15 18l-6-6 6-6",
    chevronRight: "M9 18l6-6-6-6",
    chevronDown: "M6 9l6 6 6-6",
    plus: "M12 5v14 M5 12h14",
    minus: "M5 12h14",
    expand: "M8 3H3v5 M16 3h5v5 M3 16v5h5 M21 16v5h-5 M3 3l6 6 M21 3l-6 6 M3 21l6-6 M21 21l-6-6",
    play: "M8 5v14l11-7-11-7z",
    image: "M4 5h16v14H4z M8 13l3-3 3 4 2-2 4 5 M8 8h.01",
    video: "M4 6h11v12H4z M15 10l5-3v10l-5-3z",
    audio: "M9 18V6l10-2v12 M9 18a3 3 0 1 1-2-2.83 M19 16a3 3 0 1 1-2-2.83",
    file: "M6 3h8l4 4v14H6z M14 3v5h5",
    comments: "M4 5h16v10H8l-4 4z",
    warning: "M12 3l10 18H2z M12 9v5 M12 17h.01",
    close: "M6 6l12 12 M18 6L6 18",
    copy: "M8 8h11v11H8z M5 16H4V4h12v1",
    check: "M20 6L9 17l-5-5",
    refresh: "M20 12a8 8 0 0 1-14 5 M4 12a8 8 0 0 1 14-5 M18 3v4h-4 M6 21v-4h4",
    magic: "M5 19L19 5 M14 5h5v5 M4 5l1 2 2 1-2 1-1 2-1-2-2-1 2-1z",
    link: "M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1 M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1",
  };

  return (
    <svg aria-hidden="true" className="icon" viewBox="0 0 24 24">
      <path d={paths[name] || paths.file} />
    </svg>
  );
}

function categoryIcon(category: string) {
  const name = category.toLowerCase();
  if (name.includes("audio")) return "audio";
  if (name.includes("reference") || name.includes("style")) return "image";
  if (name.includes("clip")) return "video";
  return "box";
}

function fileIcon(type: AssetFile["type"]) {
  if (type === "image") return "image";
  if (type === "video") return "video";
  if (type === "audio") return "audio";
  return "file";
}

function getVersions(prompt?: PromptRecord | null) {
  if (!prompt) return [];
  if (prompt.history?.length) return prompt.history;
  if (prompt.video_model_prompt) return [prompt];
  return [];
}

function versionLabel(version: PromptVersion, index: number) {
  const date = version.timestamp ? new Date(version.timestamp) : new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `Version ${index + 1} (${(version.provider || "AI").toUpperCase()} - ${time})`;
}

function App() {
  const [projects, setProjects] = useState<string[]>([]);
  const [activeProject, setActiveProject] = useState("");
  const [provider, setProvider] = useState<Provider>("gemini");
  const [projectData, setProjectData] = useState<ProjectData | null>(null);
  const [loadingProject, setLoadingProject] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [assetFilter, setAssetFilter] = useState("");
  const [collapsedAssets, setCollapsedAssets] = useState(false);
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  const [zoom, setZoom] = useState(1);
  const [preview, setPreview] = useState<PreviewState>(null);
  const [result, setResult] = useState<ResultState>(null);
  const [errorLog, setErrorLog] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [runningIndexes, setRunningIndexes] = useState<Set<number>>(new Set());
  const [selectedVersions, setSelectedVersions] = useState<Record<string, number>>({});
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const eventSourcesRef = useRef<Record<number, EventSource>>({});

  const notify = useCallback((message: string, type: Toast["type"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, message, type }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 4200);
  }, []);

  const loadProject = useCallback(
    async (projectName: string) => {
      setActiveProject(projectName);
      setLoadingProject(true);
      setLoadError("");
      setProjectData(null);

      try {
        const response = await fetch(apiUrl(`/api/project/${encodeURIComponent(projectName)}`));
        if (!response.ok) throw new Error("Failed to load project details");
        const data = (await response.json()) as ProjectData;
        setProjectData(data);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown project loading error";
        setLoadError(`${message}. Check that the backend server is running on ${API_BASE}.`);
      } finally {
        setLoadingProject(false);
      }
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;

    async function fetchProjects() {
      try {
        const response = await fetch(apiUrl("/api/projects"));
        if (!response.ok) throw new Error("Failed to load project list");
        const list = (await response.json()) as string[];
        if (cancelled) return;
        setProjects(list);
        if (list.length > 0) {
          void loadProject(list[0]);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not connect to backend";
        setLoadError(`${message}. Start the FastAPI backend at ${API_BASE}.`);
      }
    }

    void fetchProjects();

    return () => {
      cancelled = true;
      Object.values(eventSourcesRef.current).forEach((source) => source.close());
    };
  }, [loadProject]);

  const assetTotal = useMemo(() => {
    if (!projectData) return 0;
    return Object.values(projectData.assets).reduce((sum, files) => sum + files.length, 0);
  }, [projectData]);

  const duration = projectData
    ? projectData.total_duration_tc || `${projectData.total_duration_s.toFixed(2)}s`
    : "--";
  const providerLabel = provider === "openai" ? "OpenAI" : "Gemini";
  const timelineCount = projectData?.timeline.length || 0;

  const findPrompt = useCallback(
    (clipName: string) =>
      projectData?.prompts?.find((prompt) => clipBasename(prompt.clip_used) === clipBasename(clipName)) || null,
    [projectData],
  );

  const openResultFromVersion = useCallback((version: PromptVersion) => {
    setResult({
      prompt: version.video_model_prompt || "No video prompt generated.",
      initialPrompt: version.initial_frame_prompt,
      explanation: version.explanation || "No explanation provided.",
      initialImage: staticUrl(version.initial_frame_image_path),
      assets: version.selected_assets || [],
      quality: version.quality_report,
    });
  }, []);

  const executeWorkflow = useCallback(
    (feedbackIndex: number) => {
      if (!activeProject) return;
      eventSourcesRef.current[feedbackIndex]?.close();

      setRunningIndexes((current) => new Set(current).add(feedbackIndex));
      const logs = [
        `[SYSTEM] Initializing pipeline connection for feedback index ${feedbackIndex}...`,
        `[SYSTEM] Selected AI Provider: ${provider.toUpperCase()}`,
      ];
      notify(`Workflow initiated for feedback index ${feedbackIndex}...`);

      const source = new EventSource(
        apiUrl(
          `/api/run-workflow?project=${encodeURIComponent(activeProject)}&index=${feedbackIndex}&provider=${provider}`,
        ),
      );
      eventSourcesRef.current[feedbackIndex] = source;

      source.onmessage = (event) => {
        const line = event.data as string;
        logs.push(line);

        if (line.includes("[SUCCESS]") || line.startsWith("[SUCCESS]")) {
          source.close();
          delete eventSourcesRef.current[feedbackIndex];
          setRunningIndexes((current) => {
            const next = new Set(current);
            next.delete(feedbackIndex);
            return next;
          });
          notify("Prompt plan generated successfully.", "success");
          void loadProject(activeProject);
        }
      };

      source.onerror = () => {
        source.close();
        delete eventSourcesRef.current[feedbackIndex];
        setRunningIndexes((current) => {
          const next = new Set(current);
          next.delete(feedbackIndex);
          return next;
        });
        setErrorLog(`${logs.join("\n")}\n[CONNECTION ERROR] EventStream connection dropped or interrupted.`);
        notify(`Workflow execution failed for feedback index ${feedbackIndex}`, "error");
      };
    },
    [activeProject, loadProject, notify, provider],
  );

  const copyPrompt = useCallback(async () => {
    if (!result?.prompt) return;
    try {
      await navigator.clipboard.writeText(result.prompt);
      notify("Prompt copied to clipboard.", "success");
    } catch {
      notify("Could not copy prompt.", "error");
    }
  }, [notify, result]);

  const timelineStyle = {
    "--card-width": `${420 * zoom}px`,
    "--thumbnail-height": `${220 * zoom}px`,
  } as React.CSSProperties;

  const zoomClass = zoom < 0.75 ? "font-small" : zoom > 1.25 ? "font-large" : "";

  return (
    <div className="app-container">
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
            <select className="premium-select provider-select" value={provider} onChange={(event) => setProvider(event.target.value as Provider)}>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
            </select>
          </label>
          <label className="project-selector-wrapper">
            <span>Active Project</span>
            <select
              className="premium-select"
              value={activeProject}
              onChange={(event) => void loadProject(event.target.value)}
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

      <main className="app-body">
        {collapsedAssets && (
          <button className="expand-assets-btn" title="Expand Assets" onClick={() => setCollapsedAssets(false)}>
            <Icon name="chevronRight" />
          </button>
        )}

        <aside className={`left-panel ${collapsedAssets ? "collapsed" : ""}`}>
          <div className="panel-header">
            <h2>
              <Icon name="box" /> Project Assets <span className="badge">{assetTotal}</span>
            </h2>
            <button className="icon-btn" title="Collapse Assets" onClick={() => setCollapsedAssets(true)}>
              <Icon name="chevronLeft" />
            </button>
          </div>
          <div className="asset-search">
            <Icon name="search" />
            <input value={assetFilter} onChange={(event) => setAssetFilter(event.target.value)} placeholder="Filter assets..." />
          </div>
          <div className="assets-list">
            {loadError ? (
              <div className="empty-inline">
                <Icon name="warning" />
                <p>Assets will appear after the backend connects.</p>
              </div>
            ) : projectData ? (
              <AssetsList
                assets={projectData.assets}
                filter={assetFilter}
                collapsedCategories={collapsedCategories}
                onToggleCategory={(category) =>
                  setCollapsedCategories((current) => {
                    const next = new Set(current);
                    if (next.has(category)) next.delete(category);
                    else next.add(category);
                    return next;
                  })
                }
                onPreview={setPreview}
              />
            ) : (
              <div className="panel-loading">Loading assets...</div>
            )}
          </div>
        </aside>

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
            refEl={timelineRef}
            className={zoomClass}
            style={timelineStyle}
            loading={loadingProject}
            error={loadError}
            projectData={projectData}
            findPrompt={findPrompt}
            selectedVersions={selectedVersions}
            setSelectedVersions={setSelectedVersions}
            runningIndexes={runningIndexes}
            executeWorkflow={executeWorkflow}
            openResultFromVersion={openResultFromVersion}
          />
        </section>
      </main>

      <footer className="app-footer">
        <div>
          <span className="footer-status-dot" /> Synced to {providerLabel} - {projectData?.sequence_name || activeProject || "No Project"}
        </div>
        <div>
          Showing clips {timelineCount ? 1 : 0} - {timelineCount} of {timelineCount} scroll horizontally
        </div>
      </footer>

      {preview && <PreviewModal preview={preview} onClose={() => setPreview(null)} />}
      {errorLog && <ErrorModal errorLog={errorLog} onClose={() => setErrorLog("")} />}
      {result && <ResultModal result={result} onClose={() => setResult(null)} onCopy={copyPrompt} />}
      <ToastStack toasts={toasts} />
    </div>
  );
}

function AssetsList({
  assets,
  filter,
  collapsedCategories,
  onToggleCategory,
  onPreview,
}: {
  assets: Record<string, AssetFile[]>;
  filter: string;
  collapsedCategories: Set<string>;
  onToggleCategory: (category: string) => void;
  onPreview: (preview: PreviewState) => void;
}) {
  const query = filter.trim().toLowerCase();
  const categories = Object.keys(assets).sort();

  if (categories.length === 0) {
    return (
      <div className="empty-inline">
        <Icon name="box" />
        <p>No assets found for this project.</p>
      </div>
    );
  }

  return (
    <>
      {categories.map((category) => {
        const files = assets[category].filter((file) => !query || file.name.toLowerCase().includes(query));
        if (files.length === 0) return null;
        const cleanName = category.replace(/^\d+_/, "");
        const collapsed = collapsedCategories.has(category) && !query;

        return (
          <div key={category} className={`asset-category ${collapsed ? "collapsed" : ""}`}>
            <button className="category-header" onClick={() => onToggleCategory(category)}>
              <div className="category-title">
                <Icon name={categoryIcon(cleanName)} />
                <span>{cleanName}</span>
                <span className="badge mini">{files.length}</span>
              </div>
              <span className="chevron">
                <Icon name="chevronDown" />
              </span>
            </button>
            <div className="category-content">
              {files.map((file) => (
                <button key={`${category}-${file.path}`} className="asset-item" onClick={() => onPreview({ file })}>
                  <span className="asset-name">
                    <Icon name={fileIcon(file.type)} />
                    <span title={file.name}>{file.name}</span>
                  </span>
                  <span className="asset-size">{file.size}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </>
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
  refEl: React.RefObject<HTMLDivElement | null>;
  className: string;
  style: React.CSSProperties;
  loading: boolean;
  error: string;
  projectData: ProjectData | null;
  findPrompt: (clipName: string) => PromptRecord | null;
  selectedVersions: Record<string, number>;
  setSelectedVersions: React.Dispatch<React.SetStateAction<Record<string, number>>>;
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
            {index < projectData.timeline.length - 1 && <TimelineArrow />}
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

function TimelineArrow() {
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

function EmptyState({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="empty-state">
      <Icon name={icon} />
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function PreviewModal({ preview, onClose }: { preview: NonNullable<PreviewState>; onClose: () => void }) {
  const { file } = preview;

  return (
    <div className="modal active">
      <div className="modal-content glass-card">
        <div className="modal-header">
          <h3>{file.name}</h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body">
          {file.type === "image" && <img src={staticUrl(file.url)} alt={file.name} />}
          {file.type === "video" && <video src={staticUrl(file.url)} controls autoPlay />}
          {file.type === "audio" && (
            <div className="audio-preview">
              <Icon name="audio" />
              <audio src={staticUrl(file.url)} controls autoPlay />
            </div>
          )}
          {file.type === "other" && (
            <div className="empty-inline">
              <Icon name="file" />
              <p>No viewer available for this file extension.</p>
              <a href={staticUrl(file.url)} download className="premium-btn">
                Download Asset
              </a>
            </div>
          )}
        </div>
        <div className="modal-footer">
          Path: {file.path} | Size: {file.size}
        </div>
      </div>
    </div>
  );
}

function ErrorModal({ errorLog, onClose }: { errorLog: string; onClose: () => void }) {
  return (
    <div className="modal active">
      <div className="modal-content glass-card error-modal-content">
        <div className="modal-header error-header">
          <h3>
            <Icon name="warning" /> Workflow Execution Failed
          </h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body error-body">
          <p>The pipeline run encountered an error. See the console logs below:</p>
          <pre className="error-log-area">{errorLog}</pre>
        </div>
        <div className="modal-footer modal-actions">
          <button className="premium-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

function ResultModal({ result, onClose, onCopy }: { result: NonNullable<ResultState>; onClose: () => void; onCopy: () => void }) {
  const passed = result.quality?.passed !== false;
  const forbidden = result.quality?.forbidden_terms_found || [];

  return (
    <div className="modal active">
      <div className="modal-content glass-card result-modal-content">
        <div className="modal-header">
          <h3>
            <Icon name="timeline" /> Workflow Result Verification
          </h3>
          <button className="close-btn" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body result-modal-body">
          <div className="result-content-layout">
            <div className="result-col">
              <div className="result-section">
                <label className="result-label">
                  <Icon name="file" /> Generated Video Model Prompt
                </label>
                <div className="prompt-box-wrapper">
                  <pre className="monospace-box">{result.prompt}</pre>
                  <button className="icon-btn copy-btn" title="Copy Prompt" onClick={onCopy}>
                    <Icon name="copy" />
                  </button>
                </div>
              </div>
              {result.initialPrompt && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="image" /> Initial Frame Prompt
                  </label>
                  <pre className="monospace-box">{result.initialPrompt}</pre>
                </div>
              )}
              <div className="result-section">
                <label className="result-label">
                  <Icon name="magic" /> Generation Explanation
                </label>
                <p className="text-description">{result.explanation}</p>
              </div>
            </div>

            <div className="result-col">
              {result.initialImage && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="image" /> Generated Initial Frame Image
                  </label>
                  <div className="initial-image-preview-wrapper">
                    <img src={result.initialImage} alt="Initial Frame" />
                  </div>
                </div>
              )}
              <div className="result-section">
                <label className="result-label">
                  <Icon name="box" /> Context Assets Used
                </label>
                <div className="mini-assets-list">
                  {result.assets.length === 0 ? (
                    <div className="muted-small">No reference assets were selected for this prompt.</div>
                  ) : (
                    result.assets.map((asset) => (
                      <div className="mini-asset-item" key={asset}>
                        <Icon name="link" />
                        <span title={asset}>{basename(asset)}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
              {result.quality && (
                <div className="result-section">
                  <label className="result-label">
                    <Icon name="check" /> Quality Check Report
                  </label>
                  <div className="quality-report-card">
                    <div className="quality-status">
                      Status: <span className={`tag ${passed ? "tag-audio" : "tag-both"}`}>{passed ? "PASSED" : "WARNING"}</span>
                    </div>
                    <div className="quality-details-table">
                      <QualityRow label="Feedback Adherence:" value={result.quality.feedback_adherence || "Passed"} />
                      <QualityRow label="Clothing Consistency:" value={result.quality.clothing_consistency || "Passed"} />
                      <QualityRow label="Forbidden Terms:" value={forbidden.length ? forbidden.join(", ") : "None"} danger={forbidden.length > 0} />
                    </div>
                    {Boolean(result.quality.suggestions?.length) && (
                      <div className="quality-suggestions-list">
                        {result.quality.suggestions?.map((suggestion) => (
                          <div key={suggestion}>
                            <Icon name="warning" /> {suggestion}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="modal-footer modal-actions">
          <button className="premium-btn secondary" onClick={onClose}>
            Close Verification
          </button>
          <button className="premium-btn" onClick={onClose}>
            <Icon name="check" /> Approve & Proceed
          </button>
        </div>
      </div>
    </div>
  );
}

function QualityRow({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="q-row">
      <span>{label}</span>
      <strong className={danger ? "danger" : ""}>{value}</strong>
    </div>
  );
}

function ToastStack({ toasts }: { toasts: Toast[] }) {
  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div className={`toast ${toast.type}`} key={toast.id}>
          <Icon name={toast.type === "error" ? "warning" : toast.type === "success" ? "check" : "file"} />
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}

export default App;
