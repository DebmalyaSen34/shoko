import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import "./App.css";
import { AppHeader } from "./components/AppHeader";
import { AssetsSidebar } from "./components/assets/AssetsSidebar";
import { ErrorModal, PreviewModal, ResultModal } from "./components/modals/Modals";
import { NewProjectModal } from "./components/modals/NewProjectModal";
import { UploadAssetsModal } from "./components/modals/UploadAssetsModal";
import { UploadFeedbackModal } from "./components/modals/UploadFeedbackModal";
import { AddManualFeedbackModal } from "./components/modals/AddManualFeedbackModal";
import { TimelinePanel } from "./components/timeline/TimelinePanel";
import { ToastStack } from "./components/ToastStack";
import { apiUrl, API_BASE, staticUrl } from "./lib/api";
import { clipBasename } from "./lib/format";
import type { PreviewState, ProjectData, PromptRecord, PromptVersion, Provider, ResultState, Toast } from "./types";

function App() {
  const [projects, setProjects] = useState<string[]>([]);
  const [activeProject, setActiveProject] = useState("");
  const [provider, setProvider] = useState<Provider>("openai");
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
  const [showNewProject, setShowNewProject] = useState(false);
  const [showUploadAssets, setShowUploadAssets] = useState(false);
  const [showUploadFeedback, setShowUploadFeedback] = useState(false);
  const [showAddManualFeedback, setShowAddManualFeedback] = useState(false);
  const [activeClipForManualFeedback, setActiveClipForManualFeedback] = useState("");
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
    async (projectName: string, options?: { preserveTimelineScroll?: boolean }) => {
      const previousTimelineScroll = options?.preserveTimelineScroll ? timelineRef.current?.scrollLeft ?? 0 : null;
      setActiveProject(projectName);
      setLoadingProject(true);
      setLoadError("");
      setProjectData(null);

      try {
        const response = await fetch(apiUrl(`/api/project/${encodeURIComponent(projectName)}`));
        if (!response.ok) throw new Error("Failed to load project details");
        const data = (await response.json()) as ProjectData;
        setProjectData(data);
        if (previousTimelineScroll !== null) {
          window.requestAnimationFrame(() => {
            window.requestAnimationFrame(() => {
              if (timelineRef.current) {
                timelineRef.current.scrollLeft = previousTimelineScroll;
              }
            });
          });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown project loading error";
        setLoadError(`${message}. Check that the backend server is running on ${API_BASE}.`);
      } finally {
        setLoadingProject(false);
      }
    },
    [],
  );

  const refreshProjects = useCallback(
    async (selectProject?: string) => {
      const response = await fetch(apiUrl("/api/projects"));
      if (!response.ok) throw new Error("Failed to load project list");
      const list = (await response.json()) as string[];
      setProjects(list);

      const nextProject = selectProject || activeProject || list[0];
      if (nextProject && list.includes(nextProject)) {
        await loadProject(nextProject);
      } else if (list.length === 0) {
        setActiveProject("");
        setProjectData(null);
      }
    },
    [activeProject, loadProject],
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
    (clipName: string, clipOccurrence?: number): PromptRecord | null => {
      const prompts = projectData?.prompts || [];
      const basenameMatch = (prompt: PromptRecord) => clipBasename(prompt.clip_used) === clipBasename(clipName);
      if (typeof clipOccurrence === "number") {
        const exactOccurrence = prompts.find(
          (prompt) => basenameMatch(prompt) && prompt.clip_occurrence === clipOccurrence,
        );
        if (exactOccurrence) return exactOccurrence;
      }
      return prompts.find(basenameMatch) || null;
    },
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
      clipFrames: version.clip_frame_paths || [],
      audioTrim: {
        start: version.audio_trim_start_s,
        end: version.audio_trim_end_s,
        duration: version.audio_trim_duration_s,
        source: version.audio_trim_source,
        path:
          version.audio_reference_path ||
          version.trimmed_audio_path ||
          version.segmind_reference_audios?.[0] ||
          version.audio_url,
        error: version.audio_trim_error,
      },
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
          void loadProject(activeProject, { preserveTimelineScroll: true });
        } else if (line.includes("[ERROR]") || line.startsWith("[ERROR]")) {
          source.close();
          delete eventSourcesRef.current[feedbackIndex];
          setRunningIndexes((current) => {
            const next = new Set(current);
            next.delete(feedbackIndex);
            return next;
          });
          setErrorLog(logs.join("\n"));
          notify(`Workflow execution failed for feedback index ${feedbackIndex}`, "error");
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

  const toggleAssetCategory = useCallback((category: string) => {
    setCollapsedCategories((current) => {
      const next = new Set(current);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
  }, []);

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
    "--card-width": `${640 * zoom}px`,
    "--thumbnail-height": `${320 * zoom}px`,
  } as CSSProperties;

  const zoomClass = zoom < 0.75 ? "font-small" : zoom > 1.25 ? "font-large" : "";

  return (
    <div className="app-container">
      <AppHeader
        activeProject={activeProject}
        projects={projects}
        provider={provider}
        onNewProject={() => setShowNewProject(true)}
        onProjectChange={(project) => void loadProject(project)}
        onProviderChange={setProvider}
      />

      <main className="app-body">
        <AssetsSidebar
          assetFilter={assetFilter}
          assetTotal={assetTotal}
          assets={projectData?.assets}
          collapsed={collapsedAssets}
          collapsedCategories={collapsedCategories}
          loadError={loadError}
          onCollapse={() => setCollapsedAssets(true)}
          onExpand={() => setCollapsedAssets(false)}
          onFilterChange={setAssetFilter}
          onPreview={setPreview}
          onToggleCategory={toggleAssetCategory}
          onAddAssets={() => setShowUploadAssets(true)}
        />

        <TimelinePanel
          duration={duration}
          error={loadError}
          loading={loadingProject}
          projectData={projectData}
          refEl={timelineRef}
          runningIndexes={runningIndexes}
          selectedVersions={selectedVersions}
          setSelectedVersions={setSelectedVersions}
          timelineStyle={timelineStyle}
          zoom={zoom}
          zoomClass={zoomClass}
          executeWorkflow={executeWorkflow}
          findPrompt={findPrompt}
          openResultFromVersion={openResultFromVersion}
          setZoom={setZoom}
          onUploadFeedback={() => setShowUploadFeedback(true)}
          onAddManualFeedback={(clipName) => {
            setActiveClipForManualFeedback(clipName);
            setShowAddManualFeedback(true);
          }}
        />
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
      {showNewProject && (
        <NewProjectModal
          onClose={() => setShowNewProject(false)}
          onCreated={(projectName) => {
            setShowNewProject(false);
            notify("Project imported successfully.", "success");
            void refreshProjects(projectName);
          }}
        />
      )}
      {result && <ResultModal result={result} onClose={() => setResult(null)} onCopy={copyPrompt} />}
      {showUploadAssets && activeProject && (
        <UploadAssetsModal
          projectName={activeProject}
          onClose={() => setShowUploadAssets(false)}
          onUploaded={() => {
            setShowUploadAssets(false);
            notify("Assets uploaded successfully.", "success");
            void loadProject(activeProject);
          }}
        />
      )}
      {showUploadFeedback && activeProject && (
        <UploadFeedbackModal
          projectName={activeProject}
          onClose={() => setShowUploadFeedback(false)}
          onUploaded={() => {
            setShowUploadFeedback(false);
            notify("Feedback uploaded and parsed successfully.", "success");
            void loadProject(activeProject);
          }}
        />
      )}
      {showAddManualFeedback && activeProject && activeClipForManualFeedback && (
        <AddManualFeedbackModal
          projectName={activeProject}
          clipName={activeClipForManualFeedback}
          onClose={() => {
            setShowAddManualFeedback(false);
            setActiveClipForManualFeedback("");
          }}
          onAdded={() => {
            setShowAddManualFeedback(false);
            setActiveClipForManualFeedback("");
            notify("Manual feedback added successfully.", "success");
            void loadProject(activeProject);
          }}
        />
      )}
      <ToastStack toasts={toasts} />
    </div>
  );
}

export default App;
