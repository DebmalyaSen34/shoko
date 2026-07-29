import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import "./App.css";
import { AppShell } from "./components/shell/AppShell";
import type { RailItemId } from "./components/shell/LeftRail";
import { ProjectHomePanel, ProjectWorkflowPanel } from "./components/project/ProjectWorkflowPanel";
import { ErrorModal, GenerateVideoOptionsModal, PreviewModal, ResultModal } from "./components/modals/Modals";
import { NewProjectModal } from "./components/modals/NewProjectModal";
import { UploadFeedbackModal } from "./components/modals/UploadFeedbackModal";
import { AddManualFeedbackModal } from "./components/modals/AddManualFeedbackModal";
import { SettingsModal } from "./components/modals/SettingsModal";
import { TimelineWorkspace } from "./components/timeline/TimelineWorkspace";
import { ClipChatPanel } from "./components/chat/ClipChatPanel";
import { ToastStack } from "./components/ToastStack";
import { apiUrl, API_BASE, initializeApiBase, staticUrl } from "./lib/api";
import { clipBasename } from "./lib/format";
import type { ActiveClipChat, GenerateVideoOptions, GeneratedVideo, PreviewState, ProjectData, PromptRecord, PromptVersion, Provider, ResultState, Toast } from "./types";

const DEFAULT_VIDEO_OPTIONS: GenerateVideoOptions = {
  resolution: "720p",
  generate_audio: false,
  aspect_ratio: "9:16",
  duration: 5,
};

const VIDEO_OPTIONS_STORAGE_KEY = "loka15.video-generation.options";
const DEFAULT_TIMELINE_ZOOM = 2;

type PendingVideoRequest = {
  clipIndex: number;
  promptVersionIndex?: number;
  resolve: (result: { video: GeneratedVideo; generated_videos: GeneratedVideo[] }) => void;
  reject: (error: Error) => void;
};

function loadRememberedVideoOptions(): GenerateVideoOptions {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(VIDEO_OPTIONS_STORAGE_KEY) || "{}") as Partial<GenerateVideoOptions>;
    return {
      resolution: parsed.resolution || DEFAULT_VIDEO_OPTIONS.resolution,
      generate_audio: parsed.generate_audio ?? DEFAULT_VIDEO_OPTIONS.generate_audio,
      aspect_ratio: parsed.aspect_ratio || DEFAULT_VIDEO_OPTIONS.aspect_ratio,
      duration: parsed.duration || DEFAULT_VIDEO_OPTIONS.duration,
    };
  } catch {
    return DEFAULT_VIDEO_OPTIONS;
  }
}

function App() {
  const [projects, setProjects] = useState<string[]>([]);
  const [activeProject, setActiveProject] = useState("");
  const [provider] = useState<Provider>("openai");
  const [projectData, setProjectData] = useState<ProjectData | null>(null);
  const [loadingProject, setLoadingProject] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [zoom, setZoom] = useState(DEFAULT_TIMELINE_ZOOM);
  const [preview, setPreview] = useState<PreviewState>(null);
  const [result, setResult] = useState<ResultState>(null);
  const [errorLog, setErrorLog] = useState("");
  const [showNewProject, setShowNewProject] = useState(false);
  const [showUploadFeedback, setShowUploadFeedback] = useState(false);
  const [showAddManualFeedback, setShowAddManualFeedback] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [activeClipForManualFeedback, setActiveClipForManualFeedback] = useState("");
  const [activeRailItem, setActiveRailItem] = useState<RailItemId>("project");
  const [timelineWorkflowOpen, setTimelineWorkflowOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [runningIndexes, setRunningIndexes] = useState<Set<number>>(new Set());
  const [generatingVideoKeys, setGeneratingVideoKeys] = useState<Set<string>>(new Set());
  const [videoOptions, setVideoOptions] = useState<GenerateVideoOptions>(() => loadRememberedVideoOptions());
  const [pendingVideoRequest, setPendingVideoRequest] = useState<PendingVideoRequest | null>(null);
  const [selectedVersions, setSelectedVersions] = useState<Record<string, number>>({});
  const [activeClipChat, setActiveClipChat] = useState<ActiveClipChat>(null);
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
        await initializeApiBase();
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
  const timelineCount = projectData?.timeline.length || 0;

  const findPrompt = useCallback(
    (clipName: string, clipOccurrence?: number): PromptRecord | null => {
      const prompts = projectData?.prompts || [];
      const basenameMatch = (prompt: PromptRecord) => clipBasename(prompt.clip_used) === clipBasename(clipName);
      const sameClipPrompts = prompts.filter(basenameMatch);
      if (typeof clipOccurrence === "number") {
        const exactOccurrence = sameClipPrompts.find((prompt) => prompt.clip_occurrence === clipOccurrence);
        if (exactOccurrence) return exactOccurrence;
      }
      return sameClipPrompts.find((prompt) => prompt.clip_occurrence == null) || null;
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
      generatedVideos: version.generated_videos || [],
    });
  }, []);

  const runGenerateVideo = useCallback(
    async (clipIndex: number, promptVersionIndex: number | undefined, options: GenerateVideoOptions) => {
      if (!activeProject) throw new Error("No active project selected.");
      const key = `${clipIndex}:${promptVersionIndex ?? "latest"}`;
      setGeneratingVideoKeys((current) => new Set(current).add(key));
      notify("Video generation started with Segmind...");

      try {
        const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(activeProject)}/generate-video`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clip_index: clipIndex,
            prompt_version_index: promptVersionIndex,
            provider: "segmind",
            resolution: options.resolution,
            generate_audio: options.generate_audio,
            aspect_ratio: options.aspect_ratio,
            duration: options.duration,
          }),
        });
        if (!response.ok) {
          let message = await response.text();
          try {
            const parsed = JSON.parse(message) as { detail?: string | { message?: string; request_id?: string; next_step?: string } };
            if (typeof parsed.detail === "string") {
              message = parsed.detail;
            } else if (parsed.detail?.message) {
              message = parsed.detail.message;
              if (parsed.detail.request_id) message += ` Request id: ${parsed.detail.request_id}.`;
              if (parsed.detail.next_step) message += ` ${parsed.detail.next_step}`;
            }
          } catch {
            // Keep the raw response text.
          }
          throw new Error(message);
        }
        const data = (await response.json()) as { video: GeneratedVideo; generated_videos: GeneratedVideo[] };
        notify("Video generated successfully.", "success");
        await loadProject(activeProject, { preserveTimelineScroll: true });
        return data;
      } catch (error) {
        const message = error instanceof Error ? error.message : "Video generation failed.";
        notify(message, "error");
        throw error;
      } finally {
        setGeneratingVideoKeys((current) => {
          const next = new Set(current);
          next.delete(key);
          return next;
        });
      }
    },
    [activeProject, loadProject, notify],
  );

  const rememberVideoOptions = useCallback(
    (options: GenerateVideoOptions) => {
      setVideoOptions(options);
      window.localStorage.setItem(VIDEO_OPTIONS_STORAGE_KEY, JSON.stringify(options));
      notify("Video generation choices remembered.", "success");
    },
    [notify],
  );

  const generateVideo = useCallback(
    (clipIndex: number, promptVersionIndex?: number) =>
      new Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>((resolve, reject) => {
        setPendingVideoRequest({ clipIndex, promptVersionIndex, resolve, reject });
      }),
    [],
  );

  const closeVideoOptionsModal = useCallback(() => {
    pendingVideoRequest?.reject(new Error("Video generation cancelled."));
    setPendingVideoRequest(null);
  }, [pendingVideoRequest]);

  const submitVideoOptions = useCallback(
    async (options: GenerateVideoOptions) => {
      if (!pendingVideoRequest) return;
      setVideoOptions(options);
      const request = pendingVideoRequest;
      setPendingVideoRequest(null);
      try {
        const result = await runGenerateVideo(request.clipIndex, request.promptVersionIndex, options);
        request.resolve(result);
      } catch (error) {
        request.reject(error instanceof Error ? error : new Error("Video generation failed."));
      }
    },
    [pendingVideoRequest, runGenerateVideo],
  );

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
  const chatClip = activeClipChat && projectData ? projectData.timeline[activeClipChat.clipIndex] : null;
  const chatFeedback =
    activeClipChat && chatClip && projectData
      ? projectData.feedback.find(
          (item) =>
            item.clip_used === chatClip.clip &&
            (typeof item.clip_occurrence !== "number" || item.clip_occurrence === activeClipChat.clipIndex),
        )
      : undefined;
  const chatPrompt = activeClipChat && chatClip ? findPrompt(chatClip.clip, activeClipChat.clipIndex) : null;
  const timelineOpen = activeRailItem === "timeline";
  const projectLabel = projectData?.sequence_name || activeProject || "No Project";
  const runFirstAvailableWorkflow = useCallback(() => {
    const firstFeedback = projectData?.feedback.flatMap((group) => group.feedback_items)[0];
    if (!firstFeedback) {
      notify("Upload feedback before running the workflow.", "info");
      return;
    }
    executeWorkflow(firstFeedback.raw_index);
  }, [executeWorkflow, notify, projectData]);

  return (
    <>
      <AppShell
        activeProject={activeProject}
        activeRailItem={activeRailItem}
        clipCount={timelineCount}
        projectLabel={projectLabel}
        projects={projects}
        provider={provider}
        workflowDrawerOpen={timelineWorkflowOpen}
        onExport={() => notify("Export workflow will be added in a later implementation step.", "info")}
        onOpenSettings={() => setShowSettings(true)}
        onRunWorkflow={runFirstAvailableWorkflow}
        onProjectChange={(project) => void loadProject(project)}
        onToggleWorkflowDrawer={() => setTimelineWorkflowOpen((open) => !open)}
        onRailSelect={(item) => {
          if (item === "project" && timelineOpen) {
            setTimelineWorkflowOpen((open) => !open);
            return;
          }
          if (item === "settings") {
            setActiveRailItem("settings");
            setTimelineWorkflowOpen(false);
            setShowSettings(true);
            return;
          }
          if (item === "feedback") {
            setActiveRailItem("project");
            setTimelineWorkflowOpen(false);
            setShowUploadFeedback(true);
            return;
          }
          if (item === "workflow") {
            setActiveRailItem("timeline");
            setTimelineWorkflowOpen(false);
            runFirstAvailableWorkflow();
            return;
          }
          if (item === "exports") {
            setActiveRailItem("exports");
            setTimelineWorkflowOpen(false);
            notify("Export workflow will be added in a later implementation step.", "info");
            return;
          }
          setActiveRailItem(item);
          if (item === "timeline") {
            setTimelineWorkflowOpen(false);
          }
        }}
      >
        {(!timelineOpen || timelineWorkflowOpen) && (
          <ProjectWorkflowPanel
            activeProject={activeProject}
            assetTotal={assetTotal}
            className={timelineOpen ? "timeline-workflow-drawer" : ""}
            duration={duration}
            loading={loadingProject}
            loadError={loadError}
            projectData={projectData}
            runningIndexes={runningIndexes}
            onCollapse={timelineOpen ? () => setTimelineWorkflowOpen(false) : undefined}
            onImportProject={() => setShowNewProject(true)}
            onRunAssignment={runFirstAvailableWorkflow}
            onUploadFeedback={() => setShowUploadFeedback(true)}
            onViewResults={() => {
              setActiveRailItem("timeline");
              setTimelineWorkflowOpen(false);
            }}
          />
        )}

        {timelineOpen ? (
          <>
            <TimelineWorkspace
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
              generatingVideoKeys={generatingVideoKeys}
              onGenerateVideo={generateVideo}
              setZoom={setZoom}
              onUploadFeedback={() => setShowUploadFeedback(true)}
              onAddManualFeedback={(clipName) => {
                setActiveClipForManualFeedback(clipName);
                setShowAddManualFeedback(true);
              }}
              onOpenClipChat={(clipIndex) => setActiveClipChat({ clipIndex })}
              onPreview={setPreview}
            />

            {projectData && chatClip && activeClipChat && (
              <ClipChatPanel
                clip={chatClip}
                clipIndex={activeClipChat.clipIndex}
                feedback={chatFeedback}
                prompt={chatPrompt}
                projectData={projectData}
                provider={provider}
                runningIndexes={runningIndexes}
                onClose={() => setActiveClipChat(null)}
                onExecuteWorkflow={executeWorkflow}
                onOpenPromptDetails={openResultFromVersion}
                onGenerateVideo={generateVideo}
                onPreview={setPreview}
              />
            )}
          </>
        ) : (
          <ProjectHomePanel
            activeProject={activeProject}
            assetTotal={assetTotal}
            duration={duration}
            loading={loadingProject}
            projectData={projectData}
            onOpenTimeline={() => setActiveRailItem("timeline")}
          />
        )}
      </AppShell>

      {preview && <PreviewModal preview={preview} onClose={() => setPreview(null)} />}
      {errorLog && <ErrorModal errorLog={errorLog} onClose={() => setErrorLog("")} />}
      {pendingVideoRequest && (
        <GenerateVideoOptionsModal
          initialOptions={videoOptions}
          onClose={closeVideoOptionsModal}
          onGenerate={submitVideoOptions}
          onRemember={rememberVideoOptions}
        />
      )}
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
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSaved={() => notify("Settings saved.", "success")}
        />
      )}
      <ToastStack toasts={toasts} />
    </>
  );
}

export default App;
