import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  ClipChatAction,
  ClipAgentRun,
  ClipChatMedia,
  ClipChatMessage,
  ClipChatResponse,
  ClipState,
  GeneratedVideo,
  ClipChatSnapshot,
  FeedbackGroup,
  PromptFeedbackCategory,
  PromptFeedbackPayload,
  PromptFeedbackTarget,
  PromptRecord,
  PromptVersion,
  PreviewState,
  ProjectData,
  Provider,
} from "../../types";
import { getVersions } from "../../lib/format";
import { apiUrl, createPromptFeedback, staticUrl } from "../../lib/api";
import { Icon } from "../Icon";
import { WorkflowRunningLabel } from "../WorkflowRunningLabel";
import { PromptInputBox } from "../ui/ai-prompt-box";

type ClipChatPanelProps = {
  variant?: "panel" | "dock";
  clipIndex: number;
  feedback?: FeedbackGroup;
  prompt: PromptRecord | null;
  projectData: ProjectData;
  provider: Provider;
  runningIndexes: Set<number>;
  externalGeneratingVideo?: boolean;
  onClose?: () => void;
  onExecuteWorkflow: (feedbackIndex: number) => Promise<void> | void;
  onOpenPromptDetails: (version: PromptVersion) => void;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onPreview: (preview: PreviewState) => void;
  promptFeedbackTarget?: PromptFeedbackTarget | null;
  onClosePromptFeedback?: () => void;
  onPromptFeedbackSaved?: (clipState: ClipState) => void;
};

function localToolMessage(content: string): ClipChatMessage {
  return {
    id: `${Date.now()}-${Math.random()}`,
    role: "tool",
    content,
    created_at: new Date().toISOString(),
    metadata: {},
  };
}

function formatMessageTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const SLASH_COMMANDS = [
  { name: "/clear", description: "Clear visible chat space without deleting saved history." },
  { name: "/history", description: "Show the saved chat history again." },
  { name: "/workflow", description: "Run the feedback workflow for this clip." },
  { name: "/prepare", description: "Open the latest generated prompt details." },
  { name: "/video", description: "Generate a video from the latest prompt." },
  { name: "/help", description: "Show available chat commands." },
  { name: "/commands", description: "Show available chat commands." },
];

export function ClipChatPanel({
  variant = "panel",
  clipIndex,
  feedback,
  prompt,
  projectData,
  provider,
  runningIndexes,
  externalGeneratingVideo = false,
  onClose,
  onExecuteWorkflow,
  onOpenPromptDetails,
  onGenerateVideo,
  onPreview,
  promptFeedbackTarget,
  onClosePromptFeedback,
  onPromptFeedbackSaved,
}: ClipChatPanelProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ClipChatMessage[]>([]);
  const [clipState, setClipState] = useState<ClipState | null>(null);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [generatingVideo, setGeneratingVideo] = useState(false);
  const [error, setError] = useState("");
  const [clearedAt, setClearedAt] = useState("");
  const [expandedRunIds, setExpandedRunIds] = useState<Set<string>>(new Set());
  const [panelWidth, setPanelWidth] = useState(() => {
    const saved = window.localStorage.getItem("loka15.clip-chat.width");
    return saved ? Number(saved) || 420 : 420;
  });
  const listRef = useRef<HTMLDivElement | null>(null);

  const feedbackItems = feedback?.feedback_items || [];
  const activePrompt = clipState?.active_prompt;
  const versions = activePrompt?.versions?.length ? activePrompt.versions : getVersions(prompt);
  const latestVersion = activePrompt?.version || versions[versions.length - 1] || prompt || null;
  const runnableFeedback = feedbackItems[0];
  const activeJobs = clipState?.job_state.active_jobs || [];
  const recentJobs = clipState?.job_state.recent_jobs || [];
  const workflowJobs = activeJobs.filter((job) => job.type === "workflow");
  const videoJobs = activeJobs.filter((job) => job.type === "generate_video");
  const running = feedbackItems.some((item) => runningIndexes.has(item.raw_index)) || workflowJobs.length > 0;
  const videoBusy = generatingVideo || externalGeneratingVideo || videoJobs.length > 0;
  const activeVideo = clipState?.video_state.active_video || clipState?.video_state.latest_video;
  const selectedAssets = clipState?.asset_state.selected_assets || [];
  const promptStatus = activePrompt?.prompt_ready ? `Prompt v${(activePrompt.version_index ?? 0) + 1}` : "Prompt missing";
  const videoStatus = activeVideo ? activeVideo.label || `Video v${activeVideo.version}` : "Video missing";
  const assetStatus = `${selectedAssets.length} asset${selectedAssets.length === 1 ? "" : "s"}`;
  const jobStatus = activeJobs.length ? `${activeJobs.length} active job${activeJobs.length === 1 ? "" : "s"}` : recentJobs[0]?.status || "Idle";
  const clearStorageKey = useMemo(
    () => `loka15.clip-chat.cleared-at:${projectData.project_name}:${clipIndex}`,
    [clipIndex, projectData.project_name],
  );
  const visibleMessages = clearedAt
    ? messages.filter((message) => {
        const messageTime = new Date(message.created_at).getTime();
        const clearTime = new Date(clearedAt).getTime();
        return Number.isNaN(messageTime) || Number.isNaN(clearTime) || messageTime > clearTime;
      })
    : messages;
  const slashQuery = input.trim().startsWith("/") ? input.trim().toLowerCase() : "";
  const matchingCommands = slashQuery
    ? SLASH_COMMANDS.filter((command) => command.name.startsWith(slashQuery)).slice(0, 5)
    : [];

  useEffect(() => {
    let cancelled = false;

    async function loadChat() {
      setLoading(true);
      setError("");
      try {
        const response = await fetch(
          apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/chat/clip/${clipIndex}`),
        );
        if (!response.ok) throw new Error(await response.text());
        const data = (await response.json()) as ClipChatSnapshot;
        if (cancelled) return;
        setMessages(data.messages);
        setClipState(data.context.clip_state);
      } catch (loadError) {
        if (cancelled) return;
        const message = loadError instanceof Error ? loadError.message : "Failed to load clip chat.";
        setError(message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void loadChat();

    return () => {
      cancelled = true;
    };
  }, [clipIndex, projectData.project_name]);

  useEffect(() => {
    if (!clipState?.job_state.active_jobs.length) return;
    const timer = window.setInterval(() => {
      void refreshClipState();
    }, 1800);
    return () => window.clearInterval(timer);
  }, [clipState?.clip_state_id, clipState?.job_state.active_jobs.length]);

  useEffect(() => {
    if (running) {
      void refreshClipState();
    }
  }, [running]);

  async function refreshClipState() {
    try {
      const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/clips/${clipIndex}/state`));
      if (!response.ok) throw new Error(await response.text());
      const state = (await response.json()) as ClipState;
      setClipState(state);
    } catch {
      // Keep the last known state visible.
    }
  }

  useEffect(() => {
    setClearedAt(window.localStorage.getItem(clearStorageKey) || "");
  }, [clearStorageKey]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [visibleMessages, sending]);

  function commandHelpMessage() {
    return [
      "Available chat commands:",
      ...SLASH_COMMANDS.map((command) => `- \`${command.name}\` - ${command.description}`),
    ].join("\n");
  }

  function executeSlashCommand(text: string) {
    const command = text.trim().toLowerCase();
    if (!command.startsWith("/")) return false;

    setInput("");
    setError("");

    if (command === "/clear") {
      const marker = new Date().toISOString();
      setClearedAt(marker);
      window.localStorage.setItem(clearStorageKey, marker);
      return true;
    }

    if (command === "/history") {
      setClearedAt("");
      window.localStorage.removeItem(clearStorageKey);
      setMessages((current) => [...current, localToolMessage("Restored saved chat history.")]);
      return true;
    }

    if (command === "/workflow") {
      void runWorkflowFromChat();
      return true;
    }

    if (command === "/prepare") {
      prepareVideoGeneration();
      return true;
    }

    if (command === "/video") {
      void generateVideoFromChat();
      return true;
    }

    if (command === "/help" || command === "/commands") {
      setMessages((current) => [...current, localToolMessage(commandHelpMessage())]);
      return true;
    }

    setMessages((current) => [...current, localToolMessage(`Unknown command: \`${command}\`. Type \`/help\` to see available commands.`)]);
    return true;
  }

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || sending) return;

    setInput("");
    setSending(true);
    setError("");

    const optimistic: ClipChatMessage = {
      id: `local-${Date.now()}`,
      role: "user",
      content: trimmed,
      created_at: new Date().toISOString(),
      metadata: {},
    };
    setMessages((current) => [...current, optimistic]);

    try {
      const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/chat/clip`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_index: clipIndex, message: trimmed, provider }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as ClipChatResponse;
      setMessages(data.messages);
      setClipState(data.clip_state);
    } catch (sendError) {
      const message = sendError instanceof Error ? sendError.message : "Failed to send message.";
      setError(message);
      setMessages((current) => [...current, localToolMessage("Chat request failed. Check that the backend server and provider key are available.")]);
    } finally {
      setSending(false);
    }
  }

  async function runWorkflowFromChat(feedbackIndex?: number) {
    const index = feedbackIndex ?? runnableFeedback?.raw_index;
    if (typeof index !== "number") {
      setMessages((current) => [...current, localToolMessage("No feedback item is available for this clip yet.")]);
      return;
    }

    setMessages((current) => [...current, localToolMessage(`Workflow started for feedback index ${index} using ${provider.toUpperCase()}.`)]);
    try {
      await onExecuteWorkflow(index);
      await refreshClipState();
      setMessages((current) => [...current, localToolMessage(`Workflow finished for feedback index ${index}. The latest prompt and references are ready.`)]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Workflow execution failed.";
      setMessages((current) => [...current, localToolMessage(message)]);
    }
  }

  function prepareVideoGeneration() {
    if (!latestVersion?.video_model_prompt) {
      setMessages((current) => [
        ...current,
        localToolMessage("No generated prompt exists yet. Run the workflow first, then prepare video generation."),
      ]);
      return;
    }

    onOpenPromptDetails(latestVersion);
    setMessages((current) => [
      ...current,
      localToolMessage("Opened the generated prompt, references, audio trim, and quality report for video generation handoff."),
    ]);
  }

  async function generateVideoFromChat() {
    if (!latestVersion?.video_model_prompt) {
      setMessages((current) => [
        ...current,
        localToolMessage("No generated prompt exists yet. Run the workflow first, then generate video."),
      ]);
      return;
    }

    const promptVersionIndex = activePrompt?.version_index ?? Math.max(versions.length - 1, 0);
    setGeneratingVideo(true);
    setMessages((current) => [...current, localToolMessage("Choose video generation options to start the Segmind render.")]);
    try {
      const result = await onGenerateVideo(clipIndex, promptVersionIndex);
      const video = result.video;
      const videoMessage = localToolMessage(`Generated video v${video.version || result.generated_videos.length} is ready.`);
      videoMessage.metadata = {
        media: [
          {
            type: "video",
            label: video.label || "Generated video",
            source: "generated_videos",
            name: video.path.split(/[/\\]/).pop() || "generated-video.mp4",
            path: video.path,
            url: video.url || video.path,
          },
        ],
      };
      setMessages((current) => [...current, videoMessage]);
      void refreshClipState();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Video generation failed.";
      if (message === "Video generation cancelled.") return;
      setMessages((current) => [...current, localToolMessage(message)]);
    } finally {
      setGeneratingVideo(false);
    }
  }

  function handleAction(action: ClipChatAction) {
    if (action.type === "execute_workflow") {
      void runWorkflowFromChat(action.feedback_index);
    } else if (action.type === "generate_video" || (action.type === "prepare_video" && action.label.toLowerCase().includes("generate"))) {
      void generateVideoFromChat();
    } else if (action.type === "prepare_video") {
      prepareVideoGeneration();
    } else if (action.type === "send_message" && action.prompt) {
      void sendMessage(action.prompt);
    } else if (action.type === "set_active_prompt_version" && action.prompt_version_id) {
      const version = versions.find((candidate) => candidate.prompt_version_id === action.prompt_version_id);
      if (version) void setActivePromptVersion(version);
    } else if (action.type === "detach_asset") {
      void detachAsset(action.asset_path, action.asset_id);
    } else if (action.type === "attach_asset" || action.type === "mark_feedback_resolved" || action.type === "add_reference_frame") {
      void sendMessage(action.prompt || action.reason || action.label);
    }
  }

  async function setActivePromptVersion(version: PromptVersion) {
    const versionId = version.prompt_version_id;
    if (!versionId) return;
    try {
      const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/clips/${clipIndex}/selection`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active_prompt_version_id: versionId }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { clip_state: ClipState };
      setClipState(data.clip_state);
      setMessages((current) => [...current, localToolMessage(`Active prompt switched to version ${(version.version_index ?? 0) + 1}.`)]);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to switch prompt version.");
    }
  }

  async function detachAsset(assetPath?: string | null, assetId?: string | null) {
    if (!clipState) return;
    const nextAssets = (clipState.selection_state.selected_assets || []).filter((asset) => {
      if (assetId && asset.asset_id === assetId) return false;
      if (assetPath && asset.path === assetPath) return false;
      return true;
    });
    try {
      const response = await fetch(apiUrl(`/api/projects/${encodeURIComponent(projectData.project_name)}/clips/${clipIndex}/selection`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_assets: nextAssets }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { clip_state: ClipState };
      setClipState(data.clip_state);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to detach asset.");
    }
  }

  function actionIcon(action: ClipChatAction) {
    if (action.type === "execute_workflow") return "refresh";
    if (action.type === "prepare_video" || action.type === "generate_video") return "video";
    if (action.type === "attach_asset" || action.type === "detach_asset") return "file";
    if (action.type === "mark_feedback_resolved") return "check";
    if (action.type === "add_reference_frame") return "image";
    return "chat";
  }

  function toggleAgentRun(messageId: string) {
    setExpandedRunIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }

  function resizePanel(clientX: number) {
    const nextWidth = Math.min(760, Math.max(320, window.innerWidth - clientX));
    setPanelWidth(nextWidth);
    window.localStorage.setItem("loka15.clip-chat.width", String(nextWidth));
  }

  const panelStyle = variant === "panel" && window.innerWidth > 900 ? { width: panelWidth, minWidth: panelWidth } : undefined;

  if (promptFeedbackTarget) {
    return (
      <aside className={`clip-chat-panel prompt-feedback-panel ${variant === "dock" ? "dock-chat-panel" : ""}`} style={panelStyle} aria-label="Prompt feedback">
        {variant === "panel" && (
          <div
            className="clip-chat-resize-handle"
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize feedback"
            tabIndex={0}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              resizePanel(event.clientX);
            }}
            onPointerMove={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                resizePanel(event.clientX);
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowLeft") setPanelWidth((width) => Math.min(760, width + 24));
              if (event.key === "ArrowRight") setPanelWidth((width) => Math.max(320, width - 24));
            }}
          />
        )}
        <PromptFeedbackPanel
          projectName={projectData.project_name}
          target={promptFeedbackTarget}
          onBack={onClosePromptFeedback || (() => undefined)}
          onSaved={(state) => {
            setClipState(state);
            onPromptFeedbackSaved?.(state);
          }}
        />
      </aside>
    );
  }

  return (
    <aside className={`clip-chat-panel ${variant === "dock" ? "dock-chat-panel" : ""}`} style={panelStyle} aria-label="Clip chat">
      {variant === "panel" && (
        <div
          className="clip-chat-resize-handle"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize chat"
          tabIndex={0}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            resizePanel(event.clientX);
          }}
          onPointerMove={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              resizePanel(event.clientX);
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") setPanelWidth((width) => Math.min(760, width + 24));
            if (event.key === "ArrowRight") setPanelWidth((width) => Math.max(320, width - 24));
          }}
        />
      )}
      
      <div className="clip-chat-header">
        <div className="clip-chat-meta-row" aria-label="Clip agent state">
          <span className={!activePrompt?.prompt_ready ? "warning" : ""} title={activePrompt?.version_id || undefined}>{promptStatus}</span>
          <span className={!activeVideo ? "warning" : ""} title={activeVideo?.path || activeVideo?.url || undefined}>{videoStatus}</span>
          <span title={selectedAssets.map((asset) => `${asset.role || "asset"}: ${asset.name}`).join("\n") || undefined}>{assetStatus}</span>
          <span title={recentJobs[0]?.logs?.[recentJobs[0].logs.length - 1]?.message || undefined}>{jobStatus}</span>
        </div>
        {onClose && (
          <button className="icon-btn" type="button" title="Close Chat" onClick={onClose}>
            <Icon name="close" />
          </button>
        )}
      </div>

      <div className="clip-chat-actions">
        <button className={`premium-btn secondary ${running ? "workflow-running-button" : ""}`} type="button" disabled={!runnableFeedback || running} onClick={() => void runWorkflowFromChat()}>
          {running ? <WorkflowRunningLabel /> : <><Icon name="refresh" /> Run Workflow</>}
        </button>
        <button className="premium-btn" type="button" disabled={videoBusy} onClick={() => void generateVideoFromChat()}>
          <Icon name="video" /> {videoBusy ? "Generating..." : "Generate Video"}
        </button>
      </div>

      <div className="clip-chat-log" ref={listRef}>
        {loading && <div className="inline-loader"><span className="loader-orbit" aria-hidden="true"><span /><span /><span /></span>Loading chat...</div>}
        {!loading && clearedAt && visibleMessages.length === 0 && (
          <div className="chat-command-empty">
            <Icon name="check" />
            <span>Chat space cleared. Saved history is still available with <code>/history</code>.</span>
          </div>
        )}
        {visibleMessages.map((message) => (
          <div className={`chat-message-group ${message.role}`} key={message.id}>
            <div className={`chat-message ${message.role}`}>
              <div className="chat-message-text">
                <MarkdownMessage text={message.content} />
              </div>
              {Boolean(message.metadata?.media?.length) && (
                <ChatMediaGallery media={message.metadata?.media || []} onPreview={onPreview} />
              )}
              {Boolean(message.metadata?.actions?.length) && (
                <div className="chat-action-row">
                  {message.metadata?.actions?.map((action) => (
                    <button className="premium-btn secondary" type="button" key={`${message.id}-${action.type}-${action.label}`} onClick={() => handleAction(action)}>
                      <Icon name={actionIcon(action)} /> {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="chat-message-footer">
              {message.metadata?.agent_run ? (
                <button
                  className="chat-info-toggle"
                  type="button"
                  title={expandedRunIds.has(message.id) ? "Hide feedback review" : "Show feedback review"}
                  aria-label={expandedRunIds.has(message.id) ? "Hide feedback review" : "Show feedback review"}
                  aria-expanded={expandedRunIds.has(message.id)}
                  onClick={() => toggleAgentRun(message.id)}
                >
                  <Icon name="info" />
                </button>
              ) : (
                <span />
              )}
              <time>{formatMessageTime(message.created_at)}</time>
            </div>
            {message.metadata?.agent_run && expandedRunIds.has(message.id) && (
              <div className="chat-message-info">
                <AgentRunSummary run={message.metadata.agent_run} />
              </div>
            )}
          </div>
        ))}
        {sending && <FilmThinkingLoader provider={provider} />}
        {videoBusy && <div className="inline-loader"><span className="loader-orbit" aria-hidden="true"><span /><span /><span /></span>Generating video...</div>}
      </div>

      {error && <div className="clip-chat-error"><Icon name="warning" /> {error}</div>}

      <div className="clip-chat-input-wrapper">
        <PromptInputBox
          value={input}
          onValueChange={setInput}
          isLoading={sending}
          onSend={(messageText) => {
            if (executeSlashCommand(messageText)) return;
            void sendMessage(messageText);
          }}
          topAddon={
            matchingCommands.length > 0 && (
              <div className="slash-command-menu">
                {matchingCommands.map((command) => (
                  <button
                    type="button"
                    key={command.name}
                    onClick={() => setInput(command.name)}
                  >
                    <code>{command.name}</code>
                    <span>{command.description}</span>
                  </button>
                ))}
              </div>
            )
          }
        />
      </div>
    </aside>
  );
}

function FilmThinkingLoader({ provider }: { provider: Provider }) {
  const words = ["blocking", "framing", "lighting", "rolling", "storyboarding", "composing", "grading", "cutting", "mixing", "rendering"];

  return (
    <div className="film-thinking-loader" role="status" aria-live="polite" aria-label={`Working with ${provider.toUpperCase()}`}>
      <div className="film-clapper" aria-hidden="true">
        <span className="film-clapper-top">
          <i />
          <i />
          <i />
        </span>
        <span className="film-clapper-body">
          <b />
          <b />
        </span>
      </div>
      <div className="film-loader-copy">
        <strong>
          {words.map((word) => (
            <em key={word}>{word}</em>
          ))}
        </strong>
      </div>
    </div>
  );
}

const PROMPT_FEEDBACK_CATEGORIES: Array<{ value: PromptFeedbackCategory; label: string }> = [
  { value: "missed_feedback", label: "Missed feedback" },
  { value: "wrong_visual_detail", label: "Wrong visual detail" },
  { value: "wrong_character_or_wardrobe", label: "Character or wardrobe" },
  { value: "continuity_error", label: "Continuity" },
  { value: "bad_camera_instruction", label: "Camera" },
  { value: "bad_audio_or_dialogue", label: "Audio or dialogue" },
  { value: "unsupported_assumption", label: "Unsupported assumption" },
  { value: "format_error", label: "Format" },
  { value: "too_vague", label: "Too vague" },
  { value: "too_verbose", label: "Too verbose" },
  { value: "provider_incompatible", label: "Provider incompatible" },
  { value: "other", label: "Other" },
];

function PromptFeedbackPanel({
  projectName,
  target,
  onBack,
  onSaved,
}: {
  projectName: string;
  target: PromptFeedbackTarget;
  onBack: () => void;
  onSaved: (clipState: ClipState) => void;
}) {
  const [rating, setRating] = useState<"positive" | "negative">("negative");
  const [categories, setCategories] = useState<PromptFeedbackCategory[]>([]);
  const [severity, setSeverity] = useState(3);
  const [comment, setComment] = useState("");
  const [correction, setCorrection] = useState("");
  const [rememberNote, setRememberNote] = useState("");
  const [createEvalCase, setCreateEvalCase] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [savedMessage, setSavedMessage] = useState("");
  const issueCount = target.feedback_summary?.open_negative_count || 0;

  useEffect(() => {
    setRating("negative");
    setCategories([]);
    setSeverity(3);
    setComment("");
    setCorrection("");
    setRememberNote("");
    setCreateEvalCase(false);
    setError("");
    setSavedMessage("");
  }, [target.prompt_version_id]);

  function toggleCategory(category: PromptFeedbackCategory) {
    setCategories((current) => current.includes(category) ? current.filter((item) => item !== category) : [...current, category]);
  }

  async function submitFeedback() {
    setError("");
    setSavedMessage("");
    if (rating === "negative" && !comment.trim() && !correction.trim() && !rememberNote.trim()) {
      setError("Add a comment, correction, or remember note before saving negative feedback.");
      return;
    }
    const payload: PromptFeedbackPayload = {
      clip_index: target.clip_index,
      clip_key: target.clip_key,
      prompt_id: target.prompt_id,
      prompt_version_id: target.prompt_version_id,
      rating,
      categories,
      severity,
      comment,
      correction,
      remember_note: rememberNote,
      create_eval_case: createEvalCase,
      status: rating === "positive" ? "approved" : "open",
    };
    setSubmitting(true);
    try {
      const result = await createPromptFeedback(projectName, payload);
      onSaved(result.clip_state);
      setSavedMessage(rating === "positive" ? "Prompt approved." : "Feedback saved. This version now needs revision.");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Failed to save prompt feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <div className="clip-chat-header prompt-feedback-header">
        <button className="icon-btn" type="button" title="Back to chat" onClick={onBack}>
          <Icon name="chevronLeft" />
        </button>
        <div className="prompt-feedback-title">
          <span>Prompt Feedback</span>
          <strong>{target.prompt_label}</strong>
        </div>
      </div>

      <div className="prompt-feedback-context">
        <div>
          <span>Status</span>
          <strong>{promptFeedbackStatusLabel(target.feedback_summary?.status)}</strong>
        </div>
        <div>
          <span>Open issues</span>
          <strong>{issueCount}</strong>
        </div>
      </div>

      <div className="prompt-feedback-form">
        <div className="prompt-feedback-field">
          <span>Review</span>
          <div className="prompt-rating-toggle" role="group" aria-label="Prompt rating">
            <button className={rating === "positive" ? "active positive" : ""} type="button" onClick={() => setRating("positive")}>
              <Icon name="check" /> Works well
            </button>
            <button className={rating === "negative" ? "active negative" : ""} type="button" onClick={() => setRating("negative")}>
              <Icon name="warning" /> Needs changes
            </button>
          </div>
        </div>

        <div className="prompt-feedback-field">
          <span>Issue categories</span>
          <div className="prompt-category-grid">
            {PROMPT_FEEDBACK_CATEGORIES.map((category) => (
              <button
                className={categories.includes(category.value) ? "active" : ""}
                type="button"
                key={category.value}
                onClick={() => toggleCategory(category.value)}
              >
                {category.label}
              </button>
            ))}
          </div>
        </div>

        <label className="prompt-feedback-field">
          <span>Severity</span>
          <select value={severity} onChange={(event) => setSeverity(Number(event.target.value))}>
            {[1, 2, 3, 4, 5].map((value) => (
              <option value={value} key={value}>{value}</option>
            ))}
          </select>
        </label>

        <label className="prompt-feedback-field">
          <span>What should change?</span>
          <textarea value={comment} onChange={(event) => setComment(event.target.value)} placeholder="Describe what did not work in this prompt..." />
        </label>

        <label className="prompt-feedback-field">
          <span>Correction</span>
          <textarea value={correction} onChange={(event) => setCorrection(event.target.value)} placeholder="Write the preferred direction or replacement..." />
        </label>

        <label className="prompt-feedback-field">
          <span>Remember this</span>
          <textarea value={rememberNote} onChange={(event) => setRememberNote(event.target.value)} placeholder="A reusable note for future prompts..." />
        </label>

        <label className="prompt-feedback-check">
          <input type="checkbox" checked={createEvalCase} onChange={(event) => setCreateEvalCase(event.target.checked)} />
          <span>Create eval case from this later</span>
        </label>

        {savedMessage && <div className="prompt-feedback-success"><Icon name="check" /> {savedMessage}</div>}
        {error && <div className="clip-chat-error"><Icon name="warning" /> {error}</div>}

        <div className="prompt-feedback-submit-row">
          <button className="premium-btn secondary" type="button" onClick={onBack}>Back to Chat</button>
          <button className="premium-btn" type="button" disabled={submitting} onClick={() => void submitFeedback()}>
            <Icon name="comments" /> {submitting ? "Saving..." : "Save Feedback"}
          </button>
        </div>
      </div>
    </>
  );
}

function promptFeedbackStatusLabel(status?: string) {
  if (status === "approved") return "Approved";
  if (status === "needs_revision") return "Needs revision";
  if (status === "rejected") return "Rejected";
  return "Unreviewed";
}

function formatAgentRunStatus(value: string) {
  return value.replace(/_/g, " ");
}

function AgentRunSummary({ run }: { run: ClipAgentRun }) {
  const steps = run.plan_steps || [];
  const results = run.tool_results || [];
  return (
    <div className="agent-run-summary">
      <div className="agent-run-header">
        <div>
          <span>Agent Plan</span>
          <strong>{formatAgentRunStatus(run.intent)}</strong>
        </div>
        <em className={`agent-run-status ${run.status}`}>{formatAgentRunStatus(run.status)}</em>
      </div>
      <div className="agent-run-meta">
        <span>{formatAgentRunStatus(run.autonomy_level)}</span>
        <span>{Math.round((run.confidence || 0) * 100)}% confidence</span>
        {run.approval_required && <span>approval needed</span>}
        {run.freshness?.is_stale && <span>stale: {run.freshness.stale_reasons.join(", ")}</span>}
      </div>
      {steps.length > 0 && (
        <ol className="agent-run-steps">
          {steps.slice(0, 4).map((step) => (
            <li key={step.id} className={step.status}>
              <span>{step.label}</span>
              <em>{formatAgentRunStatus(step.status)}</em>
            </li>
          ))}
        </ol>
      )}
      {results.length > 0 && (
        <div className="agent-run-results">
          {results.slice(0, 3).map((result, index) => (
            <span key={`${run.id}-result-${index}`}>
              {formatToolResult(result)}
            </span>
          ))}
        </div>
      )}
      {run.self_evaluation && (
        <div className="agent-run-results">
          <span>{formatSelfEvaluation(run.self_evaluation)}</span>
        </div>
      )}
    </div>
  );
}

function formatToolResult(result: Record<string, unknown>) {
  const tool = typeof result.tool === "string" ? formatAgentRunStatus(result.tool) : "tool";
  const message = typeof result.message === "string" ? result.message : "";
  return message ? `${tool}: ${message}` : tool;
}

function formatSelfEvaluation(value: Record<string, unknown>) {
  const verdict = typeof value.verdict === "string" ? value.verdict.replace(/_/g, " ") : "evaluated";
  const nextAction = value.next_action && typeof value.next_action === "object" ? value.next_action as Record<string, unknown> : null;
  const nextType = typeof nextAction?.type === "string" ? nextAction.type.replace(/_/g, " ") : "";
  return nextType ? `Self-evaluation: ${verdict}. Next: ${nextType}.` : `Self-evaluation: ${verdict}.`;
}

function ChatMediaGallery({ media, onPreview }: { media: ClipChatMedia[]; onPreview: (preview: PreviewState) => void }) {
  const fallbackVideoPoster = media.find((item) => item.type === "image" && item.source === "clip_frames")?.url;

  return (
    <div className="chat-media-grid">
      {media.map((item) => {
        const posterUrl = item.thumbnail_url || (item.type === "video" ? fallbackVideoPoster : undefined);
        return (
          <button
            type="button"
            className={`chat-media-card ${item.type}`}
            key={`${item.source}-${item.url}`}
            title={item.path || item.name}
            onClick={() =>
              onPreview({
                file: {
                  name: decodeMediaName(item.name),
                  path: item.path || item.url,
                  url: item.url,
                  type: item.type,
                  size: item.size || "",
                },
              })
            }
          >
            <div className="chat-media-preview">
              {item.type === "image" && <img src={staticUrl(item.url)} alt={item.label} loading="lazy" />}
              {item.type === "video" && <ChatVideoPreview item={item} posterUrl={posterUrl} />}
              {item.type === "audio" && <Icon name="audio" />}
              {item.type === "other" && <Icon name="file" />}
            </div>
            <div className="chat-media-info">
              <strong>{item.label}</strong>
              <span>{item.name}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function decodeMediaName(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function ChatVideoPreview({ item, posterUrl }: { item: ClipChatMedia; posterUrl?: string | null }) {
  if (posterUrl) {
    return (
      <div className="chat-video-thumb">
        <img src={staticUrl(posterUrl)} alt={item.label} loading="lazy" />
        <span>
          <Icon name="video" />
        </span>
      </div>
    );
  }

  return (
    <div className="chat-video-thumb empty">
      <span>
        <Icon name="video" />
      </span>
    </div>
  );
}

function MarkdownMessage({ text }: { text: string }) {
  const markdown = normalizeChatMarkdown(text);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
}

function normalizeChatMarkdown(text: string) {
  return dedentAccidentalPromptBlock(unwrapPromptCodeFences(unwrapMarkdownFence(text)));
}

function unwrapMarkdownFence(text: string) {
  const trimmed = text.trim();
  const match = trimmed.match(/^```[\w-]*\s*\n([\s\S]*?)\n```$/i);
  return match ? match[1].trim() : text;
}

function unwrapPromptCodeFences(text: string) {
  return text.replace(
    /(^|\n)([^\n`]*(?:prompt|seedance|video model prompt)[^\n`]*:\s*)?\n?```(?:text|markdown|md)?\s*\n([\s\S]*?)\n```/gi,
    (_match, prefix: string, label: string | undefined, body: string) => {
      const cleanPrefix = prefix || "";
      const cleanLabel = (label || "").trim();
      const cleanBody = body.trim();
      return cleanLabel ? `${cleanPrefix}${cleanLabel}\n${cleanBody}` : `${cleanPrefix}${cleanBody}`;
    },
  );
}

function dedentAccidentalPromptBlock(text: string) {
  const lines = text.split("\n");
  const indentedLines = lines.filter((line) => /^ {4,}\S/.test(line));
  if (indentedLines.length < 3) return text;

  const promptLikeIndentedLine = indentedLines.some((line) =>
    /^ {4,}(REFERENCE IMAGE MAP:|image \d+:|video model prompt:|initial frame prompt:|quality report:)/i.test(line),
  );
  const promptRequestIntro = /latest prompt|generated prompt|prompt from the context|video prompt/i.test(text);
  if (!promptLikeIndentedLine && !promptRequestIntro) return text;

  return lines.map((line) => line.replace(/^ {4}/, "")).join("\n");
}
