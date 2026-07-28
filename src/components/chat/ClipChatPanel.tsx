import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  ClipChatAction,
  ClipAgentRun,
  ClipChatMedia,
  ClipChatMessage,
  ClipChatResponse,
  GeneratedVideo,
  ClipChatSnapshot,
  FeedbackGroup,
  PromptRecord,
  PromptVersion,
  PreviewState,
  ProjectData,
  Provider,
  TimelineClip,
} from "../../types";
import { getVersions } from "../../lib/format";
import { apiUrl, staticUrl } from "../../lib/api";
import { Icon } from "../Icon";

type ClipChatPanelProps = {
  clip: TimelineClip;
  clipIndex: number;
  feedback?: FeedbackGroup;
  prompt: PromptRecord | null;
  projectData: ProjectData;
  provider: Provider;
  runningIndexes: Set<number>;
  onClose: () => void;
  onExecuteWorkflow: (feedbackIndex: number) => void;
  onOpenPromptDetails: (version: PromptVersion) => void;
  onGenerateVideo: (clipIndex: number, promptVersionIndex?: number) => Promise<{ video: GeneratedVideo; generated_videos: GeneratedVideo[] }>;
  onPreview: (preview: PreviewState) => void;
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
  clip,
  clipIndex,
  feedback,
  prompt,
  projectData,
  provider,
  runningIndexes,
  onClose,
  onExecuteWorkflow,
  onOpenPromptDetails,
  onGenerateVideo,
  onPreview,
}: ClipChatPanelProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ClipChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [generatingVideo, setGeneratingVideo] = useState(false);
  const [error, setError] = useState("");
  const [clearedAt, setClearedAt] = useState("");
  const [panelWidth, setPanelWidth] = useState(() => {
    const saved = window.localStorage.getItem("loka15.clip-chat.width");
    return saved ? Number(saved) || 420 : 420;
  });
  const listRef = useRef<HTMLDivElement | null>(null);

  const feedbackItems = feedback?.feedback_items || [];
  const versions = getVersions(prompt);
  const latestVersion = versions[versions.length - 1] || prompt || null;
  const runnableFeedback = feedbackItems[0];
  const running = feedbackItems.some((item) => runningIndexes.has(item.raw_index));
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
      runWorkflowFromChat();
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
      const autonomousWorkflow = data.suggested_actions.find(
        (action) => action.type === "execute_workflow" && action.autonomous,
      );
      if (autonomousWorkflow) {
        runWorkflowFromChat(autonomousWorkflow.feedback_index);
      }
      const autonomousVideo = data.suggested_actions.find(
        (action) => action.type === "generate_video" && action.autonomous,
      );
      if (autonomousVideo) {
        void generateVideoFromChat();
      }
    } catch (sendError) {
      const message = sendError instanceof Error ? sendError.message : "Failed to send message.";
      setError(message);
      setMessages((current) => [...current, localToolMessage("Chat request failed. Check that the backend server and provider key are available.")]);
    } finally {
      setSending(false);
    }
  }

  function submitInput() {
    if (executeSlashCommand(input)) return;
    void sendMessage(input);
  }

  function runWorkflowFromChat(feedbackIndex?: number) {
    const index = feedbackIndex ?? runnableFeedback?.raw_index;
    if (typeof index !== "number") {
      setMessages((current) => [...current, localToolMessage("No feedback item is available for this clip yet.")]);
      return;
    }

    onExecuteWorkflow(index);
    setMessages((current) => [...current, localToolMessage(`Workflow started for feedback index ${index} using ${provider.toUpperCase()}.`)]);
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

    const promptVersionIndex = Math.max(versions.length - 1, 0);
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
      runWorkflowFromChat(action.feedback_index);
    } else if (action.type === "generate_video" || (action.type === "prepare_video" && action.label.toLowerCase().includes("generate"))) {
      void generateVideoFromChat();
    } else if (action.type === "prepare_video") {
      prepareVideoGeneration();
    } else if (action.type === "send_message" && action.prompt) {
      void sendMessage(action.prompt);
    }
  }

  function actionIcon(action: ClipChatAction) {
    if (action.type === "execute_workflow") return "refresh";
    if (action.type === "prepare_video" || action.type === "generate_video") return "video";
    return "chat";
  }

  function resizePanel(clientX: number) {
    const nextWidth = Math.min(760, Math.max(320, window.innerWidth - clientX));
    setPanelWidth(nextWidth);
    window.localStorage.setItem("loka15.clip-chat.width", String(nextWidth));
  }

  const panelStyle = window.innerWidth > 900 ? { width: panelWidth, minWidth: panelWidth } : undefined;

  return (
    <aside className="clip-chat-panel" style={panelStyle} aria-label="Clip chat">
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
      <div className="clip-chat-header">
        <div>
          <div className="clip-chat-kicker">Clip Chat</div>
          <h3>{clip.clip}</h3>
          <span>{projectData.sequence_name || projectData.project_name}</span>
        </div>
        <button className="icon-btn" type="button" title="Close Chat" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>

      <div className="clip-chat-actions">
        <button className="premium-btn secondary" type="button" disabled={!runnableFeedback || running} onClick={() => runWorkflowFromChat()}>
          <Icon name="refresh" /> {running ? "Running..." : "Run Workflow"}
        </button>
        <button className="premium-btn" type="button" disabled={generatingVideo} onClick={() => void generateVideoFromChat()}>
          <Icon name="video" /> {generatingVideo ? "Generating..." : "Generate Video"}
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
          <div className={`chat-message ${message.role}`} key={message.id}>
            <div className="chat-message-meta">{message.role} · {formatMessageTime(message.created_at)}</div>
            <div className="chat-message-text">
              <MarkdownMessage text={message.content} />
            </div>
            {Boolean(message.metadata?.media?.length) && (
              <ChatMediaGallery media={message.metadata?.media || []} onPreview={onPreview} />
            )}
            {message.metadata?.agent_run && <AgentRunSummary run={message.metadata.agent_run} />}
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
        ))}
        {sending && <div className="inline-loader"><span className="loader-orbit" aria-hidden="true"><span /><span /><span /></span>Thinking with {provider.toUpperCase()}...</div>}
        {generatingVideo && <div className="inline-loader"><span className="loader-orbit" aria-hidden="true"><span /><span /><span /></span>Generating video...</div>}
      </div>

      {error && <div className="clip-chat-error"><Icon name="warning" /> {error}</div>}

      <form
        className="clip-chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          submitInput();
        }}
      >
        <div className="clip-chat-input-field">
          {matchingCommands.length > 0 && (
            <div className="slash-command-menu">
              {matchingCommands.map((command) => (
                <button type="button" key={command.name} onClick={() => setInput(command.name)}>
                  <code>{command.name}</code>
                  <span>{command.description}</span>
                </button>
              ))}
            </div>
          )}
          <textarea
            value={input}
            disabled={sending}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitInput();
              }
            }}
            placeholder="Ask about feedback, assets, timeline, memory, workflow... Type /help for commands."
            rows={2}
          />
        </div>
        <button className="icon-btn chat-send-btn" type="submit" title="Send Message" disabled={sending || !input.trim()}>
          <Icon name="send" />
        </button>
      </form>
    </aside>
  );
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
    </div>
  );
}

function formatToolResult(result: Record<string, unknown>) {
  const tool = typeof result.tool === "string" ? formatAgentRunStatus(result.tool) : "tool";
  const message = typeof result.message === "string" ? result.message : "";
  return message ? `${tool}: ${message}` : tool;
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
