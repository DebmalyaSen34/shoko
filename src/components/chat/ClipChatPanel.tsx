import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ClipChatAction,
  ClipChatMemory,
  ClipChatMessage,
  ClipChatResponse,
  ClipChatSnapshot,
  FeedbackGroup,
  PromptRecord,
  PromptVersion,
  ProjectData,
  Provider,
  TimelineClip,
} from "../../types";
import { basename, getVersions } from "../../lib/format";
import { apiUrl } from "../../lib/api";
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
};

type ContextTab = "clip" | "feedback" | "assets" | "memory";

const MEMORY_OPTIONS = ["Mem0", "Letta", "Graphiti/Zep", "Cognee", "LangGraph/LangMem"];

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
}: ClipChatPanelProps) {
  const [input, setInput] = useState("");
  const [activeTab, setActiveTab] = useState<ContextTab>("clip");
  const [messages, setMessages] = useState<ClipChatMessage[]>([]);
  const [memory, setMemory] = useState<ClipChatMemory>({ project: [], clip: [] });
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  const feedbackItems = feedback?.feedback_items || [];
  const versions = getVersions(prompt);
  const latestVersion = versions[versions.length - 1] || prompt || null;
  const selectedAssets = latestVersion?.selected_assets || [];
  const runnableFeedback = feedbackItems[0];
  const running = feedbackItems.some((item) => runningIndexes.has(item.raw_index));

  const adjacentClips = useMemo(
    () => ({
      previous: clipIndex > 0 ? projectData.timeline[clipIndex - 1] : null,
      next: clipIndex < projectData.timeline.length - 1 ? projectData.timeline[clipIndex + 1] : null,
    }),
    [clipIndex, projectData.timeline],
  );

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
        setMemory(data.memory);
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
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages, sending]);

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
      setMemory(data.memory);
    } catch (sendError) {
      const message = sendError instanceof Error ? sendError.message : "Failed to send message.";
      setError(message);
      setMessages((current) => [...current, localToolMessage("Chat request failed. Check that the backend server and provider key are available.")]);
    } finally {
      setSending(false);
    }
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

  function handleAction(action: ClipChatAction) {
    if (action.type === "execute_workflow") {
      runWorkflowFromChat(action.feedback_index);
    } else if (action.type === "prepare_video") {
      prepareVideoGeneration();
    }
  }

  return (
    <aside className="clip-chat-panel" aria-label="Clip chat">
      <div className="clip-chat-header">
        <div>
          <div className="clip-chat-kicker">Clip Chat</div>
          <h3>{clip.clip}</h3>
          <span>#{clipIndex + 1} in {projectData.sequence_name || projectData.project_name}</span>
        </div>
        <button className="icon-btn" type="button" title="Close Chat" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>

      <div className="clip-chat-actions">
        <button className="premium-btn secondary" type="button" disabled={!runnableFeedback || running} onClick={() => runWorkflowFromChat()}>
          <Icon name="refresh" /> {running ? "Running..." : "Run Workflow"}
        </button>
        <button className="premium-btn" type="button" onClick={prepareVideoGeneration}>
          <Icon name="video" /> Prepare Video
        </button>
      </div>

      <div className="clip-chat-tabs" role="tablist" aria-label="Clip context">
        {(["clip", "feedback", "assets", "memory"] as ContextTab[]).map((tab) => (
          <button key={tab} type="button" className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>
            {tab}
          </button>
        ))}
      </div>

      <ContextPanel
        activeTab={activeTab}
        clip={clip}
        clipIndex={clipIndex}
        totalClips={projectData.timeline.length}
        feedbackItems={feedbackItems}
        projectAssets={projectData.assets}
        selectedAssets={selectedAssets}
        memory={memory}
        latestVersion={latestVersion}
        adjacentClips={adjacentClips}
      />

      <div className="clip-chat-log" ref={listRef}>
        {loading && <div className="inline-runner-status"><Icon name="refresh" /> Loading chat...</div>}
        {messages.map((message) => (
          <div className={`chat-message ${message.role}`} key={message.id}>
            <div className="chat-message-meta">{message.role} · {formatMessageTime(message.created_at)}</div>
            <div className="chat-message-text">{message.content}</div>
            {Boolean(message.metadata?.actions?.length) && (
              <div className="chat-action-row">
                {message.metadata?.actions?.map((action) => (
                  <button className="premium-btn secondary" type="button" key={`${message.id}-${action.type}`} onClick={() => handleAction(action)}>
                    <Icon name={action.type === "execute_workflow" ? "refresh" : "video"} /> {action.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {sending && <div className="inline-runner-status"><Icon name="refresh" /> Thinking with {provider.toUpperCase()}...</div>}
      </div>

      {error && <div className="clip-chat-error"><Icon name="warning" /> {error}</div>}

      <form
        className="clip-chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          sendMessage(input);
        }}
      >
        <textarea
          value={input}
          disabled={sending}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              sendMessage(input);
            }
          }}
          placeholder="Ask about feedback, assets, timeline, memory, workflow..."
          rows={2}
        />
        <button className="icon-btn chat-send-btn" type="submit" title="Send Message" disabled={sending || !input.trim()}>
          <Icon name="send" />
        </button>
      </form>
    </aside>
  );
}

function ContextPanel({
  activeTab,
  clip,
  clipIndex,
  totalClips,
  feedbackItems,
  projectAssets,
  selectedAssets,
  memory,
  latestVersion,
  adjacentClips,
}: {
  activeTab: ContextTab;
  clip: TimelineClip;
  clipIndex: number;
  totalClips: number;
  feedbackItems: FeedbackGroup["feedback_items"];
  projectAssets: ProjectData["assets"];
  selectedAssets: string[];
  memory: ClipChatMemory;
  latestVersion: PromptVersion | PromptRecord | null;
  adjacentClips: { previous: TimelineClip | null; next: TimelineClip | null };
}) {
  const assetTotal = Object.values(projectAssets).reduce((sum, assets) => sum + assets.length, 0);

  return (
    <div className="clip-chat-context">
      {activeTab === "clip" && (
        <div className="context-grid">
          <ContextItem label="Position" value={`${clipIndex + 1} / ${totalClips}`} />
          <ContextItem label="Timecode" value={`${clip.start_tc} - ${clip.end_tc}`} />
          <ContextItem label="Previous" value={adjacentClips.previous?.clip || "None"} />
          <ContextItem label="Next" value={adjacentClips.next?.clip || "None"} />
          <ContextItem label="Bounds" value={`${clip.start_s.toFixed(2)}s - ${clip.end_s.toFixed(2)}s`} />
          <ContextItem label="Duration" value={`${clip.duration_s.toFixed(2)}s`} />
        </div>
      )}

      {activeTab === "feedback" && (
        <div className="context-list">
          {feedbackItems.length ? feedbackItems.map((item) => (
            <div className="context-row" key={item.raw_index}>
              <span className={`tag tag-${item.category}`}>{item.category}</span>
              <p>{item.remark}</p>
            </div>
          )) : <p className="muted-small">No feedback attached to this clip.</p>}
        </div>
      )}

      {activeTab === "assets" && (
        <div className="context-list">
          <ContextItem label="Project Asset Library" value={`${assetTotal} files`} />
          <ContextItem label="Selected For Latest Plan" value={`${selectedAssets.length} files`} />
          {selectedAssets.slice(0, 5).map((asset) => (
            <div className="mini-asset-item" key={asset}>
              <Icon name="box" /> {basename(asset)}
            </div>
          ))}
          {latestVersion?.audio_reference_path && (
            <div className="mini-asset-item">
              <Icon name="audio" /> {basename(latestVersion.audio_reference_path)}
            </div>
          )}
        </div>
      )}

      {activeTab === "memory" && (
        <div className="context-list">
          <div className="memory-options">
            {MEMORY_OPTIONS.map((option) => <span className="badge mini" key={option}>{option}</span>)}
          </div>
          <ContextItem label="Durable Memory" value={`${memory.project.length} project, ${memory.clip.length} clip`} />
          {[...memory.project, ...memory.clip].length ? [...memory.project, ...memory.clip].map((item) => (
            <div className="context-row" key={item.id}>
              <Icon name="memory" />
              <p>{item.text}</p>
            </div>
          )) : <p className="muted-small">Say “remember: ...” to save a durable note for this clip.</p>}
        </div>
      )}
    </div>
  );
}

function ContextItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="context-item">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}
