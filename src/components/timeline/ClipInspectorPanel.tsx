import { useEffect, useState } from "react";
import type { AssetFile, ClipAssetReference, ClipState, FeedbackGroup, GeneratedVideo, PreviewState, ProjectData, PromptVersion, ReferencedFrameState, TimelineClip } from "../../types";
import { staticUrl } from "../../lib/api";
import { basename, versionLabel } from "../../lib/format";
import { Icon } from "../Icon";

export type ClipInspectorTab = "feedback" | "prompts" | "references" | "assets" | "videos";

type ClipInspectorPanelProps = {
  activeTab: ClipInspectorTab;
  clip: TimelineClip;
  feedbackItems: FeedbackGroup["feedback_items"];
  projectData: ProjectData;
  clipState?: ClipState | null;
  versions: PromptVersion[];
  selectedVersion?: PromptVersion;
  selectedVersionIndex: number;
  setActiveTab: (tab: ClipInspectorTab) => void;
  onAddManualFeedback: (clipName: string) => void;
  onPreview: (preview: PreviewState) => void;
};

const WAVEFORM_PATTERN = [16, 29, 42, 55, 26, 39, 52, 23, 36, 49, 20, 33, 46, 17, 30, 43, 56, 27, 40, 53];
const ASSET_PREVIEW_LIMIT = 12;
type AssetReferenceInput = string | AssetFile | ClipAssetReference;

export function ClipInspectorPanel({
  activeTab,
  clip,
  feedbackItems,
  projectData,
  clipState,
  versions,
  selectedVersion,
  selectedVersionIndex,
  setActiveTab,
  onAddManualFeedback,
  onPreview,
}: ClipInspectorPanelProps) {
  const [showAllAssets, setShowAllAssets] = useState(false);
  const [promptVersionIndex, setPromptVersionIndex] = useState(selectedVersionIndex);
  const promptVersions = versions
    .map((version, index) => ({ version, index, sections: promptSections(version) }))
    .filter((item) => item.sections.length > 0);
  const selectedPromptVersion =
    promptVersions.find((item) => item.index === promptVersionIndex) ||
    promptVersions.find((item) => item.index === selectedVersionIndex) ||
    promptVersions[promptVersions.length - 1];
  const videoPromptSection = selectedPromptVersion?.sections.find((section) => section.kind === "video");
  const supportingPromptSections = selectedPromptVersion?.sections.filter((section) => section.kind !== "video") || [];
  const referenceItems = selectedPromptVersion ? referenceItemsForVersion(selectedPromptVersion.version, clipState?.asset_state.referenced_frames || []) : [];
  const projectAssets = flattenProjectAssets(projectData);
  const backendAssets = clipState?.asset_state.selected_assets || [];
  const assets = clipState
    ? backendAssets
    : selectedVersion?.selected_assets?.length
    ? (selectedVersion.selected_assets as AssetReferenceInput[]).map((asset) => assetFromReference(asset, projectAssets))
    : projectAssets;
  const visibleAssets = showAllAssets ? assets : assets.slice(0, ASSET_PREVIEW_LIMIT);
  const generatedVideos = clipState
    ? clipState.video_state.generated_videos.map((video) => ({ video, versionIndex: selectedVersionIndex }))
    : collectGeneratedVideos(versions);

  useEffect(() => {
    setPromptVersionIndex(selectedVersionIndex);
  }, [clip.clip, selectedVersionIndex]);

  useEffect(() => {
    if (activeTab === "videos" && generatedVideos.length === 0) {
      setActiveTab("assets");
    }
  }, [activeTab, generatedVideos.length, setActiveTab]);

  return (
    <section className="clip-inspector-panel">
      <div
        className={`clip-inspector-tabs ${generatedVideos.length > 0 ? "has-videos" : ""}`}
        role="tablist"
        aria-label="Clip context"
        style={{ gridTemplateColumns: `repeat(${4 + (generatedVideos.length > 0 ? 1 : 0)}, 1fr)` }}
      >
        <button className={activeTab === "feedback" ? "active" : ""} type="button" onClick={() => setActiveTab("feedback")}>
          Feedback ({feedbackItems.length})
        </button>
        <button className={activeTab === "prompts" ? "active" : ""} type="button" onClick={() => setActiveTab("prompts")}>
          Prompt ({promptVersions.length})
        </button>
        <button className={activeTab === "references" ? "active" : ""} type="button" onClick={() => setActiveTab("references")}>
          References ({referenceItems.length})
        </button>
        <button className={activeTab === "assets" ? "active" : ""} type="button" onClick={() => setActiveTab("assets")}>
          Assets ({assets.length})
        </button>
        {generatedVideos.length > 0 && (
          <button className={activeTab === "videos" ? "active" : ""} type="button" onClick={() => setActiveTab("videos")}>
            Videos ({generatedVideos.length})
          </button>
        )}
      </div>

      {activeTab === "feedback" && (
        <div className="clip-tab-body">
          <div className="selected-section-title">
            <Icon name="comments" />
            <span>Clip Feedback</span>
          </div>
          {feedbackItems.length ? (
            feedbackItems.map((item) => (
              <div className="compact-feedback-note" key={item.raw_index}>
                <span className={`tag tag-${item.category}`}>{item.category}</span>
                <p>{item.remark}</p>
              </div>
            ))
          ) : (
            <div className="compact-feedback-empty">
              <Icon name="check" /> No pending feedback on this clip
            </div>
          )}
          <button className="shoko-ghost-button compact-action" type="button" onClick={() => onAddManualFeedback(clip.clip)}>
            <Icon name="plus" /> Add Feedback
          </button>
        </div>
      )}

      {activeTab === "prompts" && (
        <div className="clip-tab-body prompts-tab-body">
          {promptVersions.length ? (
            <div className="single-prompt-view">
              <div className="prompt-version-toolbar">
                <label>
                  <span>Version</span>
                  <select value={selectedPromptVersion.index} onChange={(event) => setPromptVersionIndex(Number(event.target.value))}>
                    {promptVersions.map(({ version, index }) => (
                      <option value={index} key={`${version.prompt_version_id || version.timestamp || "prompt"}-${index}`}>
                        {versionLabel(version, index)}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedPromptVersion.index === selectedVersionIndex && (
                  <span className="prompt-version-status" title="Selected version" aria-label="Selected version">
                    <Icon name="check" />
                  </span>
                )}
              </div>

              <article className="clip-prompt-card single">
                <section className="clip-prompt-section video-prompt-section">
                  <h4>Video Prompt</h4>
                  <pre>{videoPromptSection?.text || "No video prompt was generated for this version."}</pre>
                </section>
                {supportingPromptSections.length > 0 && (
                  <details className="supporting-prompts">
                    <summary>Supporting prompts</summary>
                    {supportingPromptSections.map((section) => (
                      <section className="clip-prompt-section" key={section.label}>
                        <h4>{section.label}</h4>
                        <pre>{section.text}</pre>
                      </section>
                    ))}
                  </details>
                )}
                {selectedPromptVersion.version.explanation && <p className="clip-prompt-explanation">{selectedPromptVersion.version.explanation}</p>}
              </article>
            </div>
          ) : (
            <div className="compact-feedback-empty">
              <Icon name="file" /> No generated prompts for this clip
            </div>
          )}
        </div>
      )}

      {activeTab === "references" && (
        <div className="clip-tab-body references-tab-body">
          {promptVersions.length ? (
            <>
              <div className="prompt-version-toolbar">
                <label>
                  <span>Version</span>
                  <select value={selectedPromptVersion.index} onChange={(event) => setPromptVersionIndex(Number(event.target.value))}>
                    {promptVersions.map(({ version, index }) => (
                      <option value={index} key={`${version.prompt_version_id || version.timestamp || "prompt"}-${index}`}>
                        {versionLabel(version, index)}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedPromptVersion.index === selectedVersionIndex && (
                  <span className="prompt-version-status" title="Selected version" aria-label="Selected version">
                    <Icon name="check" />
                  </span>
                )}
              </div>
              <div className="clip-references-grid">
                {referenceItems.length ? referenceItems.map((item) => (
                  <button
                    className="clip-reference-tile"
                    key={`${item.source}-${item.path || item.url}-${item.label}`}
                    type="button"
                    title={`Preview ${item.label}`}
                    onClick={() => onPreview({ file: referenceItemToAsset(item) })}
                  >
                    <div className="clip-reference-thumb">
                      <img src={staticUrl(item.url || item.path)} alt="" loading="lazy" onError={(event) => { event.currentTarget.hidden = true; }} />
                      <Icon name="image" className="asset-fallback-icon" />
                    </div>
                    <div className="clip-reference-meta">
                      <strong>{item.label}</strong>
                      <span>{item.meta}</span>
                    </div>
                  </button>
                )) : (
                  <div className="compact-feedback-empty">
                    <Icon name="image" /> No extracted or reference frames for this prompt version
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="compact-feedback-empty">
              <Icon name="file" /> No prompt versions available yet
            </div>
          )}
        </div>
      )}

      {activeTab === "assets" && (
        <div className="clip-tab-body assets-tab-body">
          <div className="selected-section-title">
            <Icon name="box" />
            <span>Assets used in this clip</span>
          </div>
          <div className={`clip-assets-grid ${showAllAssets ? "show-all" : ""}`}>
            {visibleAssets.length ? visibleAssets.map((asset) => (
              <button
                className={`clip-asset-tile ${asset.type}`}
                key={`${asset.name}-${asset.url || asset.path}`}
                type="button"
                title={`Preview ${asset.name}`}
                onClick={() => onPreview({ file: asset })}
              >
                <div className="clip-asset-thumb">
                  {asset.type === "image" && (asset.url || asset.path) ? (
                    <>
                      <img src={staticUrl(asset.url || asset.path)} alt="" loading="lazy" onError={(event) => { event.currentTarget.hidden = true; }} />
                      <Icon name="image" className="asset-fallback-icon" />
                    </>
                  ) : null}
                  {asset.type === "video" && (asset.url || asset.path) ? (
                    <>
                      <video
                        src={`${staticUrl(asset.url || asset.path)}#t=0.2`}
                        preload="metadata"
                        muted
                        playsInline
                        onError={(event) => { event.currentTarget.hidden = true; }}
                      />
                      <Icon name="video" className="asset-fallback-icon" />
                    </>
                  ) : null}
                  {asset.type === "audio" && <MiniWaveform />}
                  {asset.type === "other" && <Icon name="file" />}
                </div>
                <span title={asset.name}>{asset.name}</span>
              </button>
            )) : (
              <div className="compact-feedback-empty">
                <Icon name="box" /> No selected assets for this clip
              </div>
            )}
          </div>
          {assets.length > ASSET_PREVIEW_LIMIT && (
            <button className="clip-assets-view-all" type="button" onClick={() => setShowAllAssets((value) => !value)}>
              {showAllAssets ? "Show Fewer Assets" : `View All Assets (${assets.length})`}
            </button>
          )}
        </div>
      )}

      {activeTab === "videos" && generatedVideos.length > 0 && (
        <div className="clip-tab-body generated-videos-tab-body">
          <div className="generated-videos-grid">
            {generatedVideos.map(({ video, versionIndex }) => (
              <button
                className="generated-video-tile"
                key={`${video.generated_video_id || video.path}-${versionIndex}-${video.version}`}
                type="button"
                title={`Preview ${video.label || `Generated video v${video.version}`}`}
                onClick={() => onPreview({ file: generatedVideoToAsset(video) })}
              >
                <div className="generated-video-thumb">
                  <video src={`${staticUrl(video.url || video.path)}#t=0.2`} preload="metadata" muted playsInline />
                  <span>
                    <Icon name="video" />
                  </span>
                </div>
                <div className="generated-video-meta">
                  <strong>{video.label || `Generated video v${video.version}`}</strong>
                  <span>{versionLabel(versions[versionIndex], versionIndex)}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function MiniWaveform() {
  return (
    <div className="mini-waveform" aria-hidden="true">
      {WAVEFORM_PATTERN.slice(0, 18).map((height, index) => (
        <i key={index} style={{ height: Math.max(10, height * 0.72) }} />
      ))}
    </div>
  );
}

function promptSections(version: PromptVersion) {
  const sections: Array<{ label: string; text: string; kind: "video" | "provider" | "initial" }> = [];
  const videoPrompt = normalizePromptText(version.video_model_prompt);
  const segmindPrompt = normalizePromptText(version.segmind_prompt);
  const initialFramePrompt = normalizePromptText(version.initial_frame_prompt);

  if (videoPrompt) sections.push({ label: "Video Prompt", text: videoPrompt, kind: "video" });
  if (segmindPrompt && segmindPrompt !== videoPrompt) sections.push({ label: "Provider Prompt", text: segmindPrompt, kind: "provider" });
  if (initialFramePrompt) sections.push({ label: "Initial Frame Prompt", text: initialFramePrompt, kind: "initial" });
  return sections;
}

function normalizePromptText(value?: string | null) {
  return typeof value === "string" ? value.trim() : "";
}

type ReferenceItem = {
  label: string;
  meta: string;
  path: string;
  url: string;
  source: "clip_frames" | "referenced_frames";
};

function referenceItemsForVersion(version: PromptVersion, fallbackReferencedFrames: ReferencedFrameState[]): ReferenceItem[] {
  const clipFrames = (version.clip_frame_paths || []).map((path, index) => ({
    label: `Extracted frame ${index + 1}`,
    meta: basename(path),
    path,
    url: path,
    source: "clip_frames" as const,
  }));
  const versionReferencedFrames: ReferencedFrameState[] = version.referenced_frames?.length
    ? version.referenced_frames
    : (version.referenced_frame_paths || []).map((path, index) => ({
        reference_id: `${path}-${index}`,
        frame_path: path,
        reason: version.referenced_frame_labels?.[index],
      }));
  const referencedFrames = (versionReferencedFrames.length ? versionReferencedFrames : fallbackReferencedFrames).map((frame, index) => {
    const path = frame.frame_path || frame.url || "";
    return {
      label: frame.timestamp ? `Reference ${frame.timestamp}` : `Reference frame ${index + 1}`,
      meta: frame.reason || frame.clip_used || basename(path),
      path,
      url: frame.url || path,
      source: "referenced_frames" as const,
    };
  }).filter((item) => item.path || item.url);
  return [...clipFrames, ...referencedFrames];
}

function referenceItemToAsset(item: ReferenceItem): AssetFile {
  const path = item.path || item.url;
  return {
    name: item.label,
    path,
    url: item.url || path,
    type: "image",
    size: item.meta,
    source: item.source,
  };
}

function collectGeneratedVideos(versions: PromptVersion[]) {
  return versions.flatMap((version, versionIndex) =>
    (version.generated_videos || []).map((video) => ({ video, versionIndex })),
  );
}

function generatedVideoToAsset(video: GeneratedVideo): AssetFile {
  const path = video.path || video.url;
  return {
    name: video.label || basename(path),
    path,
    url: video.url || path,
    type: "video",
    size: video.resolution || "",
  };
}

function flattenProjectAssets(projectData: ProjectData): AssetFile[] {
  return Object.values(projectData.assets).flat();
}

function assetFromReference(asset: AssetReferenceInput, projectAssets: AssetFile[]): AssetFile {
  if (typeof asset === "object" && "name" in asset && "type" in asset) {
    const matchedAsset = findMatchingProjectAsset(asset, projectAssets);
    return {
      ...matchedAsset,
      ...asset,
      url: asset.url || matchedAsset?.url || normalizeAssetUrl(asset.path || matchedAsset?.path || ""),
      path: asset.path || matchedAsset?.path || asset.url || asset.name,
      size: asset.size || matchedAsset?.size || "",
    };
  }

  const rawPath = typeof asset === "string" ? asset : asset.path || "";
  const normalized = rawPath.replace(/\\/g, "/");
  const matchedAsset = findMatchingProjectAsset({ path: normalized }, projectAssets);

  if (matchedAsset) {
    return matchedAsset;
  }

  return {
    name: basename(normalized),
    path: normalized,
    url: normalizeAssetUrl(normalized),
    type: inferAssetType(normalized),
    size: "",
  };
}

function findMatchingProjectAsset(reference: Partial<AssetFile | ClipAssetReference>, projectAssets: AssetFile[]) {
  const referenceValues = new Set(
    [
      "asset_id" in reference ? reference.asset_id : null,
      "path" in reference ? reference.path : null,
      "selected_path" in reference ? reference.selected_path : null,
      "name" in reference ? reference.name : null,
    ]
      .filter(Boolean)
      .map((value) => normalizeComparableAssetPath(String(value))),
  );

  if (referenceValues.size === 0) return undefined;

  return projectAssets.find((asset) => {
    const candidates = [asset.asset_id, asset.path, asset.url, asset.selected_path, asset.name]
      .filter(Boolean)
      .map((value) => normalizeComparableAssetPath(String(value)));
    return candidates.some((candidate) => {
      if (referenceValues.has(candidate)) return true;
      return Array.from(referenceValues).some((referenceValue) => candidate.endsWith(referenceValue) || referenceValue.endsWith(candidate));
    });
  });
}

function normalizeComparableAssetPath(value: string) {
  return value
    .replace(/\\/g, "/")
    .replace(/^https?:\/\/[^/]+/, "")
    .replace(/^\/+/, "")
    .replace(/^(assets|data)\//, "")
    .toLowerCase();
}

function normalizeAssetUrl(path: string) {
  if (/^https?:\/\//.test(path)) return path;
  const assetsIndex = path.indexOf("/assets/");
  if (assetsIndex !== -1) return path.slice(assetsIndex);
  const relativeAssetsIndex = path.indexOf("assets/");
  if (relativeAssetsIndex !== -1) return `/${path.slice(relativeAssetsIndex)}`;
  return path;
}

function inferAssetType(asset: string): "image" | "video" | "audio" | "other" {
  const normalized = asset.toLowerCase();
  if (/\.(png|jpe?g|webp|gif)$/.test(normalized)) return "image";
  if (/\.(mp4|mov|webm|m4v)$/.test(normalized)) return "video";
  if (/\.(wav|mp3|m4a|aac|flac)$/.test(normalized)) return "audio";
  return "other";
}
