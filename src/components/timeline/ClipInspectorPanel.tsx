import { useState } from "react";
import type { AssetFile, ClipAssetReference, FeedbackGroup, PreviewState, ProjectData, PromptVersion, TimelineClip } from "../../types";
import { staticUrl } from "../../lib/api";
import { basename, formatSeconds, formatTimecode } from "../../lib/format";
import { Icon } from "../Icon";

type ClipInspectorTab = "feedback" | "details" | "assets";

type ClipInspectorPanelProps = {
  activeTab: ClipInspectorTab;
  clip: TimelineClip;
  clipIndex: number;
  feedbackItems: FeedbackGroup["feedback_items"];
  projectData: ProjectData;
  selectedVersion?: PromptVersion;
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
  clipIndex,
  feedbackItems,
  projectData,
  selectedVersion,
  setActiveTab,
  onAddManualFeedback,
  onPreview,
}: ClipInspectorPanelProps) {
  const [showAllAssets, setShowAllAssets] = useState(false);
  const projectAssets = flattenProjectAssets(projectData);
  const assets = selectedVersion?.selected_assets?.length
    ? (selectedVersion.selected_assets as AssetReferenceInput[]).map((asset) => assetFromReference(asset, projectAssets))
    : projectAssets;
  const visibleAssets = showAllAssets ? assets : assets.slice(0, ASSET_PREVIEW_LIMIT);

  return (
    <section className="clip-inspector-panel">
      <div className="clip-inspector-tabs" role="tablist" aria-label="Clip context">
        <button className={activeTab === "feedback" ? "active" : ""} type="button" onClick={() => setActiveTab("feedback")}>
          Feedback ({feedbackItems.length})
        </button>
        <button className={activeTab === "details" ? "active" : ""} type="button" onClick={() => setActiveTab("details")}>
          Details
        </button>
        <button className={activeTab === "assets" ? "active" : ""} type="button" onClick={() => setActiveTab("assets")}>
          Assets ({assets.length})
        </button>
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

      {activeTab === "details" && (
        <div className="clip-tab-body">
          <div className="selected-details-grid">
            <Detail label="Clip" value={basename(clip.clip)} />
            <Detail label="Sequence Position" value={`#${clipIndex + 1}`} />
            <Detail label="Timecode" value={`${formatTimecode(clip.start_tc)} - ${formatTimecode(clip.end_tc)}`} />
            <Detail label="Bounds" value={`${formatSeconds(clip.start_s)} - ${formatSeconds(clip.end_s)}`} />
          </div>
        </div>
      )}

      {activeTab === "assets" && (
        <div className="clip-tab-body assets-tab-body">
          <div className="selected-section-title">
            <Icon name="box" />
            <span>Assets used in this clip</span>
          </div>
          <div className={`clip-assets-grid ${showAllAssets ? "show-all" : ""}`}>
            {visibleAssets.map((asset) => (
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
            ))}
          </div>
          {assets.length > ASSET_PREVIEW_LIMIT && (
            <button className="clip-assets-view-all" type="button" onClick={() => setShowAllAssets((value) => !value)}>
              {showAllAssets ? "Show Fewer Assets" : `View All Assets (${assets.length})`}
            </button>
          )}
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

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="compact-detail">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
