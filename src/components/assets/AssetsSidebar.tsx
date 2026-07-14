import type { AssetFile, PreviewState } from "../../types";
import { categoryIcon, fileIcon, Icon } from "../Icon";

type AssetsSidebarProps = {
  assetFilter: string;
  assetTotal: number;
  assets?: Record<string, AssetFile[]>;
  collapsed: boolean;
  collapsedCategories: Set<string>;
  loadError: string;
  onCollapse: () => void;
  onExpand: () => void;
  onFilterChange: (filter: string) => void;
  onPreview: (preview: PreviewState) => void;
  onToggleCategory: (category: string) => void;
};

export function AssetsSidebar({
  assetFilter,
  assetTotal,
  assets,
  collapsed,
  collapsedCategories,
  loadError,
  onCollapse,
  onExpand,
  onFilterChange,
  onPreview,
  onToggleCategory,
}: AssetsSidebarProps) {
  return (
    <>
      {collapsed && (
        <button className="expand-assets-btn" title="Expand Assets" onClick={onExpand}>
          <Icon name="chevronRight" />
        </button>
      )}

      <aside className={`left-panel ${collapsed ? "collapsed" : ""}`}>
        <div className="panel-header">
          <h2>
            <Icon name="box" /> Project Assets <span className="badge">{assetTotal}</span>
          </h2>
          <button className="icon-btn" title="Collapse Assets" onClick={onCollapse}>
            <Icon name="chevronLeft" />
          </button>
        </div>
        <div className="asset-search">
          <Icon name="search" />
          <input value={assetFilter} onChange={(event) => onFilterChange(event.target.value)} placeholder="Filter assets..." />
        </div>
        <div className="assets-list">
          {loadError ? (
            <div className="empty-inline">
              <Icon name="warning" />
              <p>Assets will appear after the backend connects.</p>
            </div>
          ) : assets ? (
            <AssetsList
              assets={assets}
              filter={assetFilter}
              collapsedCategories={collapsedCategories}
              onToggleCategory={onToggleCategory}
              onPreview={onPreview}
            />
          ) : (
            <div className="panel-loading">Loading assets...</div>
          )}
        </div>
      </aside>
    </>
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
