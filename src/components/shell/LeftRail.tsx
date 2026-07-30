import { Icon } from "../Icon";

export type RailItemId = "project" | "timeline" | "feedback" | "settings";

type RailItem = {
  id: RailItemId;
  icon: string;
  label: string;
};

const RAIL_ITEMS: RailItem[] = [
  { id: "project", icon: "folder", label: "Project" },
  { id: "timeline", icon: "timeline", label: "Timeline" },
  { id: "feedback", icon: "comments", label: "Feedback" },
  { id: "settings", icon: "settings", label: "Settings" },
];

type LeftRailProps = {
  activeItem: RailItemId;
  collapsedPanelOpenLabel?: string;
  collapsedPanelOpen?: boolean;
  projectContextActive?: boolean;
  onCollapsedPanelOpen?: () => void;
  onSelect: (item: RailItemId) => void;
};

export function LeftRail({
  activeItem,
  collapsedPanelOpen = false,
  collapsedPanelOpenLabel = "Open project workflow",
  projectContextActive = false,
  onCollapsedPanelOpen,
  onSelect,
}: LeftRailProps) {
  return (
    <nav className={`shoko-left-rail ${onCollapsedPanelOpen ? "has-drawer-toggle" : ""} ${collapsedPanelOpen ? "drawer-open" : ""}`} aria-label="Primary">
      {RAIL_ITEMS.map((item) => {
        const active = projectContextActive ? item.id === "project" : activeItem === item.id;
        return (
          <button
            className={`rail-item ${active ? "active" : ""}`}
            key={item.id}
            type="button"
            onClick={() => onSelect(item.id)}
            aria-current={active ? "page" : undefined}
          >
            <Icon name={item.icon} />
            <span>{item.label}</span>
          </button>
        );
      })}
      <button className="rail-collapse" type="button" title={collapsedPanelOpenLabel} onClick={onCollapsedPanelOpen}>
        <Icon name={onCollapsedPanelOpen && !collapsedPanelOpen ? "chevronRight" : "chevronLeft"} />
      </button>
    </nav>
  );
}
