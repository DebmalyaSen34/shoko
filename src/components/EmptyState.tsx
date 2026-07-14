import { Icon } from "./Icon";

export function EmptyState({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="empty-state">
      <Icon name={icon} />
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}
