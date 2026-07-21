import { Icon } from "./Icon";

export function EmptyState({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-20 px-5 max-w-[460px] mx-auto text-text-muted">
      <Icon name={icon} className="text-[48px] mb-4" />
      <h3 className="text-lg font-bold text-text-primary mt-0 mb-2">{title}</h3>
      <p className="text-sm leading-normal m-0">{text}</p>
    </div>
  );
}
