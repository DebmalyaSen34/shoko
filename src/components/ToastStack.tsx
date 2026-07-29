import type { Toast } from "../types";
import { Icon } from "./Icon";

const borderColors: Record<Toast["type"], string> = {
  error: "border-l-error",
  success: "border-l-success",
  info: "border-l-info",
};

export function ToastStack({ toasts }: { toasts: Toast[] }) {
  return (
    <div className="fixed right-6 bottom-11 flex flex-col gap-2 z-[10000]">
      {toasts.map((toast) => (
        <div
          className={`bg-bg-tertiary border border-border-color border-l-4 ${borderColors[toast.type] || "border-l-info"} text-text-primary px-4 py-2.5 rounded-[3px] text-xs font-semibold shadow-none flex items-center gap-2 animate-[toastIn_0.25s_ease]`}
          key={toast.id}
        >
          <Icon name={toast.type === "error" ? "warning" : toast.type === "success" ? "check" : "file"} />
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
