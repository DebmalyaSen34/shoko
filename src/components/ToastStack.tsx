import type { Toast } from "../types";
import { Icon } from "./Icon";

export function ToastStack({ toasts }: { toasts: Toast[] }) {
  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div className={`toast ${toast.type}`} key={toast.id} role="status" aria-live="polite">
          <Icon name={toast.type === "error" ? "warning" : toast.type === "success" ? "check" : "file"} />
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
