import React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { ArrowUp, Square } from "lucide-react";

// Utility function for className merging
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(" ");

// Textarea Component
export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  className?: string;
}
export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => (
  <textarea
    className={cn(
      "flex w-full rounded-md border-none bg-transparent px-3 py-1.5 text-sm text-[var(--text-primary,#24292e)] placeholder:text-[var(--text-muted,rgba(36,41,46,0.5))] focus-visible:outline-none focus-visible:ring-0 disabled:cursor-not-allowed disabled:opacity-50 min-h-[36px] max-h-[140px] resize-none scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-transparent dark:scrollbar-thumb-gray-600",
      className
    )}
    ref={ref}
    rows={1}
    {...props}
  />
));
Textarea.displayName = "Textarea";

// Tooltip Components
export const TooltipProvider = TooltipPrimitive.Provider;
export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;
export const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md border border-[var(--border-color,rgba(36,41,46,0.16))] bg-[var(--bg-secondary,#ffffff)] px-2.5 py-1 text-xs text-[var(--text-primary,#24292e)] shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
      className
    )}
    {...props}
  />
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

// Button Component
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
}
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    const variantClasses = {
      default: "bg-[var(--accent-primary,#2dba4e)] hover:opacity-90 text-white shadow-sm",
      outline: "border border-[var(--border-color)] bg-transparent hover:bg-[var(--bg-hover)] text-[var(--text-primary)]",
      ghost: "bg-transparent hover:bg-[var(--bg-hover)] text-[var(--text-secondary)]",
    };
    const sizeClasses = {
      default: "h-8 px-3 py-1 text-xs rounded-md",
      sm: "h-7 px-2 text-xs rounded-md",
      lg: "h-10 px-4 text-sm rounded-lg",
      icon: "h-7 w-7 rounded-md aspect-square flex-shrink-0",
    };
    return (
      <button
        className={cn(
          "inline-flex items-center justify-center font-medium transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent-primary,#2dba4e)] disabled:pointer-events-none disabled:opacity-50 active:scale-95",
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

// PromptInput Context and Components
export interface PromptInputContextType {
  isLoading: boolean;
  value: string;
  setValue: (value: string) => void;
  maxHeight: number | string;
  onSubmit?: () => void;
  disabled?: boolean;
}
const PromptInputContext = React.createContext<PromptInputContextType>({
  isLoading: false,
  value: "",
  setValue: () => {},
  maxHeight: 140,
  onSubmit: undefined,
  disabled: false,
});
export function usePromptInput() {
  const context = React.useContext(PromptInputContext);
  if (!context) throw new Error("usePromptInput must be used within a PromptInput");
  return context;
}

export interface PromptInputProps {
  isLoading?: boolean;
  value?: string;
  onValueChange?: (value: string) => void;
  maxHeight?: number | string;
  onSubmit?: () => void;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
}
export const PromptInput = React.forwardRef<HTMLDivElement, PromptInputProps>(
  (
    {
      className,
      isLoading = false,
      maxHeight = 140,
      value,
      onValueChange,
      onSubmit,
      children,
      disabled = false,
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = React.useState(value || "");
    const handleChange = (newValue: string) => {
      setInternalValue(newValue);
      onValueChange?.(newValue);
    };
    return (
      <TooltipProvider>
        <PromptInputContext.Provider
          value={{
            isLoading,
            value: value ?? internalValue,
            setValue: onValueChange ?? handleChange,
            maxHeight,
            onSubmit,
            disabled,
          }}
        >
          <div
            ref={ref}
            className={cn(
              "rounded-lg border border-[var(--border-color,rgba(36,41,46,0.16))] bg-[var(--bg-secondary,#ffffff)] p-1.5 shadow-sm transition-all duration-200 focus-within:border-[var(--accent-primary,#2dba4e)] focus-within:ring-1 focus-within:ring-[var(--accent-primary-glow,rgba(45,186,78,0.18))]",
              isLoading && "border-amber-500/60",
              className
            )}
          >
            {children}
          </div>
        </PromptInputContext.Provider>
      </TooltipProvider>
    );
  }
);
PromptInput.displayName = "PromptInput";

export interface PromptInputTextareaProps extends React.ComponentProps<typeof Textarea> {
  disableAutosize?: boolean;
  placeholder?: string;
}
export const PromptInputTextarea: React.FC<PromptInputTextareaProps> = ({
  className,
  onKeyDown,
  disableAutosize = false,
  placeholder,
  ...props
}) => {
  const { value, setValue, maxHeight, onSubmit, disabled } = usePromptInput();
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  React.useEffect(() => {
    if (disableAutosize || !textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height =
      typeof maxHeight === "number"
        ? `${Math.min(textareaRef.current.scrollHeight, maxHeight)}px`
        : `min(${textareaRef.current.scrollHeight}px, ${maxHeight})`;
  }, [value, maxHeight, disableAutosize]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit?.();
    }
    onKeyDown?.(e);
  };

  return (
    <Textarea
      ref={textareaRef}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      className={cn("text-xs md:text-sm", className)}
      disabled={disabled}
      placeholder={placeholder}
      {...props}
    />
  );
};

export interface PromptInputActionProps extends React.ComponentProps<typeof Tooltip> {
  tooltip: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}
export const PromptInputAction: React.FC<PromptInputActionProps> = ({
  tooltip,
  children,
  className,
  side = "top",
  ...props
}) => {
  const { disabled } = usePromptInput();
  return (
    <Tooltip {...props}>
      <TooltipTrigger asChild disabled={disabled}>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} className={className}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
};

// Main PromptInputBox Component
export interface PromptInputBoxProps {
  onSend?: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
  value?: string;
  onValueChange?: (val: string) => void;
  topAddon?: React.ReactNode;
}
export const PromptInputBox = React.forwardRef<HTMLDivElement, PromptInputBoxProps>((props, ref) => {
  const {
    onSend = () => {},
    isLoading = false,
    placeholder = "Ask about feedback, assets, timeline, memory, workflow... Type /help for commands.",
    className,
    value: externalValue,
    onValueChange: externalOnValueChange,
    topAddon,
  } = props;

  const [internalInput, setInternalInput] = React.useState("");
  const input = externalValue !== undefined ? externalValue : internalInput;
  const setInput = externalOnValueChange || setInternalInput;

  const promptBoxRef = React.useRef<HTMLDivElement>(null);

  const handleSubmit = () => {
    if (input.trim()) {
      onSend(input);
      setInput("");
    }
  };

  const hasContent = input.trim() !== "";

  return (
    <PromptInput
      value={input}
      onValueChange={setInput}
      isLoading={isLoading}
      onSubmit={handleSubmit}
      className={cn(
        "w-full bg-[var(--bg-secondary,#ffffff)] border-[var(--border-color,rgba(36,41,46,0.16))] transition-all duration-200 relative",
        className
      )}
      disabled={isLoading}
      ref={ref || promptBoxRef}
    >
      {topAddon}

      <div className="flex items-end gap-2 w-full">
        <div className="flex-1 min-w-0">
          <PromptInputTextarea
            placeholder={placeholder}
            className="text-xs md:text-sm text-[var(--text-primary,#24292e)] placeholder:text-[var(--text-muted,rgba(36,41,46,0.5))]"
          />
        </div>

        <PromptInputAction tooltip={isLoading ? "Generating response..." : "Send prompt (Enter)"}>
          <Button
            variant="default"
            size="icon"
            className={cn(
              "h-7 w-7 rounded-md transition-all duration-150 flex-shrink-0 mb-1",
              hasContent
                ? "bg-[var(--accent-primary,#2dba4e)] hover:bg-[var(--accent-primary-hover,#2dba4e)] text-white shadow-xs"
                : "bg-transparent hover:bg-[var(--bg-hover,rgba(36,41,46,0.06))] text-[var(--text-secondary,#2b3137)]"
            )}
            onClick={handleSubmit}
            disabled={isLoading || !hasContent}
            type="button"
          >
            {isLoading ? (
              <Square className="h-3 w-3 fill-current animate-pulse" />
            ) : (
              <ArrowUp className="h-3.5 w-3.5" />
            )}
          </Button>
        </PromptInputAction>
      </div>
    </PromptInput>
  );
});
PromptInputBox.displayName = "PromptInputBox";
