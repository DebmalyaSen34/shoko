import { Icon } from "./Icon";

const WORKFLOW_WORDS = ["Reviewing", "Improving", "Scripting", "Checking"];

export function WorkflowRunningLabel() {
  return (
    <span className="workflow-running-label" aria-live="polite">
      <Icon name="settings" className="workflow-gear" />
      <span className="workflow-word-cycle" aria-label="Workflow running">
        {WORKFLOW_WORDS.map((word) => (
          <span key={word}>{word}</span>
        ))}
      </span>
    </span>
  );
}
