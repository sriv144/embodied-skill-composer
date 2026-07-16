import { Check, Gauge } from "lucide-react";
import type { TraceFrame } from "../types";

export function ControllerSelect({
  value,
  onChange
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="segmented controller-select" aria-label="Replay controller">
      {[
        ["sequential", "Sequential"],
        ["greedy", "Greedy"],
        ["optimized", "CP-SAT"]
      ].map(([id, label]) => (
        <button key={id} className={value === id ? "selected" : ""} onClick={() => onChange(id)}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="fact">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function FidelityBadge({
  level,
  active = false
}: {
  level: "Event Sim" | "MuJoCo" | "CoppeliaSim";
  active?: boolean;
}) {
  return (
    <span className={active ? "fidelity-badge active" : "fidelity-badge"}>
      {active ? <Check size={12} /> : <Gauge size={12} />}
      {level}
    </span>
  );
}

export function nearestFrame(frames: TraceFrame[], time: number) {
  return frames.reduce(
    (best, frame) =>
      Math.abs(frame.timestamp_s - time) < Math.abs(best.timestamp_s - time) ? frame : best,
    frames[0]
  );
}

export function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
}

export function downloadText(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}
