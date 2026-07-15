import { BrainCircuit, Pause, Play, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { ConstructionScene } from "../components/ConstructionScene";
import { ControllerSelect, FidelityBadge, formatTime, nearestFrame } from "../components/WorkbenchControls";
import type { BrainEvent, CoppeliaHealth, Project, Trace } from "../types";

export function SimulateView({
  project,
  trace,
  controller,
  coppelia,
  onController,
  onTrace
}: {
  project: Project;
  trace: Trace;
  controller: string;
  coppelia: CoppeliaHealth | null;
  onController: (controller: string) => void;
  onTrace: (trace: Trace) => void;
}) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [recovering, setRecovering] = useState(false);
  useEffect(() => setTime(0), [controller]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setTime((value) => value >= trace.metrics.makespan_s ? 0 : Math.min(value + 2, trace.metrics.makespan_s)),
      160
    );
    return () => window.clearInterval(timer);
  }, [playing, trace.metrics.makespan_s]);
  const frame = nearestFrame(trace.frames, time);
  const event = [...trace.brain_events].reverse().find((item) => item.timestamp_s <= time) ?? trace.brain_events[0];

  const inject = async (type: string, label: string) => {
    setPlaying(false);
    setRecovering(true);
    try {
      const recovered = await api.disrupt(controller, type, Math.min(time, trace.metrics.makespan_s - 1));
      onTrace(recovered);
      setNotice(label);
    } catch (reason) {
      setNotice(String(reason));
    } finally {
      setRecovering(false);
    }
  };

  return (
    <div className="simulation-layout">
      <section className="simulation-main">
        <div className="scene-toolbar simulation-toolbar">
          <div><p className="eyebrow">Digital twin</p><h2>{frame.completed_module_ids.length} / {project.plan.modules.length} modules installed</h2></div>
          <div className="simulation-toolbar-right">
            <div className="fidelity-strip"><FidelityBadge level="Event Sim" active /><FidelityBadge level="MuJoCo" /><FidelityBadge level="CoppeliaSim" active={Boolean(coppelia?.reachable)} /></div>
            <ControllerSelect value={controller} onChange={onController} />
          </div>
        </div>
        <div className="simulation-canvas">
          <ConstructionScene project={project} frame={frame} routes={trace.schedule.jobs} />
          <div className="scene-readout"><span>TRACE</span><strong>{controller === "optimized" ? "CP-SAT" : controller}</strong><small>T+{time}s</small></div>
        </div>
        <div className="timeline-controls">
          <button className="icon-button" onClick={() => setPlaying(!playing)} title={playing ? "Pause replay" : "Play replay"}>{playing ? <Pause size={18} /> : <Play size={18} />}</button>
          <span className="timecode">{formatTime(time)}</span>
          <input aria-label="Construction timeline" type="range" min="0" max={trace.metrics.makespan_s} value={time} onChange={(event) => { setPlaying(false); setTime(Number(event.target.value)); }} />
          <span>{formatTime(trace.metrics.makespan_s)}</span>
          <button className="icon-button" onClick={() => setTime(0)} title="Reset replay"><RotateCcw size={17} /></button>
        </div>
      </section>
      <aside className="brain-panel">
        <div className="brain-title"><div className="brain-icon"><BrainCircuit size={21} /></div><div><p className="eyebrow">ConstructionBrain</p><h2>Decision epoch</h2></div></div>
        <Decision event={event} />
        <h3>Fleet telemetry</h3>
        <div className="robot-state-list">
          {frame.robots.map((robot, index) => (
            <div key={robot.robot_id}><span className={`robot-swatch r${index + 1}`} /><strong>{robot.robot_id.replace("robot_", "R")}</strong><span>{robot.status}</span><small>{robot.module_id?.replaceAll("_", " ") ?? "available"}</small></div>
          ))}
        </div>
        <h3>Recovery event</h3>
        <div className="failure-actions">
          <button disabled={recovering} onClick={() => inject("obstacle", "Obstacle inserted; schedule recovered.")}>Obstacle</button>
          <button disabled={recovering} onClick={() => inject("robot_unavailable", "Robot unavailable; jobs reassigned.")}>Robot offline</button>
          <button disabled={recovering} onClick={() => inject("dropped_resource", "Payload dropped; transport retried.")}>Drop payload</button>
        </div>
        {notice && <p className="recovery-note"><RotateCcw size={15} />{notice}</p>}
        <div className="fidelity-note"><strong>{coppelia?.reachable ? "Coppelia online" : "Trace playback"}</strong><span>{coppelia?.detail ?? "Checking simulator health"}</span></div>
      </aside>
    </div>
  );
}

function Decision({ event }: { event: BrainEvent }) {
  return (
    <div className="decision-block">
      <div className="decision-time">T+{event.timestamp_s}s <span>{event.predicted_remaining_s}s remaining</span></div>
      <div className="decision-module"><strong>{event.module_id?.replaceAll("_", " ") ?? "Build complete"}</strong></div>
      <p>{event.reason}</p>
      {event.robot_ids.length > 0 && <div className="assigned-team">{event.robot_ids.map((robot) => <span key={robot}>{robot.replace("robot_", "R")}</span>)}</div>}
      <div className="candidate-line"><span>Ready alternatives</span><strong>{event.candidates.length}</strong></div>
    </div>
  );
}
