import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import {
  BarChart3,
  Blocks,
  Bot,
  BrainCircuit,
  Building2,
  Check,
  ChevronRight,
  CircleAlert,
  FileDown,
  Gauge,
  ImageUp,
  Layers3,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Route,
  Sparkles
} from "lucide-react";
import { api } from "./api";
import { ConstructionScene } from "./components/ConstructionScene";
import type { BrainEvent, BuildModule, Project, Trace, TraceFrame } from "./types";

type View = "design" | "modules" | "plan" | "simulate" | "results";
const views: Array<{ id: View; label: string; icon: typeof Building2 }> = [
  { id: "design", label: "Design", icon: Building2 },
  { id: "modules", label: "Modules", icon: Blocks },
  { id: "plan", label: "Plan", icon: BrainCircuit },
  { id: "simulate", label: "Simulate", icon: Bot },
  { id: "results", label: "Results", icon: BarChart3 }
];

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [view, setView] = useState<View>("simulate");
  const [controller, setController] = useState("optimized");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.project().then(setProject).catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    api.trace(controller).then(setTrace).catch((reason) => setError(String(reason)));
  }, [controller]);

  if (error) return <StartupState error={error} />;
  if (!project || !trace) return <StartupState />;

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand-mark"><Layers3 size={21} strokeWidth={2.2} /></div>
        <nav aria-label="Workbench views">
          {views.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "rail-button active" : "rail-button"}
              onClick={() => setView(item.id)}
              title={item.label}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-status" title="Local deterministic core ready">
          <span className="status-dot" />
          Local
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Embodied Skill Composer</p>
            <h1>{project.design.title}</h1>
          </div>
          <div className="topbar-meta">
            <span>{project.plan.modules.length} modules</span>
            <span>{project.plan.robots.length} robots</span>
            <span className="solver-ready"><Check size={14} /> CP-SAT ready</span>
          </div>
        </header>
        <motion.section
          key={view}
          className="view-stage"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.24 }}
        >
          {view === "design" && (
            <DesignView
              project={project}
              onProject={(updated) => {
                setProject(updated);
                api.trace(controller).then(setTrace);
              }}
            />
          )}
          {view === "modules" && <ModulesView project={project} />}
          {view === "plan" && <PlanView project={project} trace={trace} controller={controller} setController={setController} />}
          {view === "simulate" && <SimulateView project={project} trace={trace} controller={controller} setController={setController} onTrace={setTrace} />}
          {view === "results" && <ResultsView project={project} />}
        </motion.section>
      </main>
    </div>
  );
}

function StartupState({ error }: { error?: string }) {
  return (
    <div className="startup-state">
      {error ? <CircleAlert size={32} /> : <LoaderCircle className="spin" size={32} />}
      <h1>{error ? "Workbench API unavailable" : "Compiling construction intelligence"}</h1>
      <p>{error ?? "Loading the approved house, schedules, and execution traces."}</p>
      {error && <code>python scripts\run_construction_api.py</code>}
    </div>
  );
}

function DesignView({ project, onProject }: { project: Project; onProject: (project: Project) => void }) {
  const [parsed, setParsed] = useState<Project["design"]["floor_plan"] | null>(null);
  const [width, setWidth] = useState(project.design.footprint_width_m);
  const [depth, setDepth] = useState(project.design.footprint_depth_m);
  const [busy, setBusy] = useState(false);
  const parse = async (file?: File) => {
    if (!file) return;
    setBusy(true);
    try {
      const inferred = await api.parseFloorPlan(file, width);
      const bounds = floorPlanBounds(inferred);
      setParsed(inferred);
      setDepth(Number((bounds.height * width / bounds.width).toFixed(2)));
    } finally { setBusy(false); }
  };
  const floorPlan = parsed ?? project.design.floor_plan;
  const approve = async () => {
    setBusy(true);
    try {
      const scaled = scaleFloorPlan(floorPlan, width, depth);
      scaled.approved = true;
      scaled.warnings = [];
      const updated = await api.rebuild({
        ...project.design,
        design_id: `${project.design.design_id.replace(/_reviewed$/, "")}_reviewed`,
        footprint_width_m: width,
        footprint_depth_m: depth,
        floor_plan: scaled
      });
      setParsed(updated.design.floor_plan);
      onProject(updated);
    } finally { setBusy(false); }
  };
  return (
    <div className="design-layout">
      <section className="design-canvas">
        <div className="section-heading">
          <div><p className="eyebrow">Architectural intent</p><h2>Review the metric floor plan</h2></div>
          <span className={floorPlan.approved ? "approval approved" : "approval review"}>
            {floorPlan.approved ? <Check size={15} /> : <CircleAlert size={15} />}
            {floorPlan.approved ? "Approved" : "Review required"}
          </span>
        </div>
        <FloorPlan plan={floorPlan} />
      </section>
      <aside className="design-inspector">
        <h3>Input</h3>
        <label className="upload-zone">
          <ImageUp size={25} />
          <strong>{busy ? "Working..." : "Upload floor plan"}</strong>
          <span>PNG or JPG, high-contrast orthogonal plan</span>
          <input type="file" accept="image/png,image/jpeg" onChange={(event) => parse(event.target.files?.[0])} />
        </label>
        <div className="dimension-grid">
          <label><span>Exterior width</span><div className="unit-input"><input type="number" min="2" max="40" value={width} onChange={(event) => setWidth(Number(event.target.value))} /><span>m</span></div></label>
          <label><span>Exterior depth</span><div className="unit-input"><input type="number" min="2" max="40" value={depth} onChange={(event) => setDepth(Number(event.target.value))} /><span>m</span></div></label>
        </div>
        <div className="fact-list">
          <Fact label="Footprint" value={`${width} x ${depth} m`} />
          <Fact label="Rooms" value={String(floorPlan.rooms.length)} />
          <Fact label="Openings" value={String(floorPlan.openings.length)} />
          <Fact label="Confidence" value={`${Math.round(floorPlan.confidence * 100)}%`} />
        </div>
        {floorPlan.warnings.map((warning) => <p className="warning-line" key={warning}><CircleAlert size={14} />{warning}</p>)}
        <button className="primary-button approve-button" disabled={busy} onClick={approve}><Check size={16} />{busy ? "Compiling..." : "Approve and compile"}</button>
      </aside>
    </div>
  );
}

function FloorPlan({ plan }: { plan: Project["design"]["floor_plan"] }) {
  const points = plan.walls.flatMap((wall) => [wall.start, wall.end]);
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const scale = 70 / Math.max(maxX - minX, maxY - minY);
  const p = (point: { x: number; y: number }) => ({ x: 50 + point.x * scale, y: 50 - point.y * scale });
  return (
    <svg className="floorplan" viewBox="0 0 100 100" role="img" aria-label="Reviewed floor plan">
      <defs><pattern id="grid" width="5" height="5" patternUnits="userSpaceOnUse"><path d="M 5 0 L 0 0 0 5" fill="none" stroke="#d7ded9" strokeWidth="0.18" /></pattern></defs>
      <rect width="100" height="100" fill="url(#grid)" />
      {plan.rooms.map((room, index) => <polygon key={room.room_id} points={room.polygon.map((point) => { const q = p(point); return `${q.x},${q.y}`; }).join(" ")} fill={["#e5efe9", "#f0e7d7", "#dce9ee"][index % 3]} stroke="#79857f" strokeWidth="0.3" />)}
      {plan.walls.map((wall) => { const a = p(wall.start); const b = p(wall.end); return <line key={wall.wall_id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#17201c" strokeWidth="1.4" strokeLinecap="square" />; })}
      {plan.rooms.map((room) => { const center = room.polygon.reduce((sum, point) => ({ x: sum.x + point.x / room.polygon.length, y: sum.y + point.y / room.polygon.length }), { x: 0, y: 0 }); const q = p(center); return <text key={room.room_id} x={q.x} y={q.y} textAnchor="middle" fontSize="2.4" fill="#48544f">{room.name}</text>; })}
    </svg>
  );
}

function ModulesView({ project }: { project: Project }) {
  const [exploded, setExploded] = useState(true);
  const [selected, setSelected] = useState(project.plan.modules[0].module_id);
  const module = project.plan.modules.find((item) => item.module_id === selected)!;
  return (
    <div className="workspace-layout">
      <section className="scene-panel">
        <div className="scene-toolbar">
          <div><p className="eyebrow">Buildable representation</p><h2>{exploded ? "Exploded module system" : "Assembled cottage"}</h2></div>
          <div className="segmented"><button className={!exploded ? "selected" : ""} onClick={() => setExploded(false)}>Assembled</button><button className={exploded ? "selected" : ""} onClick={() => setExploded(true)}>Exploded</button></div>
        </div>
        <div className="scene-canvas"><ConstructionScene project={project} exploded={exploded} selectedId={selected} onSelect={setSelected} /></div>
      </section>
      <aside className="module-inspector">
        <p className="eyebrow">Selected module</p><h2>{module.module_id.replaceAll("_", " ")}</h2>
        <span className="type-tag">{module.module_type.replaceAll("_", " ")}</span>
        <div className="fact-list roomy">
          <Fact label="Dimensions" value={`${module.dimensions.width.toFixed(2)} × ${module.dimensions.depth.toFixed(2)} × ${module.dimensions.height.toFixed(2)} m`} />
          <Fact label="Mass" value={`${module.mass_kg} kg`} />
          <Fact label="Team" value={`${module.required_team_size} robot${module.required_team_size > 1 ? "s" : ""}`} />
          <Fact label="Material" value={module.material.replaceAll("_", " ")} />
          <Fact label="Dependencies" value={module.dependencies.length ? String(module.dependencies.length) : "Foundation stage"} />
        </div>
        <h3>Prerequisites</h3>
        <div className="dependency-list">{module.dependencies.length ? module.dependencies.map((item) => <span key={item}>{item}</span>) : <span>None</span>}</div>
      </aside>
    </div>
  );
}

function PlanView({ project, trace, controller, setController }: { project: Project; trace: Trace; controller: string; setController: (value: string) => void }) {
  const nodes: Node[] = project.plan.modules.map((module, index) => ({
    id: module.module_id,
    type: "moduleNode",
    position: { x: (index % 6) * 175, y: Math.floor(index / 6) * 120 },
    data: { module, job: trace.schedule.jobs.find((job) => job.module_id === module.module_id) }
  }));
  const edges: Edge[] = project.plan.modules.flatMap((module) => module.dependencies.map((dependency) => ({ id: `${dependency}-${module.module_id}`, source: dependency, target: module.module_id, animated: trace.schedule.critical_path.includes(dependency) && trace.schedule.critical_path.includes(module.module_id), style: { stroke: "#93a19b" } })));
  return (
    <div className="plan-layout">
      <section className="graph-panel">
        <div className="section-heading compact"><div><p className="eyebrow">Installation graph</p><h2>Precedence and critical path</h2></div><ControllerSelect value={controller} onChange={setController} /></div>
        <div className="graph-canvas"><ReactFlow nodes={nodes} edges={edges} nodeTypes={{ moduleNode: ModuleNode }} fitView minZoom={0.25} maxZoom={1.4}><Background gap={20} size={1} color="#d8dfdb" /><Controls showInteractive={false} /></ReactFlow></div>
      </section>
      <section className="gantt-panel"><p className="eyebrow">Robot lanes</p><h2>{trace.metrics.makespan_s}s makespan</h2><Gantt trace={trace} robots={project.plan.robots.map((robot) => robot.robot_id)} /></section>
    </div>
  );
}

function ModuleNode({ data }: NodeProps) {
  const payload = data as unknown as { module: BuildModule; job?: { critical: boolean; robot_ids: string[] } };
  return <div className={payload.job?.critical ? "graph-node critical" : "graph-node"}><Handle type="target" position={Position.Top} /><span>{payload.module.module_type.replace("_panel", "")}</span><strong>{payload.module.module_id}</strong><small>{payload.job?.robot_ids.join(" + ")}</small><Handle type="source" position={Position.Bottom} /></div>;
}

function Gantt({ trace, robots }: { trace: Trace; robots: string[] }) {
  return <div className="gantt">{robots.map((robot) => <div className="gantt-row" key={robot}><span>{robot.replace("robot_", "R")}</span><div className="gantt-track">{trace.schedule.jobs.filter((job) => job.robot_ids.includes(robot)).map((job) => <div key={job.module_id} className={job.critical ? "gantt-job critical" : "gantt-job"} title={`${job.module_id}: ${job.start_s}–${job.end_s}s`} style={{ left: `${(job.start_s / trace.metrics.makespan_s) * 100}%`, width: `${((job.end_s - job.start_s) / trace.metrics.makespan_s) * 100}%` }}><span>{job.module_id.split("_").slice(0, 2).join(" ")}</span></div>)}</div></div>)}</div>;
}

function SimulateView({ project, trace, controller, setController, onTrace }: { project: Project; trace: Trace; controller: string; setController: (value: string) => void; onTrace: (trace: Trace) => void }) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [failure, setFailure] = useState<string | null>(null);
  const [recovering, setRecovering] = useState(false);
  useEffect(() => { setTime(0); }, [controller]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setTime((value) => value >= trace.metrics.makespan_s ? 0 : Math.min(value + 2, trace.metrics.makespan_s)), 160);
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
      setFailure(label);
    } catch (reason) {
      setFailure(`Recovery failed: ${String(reason)}`);
    } finally {
      setRecovering(false);
    }
  };
  return (
    <div className="simulation-layout">
      <section className="simulation-main">
        <div className="scene-toolbar simulation-toolbar"><div><p className="eyebrow">Digital twin</p><h2>{frame.completed_module_ids.length} / {project.plan.modules.length} modules installed</h2></div><ControllerSelect value={controller} onChange={setController} /></div>
        <div className="simulation-canvas"><ConstructionScene project={project} frame={frame} /></div>
        <div className="timeline-controls"><button className="icon-button" onClick={() => setPlaying(!playing)} title={playing ? "Pause replay" : "Play replay"}>{playing ? <Pause size={18} /> : <Play size={18} />}</button><span className="timecode">{formatTime(time)}</span><input aria-label="Construction timeline" type="range" min="0" max={trace.metrics.makespan_s} value={time} onChange={(event) => { setPlaying(false); setTime(Number(event.target.value)); }} /><span>{formatTime(trace.metrics.makespan_s)}</span><button className="icon-button" onClick={() => setTime(0)} title="Reset replay"><RotateCcw size={17} /></button></div>
      </section>
      <aside className="brain-panel">
        <div className="brain-title"><div className="brain-icon"><BrainCircuit size={21} /></div><div><p className="eyebrow">ConstructionBrain</p><h2>Live decision</h2></div></div>
        <Decision event={event} />
        <h3>Robot state</h3>
        <div className="robot-state-list">{frame.robots.map((robot, index) => <div key={robot.robot_id}><span className={`robot-swatch r${index + 1}`} /><strong>{robot.robot_id.replace("robot_", "R")}</strong><span>{robot.status}</span><small>{robot.module_id?.replaceAll("_", " ") ?? "available"}</small></div>)}</div>
        <h3>Disturbance</h3>
        <div className="failure-actions"><button disabled={recovering} onClick={() => inject("obstacle", "Obstacle recovery applied to the execution trace.")}>Obstacle</button><button disabled={recovering} onClick={() => inject("robot_unavailable", "Robot health recovery applied to the execution trace.")}>Robot offline</button><button disabled={recovering} onClick={() => inject("dropped_resource", "Dropped-resource recovery applied to the execution trace.")}>Drop resource</button></div>
        {failure && <p className="recovery-note"><RotateCcw size={15} />{failure}</p>}
      </aside>
    </div>
  );
}

function Decision({ event }: { event: BrainEvent }) {
  return <div className="decision-block"><div className="decision-time">T+{event.timestamp_s}s <span>{event.predicted_remaining_s}s remaining</span></div><div className="decision-module"><ChevronRight size={17} /><strong>{event.module_id?.replaceAll("_", " ") ?? "Build complete"}</strong></div><p>{event.reason}</p>{event.robot_ids.length > 0 && <div className="assigned-team">{event.robot_ids.map((robot) => <span key={robot}>{robot.replace("robot_", "R")}</span>)}</div>}<div className="candidate-line"><span>Candidates</span><strong>{event.candidates.length}</strong></div></div>;
}

function ResultsView({ project }: { project: Project }) {
  const names = ["sequential", "greedy", "optimized"];
  const option = { animationDuration: 700, grid: { left: 42, right: 20, top: 30, bottom: 34 }, xAxis: { type: "category", data: names, axisLine: { lineStyle: { color: "#9ba6a1" } }, axisLabel: { color: "#53605b" } }, yAxis: { type: "value", name: "seconds", nameTextStyle: { color: "#718078" }, axisLabel: { color: "#718078" }, splitLine: { lineStyle: { color: "#e3e8e5" } } }, series: [{ type: "bar", data: names.map((name) => ({ value: project.controllers[name].makespan_s, itemStyle: { color: name === "optimized" ? "#f05b2a" : name === "greedy" ? "#41958d" : "#798781", borderRadius: [3, 3, 0, 0] } })), barWidth: "42%" }] };
  return <div className="results-layout"><section className="results-lead"><p className="eyebrow">Experiment result</p><h2><span>{project.optimized_improvement_percent}%</span> faster than sequential construction</h2><p>The optimized controller overlaps independent foundation and wall jobs while reserving two-robot teams for heavy roof modules.</p><div className="result-kpis"><Kpi icon={Gauge} label="Optimized makespan" value={`${project.controllers.optimized.makespan_s}s`} /><Kpi icon={Route} label="Fleet travel" value={`${project.controllers.optimized.total_travel_m.toFixed(1)}m`} /><Kpi icon={Sparkles} label="Solver" value="CP-SAT" /></div><button className="primary-button" onClick={() => api.report().then(({ markdown }) => downloadText("construction-report.md", markdown))}><FileDown size={17} /> Export research report</button></section><section className="chart-section"><div><p className="eyebrow">Controller benchmark</p><h2>Makespan comparison</h2></div><ReactECharts option={option} style={{ height: 340 }} /><div className="controller-table">{names.map((name) => <div key={name}><strong>{name}</strong><span>{project.controllers[name].makespan_s}s</span><small>{project.controllers[name].idle_robot_seconds}s idle</small></div>)}</div></section></div>;
}

function ControllerSelect({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return <div className="segmented controller-select">{["sequential", "greedy", "optimized"].map((name) => <button key={name} className={value === name ? "selected" : ""} onClick={() => onChange(name)}>{name}</button>)}</div>;
}

function Fact({ label, value }: { label: string; value: string }) { return <div className="fact"><span>{label}</span><strong>{value}</strong></div>; }
function Kpi({ icon: Icon, label, value }: { icon: typeof Gauge; label: string; value: string }) { return <div className="kpi"><Icon size={19} /><span>{label}</span><strong>{value}</strong></div>; }
function nearestFrame(frames: TraceFrame[], time: number) { return frames.reduce((best, frame) => Math.abs(frame.timestamp_s - time) < Math.abs(best.timestamp_s - time) ? frame : best, frames[0]); }
function formatTime(seconds: number) { return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`; }
function downloadText(name: string, text: string) { const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" })); const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click(); URL.revokeObjectURL(url); }

function floorPlanBounds(plan: Project["design"]["floor_plan"]) {
  const points = plan.walls.flatMap((wall) => [wall.start, wall.end]);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys)
  };
}

function scaleFloorPlan(
  plan: Project["design"]["floor_plan"],
  width: number,
  depth: number
) {
  const bounds = floorPlanBounds(plan);
  const sx = width / bounds.width;
  const sy = depth / bounds.height;
  return {
    ...plan,
    walls: plan.walls.map((wall) => ({
      ...wall,
      start: { x: wall.start.x * sx, y: wall.start.y * sy },
      end: { x: wall.end.x * sx, y: wall.end.y * sy }
    })),
    openings: plan.openings.map((opening) => ({
      ...opening,
      offset_m: opening.offset_m * (["north", "south"].includes(opening.wall_id) ? sx : sy)
    })),
    rooms: plan.rooms.map((room) => ({
      ...room,
      polygon: room.polygon.map((point) => ({ x: point.x * sx, y: point.y * sy }))
    }))
  };
}
