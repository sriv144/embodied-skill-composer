import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { BrainCircuit, Boxes, GitBranch, TimerReset } from "lucide-react";
import { useMemo, useState } from "react";
import { ConstructionScene } from "../components/ConstructionScene";
import { ControllerSelect, Fact } from "../components/WorkbenchControls";
import type { BuildModule, LabPolicy, Project, Trace } from "../types";

type BrainTab = "modules" | "graph" | "schedule";

export function BrainView({
  project,
  trace,
  controller,
  policies,
  onController
}: {
  project: Project;
  trace: Trace;
  controller: string;
  policies: LabPolicy[];
  onController: (controller: string) => void;
}) {
  const [tab, setTab] = useState<BrainTab>("graph");
  const [selectedId, setSelectedId] = useState(project.plan.modules[0].module_id);
  const module = project.plan.modules.find((item) => item.module_id === selectedId) ?? project.plan.modules[0];
  return (
    <div className="brain-layout">
      <section className="brain-workspace">
        <header className="operational-toolbar">
          <div>
            <p className="eyebrow">Construction brain</p>
            <h2>Task graph and fleet allocation</h2>
          </div>
          <div className="toolbar-actions">
            <div className="icon-tabs" aria-label="Brain workspace">
              <button className={tab === "modules" ? "active" : ""} onClick={() => setTab("modules")} title="Build modules"><Boxes size={17} /></button>
              <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")} title="Dependency graph"><GitBranch size={17} /></button>
              <button className={tab === "schedule" ? "active" : ""} onClick={() => setTab("schedule")} title="Robot schedule"><TimerReset size={17} /></button>
            </div>
            <ControllerSelect value={controller} onChange={onController} />
          </div>
        </header>
        <div className="brain-canvas">
          {tab === "modules" && (
            <ConstructionScene project={project} exploded selectedId={selectedId} onSelect={setSelectedId} />
          )}
          {tab === "graph" && <DependencyGraph project={project} trace={trace} onSelect={setSelectedId} />}
          {tab === "schedule" && <Schedule trace={trace} robots={project.plan.robots.map((item) => item.robot_id)} />}
        </div>
      </section>
      <aside className="brain-inspector-light">
        <div className="brain-title light">
          <div className="brain-icon"><BrainCircuit size={20} /></div>
          <div><p className="eyebrow">Selected module</p><h2>{module.module_id.replaceAll("_", " ")}</h2></div>
        </div>
        <div className="fact-list roomy">
          <Fact label="Type" value={module.module_type.replaceAll("_", " ")} />
          <Fact label="Mass" value={`${module.mass_kg} kg`} />
          <Fact label="Install" value={`${module.install_duration_s}s`} />
          <Fact label="Team" value={`${module.required_team_size} robot${module.required_team_size > 1 ? "s" : ""}`} />
          <Fact label="Dependencies" value={String(module.dependencies.length)} />
        </div>
        <h3>Controller bench</h3>
        <div className="policy-roster">
          <PolicyRow name="Sequential" kind="baseline" status="ready" />
          <PolicyRow name="Greedy" kind="heuristic" status="ready" />
          <PolicyRow name="Auction" kind="decentralized" status="event sim" />
          <PolicyRow name="CP-SAT" kind="optimizer" status="ready" />
          <PolicyRow name="IPPO" kind="learned" status={policies.some((item) => item.controller === "ippo") ? "checkpoint" : "not trained"} />
          <PolicyRow name="MAPPO" kind="learned" status={policies.some((item) => item.controller === "mappo") ? "checkpoint" : "not trained"} />
        </div>
        <h3>Ready candidates</h3>
        <div className="candidate-heatmap">
          {(trace.brain_events.find((item) => item.candidates.length)?.candidates ?? []).slice(0, 6).map((candidate, index) => (
            <button key={candidate} onClick={() => setSelectedId(candidate)}>
              <span>{candidate.replaceAll("_", " ")}</span>
              <i style={{ width: `${Math.max(18, 100 - index * 13)}%` }} />
            </button>
          ))}
        </div>
      </aside>
    </div>
  );
}

function DependencyGraph({ project, trace, onSelect }: { project: Project; trace: Trace; onSelect: (id: string) => void }) {
  const roofModules = useMemo(() => project.plan.modules.filter((module) => module.module_type === "roof_panel"), [project.plan.modules]);
  const regularModules = useMemo(() => project.plan.modules.filter((module) => module.module_type !== "roof_panel"), [project.plan.modules]);
  const nodes = useMemo<Node[]>(() => [
    ...regularModules.map((module, index) => ({
      id: module.module_id,
      type: "moduleNode",
      position: { x: (index % 6) * 178, y: Math.floor(index / 6) * 122 },
      data: { module, job: trace.schedule.jobs.find((job) => job.module_id === module.module_id) }
    })),
    {
      id: "envelope_complete",
      type: "milestoneNode",
      position: { x: 438, y: 500 },
      data: { label: "Envelope complete", detail: `${regularModules.length} prerequisites satisfied` }
    },
    ...roofModules.map((module, index) => ({
      id: module.module_id,
      type: "moduleNode",
      position: { x: 168 + index * 220, y: 640 },
      data: { module, job: trace.schedule.jobs.find((job) => job.module_id === module.module_id) }
    }))
  ], [regularModules, roofModules, trace.schedule.jobs]);
  const edges = useMemo<Edge[]>(() => {
    const regularEdges = regularModules.flatMap((module) => module.dependencies.map((dependency) => ({
      id: `${dependency}-${module.module_id}`,
      source: dependency,
      target: module.module_id,
      animated: trace.schedule.critical_path.includes(dependency) && trace.schedule.critical_path.includes(module.module_id),
      style: { stroke: "#9ba9a3", opacity: 0.58 }
    })));
    const gateDependencies = [...new Set(roofModules.flatMap((module) => module.dependencies))];
    const gateEdges = gateDependencies.map((dependency) => ({
      id: `${dependency}-envelope_complete`,
      source: dependency,
      target: "envelope_complete",
      style: { stroke: "#a8b4ae", opacity: 0.38 }
    }));
    const roofEdges = roofModules.map((module) => ({
      id: `envelope_complete-${module.module_id}`,
      source: "envelope_complete",
      target: module.module_id,
      animated: true,
      style: { stroke: "#e96a36", opacity: 0.75 }
    }));
    return [...regularEdges, ...gateEdges, ...roofEdges];
  }, [regularModules, roofModules, trace.schedule.critical_path]);
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={{ moduleNode: ModuleNode, milestoneNode: MilestoneNode }}
      onNodeClick={(_event, node) => { if (node.type === "moduleNode") onSelect(node.id); }}
      fitView
      minZoom={0.25}
      maxZoom={1.4}
    >
      <Background gap={20} size={1} color="#d8dfdb" />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

function MilestoneNode({ data }: NodeProps) {
  const payload = data as unknown as { label: string; detail: string };
  return (
    <div className="graph-milestone">
      <Handle type="target" position={Position.Top} />
      <GitBranch size={15} />
      <strong>{payload.label}</strong>
      <small>{payload.detail}</small>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ModuleNode({ data }: NodeProps) {
  const payload = data as unknown as { module: BuildModule; job?: { critical: boolean; robot_ids: string[] } };
  return (
    <div className={payload.job?.critical ? "graph-node critical" : "graph-node"}>
      <Handle type="target" position={Position.Top} />
      <span>{payload.module.module_type.replace("_panel", "")}</span>
      <strong>{payload.module.module_id}</strong>
      <small>{payload.job?.robot_ids.join(" + ")}</small>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function Schedule({ trace, robots }: { trace: Trace; robots: string[] }) {
  return (
    <div className="schedule-workspace">
      <div className="schedule-summary"><span>MAKESPAN</span><strong>{trace.metrics.makespan_s}s</strong><small>{trace.schedule.solver_status}</small></div>
      <div className="gantt">
        {robots.map((robot) => (
          <div className="gantt-row" key={robot}>
            <span>{robot.replace("robot_", "R")}</span>
            <div className="gantt-track">
              {trace.schedule.jobs.filter((job) => job.robot_ids.includes(robot)).map((job) => (
                <div
                  key={job.module_id}
                  className={job.critical ? "gantt-job critical" : "gantt-job"}
                  title={`${job.module_id}: ${job.start_s}-${job.end_s}s`}
                  style={{ left: `${(job.start_s / trace.metrics.makespan_s) * 100}%`, width: `${((job.end_s - job.start_s) / trace.metrics.makespan_s) * 100}%` }}
                ><span>{job.module_id.split("_").slice(0, 2).join(" ")}</span></div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PolicyRow({ name, kind, status }: { name: string; kind: string; status: string }) {
  return <div><strong>{name}</strong><span>{kind}</span><small className={status === "not trained" ? "pending" : ""}>{status}</small></div>;
}
