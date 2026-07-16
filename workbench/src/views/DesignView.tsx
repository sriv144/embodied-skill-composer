import { Check, CircleAlert, ImageUp, RefreshCw } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { Fact } from "../components/WorkbenchControls";
import type { LabMode, Project } from "../types";

export function DesignView({
  project,
  mode,
  onProject
}: {
  project: Project;
  mode: LabMode;
  onProject: (project: Project) => void;
}) {
  const [parsed, setParsed] = useState<Project["design"]["floor_plan"] | null>(null);
  const [width, setWidth] = useState(project.design.footprint_width_m);
  const [depth, setDepth] = useState(project.design.footprint_depth_m);
  const [seed, setSeed] = useState(900);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const floorPlan = parsed ?? project.design.floor_plan;

  const parse = async (file?: File) => {
    if (!file) return;
    setBusy(true);
    setNotice(null);
    try {
      const inferred = await api.parseFloorPlan(file, width);
      const bounds = floorPlanBounds(inferred);
      setParsed(inferred);
      setDepth(Number(((bounds.height * width) / bounds.width).toFixed(2)));
    } catch (reason) {
      setNotice(String(reason));
    } finally {
      setBusy(false);
    }
  };

  const approve = async () => {
    setBusy(true);
    setNotice(null);
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
      setNotice("Reviewed design compiled into a new build plan.");
    } catch (reason) {
      setNotice(String(reason));
    } finally {
      setBusy(false);
    }
  };

  const generate = async () => {
    setBusy(true);
    try {
      const scenario = await api.generateScenario(seed);
      setNotice(`Scenario ${String(scenario.scenario_id)} persisted.`);
    } catch (reason) {
      setNotice(String(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="design-layout">
      <section className="design-canvas">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Architectural intent</p>
            <h2>Metric floor plan</h2>
          </div>
          <span className={floorPlan.approved ? "approval approved" : "approval review"}>
            {floorPlan.approved ? <Check size={15} /> : <CircleAlert size={15} />}
            {floorPlan.approved ? "Approved" : "Review required"}
          </span>
        </div>
        <FloorPlan plan={floorPlan} />
      </section>
      <aside className="design-inspector">
        <p className="eyebrow">Design source</p>
        <label className={mode === "static" ? "upload-zone disabled" : "upload-zone"}>
          <ImageUp size={24} />
          <strong>{busy ? "Processing" : "Floor plan image"}</strong>
          <span>PNG or JPG</span>
          <input
            disabled={mode === "static"}
            type="file"
            accept="image/png,image/jpeg"
            onChange={(event) => parse(event.target.files?.[0])}
          />
        </label>
        <div className="dimension-grid">
          <label>
            <span>Width</span>
            <div className="unit-input">
              <input type="number" min="2" max="40" value={width} onChange={(event) => setWidth(Number(event.target.value))} />
              <span>m</span>
            </div>
          </label>
          <label>
            <span>Depth</span>
            <div className="unit-input">
              <input type="number" min="2" max="40" value={depth} onChange={(event) => setDepth(Number(event.target.value))} />
              <span>m</span>
            </div>
          </label>
        </div>
        <div className="fact-list">
          <Fact label="Footprint" value={`${width} x ${depth} m`} />
          <Fact label="Rooms" value={String(floorPlan.rooms.length)} />
          <Fact label="Openings" value={String(floorPlan.openings.length)} />
          <Fact label="Confidence" value={`${Math.round(floorPlan.confidence * 100)}%`} />
        </div>
        <button className="primary-button approve-button" disabled={busy || mode === "static"} onClick={approve}>
          <Check size={16} /> Approve and compile
        </button>
        <div className="scenario-generator">
          <h3>Procedural scenario</h3>
          <div className="inline-field">
            <input type="number" min="0" max="999" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            <button className="icon-button light" disabled={busy || mode === "static"} onClick={generate} title="Generate seeded cottage">
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
        {notice && <p className="notice-line">{notice}</p>}
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
  const position = (point: { x: number; y: number }) => ({
    x: 50 + point.x * scale,
    y: 50 - point.y * scale
  });
  return (
    <svg className="floorplan" viewBox="0 0 100 100" role="img" aria-label="Reviewed floor plan">
      <defs>
        <pattern id="grid" width="5" height="5" patternUnits="userSpaceOnUse">
          <path d="M 5 0 L 0 0 0 5" fill="none" stroke="#d7ded9" strokeWidth="0.18" />
        </pattern>
      </defs>
      <rect width="100" height="100" fill="url(#grid)" />
      {plan.rooms.map((room, index) => (
        <polygon
          key={room.room_id}
          points={room.polygon.map((point) => { const item = position(point); return `${item.x},${item.y}`; }).join(" ")}
          fill={["#e5efe9", "#f0e7d7", "#dce9ee"][index % 3]}
          stroke="#79857f"
          strokeWidth="0.3"
        />
      ))}
      {plan.walls.map((wall) => {
        const start = position(wall.start);
        const end = position(wall.end);
        return <line key={wall.wall_id} x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke="#17201c" strokeWidth="1.4" />;
      })}
      {plan.rooms.map((room) => {
        const center = room.polygon.reduce(
          (sum, point) => ({ x: sum.x + point.x / room.polygon.length, y: sum.y + point.y / room.polygon.length }),
          { x: 0, y: 0 }
        );
        const item = position(center);
        return <text key={room.room_id} x={item.x} y={item.y} textAnchor="middle" fontSize="2.4" fill="#48544f">{room.name}</text>;
      })}
    </svg>
  );
}

function floorPlanBounds(plan: Project["design"]["floor_plan"]) {
  const points = plan.walls.flatMap((wall) => [wall.start, wall.end]);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return { width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
}

function scaleFloorPlan(plan: Project["design"]["floor_plan"], width: number, depth: number) {
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
