/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- The independently scrollable inspector must be keyboard reachable. */
import {
  Check,
  CircleAlert,
  Hammer,
  ImageUp,
  RefreshCw,
  Trash2
} from "lucide-react";
import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { api } from "../api";
import { DesignEditor } from "../components/DesignEditor";
import { Fact } from "../components/WorkbenchControls";
import {
  cloneFloorPlan,
  moveOpening,
  moveWall,
  projectValidation,
  removeSelection,
  resetApproval,
  resizeOpening,
  resizeWall,
  updateOpeningDimensions,
  updateWallDimensions,
  wallAxis,
  wallLength,
  type FloorPlan,
  type Selection
} from "../editor/designEditor";
import type { LabMode, Project } from "../types";
import "./design-editor.css";

export function DesignView({
  project,
  mode,
  onProject
}: {
  project: Project;
  mode: LabMode;
  onProject: (project: Project) => void;
}) {
  const [draft, setDraft] = useState<FloorPlan>(() =>
    cloneFloorPlan(project.design.floor_plan)
  );
  const [width, setWidth] = useState(project.design.footprint_width_m);
  const [depth, setDepth] = useState(project.design.footprint_depth_m);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [seed, setSeed] = useState(900);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string>(
    mode === "static"
      ? "Read-only public preview. Connect the local lab to edit and compile."
      : "Select a tool or an existing element to begin."
  );
  const validation = useMemo(
    () => projectValidation(draft, width, depth),
    [draft, width, depth]
  );
  const readOnly = mode === "static";

  useEffect(() => {
    setDraft(cloneFloorPlan(project.design.floor_plan));
    setWidth(project.design.footprint_width_m);
    setDepth(project.design.footprint_depth_m);
    setSelection(null);
  }, [
    project.design.design_id,
    project.design.floor_plan,
    project.design.footprint_depth_m,
    project.design.footprint_width_m
  ]);

  const updateDraft = (next: FloorPlan, announcement: string) => {
    if (next !== draft) setDraft(next);
    setNotice(announcement);
  };

  const updateFootprint = (
    axis: "width" | "depth",
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const value = event.target.valueAsNumber;
    if (!Number.isFinite(value)) return;
    if (axis === "width") setWidth(value);
    else setDepth(value);
    setDraft((current) => resetApproval(current));
    setNotice("Footprint changed. Review and approve the design again.");
  };

  const parse = async (file?: File) => {
    if (!file) return;
    setBusy(true);
    setNotice("Reading floor-plan image.");
    try {
      const inferred = await api.parseFloorPlan(file, width);
      const bounds = floorPlanBounds(inferred);
      const inferredDepth =
        bounds.width > 0
          ? Number(((bounds.height * width) / bounds.width).toFixed(2))
          : depth;
      setDraft(resetApproval(inferred));
      setDepth(inferredDepth);
      setSelection(null);
      setNotice(
        "Floor plan parsed. Resolve validation issues, then approve it before compilation."
      );
    } catch (reason) {
      setNotice(`Floor-plan parsing failed: ${errorMessage(reason)}`);
    } finally {
      setBusy(false);
    }
  };

  const approve = () => {
    if (!validation.valid) {
      setNotice(
        `Approval blocked: ${validation.issues[0]?.message ?? "the design is invalid."}`
      );
      return;
    }
    const approved = cloneFloorPlan(draft);
    approved.approved = true;
    approved.warnings = [];
    setDraft(approved);
    setNotice(
      "Design approved. Geometry is unchanged; compile when you are ready."
    );
  };

  const compile = async () => {
    if (!draft.approved || !validation.valid) {
      setNotice("Compile blocked until the valid design is explicitly approved.");
      return;
    }
    setBusy(true);
    setNotice("Compiling the approved design.");
    try {
      const updated = await api.rebuild({
        ...project.design,
        design_id: `${project.design.design_id.replace(/_reviewed$/, "")}_reviewed`,
        footprint_width_m: width,
        footprint_depth_m: depth,
        floor_plan: draft
      });
      setDraft(cloneFloorPlan(updated.design.floor_plan));
      setSelection(null);
      onProject(updated);
      setNotice("Approved design compiled into a new build plan.");
    } catch (reason) {
      setNotice(`Compilation failed: ${errorMessage(reason)}`);
    } finally {
      setBusy(false);
    }
  };

  const generate = async () => {
    setBusy(true);
    setNotice(`Generating scenario from seed ${seed}.`);
    try {
      const scenario = await api.generateScenario(seed);
      setNotice(`Scenario ${String(scenario.scenario_id)} persisted.`);
    } catch (reason) {
      setNotice(`Scenario generation failed: ${errorMessage(reason)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="design-layout design-editor-layout">
      <section className="design-canvas design-editor-canvas">
        <div className="section-heading design-editor-heading">
          <div>
            <p className="eyebrow">Architectural intent</p>
            <h2>Orthogonal floor-plan editor</h2>
            <p className="design-editor-subtitle">
              Draw metric walls and place openings on a {width} × {depth} m
              footprint.
            </p>
          </div>
          <span className={draft.approved ? "approval approved" : "approval review"}>
            {draft.approved ? (
              <Check size={15} aria-hidden="true" />
            ) : (
              <CircleAlert size={15} aria-hidden="true" />
            )}
            {draft.approved ? "Approved" : "Approval required"}
          </span>
        </div>
        <DesignEditor
          plan={draft}
          footprintWidthM={width}
          footprintDepthM={depth}
          readOnly={readOnly}
          selection={selection}
          onSelection={setSelection}
          onPlan={updateDraft}
        />
      </section>

      <aside
        className="design-inspector design-editor-inspector"
        aria-label="Design inspector"
        tabIndex={0}
      >
        {readOnly && (
          <div className="design-readonly-note" role="note">
            <CircleAlert size={16} aria-hidden="true" />
            <div>
              <strong>Read-only preview</strong>
              <span>Editing and compilation require the local research lab.</span>
            </div>
          </div>
        )}

        <SelectionInspector
          plan={draft}
          selection={selection}
          disabled={busy || readOnly}
          onPlan={updateDraft}
          onSelection={setSelection}
        />

        <section className="design-inspector-section" aria-labelledby="source-heading">
          <h3 id="source-heading">Design source</h3>
          <label className={readOnly ? "upload-zone compact disabled" : "upload-zone compact"}>
            <ImageUp size={20} aria-hidden="true" />
            <span>
              <strong>{busy ? "Processing" : "Import plan image"}</strong>
              PNG or JPG
            </span>
            <input
              disabled={busy || readOnly}
              type="file"
              accept="image/png,image/jpeg"
              aria-label="Import floor-plan PNG or JPG"
              onChange={(event) => {
                void parse(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
          <div className="dimension-grid">
            <label>
              <span>Width</span>
              <div className="unit-input">
                <input
                aria-label="Footprint width in metres"
                disabled={busy || readOnly}
                type="number"
                step="any"
                  value={width}
                  onChange={(event) => updateFootprint("width", event)}
                />
                <span>m</span>
              </div>
            </label>
            <label>
              <span>Depth</span>
              <div className="unit-input">
                <input
                aria-label="Footprint depth in metres"
                disabled={busy || readOnly}
                type="number"
                step="any"
                  value={depth}
                  onChange={(event) => updateFootprint("depth", event)}
                />
                <span>m</span>
              </div>
            </label>
          </div>
        </section>

        <section className="design-inspector-section" aria-labelledby="validation-heading">
          <div className="design-validation-heading">
            <h3 id="validation-heading">Design validation</h3>
            <span className={validation.valid ? "valid" : "invalid"}>
              {validation.valid ? "Ready" : `${validation.issues.length} issues`}
            </span>
          </div>
          {validation.valid ? (
            <p className="design-validation-clear">
              <Check size={15} aria-hidden="true" />
              Bounds, intersections, openings, and dimensions pass.
            </p>
          ) : (
            <ul className="design-validation-list" aria-label="Design validation issues">
              {validation.issues.map((issue, index) => (
                <li key={`${issue.code}-${issue.path}-${index}`}>
                  <CircleAlert size={14} aria-hidden="true" />
                  <span>{issue.message}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <div className="fact-list design-editor-facts">
          <Fact label="Walls" value={String(draft.walls.length)} />
          <Fact label="Openings" value={String(draft.openings.length)} />
          <Fact label="Rooms" value={String(draft.rooms.length)} />
          <Fact label="Snap grid" value="0.25 m" />
        </div>

        <div className="design-approval-actions" aria-label="Approval and compilation">
          <button
            className="secondary-button design-approve"
            type="button"
            disabled={busy || readOnly || !validation.valid || draft.approved}
            onClick={approve}
          >
            <Check size={16} aria-hidden="true" />
            {draft.approved ? "Design approved" : "Approve design"}
          </button>
          <button
            className="primary-button design-compile"
            type="button"
            disabled={busy || readOnly || !validation.valid || !draft.approved}
            onClick={() => void compile()}
          >
            <Hammer size={16} aria-hidden="true" />
            {busy ? "Working" : "Compile build plan"}
          </button>
        </div>

        <section className="scenario-generator" aria-labelledby="scenario-heading">
          <h3 id="scenario-heading">Procedural scenario</h3>
          <div className="inline-field">
            <input
              aria-label="Scenario seed"
              disabled={busy || readOnly}
              type="number"
              min="0"
              max="999"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
            <button
              className="icon-button light"
              type="button"
              disabled={busy || readOnly}
              onClick={() => void generate()}
              aria-label="Generate seeded cottage"
              title="Generate seeded cottage"
            >
              <RefreshCw size={16} aria-hidden="true" />
            </button>
          </div>
        </section>
        <p className="notice-line design-live-status" role="status" aria-live="polite">
          {notice}
        </p>
      </aside>
    </div>
  );
}

function SelectionInspector({
  plan,
  selection,
  disabled,
  onPlan,
  onSelection
}: {
  plan: FloorPlan;
  selection: Selection | null;
  disabled: boolean;
  onPlan: (plan: FloorPlan, announcement: string) => void;
  onSelection: (selection: Selection | null) => void;
}) {
  const wall =
    selection?.kind === "wall"
      ? plan.walls.find((candidate) => candidate.wall_id === selection.id)
      : undefined;
  const opening =
    selection?.kind === "opening"
      ? plan.openings.find(
          (candidate) => candidate.opening_id === selection.id
        )
      : undefined;

  if (!wall && !opening) {
    return (
      <section className="design-inspector-section selection-empty" aria-labelledby="selection-heading">
        <h3 id="selection-heading">Selection</h3>
        <p>Select a wall, door, or window to inspect exact dimensions.</p>
      </section>
    );
  }

  const remove = () => {
    if (!selection) return;
    onPlan(
      removeSelection(plan, selection),
      `${selection.kind === "wall" ? "Wall and its openings" : "Opening"} removed. Approval reset.`
    );
    onSelection(null);
  };

  if (wall) {
    const axis = wallAxis(wall);
    const horizontal = axis === "horizontal";
    const setStart = (value: number) =>
      onPlan(
        resizeWall(
          plan,
          wall.wall_id,
          "start",
          horizontal
            ? { x: value, y: wall.start.y }
            : { x: wall.start.x, y: value }
        ),
        "Wall start changed. Approval reset."
      );
    const setEnd = (value: number) =>
      onPlan(
        resizeWall(
          plan,
          wall.wall_id,
          "end",
          horizontal
            ? { x: value, y: wall.end.y }
            : { x: wall.end.x, y: value }
        ),
        "Wall end changed. Approval reset."
      );
    const setPosition = (value: number) => {
      const current = horizontal ? wall.start.y : wall.start.x;
      onPlan(
        moveWall(
          plan,
          wall.wall_id,
          horizontal ? { x: 0, y: value - current } : { x: value - current, y: 0 }
        ),
        "Wall position changed. Approval reset."
      );
    };
    return (
      <section className="design-inspector-section selection-inspector" aria-labelledby="selection-heading">
        <div className="selection-heading">
          <div>
            <p className="eyebrow">Selected wall</p>
            <h3 id="selection-heading">{wall.wall_id}</h3>
          </div>
          <button
            type="button"
            className="selection-delete"
            disabled={disabled}
            onClick={remove}
            aria-label={`Delete wall ${wall.wall_id} and its openings`}
          >
            <Trash2 size={15} aria-hidden="true" />
          </button>
        </div>
        <p className="selection-summary">
          {axis} · {wallLength(wall).toFixed(2)} m
        </p>
        <div className="numeric-inspector-grid">
          <NumericField
            label={horizontal ? "Start X" : "Start Y"}
            value={horizontal ? wall.start.x : wall.start.y}
            disabled={disabled}
            onValue={setStart}
          />
          <NumericField
            label={horizontal ? "End X" : "End Y"}
            value={horizontal ? wall.end.x : wall.end.y}
            disabled={disabled}
            onValue={setEnd}
          />
          <NumericField
            label={horizontal ? "Y position" : "X position"}
            value={horizontal ? wall.start.y : wall.start.x}
            disabled={disabled}
            onValue={setPosition}
          />
          <NumericField
            label="Thickness"
            value={wall.thickness_m}
            min={0.01}
            disabled={disabled}
            onValue={(value) =>
              onPlan(
                updateWallDimensions(plan, wall.wall_id, {
                  thickness_m: value
                }),
                "Wall thickness changed. Approval reset."
              )
            }
          />
          <NumericField
            label="Height"
            value={wall.height_m}
            min={0.01}
            disabled={disabled}
            onValue={(value) =>
              onPlan(
                updateWallDimensions(plan, wall.wall_id, { height_m: value }),
                "Wall height changed. Approval reset."
              )
            }
          />
        </div>
      </section>
    );
  }

  return (
    <section className="design-inspector-section selection-inspector" aria-labelledby="selection-heading">
      <div className="selection-heading">
        <div>
          <p className="eyebrow">Selected {opening!.kind}</p>
          <h3 id="selection-heading">{opening!.opening_id}</h3>
        </div>
        <button
          type="button"
          className="selection-delete"
          disabled={disabled}
          onClick={remove}
          aria-label={`Delete ${opening!.kind} ${opening!.opening_id}`}
        >
          <Trash2 size={15} aria-hidden="true" />
        </button>
      </div>
      <p className="selection-summary">Attached to {opening!.wall_id}</p>
      <div className="numeric-inspector-grid">
        <NumericField
          label="Offset"
          value={opening!.offset_m}
          min={0}
          disabled={disabled}
          onValue={(value) =>
            onPlan(
              moveOpening(plan, opening!.opening_id, value),
              "Opening offset changed. Approval reset."
            )
          }
        />
        <NumericField
          label="Width"
          value={opening!.width_m}
          min={0.01}
          disabled={disabled}
          onValue={(value) =>
            onPlan(
              resizeOpening(plan, opening!.opening_id, value),
              "Opening width changed. Approval reset."
            )
          }
        />
        <NumericField
          label="Height"
          value={opening!.height_m}
          min={0.01}
          disabled={disabled}
          onValue={(value) =>
            onPlan(
              updateOpeningDimensions(plan, opening!.opening_id, {
                height_m: value
              }),
              "Opening height changed. Approval reset."
            )
          }
        />
        <NumericField
          label="Sill"
          value={opening!.sill_height_m}
          min={0}
          disabled={disabled}
          onValue={(value) =>
            onPlan(
              updateOpeningDimensions(plan, opening!.opening_id, {
                sill_height_m: value
              }),
              "Opening sill changed. Approval reset."
            )
          }
        />
      </div>
    </section>
  );
}

function NumericField({
  label,
  value,
  min,
  disabled,
  onValue
}: {
  label: string;
  value: number;
  min?: number;
  disabled: boolean;
  onValue: (value: number) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <div className="unit-input compact">
        <input
          aria-label={`${label} in metres`}
          type="number"
          step="0.25"
          min={min}
          value={value}
          disabled={disabled}
          onChange={(event) => {
            if (Number.isFinite(event.target.valueAsNumber)) {
              onValue(event.target.valueAsNumber);
            }
          }}
        />
        <span>m</span>
      </div>
    </label>
  );
}

function floorPlanBounds(plan: FloorPlan) {
  const points = plan.walls.flatMap((wall) => [wall.start, wall.end]);
  if (points.length === 0) return { width: 0, height: 0 };
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys)
  };
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}
