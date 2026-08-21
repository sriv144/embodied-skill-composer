import {
  DoorOpen,
  Grid3X3,
  Minus,
  MousePointer2,
  PanelsTopLeft,
  Trash2
} from "lucide-react";
import {
  useId,
  useState,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent
} from "react";
import {
  DEFAULT_GRID_M,
  addOpening,
  createWall,
  moveOpening,
  moveWall,
  offsetAlongWall,
  openingGeometry,
  removeSelection,
  resizeOpening,
  resizeWall,
  selectAtPoint,
  selectWallAtPoint,
  snapPoint,
  wallLength,
  type FloorPlan,
  type Point,
  type Selection,
  type WallHandle
} from "../editor/designEditor";

export type EditorTool = "select" | "wall" | "door" | "window" | "delete";

type Props = {
  plan: FloorPlan;
  footprintWidthM: number;
  footprintDepthM: number;
  readOnly: boolean;
  selection: Selection | null;
  onSelection: (selection: Selection | null) => void;
  onPlan: (plan: FloorPlan, announcement: string) => void;
};

type DragState =
  | {
      kind: "move-wall";
      id: string;
      origin: Point;
      plan: FloorPlan;
    }
  | {
      kind: "move-opening";
      id: string;
      plan: FloorPlan;
    }
  | {
      kind: "resize-wall";
      id: string;
      handle: WallHandle;
      plan: FloorPlan;
    }
  | {
      kind: "resize-opening";
      id: string;
      plan: FloorPlan;
    };

const tools: Array<{
  id: EditorTool;
  label: string;
  shortcut: string;
  icon: typeof MousePointer2;
}> = [
  { id: "select", label: "Select and move", shortcut: "V", icon: MousePointer2 },
  { id: "wall", label: "Draw wall", shortcut: "W", icon: Minus },
  { id: "door", label: "Place door", shortcut: "D", icon: DoorOpen },
  { id: "window", label: "Place window", shortcut: "N", icon: PanelsTopLeft },
  { id: "delete", label: "Delete item", shortcut: "Delete", icon: Trash2 }
];

const arrowKeys = ["arrowleft", "arrowright", "arrowup", "arrowdown"] as const;

function isArrowKey(key: string): key is (typeof arrowKeys)[number] {
  return arrowKeys.includes(key as (typeof arrowKeys)[number]);
}

function arrowDelta(key: (typeof arrowKeys)[number], step: number): Point {
  return {
    x: key === "arrowleft" ? -step : key === "arrowright" ? step : 0,
    y: key === "arrowdown" ? -step : key === "arrowup" ? step : 0
  };
}

function selectionAnchor(plan: FloorPlan, selection: Selection | null): Point {
  if (selection?.kind === "wall") {
    const wall = plan.walls.find((candidate) => candidate.wall_id === selection.id);
    if (wall) {
      return snapPoint({
        x: (wall.start.x + wall.end.x) * 0.5,
        y: (wall.start.y + wall.end.y) * 0.5
      });
    }
  }
  if (selection?.kind === "opening") {
    const opening = plan.openings.find(
      (candidate) => candidate.opening_id === selection.id
    );
    const geometry = opening ? openingGeometry(plan, opening) : null;
    if (geometry) return snapPoint(geometry.center);
  }
  return { x: 0, y: 0 };
}

export function DesignEditor({
  plan,
  footprintWidthM,
  footprintDepthM,
  readOnly,
  selection,
  onSelection,
  onPlan
}: Props) {
  const editorId = useId().replace(/:/g, "");
  const patternId = `design-grid-${editorId}`;
  const instructionsId = `design-instructions-${editorId}`;
  const [tool, setTool] = useState<EditorTool>("select");
  const [drawStart, setDrawStart] = useState<Point | null>(null);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);
  const [keyboardPoint, setKeyboardPoint] = useState<Point>(() =>
    selectionAnchor(plan, selection)
  );
  const [drag, setDrag] = useState<DragState | null>(null);
  const margin = Math.max(0.75, Math.min(2, Math.max(footprintWidthM, footprintDepthM) * 0.08));
  const viewWidth = Math.max(2, footprintWidthM + margin * 2);
  const viewHeight = Math.max(2, footprintDepthM + margin * 2);
  const minX = -footprintWidthM * 0.5 - margin;
  const minY = -footprintDepthM * 0.5 - margin;

  const activateTool = (nextTool: EditorTool) => {
    if (readOnly) return;
    setTool(nextTool);
    setDrawStart(null);
    setDrag(null);
    const anchor = selectionAnchor(plan, selection);
    setKeyboardPoint(anchor);
    setHoverPoint(anchor);
  };

  const createOpeningOnWall = (
    wall: FloorPlan["walls"][number],
    kind: "door" | "window",
    point: Point
  ) => {
    const next = addOpening(plan, wall.wall_id, kind, offsetAlongWall(wall, point));
    const created = next.openings.at(-1);
    onPlan(next, `${kind === "door" ? "Door" : "Window"} added. Approval reset.`);
    onSelection(created ? { kind: "opening", id: created.opening_id } : null);
    setTool("select");
    setDrawStart(null);
  };

  const pointFromEvent = (event: ReactPointerEvent<SVGSVGElement | SVGElement>) => {
    const svg =
      event.currentTarget instanceof SVGSVGElement
        ? event.currentTarget
        : event.currentTarget.ownerSVGElement;
    if (!svg) return null;
    const matrix = svg.getScreenCTM();
    if (!matrix) return null;
    const svgPoint = svg.createSVGPoint();
    svgPoint.x = event.clientX;
    svgPoint.y = event.clientY;
    const transformed = svgPoint.matrixTransform(matrix.inverse());
    return { x: transformed.x, y: -transformed.y };
  };

  const capturePointer = (
    event: ReactPointerEvent<SVGSVGElement | SVGElement>
  ) => {
    const svg =
      event.currentTarget instanceof SVGSVGElement
        ? event.currentTarget
        : event.currentTarget.ownerSVGElement;
    svg?.setPointerCapture(event.pointerId);
  };

  const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    event.currentTarget.focus();
    const rawPoint = pointFromEvent(event);
    if (!rawPoint) return;
    const point = snapPoint(rawPoint);
    if (readOnly) {
      onSelection(selectAtPoint(plan, rawPoint, 0.24));
      return;
    }
    if (tool === "wall") {
      if (!drawStart) {
        setDrawStart(point);
        setHoverPoint(point);
        onSelection(null);
        return;
      }
      if (
        Math.hypot(point.x - drawStart.x, point.y - drawStart.y) <
        DEFAULT_GRID_M * 0.5
      ) {
        return;
      }
      const next = createWall(plan, drawStart, point);
      const created = next.walls.at(-1);
      onPlan(next, `Wall ${created?.wall_id ?? ""} added. Approval reset.`);
      onSelection(created ? { kind: "wall", id: created.wall_id } : null);
      setDrawStart(null);
      setTool("select");
      return;
    }
    if (tool === "door" || tool === "window") {
      const wallSelection = selectWallAtPoint(plan, rawPoint, 0.35);
      if (!wallSelection) {
        onPlan(plan, `Choose a wall to place the ${tool}.`);
        return;
      }
      const wall = plan.walls.find(
        (candidate) => candidate.wall_id === wallSelection.id
      );
      if (!wall) return;
      createOpeningOnWall(wall, tool, rawPoint);
      return;
    }

    const hit = selectAtPoint(plan, rawPoint, 0.24);
    if (tool === "delete") {
      if (!hit) return;
      onPlan(
        removeSelection(plan, hit),
        `${hit.kind === "wall" ? "Wall and its openings" : "Opening"} removed. Approval reset.`
      );
      onSelection(null);
      setTool("select");
      return;
    }
    onSelection(hit);
    if (!hit) return;
    capturePointer(event);
    setDrag(
      hit.kind === "wall"
        ? { kind: "move-wall", id: hit.id, origin: point, plan }
        : { kind: "move-opening", id: hit.id, plan }
    );
  };

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const rawPoint = pointFromEvent(event);
    if (!rawPoint) return;
    const point = snapPoint(rawPoint);
    setHoverPoint(point);
    if (!drag || readOnly) return;
    if (drag.kind === "move-wall") {
      onPlan(
        moveWall(drag.plan, drag.id, {
          x: point.x - drag.origin.x,
          y: point.y - drag.origin.y
        }),
        "Wall moved. Approval reset."
      );
      return;
    }
    if (drag.kind === "resize-wall") {
      onPlan(
        resizeWall(drag.plan, drag.id, drag.handle, point),
        "Wall resized. Approval reset."
      );
      return;
    }
    if (drag.kind === "move-opening") {
      const opening = drag.plan.openings.find(
        (candidate) => candidate.opening_id === drag.id
      );
      const wall = opening
        ? drag.plan.walls.find(
            (candidate) => candidate.wall_id === opening.wall_id
          )
        : undefined;
      if (wall) {
        onPlan(
          moveOpening(drag.plan, drag.id, offsetAlongWall(wall, rawPoint)),
          "Opening moved. Approval reset."
        );
      }
      return;
    }
    const opening = drag.plan.openings.find(
      (candidate) => candidate.opening_id === drag.id
    );
    const geometry = opening ? openingGeometry(drag.plan, opening) : null;
    if (geometry) {
      const width = Math.hypot(
        rawPoint.x - geometry.center.x,
        rawPoint.y - geometry.center.y
      ) * 2;
      onPlan(
        resizeOpening(drag.plan, drag.id, width),
        "Opening resized. Approval reset."
      );
    }
  };

  const finishPointer = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDrag(null);
  };

  const startWallResize = (
    event: ReactPointerEvent<SVGCircleElement>,
    wallId: string,
    handle: WallHandle
  ) => {
    if (readOnly) return;
    event.stopPropagation();
    capturePointer(event);
    setDrag({ kind: "resize-wall", id: wallId, handle, plan });
  };

  const startOpeningResize = (
    event: ReactPointerEvent<SVGCircleElement>,
    openingId: string
  ) => {
    if (readOnly) return;
    event.stopPropagation();
    capturePointer(event);
    setDrag({ kind: "resize-opening", id: openingId, plan });
  };

  const activateWallElement = (
    event: KeyboardEvent<SVGLineElement>,
    wall: FloorPlan["walls"][number]
  ) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    onSelection({ kind: "wall", id: wall.wall_id });
    if (readOnly || tool === "select" || tool === "wall") return;
    if (tool === "delete") {
      onPlan(
        removeSelection(plan, { kind: "wall", id: wall.wall_id }),
        "Wall and its openings removed. Approval reset."
      );
      onSelection(null);
      setTool("select");
      return;
    }
    createOpeningOnWall(wall, tool, keyboardPoint);
  };

  const activateOpeningElement = (
    event: KeyboardEvent<SVGLineElement>,
    opening: FloorPlan["openings"][number]
  ) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    const nextSelection: Selection = {
      kind: "opening",
      id: opening.opening_id
    };
    if (!readOnly && tool === "delete") {
      onPlan(
        removeSelection(plan, nextSelection),
        "Opening removed. Approval reset."
      );
      onSelection(null);
      setTool("select");
      return;
    }
    onSelection(nextSelection);
  };

  const resizeWallWithKeyboard = (
    event: KeyboardEvent<SVGCircleElement>,
    wall: FloorPlan["walls"][number],
    handle: WallHandle
  ) => {
    const key = event.key.toLowerCase();
    if (!isArrowKey(key) || readOnly) return;
    event.preventDefault();
    event.stopPropagation();
    const step = event.shiftKey ? 1 : DEFAULT_GRID_M;
    const delta = arrowDelta(key, step);
    const point = {
      x: wall[handle].x + delta.x,
      y: wall[handle].y + delta.y
    };
    onPlan(
      resizeWall(plan, wall.wall_id, handle, point),
      `Wall ${handle} resized by ${step.toFixed(2)} metres. Approval reset.`
    );
  };

  const resizeOpeningWithKeyboard = (
    event: KeyboardEvent<SVGCircleElement>,
    opening: FloorPlan["openings"][number],
    edge: "start" | "end"
  ) => {
    const key = event.key.toLowerCase();
    if (!isArrowKey(key) || readOnly) return;
    const geometry = openingGeometry(plan, opening);
    if (!geometry) return;
    const length = wallLength(geometry.wall);
    if (length <= 0) return;
    event.preventDefault();
    event.stopPropagation();
    const step = event.shiftKey ? 1 : DEFAULT_GRID_M;
    const delta = arrowDelta(key, step);
    const unit = {
      x: (geometry.wall.end.x - geometry.wall.start.x) / length,
      y: (geometry.wall.end.y - geometry.wall.start.y) / length
    };
    const projectedDelta = delta.x * unit.x + delta.y * unit.y;
    if (Math.abs(projectedDelta) < 1e-9) return;
    const widthDelta = (edge === "start" ? -1 : 1) * projectedDelta * 2;
    onPlan(
      resizeOpening(plan, opening.opening_id, opening.width_m + widthDelta),
      `${opening.kind === "door" ? "Door" : "Window"} width resized. Approval reset.`
    );
  };

  const handleKeyboard = (event: KeyboardEvent<SVGSVGElement>) => {
    const key = event.key.toLowerCase();
    if (key === "escape") {
      setDrawStart(null);
      setDrag(null);
      onSelection(null);
      return;
    }
    if (!readOnly && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const shortcut = { v: "select", w: "wall", d: "door", n: "window" }[
        key
      ] as EditorTool | undefined;
      if (shortcut) {
        event.preventDefault();
        activateTool(shortcut);
        return;
      }
    }
    if (readOnly) return;
    if (
      (event.key === "Enter" || event.key === " ") &&
      event.target === event.currentTarget
    ) {
      event.preventDefault();
      if (tool === "wall") {
        if (!drawStart) {
          setDrawStart(keyboardPoint);
          setHoverPoint(keyboardPoint);
          onSelection(null);
          onPlan(
            plan,
            `Wall start set at ${keyboardPoint.x.toFixed(2)}, ${keyboardPoint.y.toFixed(2)} metres. Use arrow keys, then press Enter to finish.`
          );
          return;
        }
        if (
          Math.hypot(
            keyboardPoint.x - drawStart.x,
            keyboardPoint.y - drawStart.y
          ) <
          DEFAULT_GRID_M * 0.5
        ) {
          onPlan(plan, "Move the keyboard cursor before finishing the wall.");
          return;
        }
        const next = createWall(plan, drawStart, keyboardPoint);
        const created = next.walls.at(-1);
        onPlan(next, `Wall ${created?.wall_id ?? ""} added. Approval reset.`);
        onSelection(created ? { kind: "wall", id: created.wall_id } : null);
        setDrawStart(null);
        setTool("select");
        return;
      }
      if (tool === "door" || tool === "window") {
        const selectedWall =
          selection?.kind === "wall"
            ? plan.walls.find((wall) => wall.wall_id === selection.id)
            : undefined;
        const hit = selectedWall
          ? null
          : selectWallAtPoint(plan, keyboardPoint, 0.35);
        const nearestWall =
          selectedWall ??
          (hit
            ? plan.walls.find((wall) => wall.wall_id === hit.id)
            : undefined);
        if (!nearestWall) {
          onPlan(
            plan,
            `Select a wall before placing the ${tool} with the keyboard.`
          );
          return;
        }
        createOpeningOnWall(nearestWall, tool, keyboardPoint);
        return;
      }
    }
    if (isArrowKey(key) && ["wall", "door", "window"].includes(tool)) {
      event.preventDefault();
      const step = event.shiftKey ? 1 : DEFAULT_GRID_M;
      const delta = arrowDelta(key, step);
      const point = snapPoint({
        x: keyboardPoint.x + delta.x,
        y: keyboardPoint.y + delta.y
      });
      setKeyboardPoint(point);
      setHoverPoint(point);
      return;
    }
    if (!selection) return;
    if (key === "delete" || key === "backspace") {
      event.preventDefault();
      onPlan(
        removeSelection(plan, selection),
        `${selection.kind === "wall" ? "Wall and its openings" : "Opening"} removed. Approval reset.`
      );
      onSelection(null);
      return;
    }
    if (!isArrowKey(key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 1 : DEFAULT_GRID_M;
    if (selection.kind === "wall") {
      const delta = arrowDelta(key, step);
      onPlan(
        moveWall(plan, selection.id, delta),
        `Wall moved ${step.toFixed(2)} metres. Approval reset.`
      );
    } else {
      const opening = plan.openings.find(
        (candidate) => candidate.opening_id === selection.id
      );
      if (!opening) return;
      const direction =
        key === "arrowleft" || key === "arrowdown" ? -1 : 1;
      onPlan(
        moveOpening(plan, selection.id, opening.offset_m + direction * step),
        `Opening moved ${step.toFixed(2)} metres. Approval reset.`
      );
    }
  };

  const selectedWall =
    selection?.kind === "wall"
      ? plan.walls.find((wall) => wall.wall_id === selection.id)
      : undefined;
  const selectedOpening =
    selection?.kind === "opening"
      ? plan.openings.find(
          (opening) => opening.opening_id === selection.id
        )
      : undefined;
  const selectedOpeningGeometry = selectedOpening
    ? openingGeometry(plan, selectedOpening)
    : null;
  const previewEnd =
    drawStart && hoverPoint
      ? Math.abs(hoverPoint.x - drawStart.x) >=
        Math.abs(hoverPoint.y - drawStart.y)
        ? { x: hoverPoint.x, y: drawStart.y }
        : { x: drawStart.x, y: hoverPoint.y }
      : null;

  return (
    <div className="design-editor">
      <div className="design-editor-toolbar" role="toolbar" aria-label="Floor-plan tools">
        {tools.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={tool === item.id ? "active" : ""}
              aria-label={`${item.label} (${item.shortcut})`}
              aria-pressed={tool === item.id}
              disabled={readOnly}
              onClick={() => activateTool(item.id)}
            >
              <Icon size={16} aria-hidden="true" />
              <span>{item.label.split(" ")[0]}</span>
              <kbd>{item.shortcut}</kbd>
            </button>
          );
        })}
        <span className="design-editor-grid">
          <Grid3X3 size={14} aria-hidden="true" />
          {DEFAULT_GRID_M} m grid
        </span>
      </div>
      <div className="design-editor-surface">
        <p className="sr-only" id={instructionsId}>
          Choose a tool, then focus the floor plan. For keyboard drawing, press
          W, move the drafting cursor with the arrow keys, press Enter for the
          wall start, move again, and press Enter to finish. Focus a wall and
          press D or N followed by Enter to add an opening. Enter or Space
          activates walls and openings. Arrow keys move selections or operate
          focused resize handles by 0.25 metres; hold Shift for one metre.
        </p>
        <svg
          className={`design-editor-svg tool-${tool}`}
          viewBox={`${minX} ${minY} ${viewWidth} ${viewHeight}`}
          role="application"
          aria-label={`${readOnly ? "Read-only" : "Editable"} floor plan with ${plan.walls.length} walls and ${plan.openings.length} openings`}
          aria-describedby={instructionsId}
          aria-keyshortcuts="V W D N Delete Escape ArrowLeft ArrowRight ArrowUp ArrowDown"
          tabIndex={0}
          onKeyDown={handleKeyboard}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishPointer}
          onPointerCancel={finishPointer}
        >
          <title>
            {readOnly
              ? "Read-only orthogonal floor plan"
              : "Orthogonal floor-plan editor. Use the toolbar or keyboard shortcuts to edit."}
          </title>
          <defs>
            <pattern
              id={patternId}
              width={DEFAULT_GRID_M}
              height={DEFAULT_GRID_M}
              patternUnits="userSpaceOnUse"
            >
              <path
                d={`M ${DEFAULT_GRID_M} 0 L 0 0 0 ${DEFAULT_GRID_M}`}
                className="design-grid-line"
                vectorEffect="non-scaling-stroke"
              />
            </pattern>
          </defs>
          <rect
            x={minX}
            y={minY}
            width={viewWidth}
            height={viewHeight}
            className="design-editor-paper"
          />
          <rect
            x={-footprintWidthM * 0.5}
            y={-footprintDepthM * 0.5}
            width={footprintWidthM}
            height={footprintDepthM}
            fill={`url(#${patternId})`}
            className="design-footprint"
            vectorEffect="non-scaling-stroke"
          />
          <g className="design-rooms" aria-hidden="true">
            {plan.rooms.map((room, index) => (
              <polygon
                key={room.room_id}
                points={room.polygon
                  .map((point) => `${point.x},${-point.y}`)
                  .join(" ")}
                className={`design-room tone-${index % 3}`}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
          <g className="design-walls">
            {plan.walls.map((wall) => {
              const active =
                selection?.kind === "wall" && selection.id === wall.wall_id;
              return (
                <line
                  key={wall.wall_id}
                  x1={wall.start.x}
                  y1={-wall.start.y}
                  x2={wall.end.x}
                  y2={-wall.end.y}
                  className={active ? "design-wall selected" : "design-wall"}
                  vectorEffect="non-scaling-stroke"
                  role="button"
                  tabIndex={0}
                  aria-label={`Wall ${wall.wall_id}, ${wallLength(wall).toFixed(2)} metres`}
                  aria-pressed={active}
                  onFocus={() =>
                    onSelection({ kind: "wall", id: wall.wall_id })
                  }
                  onKeyDown={(event) => activateWallElement(event, wall)}
                >
                  <title>{`Wall ${wall.wall_id}, ${wallLength(wall).toFixed(2)} metres`}</title>
                </line>
              );
            })}
          </g>
          <g className="design-openings">
            {plan.openings.map((opening) => {
              const geometry = openingGeometry(plan, opening);
              if (!geometry) return null;
              const active =
                selection?.kind === "opening" &&
                selection.id === opening.opening_id;
              return (
                <g key={opening.opening_id}>
                  <line
                    x1={geometry.start.x}
                    y1={-geometry.start.y}
                    x2={geometry.end.x}
                    y2={-geometry.end.y}
                    className="design-opening-cut"
                    vectorEffect="non-scaling-stroke"
                  />
                  <line
                    x1={geometry.start.x}
                    y1={-geometry.start.y}
                    x2={geometry.end.x}
                    y2={-geometry.end.y}
                    className={`design-opening ${opening.kind}${active ? " selected" : ""}`}
                    vectorEffect="non-scaling-stroke"
                    role="button"
                    tabIndex={0}
                    aria-label={`${opening.kind} ${opening.opening_id}, ${opening.width_m.toFixed(2)} metres wide`}
                    aria-pressed={active}
                    onFocus={() =>
                      onSelection({
                        kind: "opening",
                        id: opening.opening_id
                      })
                    }
                    onKeyDown={(event) =>
                      activateOpeningElement(event, opening)
                    }
                  >
                    <title>{`${opening.kind} ${opening.opening_id}, ${opening.width_m.toFixed(2)} metres wide`}</title>
                  </line>
                </g>
              );
            })}
          </g>
          {drawStart && previewEnd && (
            <line
              x1={drawStart.x}
              y1={-drawStart.y}
              x2={previewEnd.x}
              y2={-previewEnd.y}
              className="design-wall-preview"
              vectorEffect="non-scaling-stroke"
              aria-hidden="true"
            />
          )}
          {["wall", "door", "window"].includes(tool) && (
            <g
              className="design-keyboard-cursor"
              transform={`translate(${keyboardPoint.x} ${-keyboardPoint.y})`}
              aria-hidden="true"
            >
              <circle r={0.11} vectorEffect="non-scaling-stroke" />
              <path
                d="M -0.24 0 H 0.24 M 0 -0.24 V 0.24"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}
          {selectedWall && (
            <g className="design-handles">
              {(["start", "end"] as const).map((handle) => (
                <circle
                  key={handle}
                  cx={selectedWall[handle].x}
                  cy={-selectedWall[handle].y}
                  r={0.13}
                  className="design-handle"
                  vectorEffect="non-scaling-stroke"
                  role="button"
                  tabIndex={readOnly ? -1 : 0}
                  aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown"
                  aria-label={`Resize ${handle} endpoint of wall ${selectedWall.wall_id}. Use arrow keys; hold Shift for one metre.`}
                  onPointerDown={(event) =>
                    startWallResize(event, selectedWall.wall_id, handle)
                  }
                  onKeyDown={(event) =>
                    resizeWallWithKeyboard(event, selectedWall, handle)
                  }
                />
              ))}
            </g>
          )}
          {selectedOpening && selectedOpeningGeometry && (
            <g className="design-handles">
              {(
                [
                  ["start", selectedOpeningGeometry.start],
                  ["end", selectedOpeningGeometry.end]
                ] as const
              ).map(
                ([edge, point]) => (
                  <circle
                    key={edge}
                    cx={point.x}
                    cy={-point.y}
                    r={0.12}
                    className="design-handle opening"
                    vectorEffect="non-scaling-stroke"
                    role="button"
                    tabIndex={readOnly ? -1 : 0}
                    aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown"
                    aria-label={`Resize ${edge} edge of ${selectedOpening.kind} ${selectedOpening.opening_id}. Use arrow keys along its wall; hold Shift for one metre.`}
                    onPointerDown={(event) =>
                      startOpeningResize(event, selectedOpening.opening_id)
                    }
                    onKeyDown={(event) =>
                      resizeOpeningWithKeyboard(
                        event,
                        selectedOpening,
                        edge
                      )
                    }
                  />
                )
              )}
            </g>
          )}
        </svg>
        <p className="design-editor-hint">
          {readOnly
            ? "Public preview · editing and compilation are available in local mode."
            : drawStart
              ? "Choose the wall endpoint. The dominant direction locks its axis."
              : "Arrows move the selection by 0.25 m · hold Shift for 1 m · Delete removes."}
        </p>
      </div>
    </div>
  );
}
