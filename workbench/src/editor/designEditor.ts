import type { Project } from "../types";

export type FloorPlan = Project["design"]["floor_plan"];
export type Wall = FloorPlan["walls"][number];
export type Opening = FloorPlan["openings"][number];
export type Point = Wall["start"];
export type OpeningKind = Opening["kind"];
export type WallHandle = "start" | "end";
export type Selection =
  | { kind: "wall"; id: string }
  | { kind: "opening"; id: string };

export type ValidationIssue = {
  code: string;
  path: string;
  message: string;
};

export type ValidationProjection = {
  valid: boolean;
  issues: ValidationIssue[];
  plan: FloorPlan;
};

export type OpeningGeometry = {
  center: Point;
  start: Point;
  end: Point;
  wall: Wall;
};

export const DEFAULT_GRID_M = 0.25;
export const EDITOR_EPSILON = 1e-9;

type CreateWallOptions = {
  wallId?: string;
  thicknessM?: number;
  heightM?: number;
  gridM?: number;
};

type AddOpeningOptions = {
  openingId?: string;
  widthM?: number;
  heightM?: number;
  sillHeightM?: number;
  gridM?: number;
};

type WallDimensionPatch = Partial<Pick<Wall, "thickness_m" | "height_m">>;
type OpeningDimensionPatch = Partial<
  Pick<Opening, "width_m" | "height_m" | "sill_height_m">
>;

export function cloneFloorPlan(plan: FloorPlan): FloorPlan {
  return {
    ...plan,
    warnings: [...plan.warnings],
    walls: plan.walls.map((wall) => ({
      ...wall,
      start: { ...wall.start },
      end: { ...wall.end }
    })),
    openings: plan.openings.map((opening) => ({ ...opening })),
    rooms: plan.rooms.map((room) => ({
      ...room,
      polygon: room.polygon.map((point) => ({ ...point }))
    }))
  };
}

export function resetApproval(plan: FloorPlan): FloorPlan {
  return editableClone(plan);
}

export function snapToGrid(value: number, gridM = DEFAULT_GRID_M): number {
  if (!Number.isFinite(value) || !Number.isFinite(gridM) || gridM <= 0) {
    return value;
  }
  const snapped = Math.round(value / gridM) * gridM;
  return Math.abs(snapped) < EDITOR_EPSILON ? 0 : Number(snapped.toFixed(9));
}

export function snapPoint(point: Point, gridM = DEFAULT_GRID_M): Point {
  return {
    x: snapToGrid(point.x, gridM),
    y: snapToGrid(point.y, gridM)
  };
}

export function wallAxis(wall: Wall): "horizontal" | "vertical" | "diagonal" {
  if (Math.abs(wall.start.y - wall.end.y) <= EDITOR_EPSILON) return "horizontal";
  if (Math.abs(wall.start.x - wall.end.x) <= EDITOR_EPSILON) return "vertical";
  return "diagonal";
}

export function wallLength(wall: Wall): number {
  return Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
}

export function createWall(
  plan: FloorPlan,
  rawStart: Point,
  rawEnd: Point,
  options: CreateWallOptions = {}
): FloorPlan {
  const gridM = options.gridM ?? DEFAULT_GRID_M;
  const start = snapPoint(rawStart, gridM);
  const candidateEnd = snapPoint(rawEnd, gridM);
  const horizontal =
    Math.abs(candidateEnd.x - start.x) >= Math.abs(candidateEnd.y - start.y);
  const end = horizontal
    ? { x: candidateEnd.x, y: start.y }
    : { x: start.x, y: candidateEnd.y };
  const next = editableClone(plan);
  next.walls.push({
    wall_id: options.wallId ?? nextIdentifier(plan.walls.map((wall) => wall.wall_id), "wall"),
    start,
    end,
    thickness_m: options.thicknessM ?? 0.2,
    height_m: options.heightM ?? 2.8
  });
  return next;
}

export function moveWall(
  plan: FloorPlan,
  wallId: string,
  rawDelta: Point,
  gridM = DEFAULT_GRID_M
): FloorPlan {
  const next = editableClone(plan);
  const wall = next.walls.find((candidate) => candidate.wall_id === wallId);
  if (!wall) return next;
  const delta = snapPoint(rawDelta, gridM);
  wall.start = {
    x: snapToGrid(wall.start.x + delta.x, gridM),
    y: snapToGrid(wall.start.y + delta.y, gridM)
  };
  wall.end = {
    x: snapToGrid(wall.end.x + delta.x, gridM),
    y: snapToGrid(wall.end.y + delta.y, gridM)
  };
  return next;
}

export function resizeWall(
  plan: FloorPlan,
  wallId: string,
  handle: WallHandle,
  rawPoint: Point,
  gridM = DEFAULT_GRID_M
): FloorPlan {
  const next = editableClone(plan);
  const wall = next.walls.find((candidate) => candidate.wall_id === wallId);
  if (!wall) return next;
  const point = snapPoint(rawPoint, gridM);
  const axis = wallAxis(wall);
  if (axis === "horizontal") {
    wall[handle] = { x: point.x, y: wall[handle].y };
  } else if (axis === "vertical") {
    wall[handle] = { x: wall[handle].x, y: point.y };
  } else {
    const fixed = handle === "start" ? wall.end : wall.start;
    wall[handle] =
      Math.abs(point.x - fixed.x) >= Math.abs(point.y - fixed.y)
        ? { x: point.x, y: fixed.y }
        : { x: fixed.x, y: point.y };
  }
  return next;
}

export function updateWallDimensions(
  plan: FloorPlan,
  wallId: string,
  patch: WallDimensionPatch
): FloorPlan {
  const next = editableClone(plan);
  const wall = next.walls.find((candidate) => candidate.wall_id === wallId);
  if (!wall) return next;
  if (patch.thickness_m !== undefined) wall.thickness_m = patch.thickness_m;
  if (patch.height_m !== undefined) wall.height_m = patch.height_m;
  return next;
}

export function removeWall(plan: FloorPlan, wallId: string): FloorPlan {
  const next = editableClone(plan);
  next.walls = next.walls.filter((wall) => wall.wall_id !== wallId);
  next.openings = next.openings.filter((opening) => opening.wall_id !== wallId);
  return next;
}

export function addOpening(
  plan: FloorPlan,
  wallId: string,
  kind: OpeningKind,
  rawOffsetM: number,
  options: AddOpeningOptions = {}
): FloorPlan {
  const gridM = options.gridM ?? DEFAULT_GRID_M;
  const next = editableClone(plan);
  next.openings.push({
    opening_id:
      options.openingId ??
      nextIdentifier(
        plan.openings.map((opening) => opening.opening_id),
        kind
      ),
    wall_id: wallId,
    kind,
    offset_m: snapToGrid(rawOffsetM, gridM),
    width_m: options.widthM ?? (kind === "door" ? 1 : 1.25),
    height_m: options.heightM ?? (kind === "door" ? 2.1 : 1.2),
    sill_height_m: options.sillHeightM ?? (kind === "door" ? 0 : 0.9)
  });
  return next;
}

export function moveOpening(
  plan: FloorPlan,
  openingId: string,
  rawOffsetM: number,
  gridM = DEFAULT_GRID_M
): FloorPlan {
  const next = editableClone(plan);
  const opening = next.openings.find(
    (candidate) => candidate.opening_id === openingId
  );
  if (opening) opening.offset_m = snapToGrid(rawOffsetM, gridM);
  return next;
}

export function resizeOpening(
  plan: FloorPlan,
  openingId: string,
  rawWidthM: number,
  gridM = DEFAULT_GRID_M
): FloorPlan {
  const next = editableClone(plan);
  const opening = next.openings.find(
    (candidate) => candidate.opening_id === openingId
  );
  if (opening) opening.width_m = snapToGrid(rawWidthM, gridM);
  return next;
}

export function updateOpeningDimensions(
  plan: FloorPlan,
  openingId: string,
  patch: OpeningDimensionPatch
): FloorPlan {
  const next = editableClone(plan);
  const opening = next.openings.find(
    (candidate) => candidate.opening_id === openingId
  );
  if (!opening) return next;
  if (patch.width_m !== undefined) opening.width_m = patch.width_m;
  if (patch.height_m !== undefined) opening.height_m = patch.height_m;
  if (patch.sill_height_m !== undefined) {
    opening.sill_height_m = patch.sill_height_m;
  }
  return next;
}

export function removeOpening(plan: FloorPlan, openingId: string): FloorPlan {
  const next = editableClone(plan);
  next.openings = next.openings.filter(
    (opening) => opening.opening_id !== openingId
  );
  return next;
}

export function removeSelection(
  plan: FloorPlan,
  selection: Selection | null
): FloorPlan {
  if (!selection) return editableClone(plan);
  return selection.kind === "wall"
    ? removeWall(plan, selection.id)
    : removeOpening(plan, selection.id);
}

export function openingGeometry(
  plan: FloorPlan,
  opening: Opening
): OpeningGeometry | null {
  const wall = plan.walls.find(
    (candidate) => candidate.wall_id === opening.wall_id
  );
  if (!wall) return null;
  const length = wallLength(wall);
  if (length <= EDITOR_EPSILON) return null;
  const unit = {
    x: (wall.end.x - wall.start.x) / length,
    y: (wall.end.y - wall.start.y) / length
  };
  const center = {
    x: wall.start.x + unit.x * opening.offset_m,
    y: wall.start.y + unit.y * opening.offset_m
  };
  return {
    wall,
    center,
    start: {
      x: center.x - unit.x * opening.width_m * 0.5,
      y: center.y - unit.y * opening.width_m * 0.5
    },
    end: {
      x: center.x + unit.x * opening.width_m * 0.5,
      y: center.y + unit.y * opening.width_m * 0.5
    }
  };
}

export function offsetAlongWall(wall: Wall, point: Point): number {
  const length = wallLength(wall);
  if (length <= EDITOR_EPSILON) return 0;
  return (
    ((point.x - wall.start.x) * (wall.end.x - wall.start.x) +
      (point.y - wall.start.y) * (wall.end.y - wall.start.y)) /
    length
  );
}

export function selectWallAtPoint(
  plan: FloorPlan,
  point: Point,
  toleranceM = 0.2
): Selection | null {
  let nearest: { id: string; distance: number } | null = null;
  for (const wall of plan.walls) {
    const distance = distanceToSegment(point, wall.start, wall.end);
    if (
      distance <= toleranceM &&
      (!nearest || distance < nearest.distance)
    ) {
      nearest = { id: wall.wall_id, distance };
    }
  }
  return nearest ? { kind: "wall", id: nearest.id } : null;
}

export function selectAtPoint(
  plan: FloorPlan,
  point: Point,
  toleranceM = 0.2
): Selection | null {
  let nearestOpening: { id: string; distance: number } | null = null;
  for (const opening of plan.openings) {
    const geometry = openingGeometry(plan, opening);
    if (!geometry) continue;
    const distance = distanceToSegment(point, geometry.start, geometry.end);
    if (
      distance <= toleranceM * 1.5 &&
      (!nearestOpening || distance < nearestOpening.distance)
    ) {
      nearestOpening = { id: opening.opening_id, distance };
    }
  }
  if (nearestOpening) {
    return { kind: "opening", id: nearestOpening.id };
  }
  return selectWallAtPoint(plan, point, toleranceM);
}

export function validateFloorPlan(
  plan: FloorPlan,
  footprintWidthM: number,
  footprintDepthM: number
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  validateFootprintDimension(
    footprintWidthM,
    "footprint_width_m",
    "Footprint width",
    issues
  );
  validateFootprintDimension(
    footprintDepthM,
    "footprint_depth_m",
    "Footprint depth",
    issues
  );
  if (plan.walls.length < 4) {
    issues.push({
      code: "insufficient_walls",
      path: "floor_plan.walls",
      message: "A floor plan requires at least four walls."
    });
  }

  const wallIds = new Map<string, string>();
  plan.walls.forEach((wall, index) => {
    const path = `floor_plan.walls[${index}]`;
    validateIdentifier(wall.wall_id, `${path}.wall_id`, "Wall", wallIds, issues);
    validateWall(wall, path, footprintWidthM, footprintDepthM, issues);
  });

  for (let first = 0; first < plan.walls.length; first += 1) {
    for (let second = first + 1; second < plan.walls.length; second += 1) {
      validateWallPair(
        plan.walls[first],
        plan.walls[second],
        `floor_plan.walls[${second}]`,
        issues
      );
    }
  }

  const openingIds = new Map<string, string>();
  plan.openings.forEach((opening, index) => {
    const path = `floor_plan.openings[${index}]`;
    validateIdentifier(
      opening.opening_id,
      `${path}.opening_id`,
      "Opening",
      openingIds,
      issues
    );
    validateOpening(opening, plan, path, issues);
  });

  for (let first = 0; first < plan.openings.length; first += 1) {
    for (let second = first + 1; second < plan.openings.length; second += 1) {
      validateOpeningPair(
        plan.openings[first],
        plan.openings[second],
        `floor_plan.openings[${second}]`,
        issues
      );
    }
  }
  const roomIds = new Map<string, string>();
  plan.rooms.forEach((room, index) => {
    validateIdentifier(
      room.room_id,
      `floor_plan.rooms[${index}].room_id`,
      "Room",
      roomIds,
      issues
    );
  });
  return issues;
}

export function projectValidation(
  plan: FloorPlan,
  footprintWidthM: number,
  footprintDepthM: number
): ValidationProjection {
  const issues = validateFloorPlan(plan, footprintWidthM, footprintDepthM);
  const projected = cloneFloorPlan(plan);
  projected.warnings = issues.map((issue) => issue.message);
  if (issues.length > 0) projected.approved = false;
  return { valid: issues.length === 0, issues, plan: projected };
}

function editableClone(plan: FloorPlan): FloorPlan {
  const next = cloneFloorPlan(plan);
  next.approved = false;
  next.warnings = [];
  return next;
}

function nextIdentifier(existing: string[], prefix: string): string {
  const taken = new Set(existing);
  let suffix = 1;
  while (taken.has(`${prefix}_${suffix}`)) suffix += 1;
  return `${prefix}_${suffix}`;
}

function validateFootprintDimension(
  value: number,
  path: string,
  label: string,
  issues: ValidationIssue[]
) {
  if (!Number.isFinite(value)) {
    issues.push({
      code: "non_finite_dimension",
      path,
      message: `${label} must be finite.`
    });
  } else if (value <= 0) {
    issues.push({
      code: "non_positive_dimension",
      path,
      message: `${label} must be greater than zero.`
    });
  }
}

function validateIdentifier(
  id: string,
  path: string,
  label: string,
  seen: Map<string, string>,
  issues: ValidationIssue[]
) {
  if (!id.trim()) {
    issues.push({ code: "invalid_id", path, message: `${label} ID cannot be blank.` });
    return;
  }
  const existing = seen.get(id);
  if (existing) {
    issues.push({
      code: "duplicate_id",
      path,
      message: `${label} ID “${id}” duplicates ${existing}.`
    });
  } else {
    seen.set(id, label.toLowerCase());
  }
}

function validateWall(
  wall: Wall,
  path: string,
  footprintWidthM: number,
  footprintDepthM: number,
  issues: ValidationIssue[]
) {
  const values = [
    wall.start.x,
    wall.start.y,
    wall.end.x,
    wall.end.y,
    wall.thickness_m,
    wall.height_m
  ];
  if (values.some((value) => !Number.isFinite(value))) {
    issues.push({
      code: "non_finite_dimension",
      path,
      message: `Wall “${wall.wall_id || "untitled"}” contains a non-finite dimension.`
    });
    return;
  }
  if (wall.thickness_m <= 0 || wall.height_m <= 0) {
    issues.push({
      code: "non_positive_dimension",
      path,
      message: `Wall “${wall.wall_id || "untitled"}” thickness and height must be positive.`
    });
  }
  const axis = wallAxis(wall);
  if (axis === "diagonal") {
    issues.push({
      code: "wall_not_axis_aligned",
      path,
      message: `Wall “${wall.wall_id || "untitled"}” must be horizontal or vertical.`
    });
  }
  if (wallLength(wall) <= EDITOR_EPSILON) {
    issues.push({
      code: "wall_zero_length",
      path,
      message: `Wall “${wall.wall_id || "untitled"}” must have a non-zero length.`
    });
  }
  if (
    Number.isFinite(footprintWidthM) &&
    footprintWidthM > 0 &&
    Number.isFinite(footprintDepthM) &&
    footprintDepthM > 0
  ) {
    const halfWidth = footprintWidthM * 0.5;
    const halfDepth = footprintDepthM * 0.5;
    if (
      [wall.start, wall.end].some(
        (point) =>
          point.x < -halfWidth - EDITOR_EPSILON ||
          point.x > halfWidth + EDITOR_EPSILON ||
          point.y < -halfDepth - EDITOR_EPSILON ||
          point.y > halfDepth + EDITOR_EPSILON
      )
    ) {
      issues.push({
        code: "wall_out_of_bounds",
        path,
        message: `Wall “${wall.wall_id || "untitled"}” extends outside the footprint.`
      });
    }
  }
}

function validateWallPair(
  first: Wall,
  second: Wall,
  path: string,
  issues: ValidationIssue[]
) {
  const firstAxis = wallAxis(first);
  const secondAxis = wallAxis(second);
  if (firstAxis === "diagonal" || secondAxis === "diagonal") return;
  if (firstAxis === secondAxis) {
    const sameLine =
      firstAxis === "horizontal"
        ? almostEqual(first.start.y, second.start.y)
        : almostEqual(first.start.x, second.start.x);
    if (!sameLine) return;
    const firstRange =
      firstAxis === "horizontal"
        ? ordered(first.start.x, first.end.x)
        : ordered(first.start.y, first.end.y);
    const secondRange =
      secondAxis === "horizontal"
        ? ordered(second.start.x, second.end.x)
        : ordered(second.start.y, second.end.y);
    if (
      Math.min(firstRange[1], secondRange[1]) -
        Math.max(firstRange[0], secondRange[0]) >
      EDITOR_EPSILON
    ) {
      issues.push({
        code: "wall_overlap",
        path,
        message: `Walls “${first.wall_id}” and “${second.wall_id}” overlap.`
      });
    }
    return;
  }

  const horizontal = firstAxis === "horizontal" ? first : second;
  const vertical = firstAxis === "vertical" ? first : second;
  const intersection = { x: vertical.start.x, y: horizontal.start.y };
  const horizontalRange = ordered(horizontal.start.x, horizontal.end.x);
  const verticalRange = ordered(vertical.start.y, vertical.end.y);
  const intersects =
    inRange(intersection.x, horizontalRange) &&
    inRange(intersection.y, verticalRange);
  if (
    intersects &&
    !(
      isEndpoint(horizontal, intersection) &&
      isEndpoint(vertical, intersection)
    )
  ) {
    issues.push({
      code: "wall_intersection",
      path,
      message: `Walls “${first.wall_id}” and “${second.wall_id}” cross without a shared endpoint.`
    });
  }
}

function validateOpening(
  opening: Opening,
  plan: FloorPlan,
  path: string,
  issues: ValidationIssue[]
) {
  const values = [
    opening.offset_m,
    opening.width_m,
    opening.height_m,
    opening.sill_height_m
  ];
  if (values.some((value) => !Number.isFinite(value))) {
    issues.push({
      code: "non_finite_dimension",
      path,
      message: `Opening “${opening.opening_id || "untitled"}” contains a non-finite dimension.`
    });
    return;
  }
  if (
    opening.width_m <= 0 ||
    opening.height_m <= 0 ||
    opening.offset_m < 0 ||
    opening.sill_height_m < 0
  ) {
    issues.push({
      code: "non_positive_dimension",
      path,
      message: `Opening “${opening.opening_id || "untitled"}” dimensions must be positive and offsets non-negative.`
    });
  }
  const wall = plan.walls.find(
    (candidate) => candidate.wall_id === opening.wall_id
  );
  if (!wall) {
    issues.push({
      code: "opening_unknown_wall",
      path,
      message: `Opening “${opening.opening_id || "untitled"}” references an unknown wall.`
    });
    return;
  }
  const halfWidth = opening.width_m * 0.5;
  if (
    opening.offset_m - halfWidth < -EDITOR_EPSILON ||
    opening.offset_m + halfWidth > wallLength(wall) + EDITOR_EPSILON
  ) {
    issues.push({
      code: "opening_out_of_bounds",
      path,
      message: `Opening “${opening.opening_id || "untitled"}” must fit within wall “${wall.wall_id}”.`
    });
  }
  if (
    opening.sill_height_m + opening.height_m >
    wall.height_m + EDITOR_EPSILON
  ) {
    issues.push({
      code: "opening_vertical_overflow",
      path,
      message: `Opening “${opening.opening_id || "untitled"}” exceeds wall “${wall.wall_id}” height.`
    });
  }
}

function validateOpeningPair(
  first: Opening,
  second: Opening,
  path: string,
  issues: ValidationIssue[]
) {
  if (first.wall_id !== second.wall_id) return;
  const firstRange: [number, number] = [
    first.offset_m - first.width_m * 0.5,
    first.offset_m + first.width_m * 0.5
  ];
  const secondRange: [number, number] = [
    second.offset_m - second.width_m * 0.5,
    second.offset_m + second.width_m * 0.5
  ];
  if (
    Math.min(firstRange[1], secondRange[1]) -
      Math.max(firstRange[0], secondRange[0]) >
    EDITOR_EPSILON
  ) {
    issues.push({
      code: "opening_overlap",
      path,
      message: `Openings “${first.opening_id}” and “${second.opening_id}” overlap.`
    });
  }
}

function distanceToSegment(point: Point, start: Point, end: Point): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= EDITOR_EPSILON) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }
  const ratio = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * dx + (point.y - start.y) * dy) /
        lengthSquared
    )
  );
  return Math.hypot(
    point.x - (start.x + ratio * dx),
    point.y - (start.y + ratio * dy)
  );
}

function almostEqual(first: number, second: number): boolean {
  return Math.abs(first - second) <= EDITOR_EPSILON;
}

function ordered(first: number, second: number): [number, number] {
  return first <= second ? [first, second] : [second, first];
}

function inRange(value: number, range: [number, number]): boolean {
  return (
    value >= range[0] - EDITOR_EPSILON &&
    value <= range[1] + EDITOR_EPSILON
  );
}

function isEndpoint(wall: Wall, point: Point): boolean {
  return (
    (almostEqual(wall.start.x, point.x) &&
      almostEqual(wall.start.y, point.y)) ||
    (almostEqual(wall.end.x, point.x) &&
      almostEqual(wall.end.y, point.y))
  );
}
