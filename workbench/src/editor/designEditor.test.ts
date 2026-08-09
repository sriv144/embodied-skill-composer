import { describe, expect, it } from "vitest";
import {
  addOpening,
  cloneFloorPlan,
  createWall,
  moveOpening,
  moveWall,
  openingGeometry,
  projectValidation,
  removeOpening,
  removeSelection,
  removeWall,
  resizeOpening,
  resizeWall,
  selectAtPoint,
  snapPoint,
  snapToGrid,
  updateOpeningDimensions,
  updateWallDimensions,
  validateFloorPlan,
  wallAxis,
  wallLength,
  type FloorPlan
} from "./designEditor";

function approvedPlan(): FloorPlan {
  return {
    approved: true,
    confidence: 1,
    warnings: [],
    walls: [
      {
        wall_id: "north",
        start: { x: -4, y: 3 },
        end: { x: 4, y: 3 },
        thickness_m: 0.2,
        height_m: 2.8
      },
      {
        wall_id: "east",
        start: { x: 4, y: 3 },
        end: { x: 4, y: -3 },
        thickness_m: 0.2,
        height_m: 2.8
      },
      {
        wall_id: "south",
        start: { x: 4, y: -3 },
        end: { x: -4, y: -3 },
        thickness_m: 0.2,
        height_m: 2.8
      },
      {
        wall_id: "west",
        start: { x: -4, y: -3 },
        end: { x: -4, y: 3 },
        thickness_m: 0.2,
        height_m: 2.8
      }
    ],
    openings: [
      {
        opening_id: "front_door",
        wall_id: "south",
        kind: "door",
        offset_m: 4,
        width_m: 1,
        height_m: 2.1,
        sill_height_m: 0
      }
    ],
    rooms: []
  };
}

describe("draft cloning and grid geometry", () => {
  it("snaps values and points without negative zero", () => {
    expect(snapToGrid(1.13, 0.25)).toBe(1.25);
    expect(snapToGrid(-0.01, 0.25)).toBe(0);
    expect(snapPoint({ x: 0.61, y: -0.62 }, 0.25)).toEqual({
      x: 0.5,
      y: -0.5
    });
  });

  it("deep clones all mutable floor-plan geometry", () => {
    const source = approvedPlan();
    const clone = cloneFloorPlan(source);
    clone.walls[0].start.x = 99;
    clone.openings[0].width_m = 99;
    expect(source.walls[0].start.x).toBe(-4);
    expect(source.openings[0].width_m).toBe(1);
  });

  it("creates snapped, axis-aligned walls without mutating the source", () => {
    const source = approvedPlan();
    const result = createWall(
      source,
      { x: -1.12, y: 0.12 },
      { x: 2.31, y: 0.91 },
      { wallId: "partition" }
    );
    expect(result).not.toBe(source);
    expect(result.approved).toBe(false);
    expect(source.walls).toHaveLength(4);
    expect(result.walls.at(-1)).toMatchObject({
      wall_id: "partition",
      start: { x: -1, y: 0 },
      end: { x: 2.25, y: 0 }
    });
    expect(wallAxis(result.walls.at(-1)!)).toBe("horizontal");
  });

  it("moves and resizes walls while preserving their axis", () => {
    const source = approvedPlan();
    const moved = moveWall(source, "north", { x: 0.49, y: -0.24 });
    expect(moved.walls[0].start).toEqual({ x: -3.5, y: 2.75 });
    expect(moved.walls[0].end).toEqual({ x: 4.5, y: 2.75 });
    const resized = resizeWall(
      moved,
      "north",
      "end",
      { x: 3.13, y: -20 }
    );
    expect(resized.walls[0].end).toEqual({ x: 3.25, y: 2.75 });
    expect(wallAxis(resized.walls[0])).toBe("horizontal");
  });

  it("removes a wall and all of its dependent openings", () => {
    const source = approvedPlan();
    const result = removeWall(source, "south");
    expect(result.walls.map((wall) => wall.wall_id)).not.toContain("south");
    expect(result.openings).toEqual([]);
    expect(source.openings).toHaveLength(1);
    expect(result.approved).toBe(false);
    expect(validateFloorPlan(result, 8, 6).map((issue) => issue.code)).toContain(
      "insufficient_walls"
    );
  });
});

describe("opening operations and selection", () => {
  it("adds, moves, resizes, updates, and removes an opening immutably", () => {
    const source = approvedPlan();
    const added = addOpening(source, "north", "window", 2.13, {
      openingId: "north_window"
    });
    expect(added.openings.at(-1)).toMatchObject({
      opening_id: "north_window",
      offset_m: 2.25,
      width_m: 1.25,
      height_m: 1.2,
      sill_height_m: 0.9
    });
    const moved = moveOpening(added, "north_window", 3.13);
    const resized = resizeOpening(moved, "north_window", 1.76);
    const updated = updateOpeningDimensions(resized, "north_window", {
      height_m: 1.4,
      sill_height_m: 0.75
    });
    expect(updated.openings.at(-1)).toMatchObject({
      offset_m: 3.25,
      width_m: 1.75,
      height_m: 1.4,
      sill_height_m: 0.75
    });
    expect(removeOpening(updated, "north_window").openings).toHaveLength(1);
    expect(source.openings).toHaveLength(1);
  });

  it("projects openings onto their parent wall and selects them first", () => {
    const plan = approvedPlan();
    const opening = plan.openings[0];
    const geometry = openingGeometry(plan, opening);
    expect(geometry?.center).toEqual({ x: 0, y: -3 });
    expect(wallLength(geometry!.wall)).toBe(8);
    expect(selectAtPoint(plan, { x: 0, y: -3 }, 0.2)).toEqual({
      kind: "opening",
      id: "front_door"
    });
    expect(selectAtPoint(plan, { x: -3, y: 3 }, 0.2)).toEqual({
      kind: "wall",
      id: "north"
    });
  });
});

describe("approval-reset invariant", () => {
  it("returns a distinct unapproved clone for every editor mutation", () => {
    const source = approvedPlan();
    const mutations = [
      createWall(source, { x: 0, y: 0 }, { x: 1, y: 0 }),
      moveWall(source, "north", { x: 0, y: -0.25 }),
      resizeWall(source, "north", "end", { x: 3, y: 3 }),
      updateWallDimensions(source, "north", { height_m: 3 }),
      removeWall(source, "north"),
      addOpening(source, "north", "window", 2),
      moveOpening(source, "front_door", 3),
      resizeOpening(source, "front_door", 1.25),
      updateOpeningDimensions(source, "front_door", { height_m: 2 }),
      removeOpening(source, "front_door"),
      removeSelection(source, { kind: "opening", id: "front_door" })
    ];
    for (const result of mutations) {
      expect(result).not.toBe(source);
      expect(result.approved).toBe(false);
      expect(result.warnings).toEqual([]);
    }
    expect(source.approved).toBe(true);
  });
});

describe("client validation projection", () => {
  it("accepts the canonical cottage shell", () => {
    expect(validateFloorPlan(approvedPlan(), 8, 6)).toEqual([]);
  });

  it("accepts a safe positive footprint below editor display-grid increments", () => {
    const plan = approvedPlan();
    plan.openings = [];
    plan.walls = [
      {
        wall_id: "north",
        start: { x: -0.1, y: 0.075 },
        end: { x: 0.1, y: 0.075 },
        thickness_m: 0.01,
        height_m: 0.2
      },
      {
        wall_id: "east",
        start: { x: 0.1, y: 0.075 },
        end: { x: 0.1, y: -0.075 },
        thickness_m: 0.01,
        height_m: 0.2
      },
      {
        wall_id: "south",
        start: { x: 0.1, y: -0.075 },
        end: { x: -0.1, y: -0.075 },
        thickness_m: 0.01,
        height_m: 0.2
      },
      {
        wall_id: "west",
        start: { x: -0.1, y: -0.075 },
        end: { x: -0.1, y: 0.075 },
        thickness_m: 0.01,
        height_m: 0.2
      }
    ];

    expect(validateFloorPlan(plan, 0.2, 0.15)).toEqual([]);
  });

  it("checks IDs within each wire-format collection", () => {
    const plan = approvedPlan();
    plan.openings[0].opening_id = "north";
    plan.rooms = [
      {
        room_id: "living",
        name: "Living",
        polygon: [
          { x: -1, y: -1 },
          { x: 1, y: -1 },
          { x: 1, y: 1 }
        ]
      },
      {
        room_id: "living",
        name: "Kitchen",
        polygon: [
          { x: -1, y: -1 },
          { x: 1, y: -1 },
          { x: 1, y: 1 }
        ]
      }
    ];
    const issues = validateFloorPlan(plan, 8, 6);
    expect(
      issues.filter((issue) => issue.code === "duplicate_id")
    ).toHaveLength(1);
    expect(issues.find((issue) => issue.code === "duplicate_id")?.path).toBe(
      "floor_plan.rooms[1].room_id"
    );
  });

  it("reports bounds, wall crossing, overlap, and dimension issues", () => {
    let plan = approvedPlan();
    plan = createWall(
      plan,
      { x: -1, y: 0 },
      { x: 1, y: 0 },
      { wallId: "cross-horizontal" }
    );
    plan = createWall(
      plan,
      { x: 0, y: -1 },
      { x: 0, y: 1 },
      { wallId: "cross-vertical" }
    );
    plan = createWall(
      plan,
      { x: -2, y: 3 },
      { x: 2, y: 3 },
      { wallId: "north-overlap" }
    );
    plan = moveWall(plan, "east", { x: 0.25, y: 0 });
    plan = updateWallDimensions(plan, "west", { thickness_m: 0 });
    const codes = validateFloorPlan(plan, 8, 6).map((issue) => issue.code);
    expect(codes).toEqual(
      expect.arrayContaining([
        "wall_intersection",
        "wall_overlap",
        "wall_out_of_bounds",
        "non_positive_dimension"
      ])
    );
  });

  it("reports unknown, overflowing, out-of-bounds, and overlapping openings", () => {
    const plan = approvedPlan();
    plan.openings = [
      {
        ...plan.openings[0],
        opening_id: "wide",
        width_m: 9
      },
      {
        ...plan.openings[0],
        opening_id: "overlap",
        offset_m: 4.25,
        height_m: 3,
        sill_height_m: 0.25
      },
      {
        ...plan.openings[0],
        opening_id: "orphan",
        wall_id: "missing"
      }
    ];
    const codes = validateFloorPlan(plan, 8, 6).map((issue) => issue.code);
    expect(codes).toEqual(
      expect.arrayContaining([
        "opening_out_of_bounds",
        "opening_overlap",
        "opening_vertical_overflow",
        "opening_unknown_wall"
      ])
    );
  });

  it("projects validation messages and revokes invalid approval", () => {
    const plan = approvedPlan();
    plan.walls[0].end.y = 2;
    const projection = projectValidation(plan, 8, 6);
    expect(projection.valid).toBe(false);
    expect(projection.plan).not.toBe(plan);
    expect(projection.plan.approved).toBe(false);
    expect(projection.plan.warnings).toContain(
      "Wall “north” must be horizontal or vertical."
    );
  });
});
