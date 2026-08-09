import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { FloorPlan, Selection } from "../editor/designEditor";
import { DesignEditor } from "./DesignEditor";

function cottagePlan(): FloorPlan {
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

function EditorHarness({
  initialSelection = null
}: {
  initialSelection?: Selection | null;
}) {
  const [plan, setPlan] = useState(cottagePlan);
  const [selection, setSelection] = useState<Selection | null>(
    initialSelection
  );
  const [announcement, setAnnouncement] = useState("");
  return (
    <>
      <DesignEditor
        plan={plan}
        footprintWidthM={8}
        footprintDepthM={6}
        readOnly={false}
        selection={selection}
        onSelection={setSelection}
        onPlan={(next, message) => {
          setPlan(next);
          setAnnouncement(message);
        }}
      />
      <output aria-label="Editor announcement">{announcement}</output>
    </>
  );
}

describe("DesignEditor keyboard drafting", () => {
  it("creates an axis-aligned wall without a pointer coordinate", async () => {
    const user = userEvent.setup();
    render(<EditorHarness />);
    const canvas = screen.getByRole("application", {
      name: /editable floor plan/i
    });

    canvas.focus();
    await user.keyboard("w");
    await user.keyboard("{Enter}");
    expect(screen.getByLabelText("Editor announcement")).toHaveTextContent(
      /wall start set/i
    );
    await user.keyboard(
      "{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{Enter}"
    );

    expect(
      screen.getByRole("button", { name: /wall wall_1, 1\.00 metres/i })
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Editor announcement")).toHaveTextContent(
      "Wall wall_1 added. Approval reset."
    );
  });

  it("places an opening on the selected wall with shortcut and Enter", async () => {
    const user = userEvent.setup();
    render(
      <EditorHarness
        initialSelection={{ kind: "wall", id: "north" }}
      />
    );
    const canvas = screen.getByRole("application", {
      name: /editable floor plan/i
    });

    canvas.focus();
    await user.keyboard("d{Enter}");

    expect(
      screen.getByRole("button", {
        name: /door door_1, 1\.00 metres wide/i
      })
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Editor announcement")).toHaveTextContent(
      "Door added. Approval reset."
    );
  });

  it("activates SVG pseudo-buttons with Enter and Space", () => {
    const onSelection = vi.fn();
    render(
      <DesignEditor
        plan={cottagePlan()}
        footprintWidthM={8}
        footprintDepthM={6}
        readOnly={false}
        selection={null}
        onSelection={onSelection}
        onPlan={vi.fn()}
      />
    );

    const north = screen.getByRole("button", { name: /wall north/i });
    fireEvent.keyDown(north, { key: "Enter" });
    const door = screen.getByRole("button", { name: /door front_door/i });
    fireEvent.keyDown(door, { key: " " });

    expect(onSelection).toHaveBeenCalledWith({ kind: "wall", id: "north" });
    expect(onSelection).toHaveBeenCalledWith({
      kind: "opening",
      id: "front_door"
    });
  });

  it("preserves pointer hit-testing and selection on the canvas", () => {
    const onSelection = vi.fn();
    render(
      <DesignEditor
        plan={cottagePlan()}
        footprintWidthM={8}
        footprintDepthM={6}
        readOnly={false}
        selection={null}
        onSelection={onSelection}
        onPlan={vi.fn()}
      />
    );
    const canvas = screen.getByRole("application", {
      name: /editable floor plan/i
    });
    Object.defineProperties(canvas, {
      getScreenCTM: {
        value: () => ({ inverse: () => ({}) })
      },
      createSVGPoint: {
        value: () => {
          const point = {
            x: 0,
            y: 0,
            matrixTransform: () => ({ x: point.x, y: point.y })
          };
          return point;
        }
      },
      setPointerCapture: {
        value: vi.fn()
      }
    });

    fireEvent.pointerDown(canvas, {
      clientX: 0,
      clientY: -3,
      pointerId: 1
    });

    expect(onSelection).toHaveBeenCalledWith({ kind: "wall", id: "north" });
  });

  it("exposes named wall handles and resizes them with arrow keys", () => {
    const onPlan = vi.fn();
    render(
      <DesignEditor
        plan={cottagePlan()}
        footprintWidthM={8}
        footprintDepthM={6}
        readOnly={false}
        selection={{ kind: "wall", id: "north" }}
        onSelection={vi.fn()}
        onPlan={onPlan}
      />
    );
    const handle = screen.getByRole("button", {
      name: /resize end endpoint of wall north.*arrow keys/i
    });

    expect(handle).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(handle, { key: "ArrowLeft" });

    const resized = onPlan.mock.calls[0][0] as FloorPlan;
    expect(resized.walls[0].end).toEqual({ x: 3.75, y: 3 });
    expect(onPlan.mock.calls[0][1]).toMatch(/approval reset/i);
  });

  it("exposes named opening edges and resizes along the parent wall", () => {
    const onPlan = vi.fn();
    render(
      <DesignEditor
        plan={cottagePlan()}
        footprintWidthM={8}
        footprintDepthM={6}
        readOnly={false}
        selection={{ kind: "opening", id: "front_door" }}
        onSelection={vi.fn()}
        onPlan={onPlan}
      />
    );
    const handle = screen.getByRole("button", {
      name: /resize start edge of door front_door.*along its wall/i
    });

    fireEvent.keyDown(handle, { key: "ArrowRight" });

    const resized = onPlan.mock.calls[0][0] as FloorPlan;
    expect(resized.openings[0].width_m).toBe(1.5);
    expect(onPlan.mock.calls[0][1]).toMatch(/width resized/i);
  });
});
