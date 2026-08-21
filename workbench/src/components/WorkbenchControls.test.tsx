import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ControllerSelect } from "./WorkbenchControls";

describe("ControllerSelect", () => {
  it("exposes the selected controller as pressed state", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ControllerSelect value="optimized" onChange={onChange} />);

    const group = screen.getByRole("group", { name: "Replay controller" });
    expect(group).toBeVisible();
    expect(screen.getByRole("button", { name: "CP-SAT" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(screen.getByRole("button", { name: "Sequential" })).toHaveAttribute(
      "aria-pressed",
      "false"
    );

    await user.click(screen.getByRole("button", { name: "Greedy" }));
    expect(onChange).toHaveBeenCalledWith("greedy");
  });
});
