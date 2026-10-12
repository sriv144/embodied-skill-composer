import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, runtimeStatusFor } from "../api";
import { deferred, projectFixture, traceFixture } from "../test/workbenchFixtures";
import type { Trace } from "../types";
import { SimulateView } from "./SimulateView";

vi.mock("../components/ConstructionScene", () => ({ ConstructionScene: () => <div>Scene fixture</div> }));
afterEach(() => vi.restoreAllMocks());

describe("recovery announcements", () => {
  it.each(["success", "failure"] as const)("announces pending recovery and its %s", async (outcome) => {
    vi.spyOn(api, "runtime").mockReturnValue(runtimeStatusFor("local"));
    const pending = deferred<Trace>();
    vi.spyOn(api, "disrupt").mockReturnValue(pending.promise);
    const onTrace = vi.fn();
    render(<SimulateView project={projectFixture()} trace={traceFixture()} controller="optimized" coppelia={null} onController={vi.fn()} onTrace={onTrace} />);
    await userEvent.click(screen.getByRole("button", { name: "Obstacle" }));
    expect(screen.getByRole("status")).toHaveTextContent("Applying recovery event.");
    if (outcome === "success") {
      await act(async () => pending.resolve(traceFixture("recovered")));
      expect(screen.getByRole("status")).toHaveTextContent("Obstacle inserted; schedule recovered.");
      expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
      expect(onTrace).toHaveBeenCalledWith(traceFixture("recovered"));
    } else {
      await act(async () => pending.reject(new Error("Recovery API unavailable")));
      expect(screen.getByRole("alert")).toHaveTextContent("Recovery API unavailable");
      expect(screen.getByRole("alert")).toHaveAttribute("aria-live", "assertive");
      expect(onTrace).not.toHaveBeenCalled();
    }
  });
});
