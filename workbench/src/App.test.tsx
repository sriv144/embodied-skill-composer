import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, runtimeStatusFor } from "./api";
import { deferred, projectFixture, traceFixture } from "./test/workbenchFixtures";
import type { Project, Trace } from "./types";

const controls = vi.hoisted(() => ({
  replace: null as ((project: Project) => void) | null,
  change: null as ((controller: string) => void) | null,
  recover: null as ((trace: Trace) => void) | null
}));

vi.mock("framer-motion", () => ({
  useReducedMotion: () => true,
  motion: { section: ({ children }: { children: ReactNode }) => <section>{children}</section> }
}));
vi.mock("./views/DesignView", () => ({
  DesignView: ({ onProject }: { onProject: (project: Project) => void }) => {
    controls.replace = onProject;
    return <p>Editor ready</p>;
  }
}));
vi.mock("./views/SimulateView", () => ({
  SimulateView: ({ trace, onController, onTrace }: { trace: Trace; onController: (value: string) => void; onTrace: (value: Trace) => void }) => {
    controls.change = onController;
    controls.recover = onTrace;
    return <p>Replay {trace.plan_id}</p>;
  }
}));

beforeEach(() => {
  controls.replace = controls.change = controls.recover = null;
  window.location.hash = "/design";
  vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  vi.spyOn(api, "mode").mockReturnValue("local");
  vi.spyOn(api, "runtime").mockReturnValue(runtimeStatusFor("local"));
  vi.spyOn(api, "project").mockResolvedValue(projectFixture());
  vi.spyOn(api, "trace").mockResolvedValue(traceFixture());
  vi.spyOn(api, "runs").mockResolvedValue([]);
  vi.spyOn(api, "policies").mockResolvedValue([]);
  vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function start() {
  render(<App />);
  await screen.findByText("Editor ready");
  await userEvent.click(screen.getByRole("button", { name: "Simulate" }));
  await screen.findByText("Replay original");
}

describe("replay revision ownership", () => {
  it("reserves a live-status grid row between the header and view before and after compilation", async () => {
    await start();
    const status = document.querySelector(".main-shell > .project-notice");
    expect(status).toBeEmptyDOMElement();
    expect(status).toHaveAttribute("role", "status");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status?.previousElementSibling).toHaveClass("topbar");
    expect(status?.nextElementSibling?.tagName).toBe("SECTION");

    vi.mocked(api.trace).mockResolvedValueOnce(traceFixture("compiled"));
    await act(async () => controls.replace!(projectFixture("compiled")));
    const updatedStatus = document.querySelector(".main-shell > .project-notice");
    expect(updatedStatus).toHaveTextContent("Approved design compiled into a new build plan.");
    expect(updatedStatus?.previousElementSibling).toHaveClass("topbar");
    expect(updatedStatus?.nextElementSibling?.tagName).toBe("SECTION");
  });

  it("makes post-compile failures visible and removes stale replay while loading", async () => {
    await start();
    const pending = deferred<Trace>();
    vi.mocked(api.trace).mockReturnValueOnce(pending.promise);
    act(() => controls.replace!(projectFixture("compiled")));
    expect(screen.queryByText("Replay original")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Loading construction trace" })).toBeInTheDocument();
    await act(async () => pending.reject(new Error("trace refresh unavailable")));
    expect(screen.getByRole("alert")).toHaveTextContent("trace refresh unavailable");
    expect(screen.queryByText("Replay original")).not.toBeInTheDocument();
  });

  it.each(["resolve", "reject"] as const)("ignores an older controller request that later %ss", async (outcome) => {
    await start();
    const older = deferred<Trace>();
    const replacement = deferred<Trace>();
    vi.mocked(api.trace).mockReturnValueOnce(older.promise).mockReturnValueOnce(replacement.promise);
    act(() => controls.change!("greedy"));
    act(() => controls.replace!(projectFixture("compiled")));
    expect(api.trace).toHaveBeenLastCalledWith("greedy");
    await act(async () => replacement.resolve(traceFixture("compiled")));
    expect(await screen.findByText("Replay compiled")).toBeInTheDocument();
    expect(screen.getByText("Approved design compiled into a new build plan.")).toHaveAttribute("role", "status");
    await act(async () => outcome === "resolve" ? older.resolve(traceFixture("stale")) : older.reject(new Error("stale failure")));
    expect(screen.getByText("Replay compiled")).toBeInTheDocument();
    expect(screen.queryByText("Replay stale")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("ignores a recovery callback captured before the compiled revision", async () => {
    await start();
    const oldRecovery = controls.recover!;
    vi.mocked(api.trace).mockResolvedValueOnce(traceFixture("compiled"));
    await act(async () => controls.replace!(projectFixture("compiled")));
    expect(await screen.findByText("Replay compiled")).toBeInTheDocument();
    act(() => oldRecovery(traceFixture("stale-recovery")));
    expect(screen.getByText("Replay compiled")).toBeInTheDocument();
    expect(screen.queryByText("Replay stale-recovery")).not.toBeInTheDocument();
  });
});
