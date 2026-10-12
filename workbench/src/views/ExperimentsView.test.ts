import { describe, expect, it } from "vitest";
import type {
  ExperimentMatrix,
  ExperimentMatrixRun,
  LabRun,
  LabRunKind,
  LabRunStatus
} from "../types";
import {
  isCancellable,
  isResumable,
  isTerminalRunEvent,
  runForLiveEvents,
  runsForLauncher
} from "./ExperimentsView";

function run(
  id: string,
  status: LabRunStatus,
  kind: LabRunKind = "training"
): LabRun {
  return {
    id,
    kind,
    status,
    config: {},
    created_at: "2026-07-29T00:00:00+00:00",
    started_at: null,
    ended_at: null,
    progress: 0,
    artifact_dir: null,
    error: null,
    latest_checkpoint: null
  };
}

function matrixWith(runRecord: LabRun): ExperimentMatrix {
  const matrixRun: ExperimentMatrixRun = {
    ...runRecord,
    run_key: `${runRecord.id}-key`,
    ordinal: 0,
    config: {}
  };
  return {
    id: "matrix-1",
    protocol_digest: "digest",
    protocol: {},
    execution_profile: "unit",
    expected_run_count: 1,
    selection_count: 0,
    status: "queued",
    status_counts: { [runRecord.status]: 1 },
    created_at: "2026-07-29T00:00:00+00:00",
    runs: [matrixRun]
  };
}

describe("experiment run-state projection", () => {
  it("separates single jobs from matrix-owned jobs", () => {
    const single = run("single", "queued");
    const matrixRun = run("matrix-run", "queued");
    const heldout = run("heldout", "queued", "matrix_evaluation");
    const matrix = matrixWith(matrixRun);

    expect(
      runsForLauncher(
        "single",
        [single, matrixRun, heldout],
        [matrix],
        matrix
      ).map((item) => item.id)
    ).toEqual(["single"]);
    expect(
      runsForLauncher(
        "matrix",
        [single, matrixRun, heldout],
        [matrix],
        matrix
      ).map((item) => item.id)
    ).toEqual(["matrix-run"]);
  });

  it("activates queued sockets and prioritizes an executing run", () => {
    const queued = run("queued", "queued");
    const running = run("running", "running");

    expect(runForLiveEvents([queued])?.id).toBe("queued");
    expect(runForLiveEvents([queued, running])?.id).toBe("running");
  });

  it("treats interruption and worker exit events as terminal refreshes", () => {
    expect(isTerminalRunEvent("run_interrupted")).toBe(true);
    expect(isTerminalRunEvent("worker_exited")).toBe(true);
    expect(isTerminalRunEvent("ppo_update")).toBe(false);
  });

  it("uses server capability flags for resume and cancel parity", () => {
    const interrupted = {
      ...run("interrupted", "interrupted"),
      can_resume: false
    };
    const resuming = {
      ...run("resuming", "resuming"),
      can_cancel: true
    };
    const running = {
      ...run("running", "running"),
      can_cancel: false
    };

    expect(isResumable(interrupted)).toBe(false);
    expect(isCancellable(resuming)).toBe(true);
    expect(isCancellable(running)).toBe(false);
  });
});
