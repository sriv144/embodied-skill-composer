import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, runtimeStatusFor } from "../api";
import type {
  CoppeliaEvidenceRun,
  CoppeliaEvidenceSummary,
  Metrics,
  Project,
  ResearchSummary
} from "../types";
import { ResultsView } from "./ResultsView";

beforeEach(() => {
  vi.spyOn(api, "evaluations").mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ResultsView evidence", () => {
  it("keeps real local evaluation records and artifacts distinct from research acceptance", async () => {
    vi.spyOn(api, "runtime").mockReturnValue(runtimeStatusFor("local"));
    const research = researchSummary();
    research.acceptance!.passed = false;
    research.acceptance!.results[0].passed = false;
    vi.spyOn(api, "researchSummary").mockResolvedValue(research);
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());
    vi.mocked(api.evaluations).mockResolvedValue([{
      id: "cpu-diagnostic", matrix_id: null, split: "diagnostic", artifact_dir: "/fixture", created_at: "2026-10-11T00:00:00Z",
      payload: { evaluation_id: "cpu-diagnostic", seeds: [800], controllers: ["mappo"], episodes: [{}], summaries: research.confidence_intervals, expected_split: null, grid_validation: null, per_training_seed: [], per_scenario_seed: [] }
    }, {
      id: "matrix-heldout", matrix_id: "matrix-1", split: "held_out", artifact_dir: "/fixture", created_at: "2026-10-11T00:00:00Z",
      payload: { evaluation_id: "matrix-heldout", seeds: [900], controllers: ["mappo"], episodes: [{}], summaries: [], expected_split: "held_out", grid_validation: null, per_training_seed: [], per_scenario_seed: [] }
    }]);
    vi.spyOn(api, "evaluationArtifacts").mockResolvedValue([{ label: "Evaluation report", href: "/api/lab/evaluations/cpu-diagnostic/artifacts/report.md", path: "report.md", media_type: "text/markdown" }]);
    render(<ResultsView project={project()} policies={[]} />);
    const record = await screen.findByRole("article", { name: "Local evaluation cpu-diagnostic" });
    expect(record).toHaveTextContent("Scenario seeds: 800 · 1 episodes · diagnostic");
    expect(record).toHaveTextContent("96.0% completion");
    expect(await screen.findByRole("link", { name: "Evaluation report" })).toHaveAttribute("href", "/api/lab/evaluations/cpu-diagnostic/artifacts/report.md");
    expect(screen.getByText(/unit evaluations do not satisfy the frozen research matrix/)).toBeInTheDocument();
    expect(screen.queryByRole("article", { name: "Local evaluation matrix-heldout" })).not.toBeInTheDocument();
    expect(screen.getByText("Frozen held-out acceptance did not pass.")).toBeInTheDocument();
  });

  it("does not request or display private local evaluations in static mode", async () => {
    vi.spyOn(api, "runtime").mockReturnValue(runtimeStatusFor("static"));
    vi.spyOn(api, "researchSummary").mockResolvedValue(researchSummary());
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());
    render(<ResultsView project={project()} policies={[]} />);
    await screen.findByText("Canonical, hash-verified release evidence.");
    expect(api.evaluations).not.toHaveBeenCalled();
    expect(screen.queryByRole("heading", { name: "Local evaluation records" })).not.toBeInTheDocument();
  });

  it("renders canonical research and independently validated simulator evidence", async () => {
    vi.spyOn(api, "researchSummary").mockResolvedValue(researchSummary());
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({
      reachable: false,
      host: "127.0.0.1",
      port: 23000,
      detail: "offline",
      controller: "dynamic_base_logical_payload"
    });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());

    render(<ResultsView project={project()} policies={[]} />);

    expect(
      await screen.findByText("Canonical, hash-verified release evidence.")
    ).toBeInTheDocument();
    expect(screen.getByText("96.0% completion", { exact: false })).toBeInTheDocument();
    expect(
      screen.getByText(
        "Nominal and unavailable-robot recovery manifests independently pass the live gate."
      )
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Primary learning curves" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Per-seed evidence" })).toBeInTheDocument();
    expect(screen.getAllByRole("cell", { name: "96.0%" })).toHaveLength(2);
    expect(screen.getByText("SUPPORTED")).toBeInTheDocument();

    expect(
      screen.getByRole("link", { name: /Held-out report/ })
    ).toHaveAttribute(
      "href",
      "/api/lab/evaluations/eval-1/artifacts/report.md"
    );
    expect(
      screen.getByLabelText("Nominal evidence measured Coppelia replay")
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Unavailable-robot recovery measured Coppelia replay")
    ).toBeInTheDocument();
    await waitFor(() => expect(api.researchSummary).toHaveBeenCalledTimes(1));
  });

  it("labels historical terminal and truncated rollouts without claiming completion", async () => {
    vi.spyOn(api, "researchSummary").mockResolvedValue(researchSummary());
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());
    render(<ResultsView project={project()} policies={[]} />);

    expect(await screen.findByRole("img", { name: /Terminal or truncated rollout fraction/ })).toBeInTheDocument();
    expect(screen.getByText(/they are not construction completion/)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /Rollout completion/ })).not.toBeInTheDocument();
  });

  it("plots actual module fraction rather than the terminal fraction when recorded", async () => {
    const summary = researchSummary();
    summary.learning_curves![0].points = [
      { transitions: 100, rollout_terminal_fraction: 1, rollout_structure_completion_rate: 0.2 },
      { transitions: 200, rollout_terminal_fraction: 1, rollout_structure_completion_rate: 0.4 }
    ];
    vi.spyOn(api, "researchSummary").mockResolvedValue(summary);
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());
    render(<ResultsView project={project()} policies={[]} />);

    const chart = await screen.findByRole("img", { name: /Installed-module fraction over transitions/ });
    expect(chart.querySelector("path")).toHaveAttribute("d", "M327.00,186.40 L604.00,144.80");
    expect(screen.getByText(/rollout endings are not used as completion/)).toBeInTheDocument();
  });

  it("separates validation completion from rollout diagnostics without relabeling either", async () => {
    const summary = researchSummary();
    const historical = summary.learning_curves![0];
    summary.learning_curves!.push({ ...historical, run_key: "validation-mappo-7", curve_kind: "validation_checkpoint_completion", points: [
      { transitions: 100, validation_completion_rate: 0.2, rollout_terminal_fraction: 1 },
      { transitions: 200, validation_completion_rate: 0.4, rollout_terminal_fraction: 1 }
    ] });
    vi.spyOn(api, "researchSummary").mockResolvedValue(summary);
    vi.spyOn(api, "coppeliaHealth").mockResolvedValue({ reachable: false, host: "127.0.0.1", port: 23000, detail: "offline", controller: "dynamic_base_logical_payload" });
    vi.spyOn(api, "coppeliaEvidence").mockResolvedValue(coppeliaSummary());
    render(<ResultsView project={project()} policies={[]} />);
    const validationChart = await screen.findByRole("img", { name: /Validation checkpoint completion over transitions/ });
    expect(validationChart.querySelectorAll("path")).toHaveLength(1);
    expect(validationChart.querySelector("path")).toHaveAttribute("d", "M327.00,186.40 L604.00,144.80");
    const historicalChart = screen.getByRole("img", { name: /Terminal or truncated rollout fraction/ });
    expect(historicalChart.querySelectorAll("path")).toHaveLength(1);
    expect(historicalChart.getAttribute("aria-labelledby")).not.toBe(validationChart.getAttribute("aria-labelledby"));
    const validationHeading = screen.getByRole("heading", { name: "Validation checkpoint completion" });
    const trainingHeading = screen.getByRole("heading", { name: "Primary learning curves" });
    expect(validationHeading.id).not.toBe(trainingHeading.id);
    expect(validationHeading.closest("section")).toHaveAttribute("aria-labelledby", validationHeading.id);
    expect(trainingHeading.closest("section")).toHaveAttribute("aria-labelledby", trainingHeading.id);
    expect(screen.getByText(/not training rollout endings or held-out release acceptance/)).toBeInTheDocument();
  });
});

function researchSummary(): ResearchSummary {
  const metric = {
    mean: 0.96,
    std: 0.01,
    bootstrap_ci95_low: 0.95,
    bootstrap_ci95_high: 0.98,
    median: 0.96
  };
  return {
    schema_version: "construction-intelligence-research-summary-v1",
    status: "validated",
    claim_allowed: true,
    matrix_id: "matrix-1",
    confidence_intervals: [
      {
        controller: "mappo",
        failure_enabled: false,
        episode_count: 25,
        metrics: {
          structure_completion_rate: metric,
          makespan_s: { ...metric, mean: 100, median: 100 }
        },
        experiment_id: "construction_intelligence_v1",
        experiment_variant: "mappo_full",
        training_seed_count: 5,
        scenario_seed_count: 5
      }
    ],
    per_training_seed: [
      {
        controller: "mappo",
        experiment_variant: "mappo_full",
        failure_enabled: false,
        training_seed: 7,
        metrics: {
          structure_completion_rate: metric,
          makespan_s: { ...metric, mean: 100 }
        }
      }
    ],
    per_scenario_seed: [
      {
        controller: "mappo",
        experiment_variant: "mappo_full",
        failure_enabled: false,
        scenario_seed: 900,
        metrics: {
          structure_completion_rate: metric,
          makespan_s: { ...metric, mean: 101 }
        }
      }
    ],
    learning_curves: [
      {
        run_key: "mappo-full-seed-7",
        controller: "mappo",
        experiment_variant: "mappo_full",
        training_seed: 7,
        status: "completed",
        points: [
          { transitions: 100, rollout_terminal_fraction: 0.5 },
          { transitions: 200, rollout_terminal_fraction: 0.96 }
        ]
      }
    ],
    acceptance: {
      passed: true,
      results: [
        {
          name: "mappo_no_failure_mean_completion",
          passed: true,
          observed: 0.96,
          threshold: 0.95,
          comparison: "min"
        }
      ]
    },
    ablations: [
      {
        hypothesis: "behavior_cloning",
        supported: true,
        completion_gain: 0.06,
        transitions_to_95_reduction: 0.21,
        failure_completion_gain: null,
        no_failure_completion_delta: null,
        completion_boundary_met: true,
        transition_boundary_met: true,
        failure_boundary_met: false,
        no_failure_safety_boundary_met: true,
        interpretation: "Behavior cloning met the pre-registered boundary."
      }
    ],
    artifact_references: [
      {
        label: "Held-out report",
        href: "/api/lab/evaluations/eval-1/artifacts/report.md",
        path: "report.md",
        media_type: "text/markdown"
      }
    ]
  };
}

function coppeliaSummary(): CoppeliaEvidenceSummary {
  return {
    schema_version: "construction-intelligence-coppelia-public-v1",
    status: "validated",
    ready: true,
    claim_allowed: false,
    payload_transport: "logical",
    source_commit: "1".repeat(40),
    nominal: coppeliaRun("nominal", "nominal-1"),
    recovery: coppeliaRun(
      "unavailable_robot_recovery",
      "recovery-1"
    ),
    limitation: "Payload transport is logical."
  };
}

function coppeliaRun(
  scenario: "nominal" | "unavailable_robot_recovery",
  runId: string
): CoppeliaEvidenceRun {
  return {
    manifest: {
      run_id: runId,
      evidence_kind: "live_coppelia",
      live_evidence: true,
      run_status: "completed",
      live_gate_passed: true,
      approval_gate_confirmed: true,
      scenario,
      scenario_seed: 900,
      source_commit: "1".repeat(40),
      payload_transport_model: "logical_carrier"
    },
    metrics: {
      status: "completed",
      live_gate_passed: true,
      expected_module_count: 12,
      installed_module_ids: Array.from({ length: 12 }, (_, index) => `m-${index}`),
      acceptance: {
        all_modules_installed: true,
        zero_post_start_pose_writes: true
      }
    },
    artifact_references: [
      {
        label: `${scenario} report`,
        href: `/api/lab/coppelia/evidence/${runId}/artifacts/report.md`,
        path: "report.md",
        media_type: "text/markdown"
      },
      {
        label: `${scenario} measured replay`,
        href: `/api/lab/coppelia/evidence/${runId}/artifacts/evidence_replay.mp4`,
        path: "evidence_replay.mp4",
        media_type: "video/mp4"
      }
    ]
  };
}

function project(): Project {
  const metric = (controller: string, makespan: number): Metrics => ({
    controller,
    makespan_s: makespan,
    total_travel_m: 42,
    total_energy_wh: 10,
    idle_robot_seconds: 8,
    robot_utilization: {},
    structure_completion_rate: 1,
    collision_count: 0,
    wasted_work_s: 0,
    recovery_cost_s: 0
  });
  return {
    design: {
      design_id: "fixture",
      title: "Fixture",
      footprint_width_m: 8,
      footprint_depth_m: 6,
      roof: { style: "gable", pitch_degrees: 25, overhang_m: 0.3 },
      wall_material: "timber",
      roof_material: "metal",
      level_count: 1,
      floor_plan: {
        approved: true,
        confidence: 1,
        warnings: [],
        walls: [],
        openings: [],
        rooms: []
      }
    },
    plan: { plan_id: "plan", modules: [], robots: [] },
    controllers: {
      sequential: metric("sequential", 200),
      greedy: metric("greedy", 150),
      optimized: metric("cp_sat", 100)
    },
    optimized_improvement_percent: 50,
    geometry_asset_url: null
  };
}
