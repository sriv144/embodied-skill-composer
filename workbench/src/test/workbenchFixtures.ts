import type { Project, Trace } from "../types";

export function projectFixture(id = "original"): Project {
  return {
    design: {
      design_id: id, title: `Design ${id}`, footprint_width_m: 8, footprint_depth_m: 6,
      roof: { style: "gable", pitch_degrees: 25, overhang_m: 0.3 },
      wall_material: "timber", roof_material: "metal", level_count: 1,
      floor_plan: { approved: true, confidence: 1, warnings: [], walls: [], openings: [], rooms: [] }
    },
    plan: { plan_id: id, modules: [], robots: [] }, controllers: {},
    optimized_improvement_percent: 0, geometry_asset_url: null
  };
}

export function traceFixture(id = "original"): Trace {
  return {
    plan_id: id,
    schedule: { controller: "optimized", jobs: [], makespan_s: 10, solver_status: "fixture", critical_path: [] },
    frames: [{ timestamp_s: 0, completed_module_ids: [], robots: [], modules: [] }],
    brain_events: [{ timestamp_s: 0, event_type: "fixture", module_id: null, robot_ids: [], candidates: [], reason: "Fixture", predicted_remaining_s: 10 }],
    metrics: { controller: "optimized", makespan_s: 10, total_travel_m: 0, total_energy_wh: 0, idle_robot_seconds: 0, robot_utilization: {}, structure_completion_rate: 1, collision_count: 0, wasted_work_s: 0, recovery_cost_s: 0 }
  };
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}
