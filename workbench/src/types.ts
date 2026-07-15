export type Vec3 = { x: number; y: number; z: number };
export type Pose = { position: Vec3; rotation_rpy_degrees: Vec3 };

export type BuildModule = {
  module_id: string;
  module_type: string;
  mesh_node: string;
  target_pose: Pose;
  staging_pose: Pose;
  dimensions: { width: number; depth: number; height: number };
  mass_kg: number;
  required_team_size: number;
  dependencies: string[];
  material: string;
};

export type Robot = {
  robot_id: string;
  role: string;
  payload_capacity_kg: number;
  speed_mps: number;
  start_pose: Pose;
};

export type Metrics = {
  controller: string;
  makespan_s: number;
  total_travel_m: number;
  total_energy_wh: number;
  idle_robot_seconds: number;
  robot_utilization: Record<string, number>;
};

export type Project = {
  design: {
    design_id: string;
    title: string;
    footprint_width_m: number;
    footprint_depth_m: number;
    roof: { style: "gable" | "hip" | "flat"; pitch_degrees: number; overhang_m: number };
    wall_material: string;
    roof_material: string;
    level_count: number;
    floor_plan: {
      approved: boolean;
      confidence: number;
      warnings: string[];
      walls: Array<{ wall_id: string; start: { x: number; y: number }; end: { x: number; y: number }; thickness_m: number; height_m: number }>;
      openings: Array<{ opening_id: string; wall_id: string; kind: "door" | "window"; offset_m: number; width_m: number; height_m: number; sill_height_m: number }>;
      rooms: Array<{ room_id: string; name: string; polygon: Array<{ x: number; y: number }> }>;
    };
  };
  plan: { plan_id: string; modules: BuildModule[]; robots: Robot[] };
  controllers: Record<string, Metrics>;
  optimized_improvement_percent: number;
  geometry_asset_url: string | null;
};

export type ScheduledJob = {
  module_id: string;
  robot_ids: string[];
  start_s: number;
  pickup_s: number;
  end_s: number;
  critical: boolean;
};

export type TraceFrame = {
  timestamp_s: number;
  completed_module_ids: string[];
  robots: Array<{ robot_id: string; position: Vec3; status: string; module_id: string | null }>;
  modules: Array<{ module_id: string; position: Vec3; status: string }>;
};

export type BrainEvent = {
  timestamp_s: number;
  event_type: string;
  module_id: string | null;
  robot_ids: string[];
  candidates: string[];
  reason: string;
  predicted_remaining_s: number;
};

export type Trace = {
  plan_id: string;
  schedule: {
    controller: string;
    jobs: ScheduledJob[];
    makespan_s: number;
    solver_status: string;
    critical_path: string[];
  };
  frames: TraceFrame[];
  brain_events: BrainEvent[];
  metrics: Metrics;
};
