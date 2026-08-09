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
  install_duration_s: number;
  grip_points: Vec3[];
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
  structure_completion_rate: number;
  collision_count: number;
  wasted_work_s: number;
  recovery_cost_s: number;
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
  robot_asset_url?: string | null;
};

export type ScheduledJob = {
  module_id: string;
  robot_ids: string[];
  start_s: number;
  pickup_s: number;
  end_s: number;
  critical: boolean;
  route?: Array<{ x: number; y: number }>;
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

export type LabMode = "local" | "static";

export type RuntimeStatus = Readonly<{
  mode: LabMode;
  source: "loopback-api" | "versioned-static-demo";
  readOnly: boolean;
  interactive: boolean;
  label: string;
  detail: string;
}>;

export type LabRunStatus =
  | "queued"
  | "running"
  | "cancel_requested"
  | "interrupted"
  | "resuming"
  | "completed"
  | "failed"
  | "cancelled";

export type LabRunKind =
  | "training"
  | "evaluation"
  | "matrix_evaluation"
  | "coppelia";

export type ResumeProvenance = {
  checkpoint_path?: string | null;
  checkpoint_sha256?: string | null;
  resumed_from_transition?: number | null;
  source_attempt?: number | null;
  [key: string]: unknown;
};

export type LabRun = {
  id: string;
  kind: LabRunKind;
  status: LabRunStatus;
  config: Record<string, unknown>;
  input?: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  progress: number;
  artifact_dir: string | null;
  error: string | null;
  cancel_requested?: boolean;
  pid?: number | null;
  process_identity?: string | null;
  attempt?: number;
  heartbeat_at?: string | null;
  event_log_path?: string | null;
  latest_checkpoint?: string | null;
  config_digest?: string | null;
  source_commit?: string | null;
  resume_provenance_history?: ResumeProvenance[];
  claim_token?: string | null;
  interrupted_at?: string | null;
  can_cancel?: boolean;
  can_resume?: boolean;
};

export type LabPolicy = {
  id: string;
  controller: "sequential" | "greedy" | "auction" | "ippo" | "mappo" | "cp_sat";
  manifest: {
    policy_id?: string;
    controller?: "sequential" | "greedy" | "auction" | "ippo" | "mappo" | "cp_sat";
    environment_schema?: string;
    git_sha?: string;
    experiment_id?: string;
    experiment_variant?: string;
    training_seed?: number | null;
    transition_count?: number;
    seed?: number;
    checkpoint_path?: string;
    checkpoint_sha256?: string;
    checkpoint_lineage?: string[];
    configuration_digest?: string | null;
    source_commit?: string | null;
    source_dirty?: boolean;
    source_tree_digest?: string | null;
    resume_provenance?: ResumeProvenance;
    environment_fingerprint?: Record<string, unknown>;
    onnx_path?: string | null;
    config?: Record<string, unknown>;
  };
  created_at: string;
};

export type LabScenario = {
  id: string;
  seed: number | null;
  split: "fixture" | "reviewed" | "train" | "validation" | "test";
  payload: Record<string, unknown>;
  created_at: string;
};

export type CoppeliaHealth = {
  reachable: boolean;
  host: string;
  port: number;
  detail: string;
  controller: string;
};

export type ArtifactReference = {
  label: string;
  href: string;
  path: string;
  media_type: string;
};

export type DesignValidationIssue = {
  code: string;
  path: string;
  message: string;
};

export type DesignValidationResult = {
  valid: boolean;
  issues: DesignValidationIssue[];
};

export type LearningCurvePoint = {
  transitions: number;
  mean_episode_return?: number;
  rollout_terminal_fraction?: number;
  loss_objective?: number;
  loss_critic?: number;
  loss_entropy?: number;
};

export type LearningCurveSeries = {
  run_key: string;
  controller: "mappo" | "ippo";
  experiment_variant: string;
  training_seed: number;
  status: LabRunStatus;
  points: LearningCurvePoint[];
};

export type ExperimentProfile = "unit" | "smoke" | "research";
export type ExperimentAlgorithm = "mappo" | "ippo";
export type ExperimentRole = "primary" | "ablation";

export type ExperimentRunConfig = Record<string, unknown> & {
  algorithm?: ExperimentAlgorithm;
  profile?: ExperimentProfile;
  experiment_id?: string;
  experiment_variant?: string;
  training_seed?: number;
  seed?: number;
  transitions?: number;
  checkpoint_fractions?: number[];
  include_training_failures?: boolean;
  protocol_run_digest?: string;
};

export type ExperimentMatrixRun = LabRun & {
  run_key: string;
  ordinal: number;
  config: ExperimentRunConfig;
};

export type ExperimentMatrixStatus =
  | "queued"
  | "running"
  | "validating"
  | "selected"
  | "attention";

export type ExperimentMatrix = {
  id: string;
  protocol_digest: string;
  protocol: Record<string, unknown> & {
    experiment_id?: string;
    protocol_version?: string;
    training_seeds?: number[];
    checkpoint_fractions?: number[];
  };
  execution_profile: ExperimentProfile;
  expected_run_count: number;
  selection_count: number;
  status: ExperimentMatrixStatus;
  status_counts: Partial<Record<LabRunStatus, number>>;
  created_at: string;
  runs: ExperimentMatrixRun[];
};

export type MatrixLaunchResponse = {
  matrix_id: string;
  protocol_digest: string;
  profile: ExperimentProfile;
  run_count: number;
  run_ids: string[];
};

export type SelectedCheckpoint = {
  checkpoint_id: string;
  experiment_id: string;
  experiment_variant: string;
  training_seed: number;
  checkpoint_fraction: number;
  transition_count: number;
  split: "validation";
  scenario_seeds: number[];
  mean_completion_rate: number;
  mean_makespan_s: number;
  checkpoint_path: string;
  checkpoint_sha256: string;
  checkpoint_lineage: string[];
  configuration_digest: string;
  source_commit: string;
  resume_provenance: ResumeProvenance;
  selection_rule: string;
  required_checkpoint_fractions: number[];
  candidate_ranking: string[];
};

export type PolicySelectionRecord = {
  run_key: string;
  selection: SelectedCheckpoint;
  frozen_at: string;
};

export type ValidationSelectionResponse = {
  matrix_id: string;
  run_key: string;
  selection: SelectedCheckpoint;
  evidence_path: string;
  matrix_evidence_path: string | null;
};

export type HeldoutLaunchResponse = {
  matrix_id: string;
  selection_count: number;
  evaluation_run_count: number;
  run_ids: string[];
  selection_evidence_path: string;
};

export type MetricSummary = {
  mean: number;
  std: number;
  bootstrap_ci95_low: number;
  bootstrap_ci95_high: number;
  median: number;
};

export type ControllerEvaluation = {
  controller: "sequential" | "greedy" | "auction" | "ippo" | "mappo" | "cp_sat";
  failure_enabled: boolean;
  episode_count: number;
  metrics: Record<string, MetricSummary>;
  experiment_id: string | null;
  experiment_variant: string | null;
  training_seed_count: number;
  scenario_seed_count: number;
};

export type EvaluationSuite = {
  evaluation_id: string;
  seeds: number[];
  controllers: ControllerEvaluation["controller"][];
  episodes: Array<Record<string, unknown>>;
  summaries: ControllerEvaluation[];
  expected_split: string | null;
  grid_validation: {
    complete: true;
    expected_episode_count: number;
    observed_episode_count: number;
    expected_split: string | null;
  } | null;
  per_training_seed: Array<Record<string, unknown>>;
  per_scenario_seed: Array<Record<string, unknown>>;
};

export type LabEvaluation = {
  id: string;
  matrix_id: string | null;
  split: string;
  payload: EvaluationSuite;
  artifact_dir: string;
  created_at: string;
};

export type AcceptanceResult = {
  name: string;
  passed: boolean;
  observed: number;
  threshold: number;
  comparison: "min" | "max";
};

export type PrimaryAcceptanceAudit = {
  passed: boolean;
  results: AcceptanceResult[];
};

export type AblationDecision = {
  hypothesis: "behavior_cloning" | "failure_curriculum";
  supported: boolean;
  completion_gain?: number | null;
  transitions_to_95_reduction?: number | null;
  failure_completion_gain?: number | null;
  no_failure_completion_delta?: number | null;
  completion_boundary_met: boolean;
  transition_boundary_met: boolean;
  failure_boundary_met: boolean;
  no_failure_safety_boundary_met: boolean;
  interpretation: string;
};

export type ResearchSummary = {
  schema_version: "construction-intelligence-research-summary-v1";
  status: "absent" | "in_progress" | "local_complete" | "validated";
  claim_allowed: boolean;
  reason?: string;
  source_commit?: string | null;
  protocol_digest?: string | null;
  configuration_digests?: string[];
  matrix_id?: string;
  training_run_count?: number;
  selection_count?: number;
  heldout_episode_count?: number;
  learned_episode_count?: number;
  baseline_episode_count?: number;
  confidence_intervals: ControllerEvaluation[];
  per_training_seed: Array<Record<string, unknown>>;
  per_scenario_seed: Array<Record<string, unknown>>;
  learning_curves?: LearningCurveSeries[];
  acceptance: PrimaryAcceptanceAudit | null;
  ablations: AblationDecision[];
  artifact_references?: ArtifactReference[];
  reproducibility_audit?: Record<string, unknown>;
  release_completeness?: Record<string, unknown>;
  fidelity_boundary?: string;
};

export type CoppeliaEvidenceRun = {
  label?: string;
  manifest: {
    run_id: string;
    evidence_kind: string;
    live_evidence: boolean;
    run_status: string;
    live_gate_passed: boolean;
    approval_gate_confirmed: boolean;
    scenario: "nominal" | "unavailable_robot_recovery";
    scenario_seed: number;
    source_commit: string;
    simulator_version?: string | null;
    payload_transport_model: "logical_carrier";
    [key: string]: unknown;
  };
  metrics: {
    status: string;
    live_gate_passed: boolean;
    expected_module_count?: number;
    installed_module_ids?: string[];
    acceptance?: Record<string, boolean>;
    recovery?: Record<string, unknown> | null;
    metrics?: Record<string, unknown>;
    [key: string]: unknown;
  };
  artifact_references?: ArtifactReference[];
};

export type CoppeliaEvidenceSummary = {
  schema_version: "construction-intelligence-coppelia-public-v1";
  status: "absent" | "validated";
  ready?: boolean;
  claim_allowed: boolean;
  payload_transport: "logical";
  source_commit?: string;
  reason?: string;
  verified_scenarios?: string[];
  nominal: CoppeliaEvidenceRun | null;
  recovery: CoppeliaEvidenceRun | null;
  limitation: string;
};

export type RunEventPayload =
  | {
      event: "run_created";
      status: LabRunStatus;
      matrix_id?: string | null;
      run_key?: string | null;
    }
  | { event: "run_claimed"; attempt: number }
  | {
      event: "run_interrupted";
      reason: "stale_worker" | "stale_worker_pid_reused" | "stale_worker_terminated";
    }
  | { event: "cancel_requested" | "cancelled"; error?: string | null }
  | { event: "failed"; error: string }
  | {
      event: "resume_requested";
      checkpoint?: string | null;
      restart_from_beginning?: boolean | null;
    }
  | { event: "worker_started"; pid: number; process_log: string }
  | { event: "worker_launch_failed"; error: string }
  | { event: "worker_exited"; return_code: number }
  | {
      event: "training_started";
      mode: "inline" | "subprocess";
      attempt?: number | null;
      pid?: number | null;
    }
  | {
      event: "training_resumed";
      transitions: number;
      checkpoint_path: string;
      updates: number;
      episode_cursor: number;
    }
  | {
      event: "checkpoint_saved" | "behavior_cloning_checkpoint";
      transitions: number;
      checkpoint_path: string;
      snapshot_path: string;
      policy_checkpoint_path: string | null;
      fraction?: number | null;
      target_transitions?: number | null;
    }
  | {
      event: "behavior_cloning_complete";
      transitions: number;
      loss: number;
      epoch: number;
    }
  | {
      event: "ppo_update";
      transitions: number;
      update: number;
      loss_objective: number;
      loss_critic: number;
      loss_entropy: number;
      mean_episode_return: number;
      rollout_terminal_fraction: number;
    }
  | { event: "training_completed" | "evaluation_completed"; artifacts: Record<string, unknown> }
  | { event: "evaluation_started" }
  | {
      event: "matrix_evaluation_started";
      mode: "subprocess";
      attempt: number;
      pid: number;
      matrix_id: string;
    }
  | {
      event: "matrix_evaluation_completed";
      matrix_id: string;
      evaluation_id: string;
      acceptance: PrimaryAcceptanceAudit;
      ablations: AblationDecision[];
    }
  | {
      event: "selection_evidence_unavailable";
      path: string;
      effect: "transitions_to_95_ablation_evidence_omitted";
    };

export type RunEventEnvelope = {
  sequence: number;
  created_at: string;
  payload: RunEventPayload;
};

export type MatrixEvaluationCompletedEvent = Extract<
  RunEventPayload,
  { event: "matrix_evaluation_completed" }
>;
