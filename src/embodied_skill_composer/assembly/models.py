from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


GridCoord = tuple[int, int]
Float3 = tuple[float, float, float]


class ScenePose(BaseModel):
    position_m: Float3
    rotation_rpy_degrees: Float3 = (0.0, 0.0, 0.0)


class AssetSource(BaseModel):
    title: str
    source_url: str
    license: str
    local_root: str | None = None
    redistributed: bool = True
    attribution: str | None = None


class AssetRobotSpec(BaseModel):
    source: str
    model_path: str
    role: str


class AssetComponentSpec(BaseModel):
    source: str
    visual_mesh: str
    collision_shape: Literal["box", "convex_hull"] = "box"
    mesh_scale: float = Field(default=1.0, gt=0.0)
    orientation_rpy_degrees: Float3 = (0.0, 0.0, 0.0)
    dimensions_m: Float3

    @model_validator(mode="after")
    def validate_dimensions(self) -> "AssetComponentSpec":
        if any(value <= 0.0 for value in self.dimensions_m):
            raise ValueError("asset component dimensions must be positive")
        return self


class AssetCatalog(BaseModel):
    version: int = Field(default=1, ge=1)
    sources: dict[str, AssetSource]
    robots: dict[str, AssetRobotSpec] = Field(default_factory=dict)
    components: dict[str, AssetComponentSpec]


class BlueprintMaterial(BaseModel):
    resource_id: str
    component_type: str
    asset_key: str
    source_cells: list[GridCoord] = Field(min_length=2, max_length=2)
    source_pose: ScenePose | None = None


class BlueprintComponent(BaseModel):
    component_id: str
    component_type: str
    asset_key: str
    required_material_id: str
    target_cells: list[GridCoord] = Field(min_length=2, max_length=2)
    target_pose: ScenePose
    depends_on: list[str] = Field(default_factory=list)
    required_team_size: int = Field(default=2, ge=1)


class ModularBlueprint(BaseModel):
    version: int = Field(default=1, ge=1)
    blueprint_id: str
    title: str
    grid_size: int = Field(default=24, ge=4)
    grid_scale_m: float = Field(default=0.35, gt=0.0)
    max_steps: int = Field(default=600, ge=1)
    option_max_primitive_steps: int = Field(default=64, ge=1)
    agent_starts: list[GridCoord] = Field(min_length=2, max_length=2)
    obstacle_cells: list[GridCoord] = Field(default_factory=list)
    materials: list[BlueprintMaterial] = Field(min_length=1)
    components: list[BlueprintComponent] = Field(min_length=1)


class BeamTask(BaseModel):
    name: str
    pickup_left: GridCoord
    pickup_right: GridCoord
    assembly_left: GridCoord
    assembly_right: GridCoord


class ConstructionResource(BaseModel):
    resource_id: str
    resource_type: str = "beam"
    source_cells: list[GridCoord]
    assigned_slot_id: str | None = None
    quantity: int = Field(default=1, ge=1)
    component_id: str | None = None
    asset_key: str | None = None
    source_pose: ScenePose | None = None
    assigned_robot_ids: list[int] = Field(default_factory=list)

    @classmethod
    def from_beam(cls, beam: BeamTask) -> "ConstructionResource":
        return cls(
            resource_id=beam.name,
            resource_type="beam",
            source_cells=[beam.pickup_left, beam.pickup_right],
            assigned_slot_id=f"{beam.name}_slot",
        )


class BlueprintSlot(BaseModel):
    slot_id: str
    resource_type: str = "beam"
    target_cells: list[GridCoord]
    required_resource_id: str | None = None
    component_id: str | None = None
    asset_key: str | None = None
    target_pose: ScenePose | None = None
    depends_on: list[str] = Field(default_factory=list)
    required_team_size: int = Field(default=2, ge=1)

    @classmethod
    def from_beam(cls, beam: BeamTask) -> "BlueprintSlot":
        return cls(
            slot_id=f"{beam.name}_slot",
            resource_type="beam",
            target_cells=[beam.assembly_left, beam.assembly_right],
            required_resource_id=beam.name,
        )


class ManipulationFailureRule(BaseModel):
    beam_name: str
    phase: Literal["grasp", "install"]
    fail_first_attempts: int = Field(default=1, ge=1)
    reason: str = "injected_manipulation_failure"


class ConstructionResourceState(ConstructionResource):
    delivered: bool = False


class BlueprintSlotState(BlueprintSlot):
    completed: bool = False


class ConstructionProgress(BaseModel):
    structure_completion_rate: float = 0.0
    resource_delivery_accuracy: float = 0.0
    energy_cost: float = 0.0
    idle_step_count: int = 0
    wasted_step_count: int = 0
    collision_count: int = 0
    obstacle_collision_count: int = 0
    manipulation_failure_count: int = 0
    manipulation_recovery_count: int = 0
    coordination_efficiency: float = 0.0


class AssemblyScenarioConfig(BaseModel):
    blueprint_id: str | None = None
    installation_order: list[str] = Field(default_factory=list)
    grid_size: int = 12
    max_steps: int = 120
    agent_starts: list[GridCoord] = Field(default_factory=lambda: [(0, 0), (0, 1)])
    beams: list[BeamTask]
    obstacle_cells: list[GridCoord] = Field(default_factory=list)
    resources: list[ConstructionResource] = Field(default_factory=list)
    blueprint_slots: list[BlueprintSlot] = Field(default_factory=list)
    manipulation_failures: list[ManipulationFailureRule] = Field(default_factory=list)
    collision_penalty: float = 0.2
    invalid_action_penalty: float = 0.1
    manipulation_failure_penalty: float = 0.1
    step_penalty: float = 0.01
    grasp_reward: float = 0.5
    install_reward: float = 1.5
    completion_reward: float = 5.0
    distance_shaping: float = 0.02
    second_beam_pickup_bonus: float = 0.05
    second_beam_install_bonus: float = 0.08
    curriculum_beam_stages: list[int] = Field(default_factory=lambda: [1, 2])
    curriculum_stage_beams: list[list[BeamTask]] = Field(default_factory=list)
    option_max_primitive_steps: int = 24

    @model_validator(mode="after")
    def validate_obstacle_geometry(self) -> "AssemblyScenarioConfig":
        obstacle_set = set(self.obstacle_cells)
        if len(obstacle_set) != len(self.obstacle_cells):
            raise ValueError("obstacle_cells must not contain duplicates")
        out_of_bounds = [
            cell
            for cell in self.obstacle_cells
            if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size)
        ]
        if out_of_bounds:
            raise ValueError(f"obstacle_cells outside grid: {out_of_bounds}")

        reserved = set(self.agent_starts)
        all_beams = list(self.beams)
        for stage in self.curriculum_stage_beams:
            all_beams.extend(stage)
        for beam in all_beams:
            reserved.update(
                [
                    beam.pickup_left,
                    beam.pickup_right,
                    beam.assembly_left,
                    beam.assembly_right,
                ]
            )
        overlap = sorted(obstacle_set & reserved)
        if overlap:
            raise ValueError(f"obstacle_cells overlap robot or beam task cells: {overlap}")

        beam_names = {beam.name for beam in all_beams}
        failure_keys: set[tuple[str, str]] = set()
        for rule in self.manipulation_failures:
            if rule.beam_name not in beam_names:
                raise ValueError(
                    f"manipulation failure references unknown beam: {rule.beam_name}"
                )
            key = (rule.beam_name, rule.phase)
            if key in failure_keys:
                raise ValueError(f"duplicate manipulation failure rule: {key}")
            failure_keys.add(key)
        return self

    def derived_resources(self) -> list[ConstructionResource]:
        if self.resources:
            return list(self.resources)
        return [ConstructionResource.from_beam(beam) for beam in self.beams]

    def derived_blueprint_slots(self) -> list[BlueprintSlot]:
        if self.blueprint_slots:
            return list(self.blueprint_slots)
        return [BlueprintSlot.from_beam(beam) for beam in self.beams]


class CompiledBlueprint(BaseModel):
    blueprint: ModularBlueprint
    scenario: AssemblyScenarioConfig
    installation_order: list[str]
    component_to_resource: dict[str, str]


class TrainingConfig(BaseModel):
    total_iterations: int = 20
    episodes_per_iteration: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 7
    behavior_cloning_epochs: int = 60
    behavior_cloning_lr: float = 1e-3
    scripted_mixing_start: float = 0.8
    scripted_mixing_end: float = 0.1
    evaluation_episodes: int = 5
    behavior_cloning_aux_coef: float = 0.2
    curriculum_stage_iterations: list[int] = Field(default_factory=lambda: [0, 6])
    option_actor_lr: float = 3e-4
    option_critic_lr: float = 1e-3
    option_behavior_cloning_epochs: int = 120
    option_behavior_cloning_lr: float = 8e-4
    option_update_epochs: int = 4
    option_entropy_coef: float = 0.01
    option_behavior_cloning_aux_coef: float = 0.1
    option_scripted_mixing_start: float = 0.7
    option_scripted_mixing_end: float = 0.05
    option_switch_penalty: float = 0.0
    option_recovery_limit: int = 3


class PhysicalSensorConfig(BaseModel):
    enabled: bool = False
    alignment_noise_std_m: float = Field(default=0.0, ge=0.0)
    force_noise_std_n: float = Field(default=0.0, ge=0.0)
    joint_position_noise_std_m: float = Field(default=0.0, ge=0.0)
    dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    ema_alpha: float = Field(default=1.0, gt=0.0, le=1.0)


class VisualPerceptionConfig(BaseModel):
    enabled: bool = False
    camera_name: str = "perception_cam"
    width: int = Field(default=256, ge=64, le=1024)
    height: int = Field(default=256, ge=64, le=1024)
    minimum_component_area_px: int = Field(default=30, ge=1)
    tracking_enabled: bool = False
    tracking_max_missed_frames: int = Field(default=32, ge=0, le=256)
    tracking_max_match_distance_m: float = Field(default=4.0, gt=0.0)
    prediction_confidence_decay: float = Field(default=0.9, gt=0.0, le=1.0)
    estimated_state_control_enabled: bool = False
    control_min_track_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    grasp_agent_resource_tolerance_m: float = Field(default=0.3, gt=0.0)
    install_agent_resource_tolerance_m: float = Field(default=0.3, gt=0.0)
    install_resource_blueprint_tolerance_m: float = Field(default=0.3, gt=0.0)


class CoppeliaSimConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=23000, ge=1, le=65535)
    executable_path: str = (
        "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe"
    )
    connection_timeout_s: float = Field(default=10.0, gt=0.0, le=30.0)
    grid_scale_m: float = Field(default=0.35, gt=0.0)
    control_steps_per_frame: int = Field(default=1, ge=1, le=100)
    camera_width: int = Field(default=512, ge=64, le=2048)
    camera_height: int = Field(default=512, ge=64, le=2048)
    rebuild_scene_on_connect: bool = True
    use_mujoco_physics: bool = True
    use_bundled_robot_model: bool = True
    robot_model_path: str = (
        "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/mobile/"
        "KUKA YouBot.ttm"
    )
    robot_model_scale: float = Field(default=0.55, gt=0.0, le=2.0)
    use_construction_meshes: bool = True
    asset_catalog_path: str = "configs/construction_asset_catalog.yaml"
    overview_camera_fov_degrees: float = Field(default=58.0, gt=10.0, lt=120.0)


class AssemblyRuntimeProfile(BaseModel):
    name: str = "local_dev"
    backend: Literal[
        "local_sandbox", "mujoco_local", "coppelia_sim", "isaac_lab"
    ] = "local_sandbox"
    device: str | None = None
    requires_linux: bool = False
    requires_nvidia_gpu: bool = False
    manipulation_alignment_tolerance_m: float = Field(default=0.03, gt=0.0)
    manipulation_min_grip_force_n: float = Field(default=25.0, ge=0.0)
    physical_sensors: PhysicalSensorConfig = Field(default_factory=PhysicalSensorConfig)
    visual_perception: VisualPerceptionConfig = Field(
        default_factory=VisualPerceptionConfig
    )
    coppelia: CoppeliaSimConfig = Field(default_factory=CoppeliaSimConfig)
    notes: str = ""


class BackendStatus(BaseModel):
    backend_name: str
    is_ready: bool
    readiness_notes: list[str] = Field(default_factory=list)


class GpuRuntimeStatus(BaseModel):
    runtime_profile: str
    backend: str
    requested_device: str | None
    torch_installed: bool
    cuda_available: bool
    selected_device: str
    device_name: str | None = None
    tensor_allocation_ok: bool = False
    notes: list[str] = Field(default_factory=list)


class TeamOption(IntEnum):
    GO_PICKUP = 0
    GRAB = 1
    GO_ASSEMBLY = 2
    INSTALL = 3
    RESET_TO_PICKUP_ROUTE = 4
    REPOSITION_AFTER_INSTALL = 5
    WAIT = 6
    ALIGN_FOR_TERMINAL_ACTION = 7


class OptionExecutionResult(BaseModel):
    option: TeamOption
    reward: float
    primitive_steps: int
    done: bool
    success: bool
    info: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class OptionTrainingSample(BaseModel):
    observation: list[float]
    action_mask: list[float]
    action: TeamOption
    stage_index: int


class OptionPolicyMetrics(BaseModel):
    success_rate: float
    mean_return: float
    mean_beams_installed: float
    mean_option_switches: float
    mean_recovery_usage: float
    mean_step_count: float = 0.0


class AssemblyPlaybackFrame(BaseModel):
    step_count: int
    current_beam_index: int
    current_beam_name: str | None = None
    carrying: bool
    agent_positions: list[GridCoord]
    pickup_targets: list[GridCoord] = Field(default_factory=list)
    assembly_targets: list[GridCoord] = Field(default_factory=list)
    selected_option: str | None = None
    primitive_step_index: int = 0
    option_reward: float = 0.0
    option_success: bool | None = None
    completed_beams: list[str] = Field(default_factory=list)
    completed_component_ids: list[str] = Field(default_factory=list)


class PolicyBenchmarkResult(BaseModel):
    policy_name: str
    success_rate: float
    mean_return: float
    mean_beams_installed: float
    mean_step_count: float = 0.0
    notes: str = ""


class AssemblyBenchmarkSummary(BaseModel):
    backend: str
    runtime_profile: str
    scripted_options: PolicyBenchmarkResult
    learned_options: PolicyBenchmarkResult
    low_level_learned: PolicyBenchmarkResult


class AssemblyMetrics(BaseModel):
    success: bool
    beams_installed: int
    total_beams: int
    step_count: int
    total_reward: float
    collision_count: int
    invalid_action_count: int
    deadlock_steps: int
    coordination_efficiency: float
    structure_completion_rate: float = 0.0
    resource_delivery_accuracy: float = 0.0
    energy_cost: float = 0.0
    idle_step_count: int = 0
    wasted_step_count: int = 0
    obstacle_collision_count: int = 0
    manipulation_failure_count: int = 0
    manipulation_recovery_count: int = 0


class EpisodeArtifact(BaseModel):
    metrics: AssemblyMetrics
    final_positions: list[GridCoord]
    carrying: bool
    completed_beams: list[str]
    policy_mode: Literal["scripted", "learned", "brain"]


class PhysicalManipulationFeedback(BaseModel):
    backend: str
    current_alignment_error_m: float | None = 0.0
    alignment_tolerance_m: float = 0.0
    required_minimum_grip_force_n: float = 0.0
    last_check_phase: Literal["grasp", "install"] | None = None
    last_check_passed: bool | None = None
    last_contact_forces_n: dict[str, float] = Field(default_factory=dict)
    active_attachment_beam: str | None = None
    gripper_state: Literal["open", "closed", "transitioning", "unknown"] = "open"
    gripper_joint_positions_m: dict[str, float] = Field(default_factory=dict)
    sensor_mode: Literal["privileged", "simulated"] = "privileged"
    sensor_fresh: bool = True
    sensor_dropped: bool = False
    sensor_age_physics_steps: int = Field(default=0, ge=0)
    sensor_sample_index: int = Field(default=0, ge=0)


class VisualObjectEstimate(BaseModel):
    track_id: str
    category: Literal["agent", "resource", "blueprint_cell"]
    centroid_px: tuple[float, float]
    position_m: tuple[float, float, float]
    bounding_box_xywh: tuple[int, int, int, int]
    pixel_area: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    is_predicted: bool = False
    track_age: int = Field(default=1, ge=1)
    missed_frames: int = Field(default=0, ge=0)


class VisualTerminalAssessment(BaseModel):
    phase: Literal["grasp", "install"]
    ready: bool = False
    reason: Literal[
        "ready",
        "insufficient_agent_tracks",
        "insufficient_resource_tracks",
        "agent_resource_misaligned",
        "insufficient_blueprint_tracks",
        "resource_blueprint_misaligned",
    ]
    agent_track_ids: list[str] = Field(default_factory=list)
    resource_track_id: str | None = None
    blueprint_track_ids: list[str] = Field(default_factory=list)
    max_agent_resource_distance_m: float | None = None
    max_resource_blueprint_distance_m: float | None = None
    minimum_track_confidence: float = 0.0
    uses_predicted_tracks: bool = False


class VisualPerceptionFeedback(BaseModel):
    camera_name: str
    resolution: tuple[int, int]
    sample_index: int = Field(ge=1)
    estimates: list[VisualObjectEstimate] = Field(default_factory=list)
    detected_counts: dict[str, int] = Field(default_factory=dict)
    tracked_counts: dict[str, int] = Field(default_factory=dict)
    predicted_estimate_count: int = 0
    mean_confidence: float = 0.0
    terminal_assessment: VisualTerminalAssessment | None = None


class VisualPerceptionEvaluation(BaseModel):
    sample_index: int = Field(ge=1)
    expected_counts: dict[str, int] = Field(default_factory=dict)
    detected_counts: dict[str, int] = Field(default_factory=dict)
    tracked_counts: dict[str, int] = Field(default_factory=dict)
    visible_matched_counts: dict[str, int] = Field(default_factory=dict)
    matched_counts: dict[str, int] = Field(default_factory=dict)
    visible_recall_by_category: dict[str, float] = Field(default_factory=dict)
    recall_by_category: dict[str, float] = Field(default_factory=dict)
    position_errors_m: list[float] = Field(default_factory=list)
    mean_position_error_m: float = 0.0
    max_position_error_m: float = 0.0


class ConstructionBrainObservation(BaseModel):
    backend: str
    step_count: int
    current_beam_index: int
    current_beam_name: str | None = None
    agent_positions: list[GridCoord]
    carrying: bool
    completed_beams: list[str] = Field(default_factory=list)
    obstacle_cells: list[GridCoord] = Field(default_factory=list)
    manipulation_attempts: dict[str, int] = Field(default_factory=dict)
    last_manipulation_failure: str | None = None
    resources: list[ConstructionResourceState] = Field(default_factory=list)
    blueprint_slots: list[BlueprintSlotState] = Field(default_factory=list)
    progress: ConstructionProgress = Field(default_factory=ConstructionProgress)
    available_options: list[TeamOption] = Field(default_factory=list)
    physical_feedback: PhysicalManipulationFeedback | None = None
    visual_feedback: VisualPerceptionFeedback | None = None


class ConstructionAssignment(BaseModel):
    resource_id: str
    slot_id: str
    beam_name: str | None = None
    estimated_cost: float = 0.0
    component_id: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    assigned_robot_ids: list[int] = Field(default_factory=list)
    status: Literal["pending", "active", "completed"] = "pending"


class ConstructionBrainDecision(BaseModel):
    option: TeamOption
    rationale: str
    assignment: ConstructionAssignment | None = None
    safety_hold_reason: Literal[
        "sensor_unavailable",
        "alignment_error",
        "visual_target_unavailable",
        "blocked_dependency",
    ] | None = None


class ConstructionBrainStep(BaseModel):
    decision_index: int
    observation: ConstructionBrainObservation
    decision: ConstructionBrainDecision
    execution: OptionExecutionResult


class ConstructionBrainEpisode(BaseModel):
    brain_name: str
    seed: int
    backend: str
    assignments: list[ConstructionAssignment] = Field(default_factory=list)
    steps: list[ConstructionBrainStep] = Field(default_factory=list)
    artifact: EpisodeArtifact
    diagnostics: dict[str, object] = Field(default_factory=dict)
