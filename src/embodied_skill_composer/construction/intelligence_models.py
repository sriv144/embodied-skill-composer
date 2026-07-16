from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_skill_composer.construction.models import BuildPlan, Pose3D, Vec2


class ScenarioSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FailureKind(StrEnum):
    OBSTACLE = "obstacle"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    DROPPED_MODULE = "dropped_module"


class ConstructionFailure(BaseModel):
    failure_id: str
    kind: FailureKind
    trigger_time_s: float = Field(ge=0)
    duration_s: float = Field(default=15.0, gt=0)
    robot_id: str | None = None
    module_id: str | None = None
    obstacle_cell: tuple[int, int] | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "ConstructionFailure":
        if self.kind == FailureKind.ROBOT_UNAVAILABLE and not self.robot_id:
            raise ValueError("robot_unavailable failures require robot_id")
        if self.kind == FailureKind.OBSTACLE and self.obstacle_cell is None:
            raise ValueError("obstacle failures require obstacle_cell")
        return self


class ScenarioManifest(BaseModel):
    schema_version: Literal["construction-intelligence-v1"] = "construction-intelligence-v1"
    scenario_id: str
    seed: int = Field(ge=0, le=999)
    split: ScenarioSplit
    generator_version: str = "cottage-family-v1"
    plan: BuildPlan
    failures: list[ConstructionFailure] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "ScenarioManifest":
        robot_ids = {item.robot_id for item in self.plan.robots}
        module_ids = {item.module_id for item in self.plan.modules}
        for failure in self.failures:
            if failure.robot_id and failure.robot_id not in robot_ids:
                raise ValueError(f"failure references unknown robot: {failure.robot_id}")
            if failure.module_id and failure.module_id not in module_ids:
                raise ValueError(f"failure references unknown module: {failure.module_id}")
            if failure.obstacle_cell:
                x, y = failure.obstacle_cell
                if not 0 <= x < self.plan.site_grid.width or not 0 <= y < self.plan.site_grid.height:
                    raise ValueError(f"failure obstacle is out of bounds: {failure.obstacle_cell}")
        return self


class SkillDistribution(BaseModel):
    success_rate: float = Field(ge=0, le=1)
    duration_mean_s: float = Field(gt=0)
    duration_std_s: float = Field(default=0, ge=0)
    peak_force_mean_n: float = Field(default=0, ge=0)
    peak_force_std_n: float = Field(default=0, ge=0)
    alignment_error_mean_m: float = Field(default=0, ge=0)
    alignment_error_std_m: float = Field(default=0, ge=0)
    sample_count: int = Field(default=0, ge=0)


class SkillProfile(BaseModel):
    profile_id: str
    source_backend: Literal["fixture", "mujoco", "coppelia"]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_artifacts: list[str] = Field(default_factory=list)
    by_module_type: dict[str, SkillDistribution]
    notes: list[str] = Field(default_factory=list)


class DecisionCandidate(BaseModel):
    module_id: str
    score: float
    probability: float | None = Field(default=None, ge=0, le=1)
    selected: bool = False
    rejection_reason: str | None = None


class SwarmBrainEvent(BaseModel):
    timestamp_s: float = Field(ge=0)
    decision_index: int = Field(ge=0)
    controller: str
    robot_id: str
    selected_module_id: str | None = None
    candidates: list[DecisionCandidate] = Field(default_factory=list)
    action_probability: float | None = Field(default=None, ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0)
    predicted_remaining_s: float = Field(ge=0)
    reason: str


class RobotCommand(BaseModel):
    timestamp_s: float = Field(ge=0)
    robot_id: str
    linear_velocity_mps: float
    angular_velocity_rps: float
    wheel_target_velocity_rad_s: tuple[float, float, float, float]
    target_position: Vec2 | None = None
    source: Literal[
        "settling",
        "path_follower",
        "collision_stop",
        "formation_hold",
        "recovery",
    ]


class RobotTelemetry(BaseModel):
    timestamp_s: float = Field(ge=0)
    robot_id: str
    measured_pose: Pose3D
    linear_velocity_mps: float
    angular_velocity_rps: float
    battery_remaining_wh: float = Field(ge=0)
    collision_stop: bool = False
    attached_module_id: str | None = None


class PolicyManifest(BaseModel):
    policy_id: str
    controller: Literal["sequential", "greedy", "auction", "ippo", "mappo", "cp_sat"]
    environment_schema: str = "construction_coordination_v1"
    git_sha: str
    seed: int
    experiment_id: str = "ad_hoc"
    experiment_variant: str = "default"
    training_seed: int | None = None
    transition_count: int = Field(ge=0)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_lineage: list[str] = Field(default_factory=list)
    configuration_digest: str | None = None
    source_commit: str | None = None
    source_dirty: bool = False
    source_tree_digest: str | None = None
    resume_provenance: dict[str, object] = Field(default_factory=dict)
    environment_fingerprint: dict[str, object] = Field(default_factory=dict)
    onnx_path: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
