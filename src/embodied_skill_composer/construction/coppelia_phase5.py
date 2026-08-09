from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path
from typing import Literal
from uuid import UUID
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
    DynamicCoppeliaExecutor,
    validate_coppelia_scene_buffer,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
    ScenarioManifest,
)
from embodied_skill_composer.construction.models import (
    BuildPlan,
    BuildModule,
    Pose3D,
    ScheduledJob,
    SiteGrid,
    Vec2,
    Vec3,
)
from embodied_skill_composer.construction.routing import (
    RoutePlan,
    RoutingAdapter,
    RoutingError,
    create_routing_adapter,
    world_to_cell,
)
from embodied_skill_composer.construction.scheduler import schedule_build


Phase5Scenario = Literal["nominal", "unavailable_robot_recovery"]
EvidenceKind = Literal["offline_harness", "live_coppelia"]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40,64}$"
_UUID_PATTERN = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_REQUIRED_ARTIFACTS = {
    "scenario.json",
    "planned_jobs.json",
    "planned_vs_measured_replay.json",
    "wheel_commands.jsonl",
    "measured_telemetry.jsonl",
    "trace.json",
    "metrics.json",
    "report.md",
    "construction_intelligence.ttt",
}
_LIVE_RUNTIME_ATTESTATION_ARTIFACT = "runtime_attestation.json"
_LIVE_REQUIRED_REMOTE_API_CAPABILITIES = (
    "getObjectUid",
    "getSimulationState",
    "getSimulationTime",
    "readCustomDataBlock",
    "saveScene",
    "writeCustomDataBlock",
)
_PHASE5_PREFLIGHT_CACHE: dict[str, tuple[float, float]] = {}
_PHASE5_ROUTE_SYNC_CACHE: dict[
    tuple[
        float,
        tuple[tuple[float, float], ...],
        tuple[tuple[float, float], ...],
    ],
    tuple[tuple[int, int], ...],
] = {}
_PHASE5_ROUTE_SYNC_STATE_LIMIT = 500_000

ClearancePhase = Literal["approach", "carry", "return"]
ClearanceSource = Literal["planned", "measured"]
ClearanceMover = Literal["robot_base", "logical_payload"]
ClearanceObstacle = Literal[
    "installed_module",
    "staged_module",
    "site_obstacle",
    "idle_robot",
    "disabled_robot",
    "active_payload",
]
_WORLD_CLEARANCE_OBSTACLES: tuple[ClearanceObstacle, ...] = (
    "installed_module",
    "staged_module",
    "site_obstacle",
    "idle_robot",
    "disabled_robot",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase5RecoveryRecord(_StrictModel):
    robot_id: str
    disabled_at_s: float = Field(ge=0)
    completion_fraction_at_disable: float = Field(ge=0, le=1)
    stop_command_index: int = Field(ge=1)
    remaining_module_count: int = Field(ge=1)
    reassigned_job_ids: list[str] = Field(default_factory=list)
    commands_after_stop: int = Field(default=0, ge=0)
    settled_at_s: float = Field(ge=0)
    settle_sample_count: int = Field(ge=2)
    settle_displacement_m: float = Field(ge=0)
    settled: bool


class Phase5ClearanceMinimum(_StrictModel):
    phase: ClearancePhase
    source: ClearanceSource
    mover_kind: ClearanceMover
    obstacle_kind: ClearanceObstacle
    minimum_surface_clearance_m: float | None = Field(default=None, ge=0)
    required_clearance_m: float = Field(gt=0)
    evaluated_pair_count: int = Field(ge=0)
    limiting_mover_id: str | None = None
    limiting_obstacle_id: str | None = None

    @model_validator(mode="after")
    def validate_clearance_inventory(self) -> Phase5ClearanceMinimum:
        has_minimum = self.minimum_surface_clearance_m is not None
        if has_minimum != (self.evaluated_pair_count > 0):
            raise ValueError(
                "clearance minimum must exist exactly when pairs were evaluated"
            )
        if has_minimum and (
            not self.limiting_mover_id or not self.limiting_obstacle_id
        ):
            raise ValueError("clearance minimum lacks a limiting pair")
        if not has_minimum and (
            self.limiting_mover_id is not None
            or self.limiting_obstacle_id is not None
        ):
            raise ValueError("empty clearance inventory names a limiting pair")
        return self


class Phase5InstallationSnap(_StrictModel):
    from_pose: Pose3D
    target_pose: Pose3D
    at_s: float = Field(ge=0)
    target_pose_digest: str = Field(pattern=_SHA256_PATTERN)
    contact_module_ids: list[str] = Field(default_factory=list)
    scope: Literal["final_target_pose_only"] = "final_target_pose_only"


class Phase5JobReplay(_StrictModel):
    job_id: str
    module_id: str
    planned_order: int = Field(ge=0)
    planned_robot_ids: list[str]
    executed_robot_ids: list[str]
    planned_start_s: int = Field(ge=0)
    planned_pickup_s: int = Field(ge=0)
    planned_end_s: int = Field(ge=0)
    approach_routes: dict[str, list[Vec2]]
    carry_route: list[Vec2]
    assigned_carry_routes: dict[str, list[Vec2]]
    return_routes: dict[str, list[Vec2]]
    formation_offsets: dict[str, Vec2]
    logical_carrier_offset: Vec3
    logical_transport_height_m: float = Field(gt=0)
    measured_carry_routes: dict[str, list[Vec2]]
    measured_carry_samples: dict[str, list[RobotTelemetry]]
    approach_robot_positions: dict[str, Vec3]
    carry_robot_positions: dict[str, Vec3]
    return_robot_positions: dict[str, Vec3]
    clearance_minima: list[Phase5ClearanceMinimum]
    installation_snap: Phase5InstallationSnap
    started_at_s: float = Field(ge=0)
    pickup_at_s: float = Field(ge=0)
    installed_at_s: float = Field(ge=0)
    returned_at_s: float = Field(ge=0)
    measured_poses: list[RobotTelemetry]
    maximum_formation_error_m: float = Field(ge=0)
    maximum_assignment_error_m: float = Field(ge=0)
    maximum_spacing_error_m: float = Field(ge=0)
    maximum_install_error_m: float = Field(ge=0)
    minimum_pickup_base_clearance_m: float = Field(ge=0)
    minimum_install_base_clearance_m: float = Field(ge=0)
    minimum_planned_payload_clearance_m: float = Field(ge=0)
    minimum_measured_payload_clearance_m: float = Field(ge=0)
    maximum_synchronized_formation_error_m: float = Field(ge=0)
    reassigned_after_unavailability: bool = False

    @model_validator(mode="after")
    def validate_measured_replay(self) -> Phase5JobReplay:
        if not (
            self.planned_start_s
            <= self.planned_pickup_s
            <= self.planned_end_s
        ):
            raise ValueError("planned job timestamps are not monotonic")
        if not (
            self.started_at_s
            <= self.pickup_at_s
            <= self.installed_at_s
            <= self.returned_at_s
        ):
            raise ValueError("measured job timestamps are not monotonic")
        expected = set(self.executed_robot_ids)
        if (
            not expected
            or set(self.approach_routes) != expected
            or set(self.assigned_carry_routes) != expected
            or set(self.return_routes) != expected
            or set(self.formation_offsets) != expected
            or set(self.measured_carry_routes) != expected
            or set(self.measured_carry_samples) != expected
        ):
            raise ValueError("replay route identities must match the executed team")
        if not self.carry_route or any(
            len(route) != len(self.carry_route)
            for route in self.assigned_carry_routes.values()
        ):
            raise ValueError(
                "assigned carry routes must share the carrier-route horizon"
            )
        for robot_id in self.executed_robot_ids:
            offset = self.formation_offsets[robot_id]
            route = self.assigned_carry_routes[robot_id]
            for carrier, assigned in zip(
                self.carry_route,
                route,
                strict=True,
            ):
                if not (
                    math.isclose(
                        assigned.x,
                        carrier.x
                        - self.logical_carrier_offset.x
                        + offset.x,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        assigned.y,
                        carrier.y
                        - self.logical_carrier_offset.y
                        + offset.y,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        "base carry route is not rigidly derived from the carrier"
                    )
        for robot_id in self.executed_robot_ids:
            samples = self.measured_carry_samples[robot_id]
            route = self.measured_carry_routes[robot_id]
            if not samples or len(samples) != len(route):
                raise ValueError("measured carry routes require one point per sample")
            if any(sample.robot_id != robot_id for sample in samples):
                raise ValueError("measured carry sample identity mismatch")
            if any(
                sample.timestamp_s < self.pickup_at_s
                or sample.timestamp_s > self.installed_at_s
                for sample in samples
            ):
                raise ValueError("measured carry sample is outside the job interval")
            if any(
                route[index].x != sample.measured_pose.position.x
                or route[index].y != sample.measured_pose.position.y
                for index, sample in enumerate(samples)
            ):
                raise ValueError("measured carry route does not match telemetry")
        if len(
            {
                len(route)
                for route in self.measured_carry_routes.values()
            }
        ) != 1:
            raise ValueError(
                "measured synchronized routes must share one horizon"
            )
        return self


class Phase5PlannedJobRecord(_StrictModel):
    job_id: str
    module_id: str
    planned_order: int = Field(ge=0)
    planned_robot_ids: list[str]
    executed_robot_ids: list[str]
    planned_start_s: int = Field(ge=0)
    planned_pickup_s: int = Field(ge=0)
    planned_end_s: int = Field(ge=0)
    approach_routes: dict[str, list[Vec2]]
    carry_route: list[Vec2]
    assigned_carry_routes: dict[str, list[Vec2]]
    return_routes: dict[str, list[Vec2]]
    formation_offsets: dict[str, Vec2]
    logical_carrier_offset: Vec3
    logical_transport_height_m: float = Field(gt=0)
    measured_carry_routes: dict[str, list[Vec2]]
    approach_robot_positions: dict[str, Vec3]
    carry_robot_positions: dict[str, Vec3]
    return_robot_positions: dict[str, Vec3]
    clearance_minima: list[Phase5ClearanceMinimum]
    installation_snap: Phase5InstallationSnap
    returned_at_s: float = Field(ge=0)
    minimum_planned_payload_clearance_m: float = Field(ge=0)
    minimum_measured_payload_clearance_m: float = Field(ge=0)


class Phase5RunResult(_StrictModel):
    schema_version: Literal["construction_intelligence.coppelia_evidence.v1"] = (
        "construction_intelligence.coppelia_evidence.v1"
    )
    scenario: Phase5Scenario
    evidence_kind: EvidenceKind
    status: Literal["completed", "failed"]
    scenario_id: str
    scenario_seed: int
    expected_module_count: int = Field(ge=1)
    installed_module_ids: list[str]
    active_robot_ids: list[str]
    replay: list[Phase5JobReplay]
    recovery: Phase5RecoveryRecord | None = None
    trace: list[dict[str, object]] = Field(default_factory=list)
    metrics: dict[str, object]
    acceptance: dict[str, bool]
    live_gate_passed: bool
    error_type: str | None = None
    error: str | None = None


class Phase5ArtifactRecord(_StrictModel):
    path: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    bytes: int = Field(gt=0)


class Phase5ProvenanceManifest(_StrictModel):
    schema_version: Literal["construction_intelligence.coppelia_bundle.v1"] = (
        "construction_intelligence.coppelia_bundle.v1"
    )
    run_id: str
    generated_at: str
    evidence_kind: EvidenceKind
    live_evidence: bool
    run_status: Literal["completed", "failed"]
    live_gate_passed: bool
    approval_gate_confirmed: bool
    scenario: Phase5Scenario
    scenario_id: str
    scenario_seed: int
    source_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    source_dirty: bool
    source_tree_digest: str = Field(pattern=_SHA256_PATTERN)
    plan_digest: str = Field(pattern=_SHA256_PATTERN)
    configuration_digest: str = Field(pattern=_SHA256_PATTERN)
    simulator_version: str | None = None
    runtime_attestation_digest: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    payload_transport_model: Literal["logical_carrier"] = "logical_carrier"
    limitations: list[str]
    artifacts: list[Phase5ArtifactRecord]

    @model_validator(mode="after")
    def validate_live_attestation(self) -> Phase5ProvenanceManifest:
        if self.live_evidence != (self.evidence_kind == "live_coppelia"):
            raise ValueError("live-evidence flag conflicts with the evidence kind")
        if self.live_gate_passed and (
            not self.live_evidence
            or self.run_status != "completed"
            or not self.approval_gate_confirmed
            or not self.simulator_version
            or not self.runtime_attestation_digest
        ):
            raise ValueError("live gate lacks a complete simulator attestation")
        if self.live_evidence != bool(self.runtime_attestation_digest):
            raise ValueError(
                "live evidence requires exactly one runtime attestation"
            )
        return self


class Phase5RuntimeAttestation(_StrictModel):
    schema_version: Literal[
        "construction_intelligence.coppelia_runtime_attestation.v1"
    ] = "construction_intelligence.coppelia_runtime_attestation.v1"
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_seed: int
    source_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    source_tree_digest: str = Field(pattern=_SHA256_PATTERN)
    plan_digest: str = Field(pattern=_SHA256_PATTERN)
    configuration_digest: str = Field(pattern=_SHA256_PATTERN)
    configuration_origin_digest: str = Field(pattern=_SHA256_PATTERN)
    transport: Literal["coppeliasim_zmq_remote_api"] = (
        "coppeliasim_zmq_remote_api"
    )
    client_implementation: Literal[
        "coppeliasim_zmqremoteapi_client.RemoteAPIClient"
    ] = "coppeliasim_zmqremoteapi_client.RemoteAPIClient"
    remote_api_client_version: str = Field(min_length=1)
    remote_api_protocol_version: int = Field(ge=1)
    client_uuid: str = Field(pattern=_UUID_PATTERN)
    client_send_count_before: int = Field(ge=0)
    client_send_count_after: int = Field(gt=0)
    endpoint_host: str = Field(min_length=1)
    endpoint_port: int = Field(ge=1, le=65_535)
    remote_api_capabilities: list[str]
    remote_api_info_sha256: str = Field(pattern=_SHA256_PATTERN)
    simulator_version: str = Field(min_length=1)
    simulator_identity_source: Literal[
        "string_parameter",
        "integer_parameters",
    ]
    simulator_program_version: int | None = Field(default=None, gt=0)
    simulator_program_revision: int | None = Field(default=None, ge=0)
    scene_root_handle: int
    scene_root_uid: int = Field(gt=0)
    simulation_stopped_state: int
    simulation_state_before: int
    simulation_state_after: int
    simulation_time_before_s: float = Field(ge=0)
    simulation_time_after_s: float = Field(ge=0)
    session_nonce: str = Field(pattern=_SHA256_PATTERN)
    challenge_tag: str = Field(min_length=1)
    challenge_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_response_sha256: str = Field(pattern=_SHA256_PATTERN)
    physics_steps: int = Field(ge=0)
    command_count: int = Field(ge=0)
    telemetry_count: int = Field(ge=0)
    command_stream_sha256: str = Field(pattern=_SHA256_PATTERN)
    telemetry_stream_sha256: str = Field(pattern=_SHA256_PATTERN)
    installed_module_ids: list[str]
    artifact_sha256: dict[str, str]
    binding_digest: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_runtime_binding(self) -> Phase5RuntimeAttestation:
        try:
            parsed_uuid = UUID(self.client_uuid)
        except ValueError as exc:
            raise ValueError("runtime client UUID is invalid") from exc
        if str(parsed_uuid) != self.client_uuid:
            raise ValueError("runtime client UUID is not canonical")
        if self.client_send_count_after <= self.client_send_count_before:
            raise ValueError("runtime attestation did not observe remote calls")
        if (
            self.simulation_state_before != self.simulation_stopped_state
            or self.simulation_state_after != self.simulation_stopped_state
        ):
            raise ValueError(
                "runtime attestation must bracket a stopped simulator"
            )
        if self.remote_api_capabilities != list(
            _LIVE_REQUIRED_REMOTE_API_CAPABILITIES
        ):
            raise ValueError("runtime attestation capability inventory differs")
        if set(self.artifact_sha256) != _REQUIRED_ARTIFACTS or any(
            re.fullmatch(_SHA256_PATTERN, digest) is None
            for digest in self.artifact_sha256.values()
        ):
            raise ValueError("runtime attestation artifact binding is incomplete")
        if self.challenge_payload_sha256 != self.challenge_response_sha256:
            raise ValueError("simulator challenge response differs from request")
        expected_challenge = _sha256_json(
            _runtime_challenge_payload(
                session_nonce=self.session_nonce,
                client_uuid=self.client_uuid,
                endpoint_host=self.endpoint_host,
                endpoint_port=self.endpoint_port,
                plan_digest=self.plan_digest,
                configuration_origin_digest=self.configuration_origin_digest,
                scene_root_uid=self.scene_root_uid,
                remote_api_info_sha256=self.remote_api_info_sha256,
                simulator_version=self.simulator_version,
            )
        )
        if self.challenge_payload_sha256 != expected_challenge:
            raise ValueError("runtime challenge is not bound to its origin fields")
        if self.simulator_identity_source == "string_parameter":
            if (
                self.simulator_program_version is not None
                or self.simulator_program_revision is not None
            ):
                raise ValueError(
                    "string simulator identity cannot include integer identity"
                )
        elif (
            self.simulator_program_version is None
            or self.simulator_program_revision is None
            or self.simulator_version
            != _format_simulator_integer_version(
                self.simulator_program_version,
                self.simulator_program_revision,
            )
        ):
            raise ValueError("integer simulator identity is inconsistent")
        expected_binding = _sha256_json(
            self.model_dump(mode="json", exclude={"binding_digest"})
        )
        if self.binding_digest != expected_binding:
            raise ValueError("runtime attestation binding digest differs")
        return self


@dataclass(frozen=True)
class _OfficialRemoteClientIdentity:
    client_uuid: str
    protocol_version: int
    package_version: str
    send_count: int


@dataclass
class _LiveRuntimeSession:
    scenario_id: str
    scenario_seed: int
    plan_digest: str
    configuration_origin_digest: str
    client_uuid: str
    protocol_version: int
    package_version: str
    client_send_count_before: int
    endpoint_host: str
    endpoint_port: int
    remote_api_info_sha256: str
    simulator_version: str
    simulator_identity_source: Literal[
        "string_parameter",
        "integer_parameters",
    ]
    simulator_program_version: int | None
    simulator_program_revision: int | None
    scene_root_handle: int
    scene_root_uid: int
    simulation_stopped_state: int
    simulation_state_before: int
    simulation_time_before_s: float
    session_nonce: str
    challenge_tag: str
    challenge_payload: bytes
    challenge_payload_sha256: str
    simulation_state_after: int | None = None
    simulation_time_after_s: float | None = None
    client_send_count_after: int | None = None
    physics_steps: int | None = None
    command_count: int | None = None
    telemetry_count: int | None = None
    command_stream_sha256: str | None = None
    telemetry_stream_sha256: str | None = None
    installed_module_ids: list[str] | None = None
    finalized: bool = False
    consumed: bool = False


_LIVE_RUNTIME_SESSIONS: WeakKeyDictionary[
    DynamicCoppeliaExecutor,
    _LiveRuntimeSession,
] = WeakKeyDictionary()


class Phase5PhysicalYardConfig(_StrictModel):
    schema_version: Literal["construction_intelligence.phase5_yard_config.v1"] = (
        "construction_intelligence.phase5_yard_config.v1"
    )
    allocator: Literal["deterministic_shelf_yard"] = (
        "deterministic_shelf_yard"
    )
    grid_resolution_m: float = Field(default=0.5, gt=0)
    yard_shelf_width_m: float = Field(default=14.0, ge=8.0)
    module_gap_m: float = Field(default=4.0, ge=0.8)
    yard_house_aisle_m: float = Field(default=2.0, ge=1.0)
    boundary_margin_m: float = Field(default=8.0, ge=1.0)
    dispatch_lane_gap_m: float = Field(default=1.5, ge=0.8)
    dispatch_robot_spacing_m: float = Field(default=1.0, ge=0.6)
    robot_footprint_radius_m: float = Field(default=0.12, ge=0.08)
    route_clearance_m: float = Field(default=0.16, ge=0.1)
    formation_clearance_m: float = Field(default=0.5, ge=0.3)
    maximum_formation_expansion_m: float = Field(
        default=4.0,
        ge=0,
        le=8,
    )
    carry_sample_spacing_m: float = Field(default=0.1, gt=0, le=0.25)
    site_obstacle_height_m: float = Field(default=0.6, gt=0)
    maximum_logical_transport_height_m: float = Field(default=12.0, gt=1)
    payload_robot_clearance_model: Literal["infinite_vertical_column"] = (
        "infinite_vertical_column"
    )
    clearance_model: Literal["piecewise_linear_swept_aabb3_v1"] = (
        "piecewise_linear_swept_aabb3_v1"
    )
    phase5_team_size: Literal[2] = 2


class Phase5PhysicalYardManifest(_StrictModel):
    schema_version: Literal["construction_intelligence.phase5_yard.v1"] = (
        "construction_intelligence.phase5_yard.v1"
    )
    source_plan_digest: str = Field(pattern=_SHA256_PATTERN)
    transformed_plan_digest: str = Field(pattern=_SHA256_PATTERN)
    allocator_configuration_digest: str = Field(pattern=_SHA256_PATTERN)
    configuration: Phase5PhysicalYardConfig
    shelf_row_count: int = Field(ge=1)
    staging_bounds_xy: tuple[float, float, float, float]
    grid_origin: Vec2
    grid_width: int = Field(gt=0)
    grid_height: int = Field(gt=0)
    module_count: int = Field(gt=0)
    robot_count: int = Field(gt=1)
    obstacle_count: int = Field(ge=1)
    initial_clearance_preflight_passed: bool
    sequential_route_preflight_passed: bool
    all_robots_active_preflight_passed: bool
    continuous_world_clearance_preflight_passed: bool
    recovery_route_preflight_passed: bool
    maximum_preflight_formation_offset_m: float = Field(ge=0)
    maximum_preflight_transport_height_m: float = Field(ge=0)


@dataclass(frozen=True)
class _Aabb3:
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float
    minimum_z: float
    maximum_z: float

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        return (
            self.minimum_x,
            self.maximum_x,
            self.minimum_y,
            self.maximum_y,
        )


def prepare_phase5_physical_yard(
    scenario: ScenarioManifest,
    *,
    config: Phase5PhysicalYardConfig | None = None,
) -> tuple[ScenarioManifest, Phase5PhysicalYardManifest]:
    """Repack a deep-copied plan into a collision-clear, routeable physical yard."""
    config = config or Phase5PhysicalYardConfig()
    transformed = scenario.model_copy(deep=True)
    plan = transformed.plan
    source_plan_digest = _sha256_json(plan.model_dump(mode="json"))
    obstacle_count = max(1, len(plan.site_grid.obstacle_cells))

    packed, row_count, local_height = _pack_staging_shelves(
        plan.modules,
        config,
    )
    target_bounds = _combined_target_bounds(plan.modules)
    yard_right = target_bounds[0] - config.yard_house_aisle_m
    yard_left = yard_right - config.yard_shelf_width_m
    yard_bottom = -local_height / 2
    for module in plan.modules:
        local_x, local_y = packed[module.module_id]
        target_rotation = module.target_pose.rotation_rpy_degrees
        _half_x, _half_y, half_z = _oriented_half_extents_xyz(
            module,
            target_rotation,
        )
        module.staging_pose.position = Vec3(
            x=yard_right - local_x,
            y=yard_bottom + local_y,
            z=half_z,
        )
        module.staging_pose.rotation_rpy_degrees = (
            target_rotation.model_copy(deep=True)
        )

    dispatch_x = yard_left - config.dispatch_lane_gap_m
    fleet_span = config.dispatch_robot_spacing_m * (len(plan.robots) - 1)
    for index, robot in enumerate(sorted(plan.robots, key=lambda item: item.robot_id)):
        robot.start_pose.position = Vec3(
            x=dispatch_x,
            y=-fleet_span / 2 + index * config.dispatch_robot_spacing_m,
            z=0,
        )
        robot.start_pose.rotation_rpy_degrees = Vec3(x=0, y=0, z=0)

    yard_top = yard_bottom + local_height
    min_x = min(dispatch_x, yard_left, target_bounds[0]) - config.boundary_margin_m
    max_x = target_bounds[1] + config.boundary_margin_m + 3.0
    min_y = min(yard_bottom, target_bounds[2], -fleet_span / 2) - (
        config.boundary_margin_m
    )
    max_y = max(yard_top, target_bounds[3], fleet_span / 2) + (
        config.boundary_margin_m
    )
    resolution = config.grid_resolution_m
    origin = Vec2(
        x=math.floor(min_x / resolution) * resolution,
        y=math.floor(min_y / resolution) * resolution,
    )
    width = math.ceil((max_x - origin.x) / resolution) + 1
    height = math.ceil((max_y - origin.y) / resolution) + 1
    plan.site_grid = SiteGrid(
        width=width,
        height=height,
        resolution_m=resolution,
        origin=origin,
        obstacle_cells=_perimeter_obstacle_cells(
            width=width,
            height=height,
            count=obstacle_count,
        ),
    )
    if not plan.plan_id.endswith("-phase5-yard-v1"):
        plan.plan_id = f"{plan.plan_id}-phase5-yard-v1"
    if "phase5_physical_yard_v1" not in transformed.tags:
        transformed.tags.append("phase5_physical_yard_v1")

    _validate_initial_physical_clearance(plan, config)
    (
        maximum_preflight_formation_offset_m,
        maximum_preflight_transport_height_m,
    ) = (
        _preflight_phase5_sequential_routes(plan, config)
    )
    transformed_plan_digest = _sha256_json(plan.model_dump(mode="json"))
    manifest = Phase5PhysicalYardManifest(
        source_plan_digest=source_plan_digest,
        transformed_plan_digest=transformed_plan_digest,
        allocator_configuration_digest=_sha256_json(
            config.model_dump(mode="json")
        ),
        configuration=config,
        shelf_row_count=row_count,
        staging_bounds_xy=(yard_left, yard_right, yard_bottom, yard_top),
        grid_origin=origin,
        grid_width=width,
        grid_height=height,
        module_count=len(plan.modules),
        robot_count=len(plan.robots),
        obstacle_count=len(plan.site_grid.obstacle_cells),
        initial_clearance_preflight_passed=True,
        sequential_route_preflight_passed=True,
        all_robots_active_preflight_passed=True,
        continuous_world_clearance_preflight_passed=True,
        recovery_route_preflight_passed=True,
        maximum_preflight_formation_offset_m=(
            maximum_preflight_formation_offset_m
        ),
        maximum_preflight_transport_height_m=(
            maximum_preflight_transport_height_m
        ),
    )
    return transformed, manifest


def _pack_staging_shelves(
    modules: list[BuildModule],
    config: Phase5PhysicalYardConfig,
) -> tuple[dict[str, tuple[float, float]], int, float]:
    packed: dict[str, tuple[float, float]] = {}
    cursor_x = 0.0
    row_bottom = 0.0
    row_height = 0.0
    row_count = 1
    ordered = sorted(
        modules,
        key=lambda item: (
            -_oriented_size_xy(
                item,
                item.target_pose.rotation_rpy_degrees,
            )[1],
            -_oriented_size_xy(
                item,
                item.target_pose.rotation_rpy_degrees,
            )[0],
            item.module_id,
        ),
    )
    for module in ordered:
        width, depth = _oriented_size_xy(
            module,
            module.target_pose.rotation_rpy_degrees,
        )
        if width > config.yard_shelf_width_m:
            raise ValueError(
                f"{module.module_id} is wider than the Phase 5 staging shelf"
            )
        if (
            cursor_x > 0
            and cursor_x + width > config.yard_shelf_width_m
        ):
            row_bottom += row_height + config.module_gap_m
            cursor_x = 0.0
            row_height = 0.0
            row_count += 1
        packed[module.module_id] = (
            cursor_x + width / 2,
            row_bottom + depth / 2,
        )
        cursor_x += width + config.module_gap_m
        row_height = max(row_height, depth)
    return packed, row_count, row_bottom + row_height


def _combined_target_bounds(
    modules: list[BuildModule],
) -> tuple[float, float, float, float]:
    bounds = [
        _module_footprint(module, installed=True)
        for module in modules
    ]
    return (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def _perimeter_obstacle_cells(
    *,
    width: int,
    height: int,
    count: int,
) -> list[tuple[int, int]]:
    candidates = [
        (width - 3 - column, 2 + row * max(2, (height - 5) // max(1, count)))
        for row in range(count)
        for column in range(2)
    ]
    selected: list[tuple[int, int]] = []
    for cell in candidates:
        bounded = (
            min(max(cell[0], 2), width - 3),
            min(max(cell[1], 2), height - 3),
        )
        if bounded not in selected:
            selected.append(bounded)
        if len(selected) == count:
            return sorted(selected)
    raise ValueError("could not allocate the required Phase 5 site obstacles")


def _module_footprint(
    module: BuildModule,
    *,
    installed: bool,
) -> tuple[float, float, float, float]:
    pose = module.target_pose if installed else module.staging_pose
    projected_width, projected_depth = _oriented_size_xy(
        module,
        pose.rotation_rpy_degrees,
    )
    projected_x = projected_width / 2
    projected_y = projected_depth / 2
    return (
        pose.position.x - projected_x,
        pose.position.x + projected_x,
        pose.position.y - projected_y,
        pose.position.y + projected_y,
    )


def _oriented_size_xy(
    module: BuildModule,
    rotation: float | Vec3,
) -> tuple[float, float]:
    if isinstance(rotation, Vec3):
        half_x, half_y, _half_z = _oriented_half_extents_xyz(
            module,
            rotation,
        )
        return 2 * half_x, 2 * half_y
    yaw = math.radians(rotation)
    half_width = module.dimensions.width / 2
    half_depth = module.dimensions.depth / 2
    projected_x = (
        abs(math.cos(yaw)) * half_width
        + abs(math.sin(yaw)) * half_depth
    )
    projected_y = (
        abs(math.sin(yaw)) * half_width
        + abs(math.cos(yaw)) * half_depth
    )
    return 2 * projected_x, 2 * projected_y


def _oriented_half_extents_xyz(
    module: BuildModule,
    rotation_rpy_degrees: Vec3,
) -> tuple[float, float, float]:
    roll = math.radians(rotation_rpy_degrees.x)
    pitch = math.radians(rotation_rpy_degrees.y)
    yaw = math.radians(rotation_rpy_degrees.z)
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    rotation = (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
    local = (
        module.dimensions.width / 2,
        module.dimensions.depth / 2,
        module.dimensions.height / 2,
    )
    return tuple(
        sum(abs(rotation[axis][local_axis]) * local[local_axis] for local_axis in range(3))
        for axis in range(3)
    )  # type: ignore[return-value]


def _module_aabb3(module: BuildModule, *, installed: bool) -> _Aabb3:
    pose = module.target_pose if installed else module.staging_pose
    half_x, half_y, half_z = _oriented_half_extents_xyz(
        module,
        pose.rotation_rpy_degrees,
    )
    return _Aabb3(
        minimum_x=pose.position.x - half_x,
        maximum_x=pose.position.x + half_x,
        minimum_y=pose.position.y - half_y,
        maximum_y=pose.position.y + half_y,
        minimum_z=pose.position.z - half_z,
        maximum_z=pose.position.z + half_z,
    )


def _inflate_bounds(
    bounds: tuple[float, float, float, float],
    clearance: float,
) -> tuple[float, float, float, float]:
    return (
        bounds[0] - clearance,
        bounds[1] + clearance,
        bounds[2] - clearance,
        bounds[3] + clearance,
    )


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return (
        left[0] < right[1]
        and right[0] < left[1]
        and left[2] < right[3]
        and right[2] < left[3]
    )


def _point_overlaps_bounds(
    point: Vec2,
    bounds: tuple[float, float, float, float],
    clearance: float,
) -> bool:
    nearest_x = min(max(point.x, bounds[0]), bounds[1])
    nearest_y = min(max(point.y, bounds[2]), bounds[3])
    return math.hypot(point.x - nearest_x, point.y - nearest_y) < clearance


def _point_clearance_to_bounds(
    point: Vec2,
    bounds: tuple[float, float, float, float],
) -> float:
    delta_x = max(bounds[0] - point.x, 0.0, point.x - bounds[1])
    delta_y = max(bounds[2] - point.y, 0.0, point.y - bounds[3])
    return math.hypot(delta_x, delta_y)


def _point_segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    delta_x = end.x - start.x
    delta_y = end.y - start.y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 1e-18:
        return _distance(point, start)
    projection = (
        (point.x - start.x) * delta_x
        + (point.y - start.y) * delta_y
    ) / length_squared
    fraction = min(max(projection, 0.0), 1.0)
    nearest = Vec2(
        x=start.x + fraction * delta_x,
        y=start.y + fraction * delta_y,
    )
    return _distance(point, nearest)


def _cross(left: Vec2, middle: Vec2, right: Vec2) -> float:
    return (
        (middle.x - left.x) * (right.y - left.y)
        - (middle.y - left.y) * (right.x - left.x)
    )


def _segments_intersect(
    first_start: Vec2,
    first_end: Vec2,
    second_start: Vec2,
    second_end: Vec2,
) -> bool:
    first_a = _cross(first_start, first_end, second_start)
    first_b = _cross(first_start, first_end, second_end)
    second_a = _cross(second_start, second_end, first_start)
    second_b = _cross(second_start, second_end, first_end)
    tolerance = 1e-12
    if (
        first_a * first_b < -tolerance
        and second_a * second_b < -tolerance
    ):
        return True
    return any(
        abs(cross) <= tolerance
        and min(start.x, end.x) - tolerance
        <= point.x
        <= max(start.x, end.x) + tolerance
        and min(start.y, end.y) - tolerance
        <= point.y
        <= max(start.y, end.y) + tolerance
        for cross, start, end, point in (
            (first_a, first_start, first_end, second_start),
            (first_b, first_start, first_end, second_end),
            (second_a, second_start, second_end, first_start),
            (second_b, second_start, second_end, first_end),
        )
    )


def _segment_segment_distance(
    first_start: Vec2,
    first_end: Vec2,
    second_start: Vec2,
    second_end: Vec2,
) -> float:
    if _segments_intersect(
        first_start,
        first_end,
        second_start,
        second_end,
    ):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _segment_bounds_clearance(
    start: Vec2,
    end: Vec2,
    bounds: tuple[float, float, float, float],
) -> float:
    if _point_clearance_to_bounds(start, bounds) == 0.0 or (
        _point_clearance_to_bounds(end, bounds) == 0.0
    ):
        return 0.0
    lower_left = Vec2(x=bounds[0], y=bounds[2])
    lower_right = Vec2(x=bounds[1], y=bounds[2])
    upper_right = Vec2(x=bounds[1], y=bounds[3])
    upper_left = Vec2(x=bounds[0], y=bounds[3])
    return min(
        _segment_segment_distance(start, end, edge_start, edge_end)
        for edge_start, edge_end in (
            (lower_left, lower_right),
            (lower_right, upper_right),
            (upper_right, upper_left),
            (upper_left, lower_left),
        )
    )


def _route_bounds_clearance(
    route: list[Vec2],
    bounds: tuple[float, float, float, float],
) -> float:
    if not route:
        raise RoutingError("cannot validate an empty route")
    if len(route) == 1:
        return _point_clearance_to_bounds(route[0], bounds)
    return min(
        _segment_bounds_clearance(start, end, bounds)
        for start, end in zip(route, route[1:], strict=False)
    )


def _route_point_clearance(route: list[Vec2], point: Vec2) -> float:
    if not route:
        raise RoutingError("cannot validate an empty route")
    if len(route) == 1:
        return _distance(route[0], point)
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(route, route[1:], strict=False)
    )


def _synchronize_safe_route_pair(
    routes: Mapping[str, list[Vec2]],
    *,
    minimum_separation_m: float,
) -> dict[str, list[Vec2]]:
    robot_ids = sorted(routes)
    if len(robot_ids) != 2:
        raise RoutingError("safe route-time scheduling requires two routes")
    left_id, right_id = robot_ids
    left = routes[left_id]
    right = routes[right_id]
    if not left or not right:
        raise RoutingError("safe route-time scheduling received an empty route")
    cache_key = (
        minimum_separation_m,
        tuple((point.x, point.y) for point in left),
        tuple((point.x, point.y) for point in right),
    )
    cached_states = _PHASE5_ROUTE_SYNC_CACHE.get(cache_key)
    if cached_states is not None:
        return {
            left_id: [left[left_index] for left_index, _ in cached_states],
            right_id: [right[right_index] for _, right_index in cached_states],
        }
    start = (0, 0)
    goal = (len(left) - 1, len(right) - 1)
    if (
        _distance(left[0], right[0]) + 1e-9
        < minimum_separation_m
    ):
        raise RoutingError(
            "two-base paths start below the required separation"
        )
    if start == goal:
        singleton_states = (start,)
        _PHASE5_ROUTE_SYNC_CACHE[cache_key] = singleton_states
        return {left_id: [left[0]], right_id: [right[0]]}

    def candidates(
        state: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        left_index, right_index = state
        values = (
            (
                min(left_index + 1, goal[0]),
                min(right_index + 1, goal[1]),
            ),
            (min(left_index + 1, goal[0]), right_index),
            (left_index, min(right_index + 1, goal[1])),
        )
        return tuple(dict.fromkeys(value for value in values if value != state))

    # Depth-first search reaches the common no-conflict case in O(max(n, m))
    # states, while stable backtracking retains completeness inside the finite
    # product graph. Breadth-first exploration needlessly visited most of the
    # O(n*m) grid for long, otherwise safe yard routes.
    frontier: list[tuple[tuple[int, int], int]] = [(start, 0)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        state, candidate_index = frontier[-1]
        if state == goal:
            break
        available = candidates(state)
        if candidate_index >= len(available):
            frontier.pop()
            continue
        candidate = available[candidate_index]
        frontier[-1] = (state, candidate_index + 1)
        if candidate in parent:
            continue
        separation = _segment_segment_distance(
            left[state[0]],
            left[candidate[0]],
            right[state[1]],
            right[candidate[1]],
        )
        if separation + 1e-9 < minimum_separation_m:
            continue
        parent[candidate] = state
        if len(parent) > _PHASE5_ROUTE_SYNC_STATE_LIMIT:
            raise RoutingError(
                "two-base route-time scheduling exceeded its deterministic "
                f"{_PHASE5_ROUTE_SYNC_STATE_LIMIT}-state bound"
            )
        frontier.append((candidate, 0))
    if goal not in parent:
        raise RoutingError(
            "two-base paths have no collision-clear shared-time schedule"
        )
    states: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        states.append(cursor)
        cursor = parent[cursor]
    states.reverse()
    _PHASE5_ROUTE_SYNC_CACHE[cache_key] = tuple(states)
    return {
        left_id: [left[left_index] for left_index, _right_index in states],
        right_id: [right[right_index] for _left_index, right_index in states],
    }


def _aabb3_separation(left: _Aabb3, right: _Aabb3) -> float:
    delta_x = max(
        right.minimum_x - left.maximum_x,
        left.minimum_x - right.maximum_x,
        0.0,
    )
    delta_y = max(
        right.minimum_y - left.maximum_y,
        left.minimum_y - right.maximum_y,
        0.0,
    )
    delta_z = max(
        right.minimum_z - left.maximum_z,
        left.minimum_z - right.maximum_z,
        0.0,
    )
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def _obstacle_bounds(
    cell: tuple[int, int],
    grid: SiteGrid,
) -> tuple[float, float, float, float]:
    center_x = grid.origin.x + cell[0] * grid.resolution_m
    center_y = grid.origin.y + cell[1] * grid.resolution_m
    half_extent = grid.resolution_m * 0.4
    return (
        center_x - half_extent,
        center_x + half_extent,
        center_y - half_extent,
        center_y + half_extent,
    )


def _validate_initial_physical_clearance(
    plan: BuildPlan,
    config: Phase5PhysicalYardConfig,
) -> None:
    for module in plan.modules:
        staging_rotation = module.staging_pose.rotation_rpy_degrees
        target_rotation = module.target_pose.rotation_rpy_degrees
        deltas = tuple(
            (staging - target + 180) % 360 - 180
            for staging, target in (
                (staging_rotation.x, target_rotation.x),
                (staging_rotation.y, target_rotation.y),
                (staging_rotation.z, target_rotation.z),
            )
        )
        if any(abs(delta) > 1e-9 for delta in deltas):
            raise ValueError(
                f"{module.module_id} requires in-transit payload rotation"
            )
    inflated_modules = {
        module.module_id: _inflate_bounds(
            _module_footprint(module, installed=False),
            config.route_clearance_m / 2,
        )
        for module in plan.modules
    }
    for left, right in combinations(sorted(inflated_modules), 2):
        if _bounds_overlap(
            inflated_modules[left],
            inflated_modules[right],
        ):
            raise ValueError(
                f"Phase 5 staging modules overlap: {left}, {right}"
            )
    robot_positions = {
        robot.robot_id: Vec2.model_validate(robot.start_pose.position)
        for robot in plan.robots
    }
    required_robot_gap = (
        2 * config.robot_footprint_radius_m + config.route_clearance_m
    )
    for left, right in combinations(sorted(robot_positions), 2):
        first = robot_positions[left]
        second = robot_positions[right]
        if math.hypot(first.x - second.x, first.y - second.y) < (
            required_robot_gap
        ):
            raise ValueError(
                f"Phase 5 dispatch robots overlap: {left}, {right}"
            )
    for robot_id, position in robot_positions.items():
        for module_id, bounds in inflated_modules.items():
            if _point_overlaps_bounds(
                position,
                bounds,
                config.robot_footprint_radius_m,
            ):
                raise ValueError(
                    f"{robot_id} overlaps staged module {module_id}"
                )
    target_bounds = _inflate_bounds(
        _combined_target_bounds(plan.modules),
        config.route_clearance_m,
    )
    for cell in plan.site_grid.obstacle_cells:
        obstacle = _obstacle_bounds(cell, plan.site_grid)
        if _bounds_overlap(obstacle, target_bounds):
            raise ValueError("Phase 5 obstacle overlaps the cottage footprint")
        for bounds in inflated_modules.values():
            if _bounds_overlap(obstacle, bounds):
                raise ValueError(
                    "Phase 5 obstacle overlaps a staged module footprint"
                )
        for robot_id, position in robot_positions.items():
            if _point_overlaps_bounds(
                position,
                obstacle,
                config.robot_footprint_radius_m,
            ):
                raise ValueError(
                    f"Phase 5 obstacle overlaps dispatch robot {robot_id}"
                )


def _rasterize_bounds(
    grid: SiteGrid,
    bounds: tuple[float, float, float, float],
) -> set[tuple[int, int]]:
    resolution = grid.resolution_m
    minimum_x = max(
        0,
        math.floor((bounds[0] - grid.origin.x) / resolution),
    )
    maximum_x = min(
        grid.width - 1,
        math.ceil((bounds[1] - grid.origin.x) / resolution),
    )
    minimum_y = max(
        0,
        math.floor((bounds[2] - grid.origin.y) / resolution),
    )
    maximum_y = min(
        grid.height - 1,
        math.ceil((bounds[3] - grid.origin.y) / resolution),
    )
    cells: set[tuple[int, int]] = set()
    for x in range(minimum_x, maximum_x + 1):
        world_x = grid.origin.x + x * resolution
        for y in range(minimum_y, maximum_y + 1):
            world_y = grid.origin.y + y * resolution
            if (
                bounds[0] <= world_x <= bounds[1]
                and bounds[2] <= world_y <= bounds[3]
            ):
                cells.add((x, y))
    return cells


def _phase5_route_grid(
    plan: BuildPlan,
    *,
    active_module_id: str,
    active_module_is_logical_payload: bool,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> SiteGrid:
    grid: SiteGrid = plan.site_grid.model_copy(deep=True)
    obstacles: set[tuple[int, int]] = set()
    base_world_clearance = (
        config.robot_footprint_radius_m + config.route_clearance_m
    )
    for cell in plan.site_grid.obstacle_cells:
        obstacles.update(
            _rasterize_bounds(
                grid,
                _inflate_bounds(
                    _obstacle_bounds(cell, grid),
                    base_world_clearance,
                ),
            )
        )
    for module in plan.modules:
        if (
            module.module_id == active_module_id
            and active_module_is_logical_payload
        ):
            continue
        bounds = _module_footprint(
            module,
            installed=module.module_id in installed_module_ids,
        )
        obstacles.update(
            _rasterize_bounds(
                grid,
                _inflate_bounds(bounds, base_world_clearance),
            )
        )
    idle_clearance = (
        2 * config.robot_footprint_radius_m + config.route_clearance_m
    )
    for robot_id, position in robot_positions.items():
        if robot_id in active_robot_ids:
            continue
        obstacles.update(
            _rasterize_bounds(
                grid,
                (
                    position.x - idle_clearance,
                    position.x + idle_clearance,
                    position.y - idle_clearance,
                    position.y + idle_clearance,
                ),
            )
        )
    grid.obstacle_cells = sorted(obstacles)
    return grid


def _phase5_team(
    plan: BuildPlan,
    module: BuildModule,
    *,
    unavailable: set[str],
    usage: Mapping[str, int],
    config: Phase5PhysicalYardConfig,
) -> list[str]:
    available = [
        robot
        for robot in plan.robots
        if robot.robot_id not in unavailable
    ]
    groups = [
        group
        for group in combinations(
            available,
            config.phase5_team_size,
        )
        if sum(robot.payload_capacity_kg for robot in group) >= module.mass_kg
    ]
    if not groups:
        raise DynamicCoppeliaError(
            f"no two-base Phase 5 team can carry {module.module_id}"
        )
    groups.sort(
        key=lambda group: (
            sum(usage.get(robot.robot_id, 0) for robot in group),
            max(usage.get(robot.robot_id, 0) for robot in group),
            tuple(robot.robot_id for robot in group),
        )
    )
    return [robot.robot_id for robot in groups[0]]


def _oriented_module_clearance(
    point: Vec2,
    center: Vec2,
    module: BuildModule,
    yaw_degrees: float,
) -> float:
    yaw = math.radians(yaw_degrees)
    delta_x = point.x - center.x
    delta_y = point.y - center.y
    local_x = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
    local_y = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
    outside_x = max(abs(local_x) - module.dimensions.width / 2, 0.0)
    outside_y = max(abs(local_y) - module.dimensions.depth / 2, 0.0)
    return math.hypot(outside_x, outside_y)


def _formation_offset_candidates(
    module: BuildModule,
    robot_ids: list[str],
    *,
    grid: SiteGrid,
    config: Phase5PhysicalYardConfig,
) -> Iterable[dict[str, Vec2]]:
    if len(robot_ids) != config.phase5_team_size:
        raise DynamicCoppeliaError(
            "Phase 5 physical formations require exactly two bases"
        )
    ordered_ids = sorted(robot_ids)
    yaws = (
        module.staging_pose.rotation_rpy_degrees.z,
        module.target_pose.rotation_rpy_degrees.z,
    )
    candidates: list[tuple[float, float, float, float]] = []
    normal = (
        module.target_pose.rotation_rpy_degrees.z + 90
    ) % 180
    angles = sorted(
        {
            round(value % 180, 6)
            for value in (
                normal,
                normal - 45,
                normal + 45,
                normal - 30,
                normal + 30,
                0,
                45,
                90,
                135,
            )
        }
    )
    maximum_increment = math.floor(
        config.maximum_formation_expansion_m
        / grid.resolution_m
    )
    for angle_degrees in angles:
        angle = math.radians(angle_degrees)
        unit_x = math.cos(angle)
        unit_y = math.sin(angle)
        support = max(
            (
                module.dimensions.width
                / 2
                * abs(
                    unit_x * math.cos(math.radians(yaw))
                    + unit_y * math.sin(math.radians(yaw))
                )
                + module.dimensions.depth
                / 2
                * abs(
                    -unit_x * math.sin(math.radians(yaw))
                    + unit_y * math.cos(math.radians(yaw))
                )
            )
            for yaw in yaws
        )
        minimum_offset = support + config.formation_clearance_m
        for increment in range(maximum_increment + 1):
            distance = minimum_offset + increment * grid.resolution_m
            candidates.append(
                (distance, float(angle_degrees), unit_x, unit_y)
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    for distance, _angle, unit_x, unit_y in candidates:
        offset = Vec2(x=unit_x * distance, y=unit_y * distance)
        yield {
            ordered_ids[0]: Vec2(x=-offset.x, y=-offset.y),
            ordered_ids[1]: offset,
        }


def _shifted_forbidden_bounds(
    bounds: tuple[float, float, float, float],
    *,
    relative_position: Vec2,
    clearance: float,
    raster_padding: float,
) -> tuple[float, float, float, float]:
    padding = clearance + raster_padding
    return (
        bounds[0] - relative_position.x - padding,
        bounds[1] - relative_position.x + padding,
        bounds[2] - relative_position.y - padding,
        bounds[3] - relative_position.y + padding,
    )


def _obstacle_aabb3(
    cell: tuple[int, int],
    grid: SiteGrid,
    config: Phase5PhysicalYardConfig,
) -> _Aabb3:
    bounds = _obstacle_bounds(cell, grid)
    return _Aabb3(
        minimum_x=bounds[0],
        maximum_x=bounds[1],
        minimum_y=bounds[2],
        maximum_y=bounds[3],
        minimum_z=0.0,
        maximum_z=config.site_obstacle_height_m,
    )


def _logical_transport_height(
    plan: BuildPlan,
    *,
    module: BuildModule,
    installed_module_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> float:
    _half_x, _half_y, payload_half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    world_tops = [config.site_obstacle_height_m]
    world_tops.extend(
        _module_aabb3(
            other,
            installed=other.module_id in installed_module_ids,
        ).maximum_z
        for other in plan.modules
        if other.module_id != module.module_id
    )
    height = max(world_tops, default=0.0) + payload_half_z + config.route_clearance_m
    if height > config.maximum_logical_transport_height_m:
        raise RoutingError(
            f"{module.module_id} requires logical transport height "
            f"{height:.3f} m above the configured bound"
        )
    return height


def _installation_contact_module_ids(
    plan: BuildPlan,
    *,
    module: BuildModule,
    installed_module_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> list[str]:
    target = _module_aabb3(module, installed=True)
    return sorted(
        other.module_id
        for other in plan.modules
        if other.module_id in installed_module_ids
        and _aabb3_separation(
            target,
            _module_aabb3(other, installed=True),
        )
        <= config.route_clearance_m + 1e-9
    )


def _payload_aabb3(
    module: BuildModule,
    *,
    center: Vec3,
) -> _Aabb3:
    half_x, half_y, half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    return _Aabb3(
        minimum_x=center.x - half_x,
        maximum_x=center.x + half_x,
        minimum_y=center.y - half_y,
        maximum_y=center.y + half_y,
        minimum_z=center.z - half_z,
        maximum_z=center.z + half_z,
    )


def _swept_payload_aabb3_clearance(
    module: BuildModule,
    *,
    carrier_route: list[Vec2],
    transport_height_m: float,
    obstacle: _Aabb3,
) -> float:
    half_x, half_y, half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    horizontal = _route_bounds_clearance(
        carrier_route,
        (
            obstacle.minimum_x - half_x,
            obstacle.maximum_x + half_x,
            obstacle.minimum_y - half_y,
            obstacle.maximum_y + half_y,
        ),
    )
    vertical = max(
        obstacle.minimum_z - (transport_height_m + half_z),
        (transport_height_m - half_z) - obstacle.maximum_z,
        0.0,
    )
    return math.hypot(horizontal, vertical)


def _swept_payload_robot_clearance(
    module: BuildModule,
    *,
    carrier_route: list[Vec2],
    robot_position: Vec2,
    robot_radius_m: float,
) -> float:
    half_x, half_y, _half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    clearance = _route_bounds_clearance(
        carrier_route,
        (
            robot_position.x - half_x,
            robot_position.x + half_x,
            robot_position.y - half_y,
            robot_position.y + half_y,
        ),
    )
    return max(0.0, clearance - robot_radius_m)


def _phase5_carrier_route_grid(
    plan: BuildPlan,
    *,
    module: BuildModule,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    formation_offsets: Mapping[str, Vec2],
    carrier_offset: Vec2,
    transport_height_m: float,
    config: Phase5PhysicalYardConfig,
) -> SiteGrid:
    grid = plan.site_grid.model_copy(deep=True)
    obstacles: set[tuple[int, int]] = set()
    # Geometric validation below checks every 0.1 m transport sample. The
    # raster grid therefore represents the exact Minkowski forbidden set once;
    # adding a second half-cell pad would isolate otherwise valid install bays.
    raster_padding = 0.0
    base_clearance = (
        config.robot_footprint_radius_m + config.route_clearance_m
    )
    base_relatives = [
        Vec2(
            x=offset.x - carrier_offset.x,
            y=offset.y - carrier_offset.y,
        )
        for offset in formation_offsets.values()
    ]
    active_width, active_depth = _oriented_size_xy(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    _half_x, _half_y, active_half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )

    def payload_needs_horizontal_avoidance(obstacle: _Aabb3) -> bool:
        vertical_gap = max(
            obstacle.minimum_z - (transport_height_m + active_half_z),
            (transport_height_m - active_half_z) - obstacle.maximum_z,
            0.0,
        )
        return vertical_gap < config.route_clearance_m

    def add_base_forbidden(
        bounds: tuple[float, float, float, float],
    ) -> None:
        for relative in base_relatives:
            obstacles.update(
                _rasterize_bounds(
                    grid,
                    _shifted_forbidden_bounds(
                        bounds,
                        relative_position=relative,
                        clearance=base_clearance,
                        raster_padding=raster_padding,
                    ),
                )
            )

    def add_payload_forbidden(
        bounds: tuple[float, float, float, float],
        *,
        clearance: float,
    ) -> None:
        obstacles.update(
            _rasterize_bounds(
                grid,
                (
                    bounds[0]
                    - active_width / 2
                    - clearance
                    - raster_padding,
                    bounds[1]
                    + active_width / 2
                    + clearance
                    + raster_padding,
                    bounds[2]
                    - active_depth / 2
                    - clearance
                    - raster_padding,
                    bounds[3]
                    + active_depth / 2
                    + clearance
                    + raster_padding,
                ),
            )
        )

    for cell in plan.site_grid.obstacle_cells:
        bounds = _obstacle_bounds(cell, grid)
        add_base_forbidden(bounds)
        obstacle = _obstacle_aabb3(cell, grid, config)
        if payload_needs_horizontal_avoidance(obstacle):
            add_payload_forbidden(
                bounds,
                clearance=config.route_clearance_m,
            )
    for other in plan.modules:
        if other.module_id == module.module_id:
            continue
        installed = other.module_id in installed_module_ids
        bounds = _module_footprint(other, installed=installed)
        add_base_forbidden(bounds)
        if payload_needs_horizontal_avoidance(
            _module_aabb3(other, installed=installed)
        ):
            add_payload_forbidden(
                bounds,
                clearance=config.route_clearance_m,
            )
    idle_radius = (
        2 * config.robot_footprint_radius_m + config.route_clearance_m
    )
    for robot_id, position in robot_positions.items():
        if robot_id in active_robot_ids:
            continue
        bounds = (position.x, position.x, position.y, position.y)
        for relative in base_relatives:
            obstacles.update(
                _rasterize_bounds(
                    grid,
                    _shifted_forbidden_bounds(
                        bounds,
                        relative_position=relative,
                        clearance=idle_radius,
                        raster_padding=raster_padding,
                    ),
                )
            )
        add_payload_forbidden(
            bounds,
            clearance=(
                config.robot_footprint_radius_m
                + config.route_clearance_m
            ),
        )

    minimum_x = grid.origin.x
    maximum_x = grid.origin.x + (grid.width - 1) * grid.resolution_m
    minimum_y = grid.origin.y
    maximum_y = grid.origin.y + (grid.height - 1) * grid.resolution_m
    payload_half_x = active_width / 2 + config.route_clearance_m
    payload_half_y = active_depth / 2 + config.route_clearance_m
    for x in range(grid.width):
        world_x = grid.origin.x + x * grid.resolution_m
        for y in range(grid.height):
            world_y = grid.origin.y + y * grid.resolution_m
            if (
                world_x - payload_half_x < minimum_x
                or world_x + payload_half_x > maximum_x
                or world_y - payload_half_y < minimum_y
                or world_y + payload_half_y > maximum_y
                or any(
                    world_x + relative.x
                    - config.robot_footprint_radius_m
                    < minimum_x
                    or world_x + relative.x
                    + config.robot_footprint_radius_m
                    > maximum_x
                    or world_y + relative.y
                    - config.robot_footprint_radius_m
                    < minimum_y
                    or world_y + relative.y
                    + config.robot_footprint_radius_m
                    > maximum_y
                    for relative in base_relatives
                )
            ):
                obstacles.add((x, y))
    grid.obstacle_cells = sorted(obstacles)
    return grid


def _densify_route(
    points: list[Vec2],
    *,
    maximum_spacing_m: float,
) -> list[Vec2]:
    if not points:
        raise DynamicCoppeliaError("cannot densify an empty carrier route")
    dense = [points[0]]
    for left, right in zip(points, points[1:], strict=False):
        distance = _distance(left, right)
        steps = max(1, math.ceil(distance / maximum_spacing_m))
        for step in range(1, steps + 1):
            fraction = step / steps
            point = Vec2(
                x=left.x + (right.x - left.x) * fraction,
                y=left.y + (right.y - left.y) * fraction,
            )
            if _distance(dense[-1], point) > 1e-12:
                dense.append(point)
    return dense


def _static_carrier_route(
    grid: SiteGrid,
    *,
    start: Vec2,
    goal: Vec2,
    blocked_edges: set[frozenset[tuple[int, int]]] | None = None,
) -> list[Vec2]:
    start_cell = world_to_cell(start, grid)
    goal_cell = world_to_cell(goal, grid)
    obstacles = set(grid.obstacle_cells)
    if start_cell in obstacles or goal_cell in obstacles:
        raise RoutingError(
            f"rigid carrier endpoint is blocked: {start_cell} -> {goal_cell}"
        )
    blocked = blocked_edges or set()
    frontier = deque([start_cell])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {
        start_cell: None
    }
    while frontier:
        cell = frontier.popleft()
        if cell == goal_cell:
            break
        x, y = cell
        for neighbor in (
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y - 1),
        ):
            if (
                0 <= neighbor[0] < grid.width
                and 0 <= neighbor[1] < grid.height
                and neighbor not in obstacles
                and neighbor not in parent
                and frozenset((cell, neighbor)) not in blocked
            ):
                parent[neighbor] = cell
                frontier.append(neighbor)
    if goal_cell not in parent:
        raise RoutingError(
            f"no rigid swept-footprint path found: {start_cell} -> {goal_cell}"
        )
    cells: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal_cell
    while cursor is not None:
        cells.append(cursor)
        cursor = parent[cursor]
    cells.reverse()
    return [
        Vec2(
            x=grid.origin.x + x * grid.resolution_m,
            y=grid.origin.y + y * grid.resolution_m,
        )
        for x, y in cells
    ]


def _payload_clearances_for_routes(
    *,
    module: BuildModule,
    carrier_route: list[Vec2],
    base_routes: Mapping[str, list[Vec2]],
) -> list[float]:
    yaws = {
        module.staging_pose.rotation_rpy_degrees.z,
        module.target_pose.rotation_rpy_degrees.z,
    }
    return [
        min(
            _oriented_module_clearance(
                base_routes[robot_id][index],
                carrier,
                module,
                yaw,
            )
            for robot_id in sorted(base_routes)
            for yaw in yaws
        )
        for index, carrier in enumerate(carrier_route)
    ]


def _minimum_record(
    *,
    phase: ClearancePhase,
    source: ClearanceSource,
    mover_kind: ClearanceMover,
    obstacle_kind: ClearanceObstacle,
    values: list[tuple[float, str, str]],
    config: Phase5PhysicalYardConfig,
) -> Phase5ClearanceMinimum:
    if not values:
        return Phase5ClearanceMinimum(
            phase=phase,
            source=source,
            mover_kind=mover_kind,
            obstacle_kind=obstacle_kind,
            minimum_surface_clearance_m=None,
            required_clearance_m=config.route_clearance_m,
            evaluated_pair_count=0,
        )
    minimum, mover_id, obstacle_id = min(
        values,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return Phase5ClearanceMinimum(
        phase=phase,
        source=source,
        mover_kind=mover_kind,
        obstacle_kind=obstacle_kind,
        minimum_surface_clearance_m=max(0.0, minimum),
        required_clearance_m=config.route_clearance_m,
        evaluated_pair_count=len(values),
        limiting_mover_id=mover_id,
        limiting_obstacle_id=obstacle_id,
    )


def _base_world_clearance_minima(
    plan: BuildPlan,
    *,
    phase: ClearancePhase,
    source: ClearanceSource,
    routes: Mapping[str, list[Vec2]],
    active_module_id: str,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    disabled_robot_ids: set[str],
    active_module_is_payload: bool,
    config: Phase5PhysicalYardConfig,
) -> list[Phase5ClearanceMinimum]:
    values: dict[ClearanceObstacle, list[tuple[float, str, str]]] = {
        "installed_module": [],
        "staged_module": [],
        "site_obstacle": [],
        "idle_robot": [],
        "disabled_robot": [],
    }
    for robot_id, route in sorted(routes.items()):
        for other in plan.modules:
            if (
                active_module_is_payload
                and other.module_id == active_module_id
            ):
                continue
            installed = other.module_id in installed_module_ids
            kind: ClearanceObstacle = (
                "installed_module" if installed else "staged_module"
            )
            clearance = max(
                0.0,
                _route_bounds_clearance(
                    route,
                    _module_footprint(other, installed=installed),
                )
                - config.robot_footprint_radius_m,
            )
            values[kind].append((clearance, robot_id, other.module_id))
        for cell in plan.site_grid.obstacle_cells:
            obstacle_id = f"{cell[0]},{cell[1]}"
            clearance = max(
                0.0,
                _route_bounds_clearance(
                    route,
                    _obstacle_bounds(cell, plan.site_grid),
                )
                - config.robot_footprint_radius_m,
            )
            values["site_obstacle"].append(
                (clearance, robot_id, obstacle_id)
            )
        for other_id, position in sorted(robot_positions.items()):
            if other_id in active_robot_ids:
                continue
            kind = (
                "disabled_robot"
                if other_id in disabled_robot_ids
                else "idle_robot"
            )
            clearance = max(
                0.0,
                _route_point_clearance(route, position)
                - 2 * config.robot_footprint_radius_m,
            )
            values[kind].append((clearance, robot_id, other_id))
    return [
        _minimum_record(
            phase=phase,
            source=source,
            mover_kind="robot_base",
            obstacle_kind=kind,
            values=values[kind],
            config=config,
        )
        for kind in _WORLD_CLEARANCE_OBSTACLES
    ]


def _payload_world_clearance_minima(
    plan: BuildPlan,
    *,
    phase: Literal["carry"],
    source: ClearanceSource,
    module: BuildModule,
    carrier_route: list[Vec2],
    transport_height_m: float,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    disabled_robot_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> list[Phase5ClearanceMinimum]:
    values: dict[ClearanceObstacle, list[tuple[float, str, str]]] = {
        "installed_module": [],
        "staged_module": [],
        "site_obstacle": [],
        "idle_robot": [],
        "disabled_robot": [],
    }
    for other in plan.modules:
        if other.module_id == module.module_id:
            continue
        installed = other.module_id in installed_module_ids
        kind: ClearanceObstacle = (
            "installed_module" if installed else "staged_module"
        )
        clearance = _swept_payload_aabb3_clearance(
            module,
            carrier_route=carrier_route,
            transport_height_m=transport_height_m,
            obstacle=_module_aabb3(other, installed=installed),
        )
        values[kind].append((clearance, module.module_id, other.module_id))
    for cell in plan.site_grid.obstacle_cells:
        obstacle_id = f"{cell[0]},{cell[1]}"
        clearance = _swept_payload_aabb3_clearance(
            module,
            carrier_route=carrier_route,
            transport_height_m=transport_height_m,
            obstacle=_obstacle_aabb3(cell, plan.site_grid, config),
        )
        values["site_obstacle"].append(
            (clearance, module.module_id, obstacle_id)
        )
    for robot_id, position in sorted(robot_positions.items()):
        if robot_id in active_robot_ids:
            continue
        kind = (
            "disabled_robot"
            if robot_id in disabled_robot_ids
            else "idle_robot"
        )
        clearance = _swept_payload_robot_clearance(
            module,
            carrier_route=carrier_route,
            robot_position=position,
            robot_radius_m=config.robot_footprint_radius_m,
        )
        values[kind].append((clearance, module.module_id, robot_id))
    return [
        _minimum_record(
            phase=phase,
            source=source,
            mover_kind="logical_payload",
            obstacle_kind=kind,
            values=values[kind],
            config=config,
        )
        for kind in _WORLD_CLEARANCE_OBSTACLES
    ]


def _require_clearance_records(
    records: Iterable[Phase5ClearanceMinimum],
) -> None:
    failed = [
        record
        for record in records
        if record.minimum_surface_clearance_m is not None
        and record.minimum_surface_clearance_m + 1e-9
        < record.required_clearance_m
    ]
    if failed:
        first = failed[0]
        raise RoutingError(
            f"continuous {first.phase} {first.mover_kind} clearance to "
            f"{first.obstacle_kind} is {first.minimum_surface_clearance_m:.3f} m"
        )


def _validate_rigid_carrier_route(
    plan: BuildPlan,
    *,
    module: BuildModule,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    carrier_route: list[Vec2],
    base_routes: Mapping[str, list[Vec2]],
    transport_height_m: float,
    disabled_robot_ids: set[str],
    source: ClearanceSource,
    config: Phase5PhysicalYardConfig,
) -> list[Phase5ClearanceMinimum]:
    if any(len(route) != len(carrier_route) for route in base_routes.values()):
        raise RoutingError("rigid base routes do not share the carrier horizon")
    minimum_x = plan.site_grid.origin.x
    maximum_x = minimum_x + (
        plan.site_grid.width - 1
    ) * plan.site_grid.resolution_m
    minimum_y = plan.site_grid.origin.y
    maximum_y = minimum_y + (
        plan.site_grid.height - 1
    ) * plan.site_grid.resolution_m
    active_width, active_depth = _oriented_size_xy(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    for carrier in carrier_route:
        if (
            carrier.x - active_width / 2 < minimum_x
            or carrier.x + active_width / 2 > maximum_x
            or carrier.y - active_depth / 2 < minimum_y
            or carrier.y + active_depth / 2 > maximum_y
        ):
            raise RoutingError("logical payload leaves the site grid")
    for robot_id, route in base_routes.items():
        for point in route:
            if (
                point.x - config.robot_footprint_radius_m < minimum_x
                or point.x + config.robot_footprint_radius_m > maximum_x
                or point.y - config.robot_footprint_radius_m < minimum_y
                or point.y + config.robot_footprint_radius_m > maximum_y
            ):
                raise RoutingError(f"{robot_id} leaves the site grid")
    records = _base_world_clearance_minima(
        plan,
        phase="carry",
        source=source,
        routes=base_routes,
        active_module_id=module.module_id,
        installed_module_ids=installed_module_ids,
        robot_positions=robot_positions,
        active_robot_ids=active_robot_ids,
        disabled_robot_ids=disabled_robot_ids,
        active_module_is_payload=True,
        config=config,
    )
    records.extend(
        _payload_world_clearance_minima(
            plan,
            phase="carry",
            source=source,
            module=module,
            carrier_route=carrier_route,
            transport_height_m=transport_height_m,
            installed_module_ids=installed_module_ids,
            robot_positions=robot_positions,
            active_robot_ids=active_robot_ids,
            disabled_robot_ids=disabled_robot_ids,
            config=config,
        )
    )
    half_x, half_y, _half_z = _oriented_half_extents_xyz(
        module,
        module.staging_pose.rotation_rpy_degrees,
    )
    payload_values: list[tuple[float, str, str]] = []
    for robot_id, route in sorted(base_routes.items()):
        relative_route = [
            Vec2(x=base.x - carrier.x, y=base.y - carrier.y)
            for base, carrier in zip(route, carrier_route, strict=True)
        ]
        clearance = max(
            0.0,
            _route_bounds_clearance(
                relative_route,
                (-half_x, half_x, -half_y, half_y),
            )
            - config.robot_footprint_radius_m,
        )
        payload_values.append((clearance, robot_id, module.module_id))
    records.append(
        _minimum_record(
            phase="carry",
            source=source,
            mover_kind="robot_base",
            obstacle_kind="active_payload",
            values=payload_values,
            config=config,
        )
    )
    _require_clearance_records(records)
    return records


def _plan_rigid_carrier_route(
    _router: RoutingAdapter,
    plan: BuildPlan,
    *,
    module: BuildModule,
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    formation_offsets: Mapping[str, Vec2],
    carrier_offset: Vec2,
    disabled_robot_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> tuple[
    list[Vec2],
    dict[str, list[Vec2]],
    str,
    SiteGrid,
    float,
    list[Phase5ClearanceMinimum],
]:
    start = Vec2.model_validate(module.staging_pose.position)
    goal = Vec2.model_validate(module.target_pose.position)
    transport_height_m = _logical_transport_height(
        plan,
        module=module,
        installed_module_ids=installed_module_ids,
        config=config,
    )
    grid = _phase5_carrier_route_grid(
        plan,
        module=module,
        installed_module_ids=installed_module_ids,
        robot_positions=robot_positions,
        active_robot_ids=active_robot_ids,
        formation_offsets=formation_offsets,
        carrier_offset=carrier_offset,
        transport_height_m=transport_height_m,
        config=config,
    )
    start_cell = world_to_cell(start, grid)
    goal_cell = world_to_cell(goal, grid)
    grid.obstacle_cells = [
        cell
        for cell in grid.obstacle_cells
        if cell not in {start_cell, goal_cell}
    ]
    def derive_base_routes(
        carrier_points: list[Vec2],
    ) -> dict[str, list[Vec2]]:
        return {
            robot_id: [
                Vec2(
                    x=(
                        carrier.x
                        - carrier_offset.x
                        + offset.x
                    ),
                    y=(
                        carrier.y
                        - carrier_offset.y
                        + offset.y
                    ),
                )
                for carrier in carrier_points
            ]
            for robot_id, offset in formation_offsets.items()
        }

    blocked_edges: set[frozenset[tuple[int, int]]] = set()
    for _attempt in range(256):
        routed = _static_carrier_route(
            grid,
            start=start,
            goal=goal,
            blocked_edges=blocked_edges,
        )
        coarse_route = [
            start,
            *routed[1:-1],
            goal,
        ]
        carrier_route = _densify_route(
            coarse_route,
            maximum_spacing_m=config.carry_sample_spacing_m,
        )
        base_routes = derive_base_routes(carrier_route)
        try:
            clearance_records = _validate_rigid_carrier_route(
                plan,
                module=module,
                installed_module_ids=installed_module_ids,
                robot_positions=robot_positions,
                active_robot_ids=active_robot_ids,
                carrier_route=carrier_route,
                base_routes=base_routes,
                transport_height_m=transport_height_m,
                disabled_robot_ids=disabled_robot_ids,
                source="planned",
                config=config,
            )
        except RoutingError:
            blocked_edge: frozenset[tuple[int, int]] | None = None
            for left, right in zip(
                coarse_route,
                coarse_route[1:],
                strict=False,
            ):
                segment = _densify_route(
                    [left, right],
                    maximum_spacing_m=config.carry_sample_spacing_m,
                )
                try:
                    _validate_rigid_carrier_route(
                        plan,
                        module=module,
                        installed_module_ids=installed_module_ids,
                        robot_positions=robot_positions,
                        active_robot_ids=active_robot_ids,
                        carrier_route=segment,
                        base_routes=derive_base_routes(segment),
                        transport_height_m=transport_height_m,
                        disabled_robot_ids=disabled_robot_ids,
                        source="planned",
                        config=config,
                    )
                except RoutingError:
                    blocked_edge = frozenset(
                        (
                            world_to_cell(left, grid),
                            world_to_cell(right, grid),
                        )
                    )
                    break
            if blocked_edge is None or len(blocked_edge) != 2:
                raise
            blocked_edges.add(blocked_edge)
            continue
        return (
            carrier_route,
            base_routes,
            "rigid_swept_bfs",
            grid,
            transport_height_m,
            clearance_records,
        )
    raise RoutingError("rigid swept-footprint edge repair exceeded its bound")


def _plan_rigid_phase5_job(
    router: RoutingAdapter,
    plan: BuildPlan,
    *,
    module: BuildModule,
    starts: Mapping[str, Vec2],
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    disabled_robot_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> tuple[
    RoutePlan,
    dict[str, Vec2],
    dict[str, Vec2],
    list[Phase5ClearanceMinimum],
]:
    pickup_grid = _phase5_route_grid(
        plan,
        active_module_id=module.module_id,
        active_module_is_logical_payload=False,
        installed_module_ids=installed_module_ids,
        robot_positions=robot_positions,
        active_robot_ids=active_robot_ids,
        config=config,
    )
    staging = Vec2.model_validate(module.staging_pose.position)
    failure_counts: Counter[str] = Counter()
    for formation_offsets in _formation_offset_candidates(
        module,
        list(starts),
        grid=pickup_grid,
        config=config,
    ):
        pickup_goals = {
            robot_id: Vec2(
                x=staging.x + offset.x,
                y=staging.y + offset.y,
            )
            for robot_id, offset in formation_offsets.items()
        }
        try:
            pickup_endpoint_clearances = _base_world_clearance_minima(
                plan,
                phase="approach",
                source="planned",
                routes={
                    robot_id: [goal]
                    for robot_id, goal in pickup_goals.items()
                },
                active_module_id=module.module_id,
                installed_module_ids=installed_module_ids,
                robot_positions=robot_positions,
                active_robot_ids=active_robot_ids,
                disabled_robot_ids=disabled_robot_ids,
                active_module_is_payload=False,
                config=config,
            )
            _require_clearance_records(pickup_endpoint_clearances)
            target = Vec2.model_validate(module.target_pose.position)
            install_goals = {
                robot_id: Vec2(
                    x=target.x + offset.x,
                    y=target.y + offset.y,
                )
                for robot_id, offset in formation_offsets.items()
            }
            install_endpoint_clearances = _base_world_clearance_minima(
                plan,
                phase="return",
                source="planned",
                routes={
                    robot_id: [goal]
                    for robot_id, goal in install_goals.items()
                },
                active_module_id=module.module_id,
                installed_module_ids={
                    *installed_module_ids,
                    module.module_id,
                },
                robot_positions=robot_positions,
                active_robot_ids=active_robot_ids,
                disabled_robot_ids=disabled_robot_ids,
                active_module_is_payload=False,
                config=config,
            )
            _require_clearance_records(install_endpoint_clearances)
            approach = router.route_many(
                pickup_grid,
                starts,
                pickup_goals,
            )
            approach_routes = {
                robot_id: _with_exact_endpoint(
                    approach.world_paths[robot_id],
                    pickup_goals[robot_id],
                )
                for robot_id in sorted(starts)
            }
            approach_routes = _synchronize_safe_route_pair(
                approach_routes,
                minimum_separation_m=(
                    2 * config.robot_footprint_radius_m
                    + config.route_clearance_m
                ),
            )
            approach = approach.model_copy(
                update={"world_paths": approach_routes}
            )
            approach_clearances = _base_world_clearance_minima(
                plan,
                phase="approach",
                source="planned",
                routes=approach_routes,
                active_module_id=module.module_id,
                installed_module_ids=installed_module_ids,
                robot_positions=robot_positions,
                active_robot_ids=active_robot_ids,
                disabled_robot_ids=disabled_robot_ids,
                active_module_is_payload=False,
                config=config,
            )
            _require_clearance_records(approach_clearances)
            (
                _carrier_route,
                base_routes,
                _backend,
                _grid,
                _transport_height,
                _carry_clearances,
            ) = _plan_rigid_carrier_route(
                router,
                plan,
                module=module,
                installed_module_ids=installed_module_ids,
                robot_positions={
                    **robot_positions,
                    **pickup_goals,
                },
                active_robot_ids=active_robot_ids,
                formation_offsets=formation_offsets,
                carrier_offset=Vec2(x=0, y=0),
                disabled_robot_ids=disabled_robot_ids,
                config=config,
            )
            simulated_positions = {
                **robot_positions,
                **{
                    robot_id: route[-1]
                    for robot_id, route in base_routes.items()
                },
            }
            _plan_phase5_dispatch_return(
                router,
                plan,
                module=module,
                starts={
                    robot_id: base_routes[robot_id][-1]
                    for robot_id in sorted(base_routes)
                },
                installed_module_ids={
                    *installed_module_ids,
                    module.module_id,
                },
                robot_positions=simulated_positions,
                active_robot_ids=active_robot_ids,
                disabled_robot_ids=disabled_robot_ids,
                config=config,
            )
        except (RoutingError, RuntimeError, ValueError) as exc:
            failure_counts[f"{type(exc).__name__}: {exc}"] += 1
            continue
        return (
            approach,
            pickup_goals,
            formation_offsets,
            approach_clearances,
        )
    failure_summary = "; ".join(
        f"{count}x {message}"
        for message, count in failure_counts.most_common(3)
    )
    raise DynamicCoppeliaError(
        f"no rigid swept-footprint route is available for "
        f"{module.module_id}: {failure_summary}"
    )


def _plan_phase5_dispatch_return(
    router: RoutingAdapter,
    plan: BuildPlan,
    *,
    module: BuildModule,
    starts: Mapping[str, Vec2],
    installed_module_ids: set[str],
    robot_positions: Mapping[str, Vec2],
    active_robot_ids: set[str],
    disabled_robot_ids: set[str],
    config: Phase5PhysicalYardConfig,
) -> tuple[
    RoutePlan,
    dict[str, Vec2],
    list[Phase5ClearanceMinimum],
]:
    parking_goals = {
        robot.robot_id: Vec2.model_validate(robot.start_pose.position)
        for robot in plan.robots
        if robot.robot_id in active_robot_ids
    }
    parking_grid = _phase5_route_grid(
        plan,
        active_module_id=module.module_id,
        active_module_is_logical_payload=False,
        installed_module_ids=installed_module_ids,
        robot_positions=robot_positions,
        active_robot_ids=active_robot_ids,
        config=config,
    )
    endpoint_cells = {
        world_to_cell(point, parking_grid)
        for point in (*starts.values(), *parking_goals.values())
    }
    parking_grid.obstacle_cells = [
        cell
        for cell in parking_grid.obstacle_cells
        if cell not in endpoint_cells
    ]
    try:
        route = router.route_many(
            parking_grid,
            starts,
            parking_goals,
        )
        exact_routes = {
            robot_id: _with_exact_endpoint(
                route.world_paths[robot_id],
                parking_goals[robot_id],
            )
            for robot_id in sorted(starts)
        }
        exact_routes = _synchronize_safe_route_pair(
            exact_routes,
            minimum_separation_m=(
                2 * config.robot_footprint_radius_m
                + config.route_clearance_m
            ),
        )
        route = route.model_copy(update={"world_paths": exact_routes})
        clearance_records = _base_world_clearance_minima(
            plan,
            phase="return",
            source="planned",
            routes=exact_routes,
            active_module_id=module.module_id,
            installed_module_ids=installed_module_ids,
            robot_positions=robot_positions,
            active_robot_ids=active_robot_ids,
            disabled_robot_ids=disabled_robot_ids,
            active_module_is_payload=False,
            config=config,
        )
        _require_clearance_records(clearance_records)
    except (RoutingError, RuntimeError, ValueError) as exc:
        raise DynamicCoppeliaError(
            f"dispatch return is unreachable after {module.module_id}"
        ) from exc
    return route, parking_goals, clearance_records


def _preflight_phase5_sequential_routes(
    plan: BuildPlan,
    config: Phase5PhysicalYardConfig,
) -> tuple[float, float]:
    cache_key = _sha256_json(
        {
            "plan": plan.model_dump(mode="json"),
            "configuration": config.model_dump(mode="json"),
            "scenarios": ["nominal", "unavailable_robot_recovery"],
        }
    )
    if cache_key in _PHASE5_PREFLIGHT_CACHE:
        return _PHASE5_PREFLIGHT_CACHE[cache_key]
    nominal_formation, nominal_transport = _preflight_phase5_route_scenario(
        plan,
        config,
        scenario="nominal",
    )
    recovery_formation, recovery_transport = _preflight_phase5_route_scenario(
        plan,
        config,
        scenario="unavailable_robot_recovery",
    )
    result = (
        max(nominal_formation, recovery_formation),
        max(nominal_transport, recovery_transport),
    )
    _PHASE5_PREFLIGHT_CACHE[cache_key] = result
    return result


def _preflight_phase5_route_scenario(
    plan: BuildPlan,
    config: Phase5PhysicalYardConfig,
    *,
    scenario: Phase5Scenario,
) -> tuple[float, float]:
    router = create_routing_adapter(seed=0, prefer_w9=False)
    schedule = schedule_build(plan, "sequential")
    modules = {module.module_id: module for module in plan.modules}
    positions = {
        robot.robot_id: Vec2.model_validate(robot.start_pose.position)
        for robot in plan.robots
    }
    usage = {robot.robot_id: 0 for robot in plan.robots}
    installed: set[str] = set()
    active_across_run: set[str] = set()
    unavailable: set[str] = set()
    recovery_clearance_observed = False
    maximum_formation_offset_m = 0.0
    maximum_transport_height_m = 0.0
    for job_index, job in enumerate(schedule.jobs):
        if (
            scenario == "unavailable_robot_recovery"
            and not unavailable
            and len(installed) / len(plan.modules) >= 0.25
        ):
            remaining_jobs = schedule.jobs[job_index:]
            frequency = Counter(
                robot_id
                for remaining_job in remaining_jobs
                for robot_id in remaining_job.robot_ids
            )
            candidates = (
                sorted(set(frequency) & active_across_run)
                or sorted(frequency)
            )
            if not candidates:
                raise ValueError(
                    "recovery preflight has no remaining robot to disable"
                )
            unavailable.add(
                sorted(
                    candidates,
                    key=lambda robot_id: (-frequency[robot_id], robot_id),
                )[0]
            )
        module = modules[job.module_id]
        team = _phase5_team(
            plan,
            module,
            unavailable=unavailable,
            usage=usage,
            config=config,
        )
        active_across_run.update(team)
        starts = {robot_id: positions[robot_id] for robot_id in team}
        pickup_route, pickup_goals, formation_offsets, _approach_clearances = (
            _plan_rigid_phase5_job(
                router,
                plan,
                module=module,
                starts=starts,
                installed_module_ids=installed,
                robot_positions=positions,
                active_robot_ids=set(team),
                disabled_robot_ids=unavailable,
                config=config,
            )
        )
        if not pickup_route.world_paths:
            raise ValueError(f"{module.module_id} staging bay is unreachable")
        maximum_formation_offset_m = max(
            maximum_formation_offset_m,
            *(
                math.hypot(offset.x, offset.y)
                for offset in formation_offsets.values()
            ),
        )
        positions.update(pickup_goals)
        (
            carrier_route,
            base_routes,
            _backend,
            _grid,
            transport_height_m,
            carry_clearances,
        ) = (
            _plan_rigid_carrier_route(
                router,
                plan,
                module=module,
                installed_module_ids=installed,
                robot_positions=positions,
                active_robot_ids=set(team),
                formation_offsets=formation_offsets,
                carrier_offset=Vec2(x=0, y=0),
                disabled_robot_ids=unavailable,
                config=config,
            )
        )
        recovery_clearance_observed = recovery_clearance_observed or any(
            record.obstacle_kind == "disabled_robot"
            and record.evaluated_pair_count > 0
            for record in carry_clearances
        )
        maximum_transport_height_m = max(
            maximum_transport_height_m,
            transport_height_m,
        )
        if not carrier_route or any(
            len(route) != len(carrier_route)
            for route in base_routes.values()
        ):
            raise ValueError(
                f"{module.module_id} rigid install formation is unreachable"
            )
        positions.update(
            {
                robot_id: route[-1]
                for robot_id, route in base_routes.items()
            }
        )
        installed.add(module.module_id)
        parking_route, parking_goals, _return_clearances = _plan_phase5_dispatch_return(
            router,
            plan,
            module=module,
            starts={
                robot_id: positions[robot_id]
                for robot_id in team
            },
            installed_module_ids=installed,
            robot_positions=positions,
            active_robot_ids=set(team),
            disabled_robot_ids=unavailable,
            config=config,
        )
        if not parking_route.world_paths:
            raise ValueError(
                f"{module.module_id} dispatch return is unreachable"
            )
        positions.update(parking_goals)
        for robot_id in team:
            usage[robot_id] += 1
    expected_robots = {robot.robot_id for robot in plan.robots}
    if scenario == "nominal" and active_across_run != expected_robots:
        raise ValueError(
            "Phase 5 balanced two-base allocation did not activate every robot"
        )
    if scenario == "unavailable_robot_recovery" and (
        not unavailable or not recovery_clearance_observed
    ):
        raise ValueError(
            "Phase 5 recovery preflight did not exercise disabled-robot "
            "payload clearance"
        )
    return maximum_formation_offset_m, maximum_transport_height_m


def _telemetry_xy_routes(
    telemetry: Iterable[RobotTelemetry],
    *,
    robot_ids: Iterable[str],
    start_s: float,
    end_s: float,
) -> dict[str, list[Vec2]]:
    routes: dict[str, list[Vec2]] = {
        robot_id: [] for robot_id in sorted(set(robot_ids))
    }
    for sample in telemetry:
        if (
            sample.robot_id in routes
            and start_s <= sample.timestamp_s <= end_s
        ):
            point = Vec2.model_validate(sample.measured_pose.position)
            if not routes[sample.robot_id] or _distance(
                routes[sample.robot_id][-1], point
            ) > 1e-12:
                routes[sample.robot_id].append(point)
    missing = sorted(
        robot_id for robot_id, route in routes.items() if not route
    )
    if missing:
        raise DynamicCoppeliaError(
            "measured phase route is empty for " + ", ".join(missing)
        )
    return routes


def _official_remote_client_identity(
    executor: DynamicCoppeliaExecutor,
) -> _OfficialRemoteClientIdentity:
    if executor.client_factory is not None:
        raise DynamicCoppeliaError(
            "injected simulator clients cannot establish live runtime attestation"
        )
    if (
        not executor.is_ready
        or executor.client is None
        or executor.sim is None
        or executor.root_handle is None
    ):
        raise DynamicCoppeliaError(
            "live runtime attestation requires a connected generated scene"
        )
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as exc:  # pragma: no cover - deployment dependency guard.
        raise DynamicCoppeliaError(
            "the official Coppelia ZeroMQ remote API client is unavailable"
        ) from exc
    if type(executor.client) is not RemoteAPIClient:
        raise DynamicCoppeliaError(
            "live runtime attestation requires the official RemoteAPIClient"
        )
    client_uuid = getattr(executor.client, "uuid", None)
    protocol_version = getattr(executor.client, "VERSION", None)
    send_count = getattr(executor.client, "sendCnt", None)
    if (
        not isinstance(client_uuid, str)
        or re.fullmatch(_UUID_PATTERN, client_uuid) is None
        or not isinstance(protocol_version, int)
        or isinstance(protocol_version, bool)
        or protocol_version < 1
        or not isinstance(send_count, int)
        or isinstance(send_count, bool)
        or send_count < 0
    ):
        raise DynamicCoppeliaError(
            "the remote API client identity is incomplete"
        )
    try:
        package_version = version("coppeliasim_zmqremoteapi_client")
    except PackageNotFoundError as exc:
        raise DynamicCoppeliaError(
            "the remote API client package version is unavailable"
        ) from exc
    if not package_version.strip():
        raise DynamicCoppeliaError(
            "the remote API client package version is empty"
        )
    return _OfficialRemoteClientIdentity(
        client_uuid=client_uuid,
        protocol_version=protocol_version,
        package_version=package_version.strip(),
        send_count=send_count,
    )


def _runtime_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_hex": bytes(value).hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _runtime_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_runtime_jsonable(item) for item in value]
    raise DynamicCoppeliaError(
        f"remote API metadata contains unsupported {type(value).__name__} data"
    )


def _capture_remote_api_info(executor: DynamicCoppeliaExecutor) -> str:
    call = getattr(executor.client, "call", None)
    if not callable(call):
        raise DynamicCoppeliaError(
            "the remote API client cannot query simulator capabilities"
        )
    try:
        info = call("zmqRemoteApi.info", ["sim"])
    except Exception as exc:
        raise DynamicCoppeliaError(
            "the simulator capability inventory query failed"
        ) from exc
    if not isinstance(info, Mapping):
        raise DynamicCoppeliaError(
            "the simulator capability inventory is not a mapping"
        )
    missing = sorted(
        set(_LIVE_REQUIRED_REMOTE_API_CAPABILITIES) - set(info)
    )
    if missing:
        raise DynamicCoppeliaError(
            f"the simulator lacks live-attestation capabilities: {missing}"
        )
    return _sha256_json(_runtime_jsonable(info))


def _format_simulator_integer_version(
    program_version: int,
    program_revision: int,
) -> str:
    major = program_version // 10_000
    minor = (program_version // 100) % 100
    patch = program_version % 100
    return f"{major}.{minor}.{patch} rev {program_revision}"


def _query_simulator_identity(
    sim: object,
) -> tuple[
    str,
    Literal["string_parameter", "integer_parameters"],
    int | None,
    int | None,
]:
    get_string = getattr(sim, "getStringParam", None)
    string_parameter = getattr(sim, "stringparam_application_version", None)
    if callable(get_string) and string_parameter is not None:
        try:
            string_version = get_string(string_parameter)
        except Exception:
            string_version = None
        if isinstance(string_version, str) and string_version.strip():
            return string_version.strip(), "string_parameter", None, None

    get_integer = getattr(sim, "getInt32Param", None)
    version_parameter = getattr(sim, "intparam_program_version", None)
    revision_parameter = getattr(sim, "intparam_program_revision", None)
    if (
        not callable(get_integer)
        or version_parameter is None
        or revision_parameter is None
    ):
        raise DynamicCoppeliaError(
            "the simulator exposes no definitive application identity"
        )
    try:
        program_version = get_integer(version_parameter)
        program_revision = get_integer(revision_parameter)
    except Exception as exc:
        raise DynamicCoppeliaError(
            "the simulator application identity query failed"
        ) from exc
    if (
        not isinstance(program_version, int)
        or isinstance(program_version, bool)
        or program_version <= 0
        or not isinstance(program_revision, int)
        or isinstance(program_revision, bool)
        or program_revision < 0
    ):
        raise DynamicCoppeliaError(
            "the simulator integer application identity is invalid"
        )
    return (
        _format_simulator_integer_version(program_version, program_revision),
        "integer_parameters",
        program_version,
        program_revision,
    )


def _runtime_number(sim: object, method_name: str) -> float:
    method = getattr(sim, method_name, None)
    if not callable(method):
        raise DynamicCoppeliaError(
            f"the simulator lacks {method_name} for live attestation"
        )
    try:
        value = method()
    except Exception as exc:
        raise DynamicCoppeliaError(
            f"the simulator {method_name} query failed"
        ) from exc
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DynamicCoppeliaError(
            f"the simulator {method_name} value is invalid"
        )
    return float(value)


def _runtime_integer(sim: object, method_name: str, *args: object) -> int:
    method = getattr(sim, method_name, None)
    if not callable(method):
        raise DynamicCoppeliaError(
            f"the simulator lacks {method_name} for live attestation"
        )
    try:
        value = method(*args)
    except Exception as exc:
        raise DynamicCoppeliaError(
            f"the simulator {method_name} query failed"
        ) from exc
    if not isinstance(value, int) or isinstance(value, bool):
        raise DynamicCoppeliaError(
            f"the simulator {method_name} value is invalid"
        )
    return value


def _runtime_challenge_payload(
    *,
    session_nonce: str,
    client_uuid: str,
    endpoint_host: str,
    endpoint_port: int,
    plan_digest: str,
    configuration_origin_digest: str,
    scene_root_uid: int,
    remote_api_info_sha256: str,
    simulator_version: str,
) -> dict[str, object]:
    return {
        "schema_version": (
            "construction_intelligence.coppelia_runtime_challenge.v1"
        ),
        "session_nonce": session_nonce,
        "client_uuid": client_uuid,
        "endpoint": {"host": endpoint_host, "port": endpoint_port},
        "plan_digest": plan_digest,
        "configuration_origin_digest": configuration_origin_digest,
        "scene_root_uid": scene_root_uid,
        "remote_api_info_sha256": remote_api_info_sha256,
        "simulator_version": simulator_version,
    }


def _runtime_response_bytes(value: object) -> bytes:
    if isinstance(value, str):
        return value.encode()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise DynamicCoppeliaError(
        "the simulator challenge response is not a byte buffer"
    )


def _begin_live_runtime_attestation(
    executor: DynamicCoppeliaExecutor,
    scenario: ScenarioManifest,
    physical_yard: Phase5PhysicalYardManifest,
) -> _LiveRuntimeSession:
    if executor in _LIVE_RUNTIME_SESSIONS:
        raise DynamicCoppeliaError(
            "this executor already owns a live runtime attestation"
        )
    if (
        executor.started
        or executor.physics_steps != 0
        or executor.commands
        or executor.telemetry
        or executor.installed_modules
    ):
        raise DynamicCoppeliaError(
            "live runtime attestation must begin before simulator execution"
        )
    identity = _official_remote_client_identity(executor)
    remote_api_info_sha256 = _capture_remote_api_info(executor)
    simulator_version, identity_source, program_version, program_revision = (
        _query_simulator_identity(executor.sim)
    )
    root_handle = executor.root_handle
    if root_handle is None:  # Narrow type guard after the official-client probe.
        raise DynamicCoppeliaError("the generated scene root is unavailable")
    root_uid = _runtime_integer(executor.sim, "getObjectUid", root_handle)
    if root_uid <= 0:
        raise DynamicCoppeliaError("the generated scene root UID is invalid")
    stopped_state = getattr(executor.sim, "simulation_stopped", None)
    if not isinstance(stopped_state, int) or isinstance(stopped_state, bool):
        raise DynamicCoppeliaError(
            "the simulator stopped-state constant is invalid"
        )
    state_before = _runtime_integer(executor.sim, "getSimulationState")
    if state_before != stopped_state:
        raise DynamicCoppeliaError(
            "live runtime attestation must begin while simulation is stopped"
        )
    time_before = _runtime_number(executor.sim, "getSimulationTime")
    if time_before < 0:
        raise DynamicCoppeliaError("simulator time cannot be negative")
    plan_digest = _sha256_json(scenario.plan.model_dump(mode="json"))
    configuration_origin_digest = _sha256_json(
        {
            "executor": executor.config.model_dump(mode="json"),
            "physical_yard": physical_yard.model_dump(mode="json"),
            "evidence_kind": "live_coppelia",
        }
    )
    session_nonce = secrets.token_hex(32)
    challenge_tag = (
        "construction_intelligence.phase5.live_attestation."
        f"{session_nonce[:16]}"
    )
    challenge_payload = _canonical_json(
        _runtime_challenge_payload(
            session_nonce=session_nonce,
            client_uuid=identity.client_uuid,
            endpoint_host=executor.config.host,
            endpoint_port=executor.config.port,
            plan_digest=plan_digest,
            configuration_origin_digest=configuration_origin_digest,
            scene_root_uid=root_uid,
            remote_api_info_sha256=remote_api_info_sha256,
            simulator_version=simulator_version,
        )
    ).encode()
    write_custom_data = getattr(executor.sim, "writeCustomDataBlock", None)
    read_custom_data = getattr(executor.sim, "readCustomDataBlock", None)
    if not callable(write_custom_data) or not callable(read_custom_data):
        raise DynamicCoppeliaError(
            "the simulator cannot round-trip a live attestation challenge"
        )
    try:
        write_custom_data(root_handle, challenge_tag, challenge_payload)
        response = _runtime_response_bytes(
            read_custom_data(root_handle, challenge_tag)
        )
    except DynamicCoppeliaError:
        raise
    except Exception as exc:
        raise DynamicCoppeliaError(
            "the simulator live-attestation challenge round-trip failed"
        ) from exc
    if response != challenge_payload:
        raise DynamicCoppeliaError(
            "the simulator live-attestation challenge response differs"
        )
    challenge_sha256 = hashlib.sha256(challenge_payload).hexdigest()
    identity_after = _official_remote_client_identity(executor)
    session = _LiveRuntimeSession(
        scenario_id=scenario.scenario_id,
        scenario_seed=scenario.seed,
        plan_digest=plan_digest,
        configuration_origin_digest=configuration_origin_digest,
        client_uuid=identity.client_uuid,
        protocol_version=identity.protocol_version,
        package_version=identity.package_version,
        client_send_count_before=identity.send_count,
        endpoint_host=executor.config.host,
        endpoint_port=executor.config.port,
        remote_api_info_sha256=remote_api_info_sha256,
        simulator_version=simulator_version,
        simulator_identity_source=identity_source,
        simulator_program_version=program_version,
        simulator_program_revision=program_revision,
        scene_root_handle=root_handle,
        scene_root_uid=root_uid,
        simulation_stopped_state=stopped_state,
        simulation_state_before=state_before,
        simulation_time_before_s=time_before,
        session_nonce=session_nonce,
        challenge_tag=challenge_tag,
        challenge_payload=challenge_payload,
        challenge_payload_sha256=challenge_sha256,
    )
    if (
        identity_after.client_uuid != session.client_uuid
        or identity_after.protocol_version != session.protocol_version
        or identity_after.package_version != session.package_version
        or identity_after.send_count <= session.client_send_count_before
    ):
        raise DynamicCoppeliaError(
            "the remote client identity changed during challenge capture"
        )
    executor.runtime_events.append(
        {
            "timestamp_s": executor.simulation_time_s,
            "event": "live_runtime_attestation_started",
            "client_uuid": session.client_uuid,
            "remote_api_info_sha256": session.remote_api_info_sha256,
            "simulator_version": session.simulator_version,
            "scene_root_uid": session.scene_root_uid,
            "challenge_payload_sha256": session.challenge_payload_sha256,
            "simulation_state": session.simulation_state_before,
        }
    )
    _LIVE_RUNTIME_SESSIONS[executor] = session
    return session


def _finalize_live_runtime_attestation(
    executor: DynamicCoppeliaExecutor,
) -> _LiveRuntimeSession:
    session = _LIVE_RUNTIME_SESSIONS.get(executor)
    if session is None:
        raise DynamicCoppeliaError(
            "live runtime attestation was not established before execution"
        )
    if session.finalized:
        return session
    identity = _official_remote_client_identity(executor)
    if (
        identity.client_uuid != session.client_uuid
        or identity.protocol_version != session.protocol_version
        or identity.package_version != session.package_version
    ):
        raise DynamicCoppeliaError(
            "the remote client identity changed during the live run"
        )
    if _capture_remote_api_info(executor) != session.remote_api_info_sha256:
        raise DynamicCoppeliaError(
            "the simulator capability inventory changed during the live run"
        )
    simulator_identity = _query_simulator_identity(executor.sim)
    if simulator_identity != (
        session.simulator_version,
        session.simulator_identity_source,
        session.simulator_program_version,
        session.simulator_program_revision,
    ):
        raise DynamicCoppeliaError(
            "the simulator application identity changed during the live run"
        )
    if (
        _runtime_integer(
            executor.sim,
            "getObjectUid",
            session.scene_root_handle,
        )
        != session.scene_root_uid
    ):
        raise DynamicCoppeliaError(
            "the generated scene root changed during the live run"
        )
    read_custom_data = getattr(executor.sim, "readCustomDataBlock", None)
    if not callable(read_custom_data):
        raise DynamicCoppeliaError(
            "the simulator cannot re-read the live attestation challenge"
        )
    try:
        response = _runtime_response_bytes(
            read_custom_data(
                session.scene_root_handle,
                session.challenge_tag,
            )
        )
    except DynamicCoppeliaError:
        raise
    except Exception as exc:
        raise DynamicCoppeliaError(
            "the simulator live-attestation challenge re-read failed"
        ) from exc
    if response != session.challenge_payload:
        raise DynamicCoppeliaError(
            "the simulator live-attestation challenge did not persist"
        )
    state_after = _runtime_integer(executor.sim, "getSimulationState")
    if state_after != session.simulation_stopped_state:
        raise DynamicCoppeliaError(
            "live runtime attestation must end while simulation is stopped"
        )
    time_after = _runtime_number(executor.sim, "getSimulationTime")
    if time_after < 0:
        raise DynamicCoppeliaError("simulator time cannot be negative")
    identity_after = _official_remote_client_identity(executor)
    if identity_after.send_count <= session.client_send_count_before:
        raise DynamicCoppeliaError(
            "the remote client call counter did not advance"
        )
    session.simulation_state_after = state_after
    session.simulation_time_after_s = time_after
    session.client_send_count_after = identity_after.send_count
    session.physics_steps = executor.physics_steps
    session.command_count = len(executor.commands)
    session.telemetry_count = len(executor.telemetry)
    session.command_stream_sha256 = _sha256_json(
        [item.model_dump(mode="json") for item in executor.commands]
    )
    session.telemetry_stream_sha256 = _sha256_json(
        [item.model_dump(mode="json") for item in executor.telemetry]
    )
    session.installed_module_ids = sorted(executor.installed_modules)
    session.finalized = True
    executor.runtime_events.append(
        {
            "timestamp_s": executor.simulation_time_s,
            "event": "live_runtime_attestation_finalized",
            "client_uuid": session.client_uuid,
            "scene_root_uid": session.scene_root_uid,
            "challenge_response_sha256": hashlib.sha256(response).hexdigest(),
            "simulation_state": state_after,
            "physics_steps": session.physics_steps,
            "command_count": session.command_count,
            "telemetry_count": session.telemetry_count,
            "client_send_count_after": session.client_send_count_after,
        }
    )
    return session


class Phase5FullCottageRunner:
    """Deterministic full-cottage wheel-control evidence orchestration."""

    def __init__(
        self,
        executor: DynamicCoppeliaExecutor,
        scenario: ScenarioManifest,
        *,
        routing_adapter: RoutingAdapter | None = None,
        evidence_kind: EvidenceKind = "offline_harness",
        physical_yard: Phase5PhysicalYardManifest | None = None,
    ) -> None:
        if executor.plan.plan_id != scenario.plan.plan_id:
            raise ValueError("executor and scenario plans must match")
        if evidence_kind == "live_coppelia" and executor.client_factory is not None:
            raise ValueError(
                "injected simulator clients cannot be attested as live Coppelia evidence"
            )
        if physical_yard is None:
            raise ValueError(
                "Phase 5 requires a verified physical-yard transformation"
            )
        if (
            physical_yard.transformed_plan_digest
            != _sha256_json(scenario.plan.model_dump(mode="json"))
            or physical_yard.allocator_configuration_digest
            != _sha256_json(
                physical_yard.configuration.model_dump(mode="json")
            )
            or not physical_yard.initial_clearance_preflight_passed
            or not physical_yard.sequential_route_preflight_passed
            or not physical_yard.all_robots_active_preflight_passed
            or not physical_yard.continuous_world_clearance_preflight_passed
            or not physical_yard.recovery_route_preflight_passed
        ):
            raise ValueError(
                "Phase 5 physical-yard manifest does not match the scenario plan"
            )
        self.executor = executor
        self.scenario = scenario.model_copy(deep=True)
        self.plan = executor.plan
        self.router = routing_adapter or create_routing_adapter(
            seed=scenario.seed,
            prefer_w9=False,
        )
        self.evidence_kind = evidence_kind
        self.physical_yard = physical_yard.model_copy(deep=True)
        self.yard_config = physical_yard.configuration.model_copy(deep=True)
        self.trace: list[dict[str, object]] = []
        self.replay: list[Phase5JobReplay] = []
        self.recovery: Phase5RecoveryRecord | None = None
        self.team_usage = {robot.robot_id: 0 for robot in self.plan.robots}
        self.trace.append(
            {
                "timestamp_s": 0.0,
                "event": "physical_yard_preflight_passed",
                "transformed_plan_digest": (
                    physical_yard.transformed_plan_digest
                ),
                "allocator_configuration_digest": (
                    physical_yard.allocator_configuration_digest
                ),
                "phase5_team_size": self.yard_config.phase5_team_size,
            }
        )

        self.live_runtime_session: _LiveRuntimeSession | None = None
        if self.evidence_kind == "live_coppelia":
            self.live_runtime_session = _begin_live_runtime_attestation(
                self.executor,
                self.scenario,
                self.physical_yard,
            )

    def run(self, scenario: Phase5Scenario) -> Phase5RunResult:
        baseline = schedule_build(self.plan, "sequential")
        modules = {module.module_id: module for module in self.plan.modules}
        active_robot_ids: set[str] = set()
        error: Exception | None = None
        try:
            self.executor.start()
            for index, planned_job in enumerate(baseline.jobs):
                if scenario == "unavailable_robot_recovery":
                    self._maybe_disable_robot(index, baseline.jobs)
                module = modules[planned_job.module_id]
                executed_team = self._select_team(module)
                active_robot_ids.update(executed_team)
                replay = self._execute_job(
                    index=index,
                    module=module,
                    planned_robot_ids=planned_job.robot_ids,
                    executed_robot_ids=executed_team,
                    planned_start_s=planned_job.start_s,
                    planned_pickup_s=planned_job.pickup_s,
                    planned_end_s=planned_job.end_s,
                )
                self.replay.append(replay)
                if replay.reassigned_after_unavailability and self.recovery is not None:
                    self.recovery.reassigned_job_ids.append(replay.job_id)
        except Exception as exc:  # Preserve partial measured evidence for diagnosis.
            error = exc
        finally:
            try:
                self.executor.stop()
            except Exception as stop_error:
                if error is None:
                    error = stop_error
            if self.evidence_kind == "live_coppelia":
                try:
                    self.live_runtime_session = (
                        _finalize_live_runtime_attestation(self.executor)
                    )
                except Exception as attestation_error:
                    if error is None:
                        error = attestation_error

        if self.recovery is not None:
            cutoff = self.recovery.stop_command_index
            self.recovery.commands_after_stop = sum(
                command.robot_id == self.recovery.robot_id
                for command in self.executor.commands[cutoff:]
            )

        acceptance = self._acceptance(scenario, active_robot_ids)
        status: Literal["completed", "failed"] = (
            "completed" if error is None and all(acceptance.values()) else "failed"
        )
        live_gate_passed = self.evidence_kind == "live_coppelia" and status == "completed"
        metrics = self.executor.diagnostics()
        metrics.update(
            {
                "scenario": scenario,
                "scenario_id": self.scenario.scenario_id,
                "expected_module_count": len(self.plan.modules),
                "installed_module_count": len(self.executor.installed_modules),
                "replay_job_count": len(self.replay),
                "active_robot_ids": sorted(active_robot_ids),
                "maximum_formation_offset_m": max(
                    (
                        math.hypot(offset.x, offset.y)
                        for item in self.replay
                        for offset in item.formation_offsets.values()
                    ),
                    default=0.0,
                ),
                "maximum_preflight_formation_offset_m": (
                    self.physical_yard.maximum_preflight_formation_offset_m
                ),
                "maximum_logical_transport_height_m": max(
                    (
                        item.logical_transport_height_m
                        for item in self.replay
                    ),
                    default=0.0,
                ),
                "maximum_preflight_transport_height_m": (
                    self.physical_yard.maximum_preflight_transport_height_m
                ),
                "payload_transport": "logical_carrier",
                "physical_yard": self.physical_yard.model_dump(mode="json"),
                "physical_yard_digest": _sha256_json(
                    self.physical_yard.model_dump(mode="json")
                ),
                "live_evidence": self.evidence_kind == "live_coppelia",
                "live_gate_passed": live_gate_passed,
            }
        )
        trace = sorted(
            [*self.executor.runtime_events, *self.trace],
            key=_event_sort_key,
        )
        return Phase5RunResult(
            scenario=scenario,
            evidence_kind=self.evidence_kind,
            status=status,
            scenario_id=self.scenario.scenario_id,
            scenario_seed=self.scenario.seed,
            expected_module_count=len(self.plan.modules),
            installed_module_ids=sorted(self.executor.installed_modules),
            active_robot_ids=sorted(active_robot_ids),
            replay=self.replay,
            recovery=self.recovery,
            trace=trace,
            metrics=metrics,
            acceptance=acceptance,
            live_gate_passed=live_gate_passed,
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
        )

    def _return_team_to_dispatch(
        self,
        module: BuildModule,
        robot_ids: list[str],
        job_id: str,
    ) -> tuple[
        dict[str, list[Vec2]],
        dict[str, Vec2],
        list[Phase5ClearanceMinimum],
        float,
    ]:
        approach_robot_positions = self._measured_robot_pose_positions()
        measured_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in approach_robot_positions.items()
        }
        route, goals, clearance_records = _plan_phase5_dispatch_return(
            self.router,
            self.plan,
            module=module,
            starts={
                robot_id: measured_positions[robot_id]
                for robot_id in robot_ids
            },
            installed_module_ids=set(self.executor.installed_modules),
            robot_positions=measured_positions,
            active_robot_ids=set(robot_ids),
            disabled_robot_ids=set(self.executor.disabled_robots),
            config=self.yard_config,
        )
        routes = {
            robot_id: _with_exact_endpoint(
                route.world_paths[robot_id],
                goals[robot_id],
            )
            for robot_id in robot_ids
        }
        self.executor.follow_routes(routes)
        self.trace.append(
            {
                "timestamp_s": self.executor.simulation_time_s,
                "event": "team_returned_to_dispatch",
                "job_id": job_id,
                "module_id": module.module_id,
                "robot_ids": sorted(robot_ids),
                "routing_backend": route.backend,
            }
        )
        return (
            routes,
            measured_positions,
            clearance_records,
            self.executor.simulation_time_s,
        )

    def _maybe_disable_robot(
        self,
        next_job_index: int,
        jobs: list[ScheduledJob],
    ) -> None:
        if self.recovery is not None:
            return
        completion = len(self.executor.installed_modules) / len(self.plan.modules)
        if completion < 0.25:
            return
        remaining_jobs = jobs[next_job_index:]
        remaining_teams = [job.robot_ids for job in remaining_jobs]
        frequency = Counter(robot_id for team in remaining_teams for robot_id in team)
        if not frequency:
            raise DynamicCoppeliaError("recovery gate has no remaining work to reassign")
        previously_active = {
            robot_id for replay in self.replay for robot_id in replay.executed_robot_ids
        }
        candidates = sorted(set(frequency) & previously_active) or sorted(frequency)
        robot_id = sorted(candidates, key=lambda item: (-frequency[item], item))[0]
        self.executor.disable_robot(robot_id)
        disabled_at_s = self.executor.simulation_time_s
        settle_samples = self.executor.settle_disabled_robot(robot_id)
        self.recovery = Phase5RecoveryRecord(
            robot_id=robot_id,
            disabled_at_s=disabled_at_s,
            completion_fraction_at_disable=completion,
            stop_command_index=len(self.executor.commands),
            remaining_module_count=len(remaining_jobs),
            settled_at_s=self.executor.simulation_time_s,
            settle_sample_count=len(settle_samples),
            settle_displacement_m=self.executor.disabled_settle_displacement_m[
                robot_id
            ],
            settled=self.executor.disabled_settled[robot_id],
        )
        self.trace.append(
            {
                "timestamp_s": disabled_at_s,
                "event": "robot_unavailable",
                "robot_id": robot_id,
                "completion_fraction": completion,
                "remaining_module_count": len(remaining_jobs),
                "settled_at_s": self.executor.simulation_time_s,
                "settle_sample_count": len(settle_samples),
            }
        )

    def _select_team(self, module: BuildModule) -> list[str]:
        team = _phase5_team(
            self.plan,
            module,
            unavailable=set(self.executor.disabled_robots),
            usage=self.team_usage,
            config=self.yard_config,
        )
        for robot_id in team:
            self.team_usage[robot_id] += 1
        return team

    def _execute_job(
        self,
        *,
        index: int,
        module: BuildModule,
        planned_robot_ids: list[str],
        executed_robot_ids: list[str],
        planned_start_s: int,
        planned_pickup_s: int,
        planned_end_s: int,
    ) -> Phase5JobReplay:
        job_id = f"job-{index:03d}-{module.module_id}"
        installed_before = set(self.executor.installed_modules)
        started_at_s = self.executor.simulation_time_s
        telemetry_start = len(self.executor.telemetry)
        formation_error_start = len(self.executor.formation_errors_m)
        assignment_error_start = len(
            self.executor.formation_assignment_errors_m
        )
        spacing_error_start = len(self.executor.formation_spacing_errors_m)
        install_error_start = len(self.executor.install_errors_m)
        approach_robot_positions = self._measured_robot_pose_positions()
        measured_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in approach_robot_positions.items()
        }
        starts = {
            robot_id: measured_positions[robot_id]
            for robot_id in executed_robot_ids
        }
        (
            approach,
            pickup_goals,
            planned_formation_offsets,
            planned_approach_clearances,
        ) = (
            _plan_rigid_phase5_job(
                self.router,
                self.plan,
                module=module,
                starts=starts,
                installed_module_ids=set(
                    self.executor.installed_modules
                ),
                robot_positions=measured_positions,
                active_robot_ids=set(executed_robot_ids),
                disabled_robot_ids=set(self.executor.disabled_robots),
                config=self.yard_config,
            )
        )
        approach_grid = _phase5_route_grid(
            self.plan,
            active_module_id=module.module_id,
            active_module_is_logical_payload=False,
            installed_module_ids=set(self.executor.installed_modules),
            robot_positions=measured_positions,
            active_robot_ids=set(executed_robot_ids),
            config=self.yard_config,
        )
        approach_routes = {
            robot_id: _with_exact_endpoint(
                approach.world_paths[robot_id],
                pickup_goals[robot_id],
            )
            for robot_id in executed_robot_ids
        }
        self.trace.append(
            {
                "timestamp_s": started_at_s,
                "event": "job_started",
                "job_id": job_id,
                "module_id": module.module_id,
                "planned_robot_ids": planned_robot_ids,
                "executed_robot_ids": executed_robot_ids,
                "approach_routing_backend": approach.backend,
                "approach_dynamic_obstacle_cells": len(
                    approach_grid.obstacle_cells
                ),
            }
        )
        self.executor.follow_routes(
            approach_routes,
            waypoint_tolerance_m=min(
                self.executor.config.waypoint_tolerance_m,
                self.yard_config.carry_sample_spacing_m / 2,
            ),
        )
        pickup_clearance = self._minimum_active_module_clearance(
            module,
            executed_robot_ids,
            installed=False,
        )
        transport_height_m = _logical_transport_height(
            self.plan,
            module=module,
            installed_module_ids=set(self.executor.installed_modules),
            config=self.yard_config,
        )
        self.executor.attach_logical_payload(
            module.module_id,
            executed_robot_ids,
            assigned_targets={
                robot_id: approach_routes[robot_id][-1]
                for robot_id in executed_robot_ids
            },
            transport_height_m=transport_height_m,
        )
        pickup_at_s = self.executor.simulation_time_s
        carry_telemetry_start = len(self.executor.telemetry)
        self.trace.append(
            {
                "timestamp_s": pickup_at_s,
                "event": "logical_payload_attached",
                "job_id": job_id,
                "module_id": module.module_id,
            }
        )
        carry_robot_positions = self._measured_robot_pose_positions()
        measured_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in carry_robot_positions.items()
        }
        measured_centroid = _centroid(
            measured_positions[robot_id]
            for robot_id in executed_robot_ids
        )
        formation_offsets = {
            robot_id: Vec2(
                x=(
                    measured_positions[robot_id].x
                    - measured_centroid.x
                ),
                y=(
                    measured_positions[robot_id].y
                    - measured_centroid.y
                ),
            )
            for robot_id in executed_robot_ids
        }
        raw_carrier_offset = self.executor.logical_carrier_offsets[
            module.module_id
        ]
        carrier_offset_xy = Vec2(
            x=raw_carrier_offset.x,
            y=raw_carrier_offset.y,
        )
        (
            carry_route,
            assigned_carry_routes,
            carry_backend,
            carry_grid,
            planned_transport_height_m,
            planned_carry_clearances,
        ) = _plan_rigid_carrier_route(
            self.router,
            self.plan,
            module=module,
            installed_module_ids=set(self.executor.installed_modules),
            robot_positions=measured_positions,
            active_robot_ids=set(executed_robot_ids),
            formation_offsets=formation_offsets,
            carrier_offset=carrier_offset_xy,
            disabled_robot_ids=set(self.executor.disabled_robots),
            config=self.yard_config,
        )
        if not math.isclose(
            transport_height_m,
            planned_transport_height_m,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise DynamicCoppeliaError(
                "logical transport height changed after attachment"
            )
        planned_payload_clearances = _payload_clearances_for_routes(
            module=module,
            carrier_route=carry_route,
            base_routes=assigned_carry_routes,
        )
        self.trace.append(
            {
                "timestamp_s": pickup_at_s,
                "event": "carry_route_planned",
                "job_id": job_id,
                "module_id": module.module_id,
                "carry_routing_backend": carry_backend,
                "carry_dynamic_obstacle_cells": len(
                    carry_grid.obstacle_cells
                ),
                "carry_sample_count": len(carry_route),
                "maximum_planned_formation_offset_m": max(
                    (
                        math.hypot(offset.x, offset.y)
                        for offset in planned_formation_offsets.values()
                    ),
                    default=0.0,
                ),
                "maximum_measured_formation_offset_m": max(
                    (
                        math.hypot(offset.x, offset.y)
                        for offset in formation_offsets.values()
                    ),
                    default=0.0,
                ),
                "minimum_planned_payload_clearance_m": min(
                    planned_payload_clearances
                ),
            }
        )
        self.executor.follow_synchronized_routes(
            assigned_carry_routes
        )
        install_clearance = self._minimum_active_module_clearance(
            module,
            executed_robot_ids,
            installed=True,
        )
        contact_module_ids = _installation_contact_module_ids(
            self.plan,
            module=module,
            installed_module_ids=set(self.executor.installed_modules),
            config=self.yard_config,
        )
        self.executor.install_logical_payload(
            module.module_id,
            assigned_targets={
                robot_id: assigned_carry_routes[robot_id][-1]
                for robot_id in executed_robot_ids
            },
            contact_module_ids=contact_module_ids,
        )
        installed_at_s = self.executor.simulation_time_s
        if not self.executor.logical_installation_snap_records:
            raise DynamicCoppeliaError(
                "logical installation snap was not recorded"
            )
        runtime_snap = self.executor.logical_installation_snap_records[-1]
        raw_contact_module_ids = runtime_snap.get("contact_module_ids")
        if not isinstance(raw_contact_module_ids, list) or not all(
            isinstance(value, str) for value in raw_contact_module_ids
        ):
            raise DynamicCoppeliaError(
                "logical installation snap has invalid contact module IDs"
            )
        raw_snap_scope = runtime_snap.get("scope")
        if raw_snap_scope != "final_target_pose_only":
            raise DynamicCoppeliaError(
                "logical installation contact exceeded the final target pose"
            )
        installation_snap = Phase5InstallationSnap(
            from_pose=Pose3D.model_validate(runtime_snap.get("from_pose")),
            target_pose=Pose3D.model_validate(runtime_snap.get("target_pose")),
            at_s=_number_value(runtime_snap.get("timestamp_s")),
            target_pose_digest=str(runtime_snap.get("target_pose_sha256")),
            contact_module_ids=raw_contact_module_ids,
            scope="final_target_pose_only",
        )
        if installation_snap.at_s != installed_at_s:
            raise DynamicCoppeliaError(
                "logical installation snap timestamp differs from install"
            )
        self.trace.append(
            {
                "timestamp_s": installed_at_s,
                "event": "module_installed",
                "job_id": job_id,
                "module_id": module.module_id,
            }
        )
        (
            return_routes,
            return_robot_positions_xy,
            planned_return_clearances,
            returned_at_s,
        ) = self._return_team_to_dispatch(
            module,
            executed_robot_ids,
            job_id,
        )
        return_robot_positions = {
            robot_id: Vec3(x=point.x, y=point.y, z=0.0)
            for robot_id, point in return_robot_positions_xy.items()
        }
        measured = self.executor.telemetry[telemetry_start:]
        carry_measurements = self.executor.telemetry[carry_telemetry_start:]
        measured_carry_samples = {
            robot_id: [
                item
                for item in carry_measurements
                if item.robot_id == robot_id
                and pickup_at_s <= item.timestamp_s <= installed_at_s
            ]
            for robot_id in executed_robot_ids
        }
        measured_carry_routes = {
            robot_id: [
                Vec2(
                    x=item.measured_pose.position.x,
                    y=item.measured_pose.position.y,
                )
                for item in samples
            ]
            for robot_id, samples in measured_carry_samples.items()
        }
        measured_horizons = {
            len(route) for route in measured_carry_routes.values()
        }
        if len(measured_horizons) != 1 or not measured_horizons:
            raise DynamicCoppeliaError(
                "measured synchronized routes do not share one horizon"
            )
        measured_payload_clearances: list[float] = []
        measured_synchronized_errors: list[float] = []
        measured_carrier_route: list[Vec2] = []
        measured_yaws = {
            module.staging_pose.rotation_rpy_degrees.z,
            module.target_pose.rotation_rpy_degrees.z,
        }
        for sample_index in range(measured_horizons.pop()):
            measured_points = {
                robot_id: measured_carry_routes[robot_id][sample_index]
                for robot_id in executed_robot_ids
            }
            measured_center = _centroid(measured_points.values())
            measured_carrier = Vec2(
                x=measured_center.x + carrier_offset_xy.x,
                y=measured_center.y + carrier_offset_xy.y,
            )
            measured_carrier_route.append(measured_carrier)
            measured_payload_clearances.append(
                min(
                    _oriented_module_clearance(
                        point,
                        measured_carrier,
                        module,
                        yaw,
                    )
                    for point in measured_points.values()
                    for yaw in measured_yaws
                )
            )
            first_id, second_id = sorted(executed_robot_ids)
            measured_synchronized_errors.append(
                math.hypot(
                    (
                        measured_points[second_id].x
                        - measured_points[first_id].x
                    )
                    - (
                        formation_offsets[second_id].x
                        - formation_offsets[first_id].x
                    ),
                    (
                        measured_points[second_id].y
                        - measured_points[first_id].y
                    )
                    - (
                        formation_offsets[second_id].y
                        - formation_offsets[first_id].y
                    ),
                )
            )
        disabled_robot_ids = set(self.executor.disabled_robots)
        measured_approach_routes = _telemetry_xy_routes(
            measured,
            robot_ids=executed_robot_ids,
            start_s=started_at_s,
            end_s=pickup_at_s,
        )
        measured_return_routes = _telemetry_xy_routes(
            measured,
            robot_ids=executed_robot_ids,
            start_s=installed_at_s,
            end_s=returned_at_s,
        )
        measured_approach_clearances = _base_world_clearance_minima(
            self.plan,
            phase="approach",
            source="measured",
            routes=measured_approach_routes,
            active_module_id=module.module_id,
            installed_module_ids=installed_before,
            robot_positions={
                robot_id: Vec2.model_validate(position)
                for robot_id, position in approach_robot_positions.items()
            },
            active_robot_ids=set(executed_robot_ids),
            disabled_robot_ids=disabled_robot_ids,
            active_module_is_payload=False,
            config=self.yard_config,
        )
        measured_carry_clearances = _validate_rigid_carrier_route(
            self.plan,
            module=module,
            installed_module_ids=installed_before,
            robot_positions={
                robot_id: Vec2.model_validate(position)
                for robot_id, position in carry_robot_positions.items()
            },
            active_robot_ids=set(executed_robot_ids),
            carrier_route=measured_carrier_route,
            base_routes=measured_carry_routes,
            transport_height_m=transport_height_m,
            disabled_robot_ids=disabled_robot_ids,
            source="measured",
            config=self.yard_config,
        )
        measured_return_clearances = _base_world_clearance_minima(
            self.plan,
            phase="return",
            source="measured",
            routes=measured_return_routes,
            active_module_id=module.module_id,
            installed_module_ids={*installed_before, module.module_id},
            robot_positions=return_robot_positions_xy,
            active_robot_ids=set(executed_robot_ids),
            disabled_robot_ids=disabled_robot_ids,
            active_module_is_payload=False,
            config=self.yard_config,
        )
        clearance_minima = [
            *planned_approach_clearances,
            *planned_carry_clearances,
            *planned_return_clearances,
            *measured_approach_clearances,
            *measured_carry_clearances,
            *measured_return_clearances,
        ]
        _require_clearance_records(clearance_minima)
        formation_errors = self.executor.formation_errors_m[formation_error_start:]
        assignment_errors = self.executor.formation_assignment_errors_m[
            assignment_error_start:
        ]
        spacing_errors = self.executor.formation_spacing_errors_m[
            spacing_error_start:
        ]
        install_errors = self.executor.install_errors_m[install_error_start:]
        return Phase5JobReplay(
            job_id=job_id,
            module_id=module.module_id,
            planned_order=index,
            planned_robot_ids=planned_robot_ids,
            executed_robot_ids=executed_robot_ids,
            planned_start_s=planned_start_s,
            planned_pickup_s=planned_pickup_s,
            planned_end_s=planned_end_s,
            approach_routes=approach_routes,
            carry_route=carry_route,
            assigned_carry_routes=assigned_carry_routes,
            return_routes=return_routes,
            formation_offsets=formation_offsets,
            logical_carrier_offset=raw_carrier_offset,
            logical_transport_height_m=transport_height_m,
            measured_carry_routes=measured_carry_routes,
            measured_carry_samples=measured_carry_samples,
            approach_robot_positions=approach_robot_positions,
            carry_robot_positions=carry_robot_positions,
            return_robot_positions=return_robot_positions,
            clearance_minima=clearance_minima,
            installation_snap=installation_snap,
            started_at_s=started_at_s,
            pickup_at_s=pickup_at_s,
            installed_at_s=installed_at_s,
            returned_at_s=returned_at_s,
            measured_poses=measured,
            maximum_formation_error_m=max(formation_errors, default=0.0),
            maximum_assignment_error_m=max(assignment_errors, default=0.0),
            maximum_spacing_error_m=max(spacing_errors, default=0.0),
            maximum_install_error_m=max(install_errors, default=0.0),
            minimum_pickup_base_clearance_m=pickup_clearance,
            minimum_install_base_clearance_m=install_clearance,
            minimum_planned_payload_clearance_m=min(
                planned_payload_clearances
            ),
            minimum_measured_payload_clearance_m=min(
                measured_payload_clearances
            ),
            maximum_synchronized_formation_error_m=max(
                measured_synchronized_errors,
                default=0.0,
            ),
            reassigned_after_unavailability=(
                self.recovery is not None
                and self.recovery.robot_id in planned_robot_ids
                and self.recovery.robot_id not in executed_robot_ids
            ),
        )

    def _measured_robot_positions(self) -> dict[str, Vec2]:
        return {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in (
                self._measured_robot_pose_positions().items()
            )
        }

    def _measured_robot_pose_positions(self) -> dict[str, Vec3]:
        return {
            robot.robot_id: self.executor.sample_telemetry(
                robot.robot_id
            ).measured_pose.position.model_copy(deep=True)
            for robot in self.plan.robots
        }

    def _minimum_active_module_clearance(
        self,
        module: BuildModule,
        robot_ids: list[str],
        *,
        installed: bool,
    ) -> float:
        bounds = _module_footprint(module, installed=installed)
        clearances = []
        for robot_id in robot_ids:
            position = Vec2.model_validate(
                self.executor.sample_telemetry(
                    robot_id
                ).measured_pose.position
            )
            clearances.append(_point_clearance_to_bounds(position, bounds))
        return min(clearances)

    def _acceptance(
        self,
        scenario: Phase5Scenario,
        active_robot_ids: set[str],
    ) -> dict[str, bool]:
        command_activity = {
            robot_id: any(
                command.robot_id == robot_id
                and any(abs(value) > 1e-9 for value in command.wheel_target_velocity_rad_s)
                for command in self.executor.commands
            )
            for robot_id in active_robot_ids
        }
        telemetry_activity = {
            robot_id: any(item.robot_id == robot_id for item in self.executor.telemetry)
            for robot_id in active_robot_ids
        }
        command_response = {
            robot_id: (
                self.executor.command_response_displacement_m.get(robot_id, 0.0)
                >= self.executor.config.command_response_min_displacement_m
                and robot_id
                in self.executor.command_response_observed_at_s
            )
            for robot_id in active_robot_ids
        }
        replay_ids = [item.module_id for item in self.replay]
        replay_complete = (
            len(self.replay) == len(self.plan.modules)
            and sorted(replay_ids)
            == sorted(module.module_id for module in self.plan.modules)
            and all(
                item.measured_poses
                and set(item.measured_carry_samples)
                == set(item.executed_robot_ids)
                and all(item.measured_carry_samples.values())
                for item in self.replay
            )
        )
        formation_maximum = max(
            [
                *self.executor.formation_errors_m,
                *self.executor.formation_assignment_errors_m,
                *self.executor.formation_spacing_errors_m,
                *self.executor.synchronized_formation_errors_m,
            ],
            default=0.0,
        )
        expected_obstacles = len(self.plan.site_grid.obstacle_cells)
        maximum_allowed_formation_offset = max(
            math.hypot(
                module.dimensions.width / 2,
                module.dimensions.depth / 2,
            )
            + self.yard_config.formation_clearance_m
            + self.yard_config.maximum_formation_expansion_m
            + self.executor.config.waypoint_tolerance_m
            for module in self.plan.modules
        )
        def phase_clearance_passed(phase: ClearancePhase) -> bool:
            records = [
                record
                for item in self.replay
                for record in item.clearance_minima
                if record.phase == phase
            ]
            return bool(records) and all(
                record.minimum_surface_clearance_m is None
                or record.minimum_surface_clearance_m + 1e-9
                >= record.required_clearance_m
                for record in records
            )

        installed_before: set[str] = set()
        final_snap_only = bool(self.replay)
        logical_lifts_attested = bool(self.replay)
        transition_stop_proofs_passed = bool(self.replay)
        modules_by_id = {
            module.module_id: module for module in self.plan.modules
        }

        def stop_proof_passed(
            raw_event: Mapping[str, object],
            robot_ids: list[str],
        ) -> bool:
            raw_proof = raw_event.get("team_zero_motion_proof")
            minimum_separation = (
                raw_proof.get(
                    "minimum_measured_team_to_scene_separation_m"
                )
                if isinstance(raw_proof, Mapping)
                else None
            )
            return (
                isinstance(raw_proof, Mapping)
                and raw_proof.get("robot_ids") == sorted(robot_ids)
                and raw_proof.get("latest_commands_zero") is True
                and raw_proof.get("measured_team_still") is True
                and raw_proof.get("no_nonzero_team_command_after_hold") is True
                and raw_proof.get("observed_consecutive_still_samples")
                == self.executor.config.logical_transition_settle_consecutive_samples
                and isinstance(minimum_separation, (int, float))
                and not isinstance(minimum_separation, bool)
                and float(minimum_separation) + 1e-9
                >= self.executor.config.safety_distance_m
                and raw_proof.get("safety_distance_m")
                == self.executor.config.safety_distance_m
            )

        for item in self.replay:
            module = modules_by_id[item.module_id]
            expected_contacts = _installation_contact_module_ids(
                self.plan,
                module=module,
                installed_module_ids=installed_before,
                config=self.yard_config,
            )
            final_snap_only = final_snap_only and (
                item.installation_snap.scope == "final_target_pose_only"
                and item.installation_snap.at_s == item.installed_at_s
                and item.installation_snap.target_pose == module.target_pose
                and item.installation_snap.target_pose_digest
                == _sha256_json(module.target_pose.model_dump(mode="json"))
                and item.installation_snap.contact_module_ids
                == expected_contacts
            )
            lift_events = [
                record
                for record in self.executor.logical_payload_lift_records
                if record.get("module_id") == item.module_id
            ]
            logical_lifts_attested = logical_lifts_attested and (
                len(lift_events) == 1
                and lift_events[0].get("timestamp_s") == item.pickup_at_s
                and lift_events[0].get("robot_ids")
                == sorted(item.executed_robot_ids)
                and lift_events[0].get("transport_height_m")
                == item.logical_transport_height_m
                and lift_events[0].get("logical_centroid_offset")
                == item.logical_carrier_offset.model_dump(mode="json")
                and lift_events[0].get("scope") == "logical_transport_only"
                and lift_events[0].get("physical_lift_claimed") is False
                and lift_events[0].get("physical_descent_claimed") is False
            )
            snap_events = [
                record
                for record in self.executor.logical_installation_snap_records
                if record.get("module_id") == item.module_id
            ]
            transition_stop_proofs_passed = (
                transition_stop_proofs_passed
                and len(lift_events) == 1
                and len(snap_events) == 1
                and stop_proof_passed(
                    lift_events[0],
                    item.executed_robot_ids,
                )
                and stop_proof_passed(
                    snap_events[0],
                    item.executed_robot_ids,
                )
            )
            installed_before.add(item.module_id)
        acceptance = {
            "every_module_installed": (
                len(self.executor.installed_modules) == len(self.plan.modules)
                and self.executor.installed_modules
                == {module.module_id for module in self.plan.modules}
            ),
            "zero_post_start_robot_pose_writes": (
                self.executor.post_start_robot_pose_writes == 0
            ),
            "active_robots_received_wheel_commands": bool(active_robot_ids)
            and all(command_activity.values()),
            "active_robots_have_measured_telemetry": bool(active_robot_ids)
            and all(telemetry_activity.values()),
            "active_robots_show_measured_command_response": bool(
                active_robot_ids
            )
            and all(command_response.values()),
            "formation_tolerance_passed": (
                formation_maximum
                <= self.executor.config.formation_tolerance_m
            ),
            "per_robot_formation_tolerance_passed": max(
                self.executor.formation_assignment_errors_m,
                default=0.0,
            )
            <= self.executor.config.formation_tolerance_m,
            "install_tolerance_passed": max(
                self.executor.install_errors_m,
                default=0.0,
            )
            <= self.executor.config.install_tolerance_m,
            "planned_measured_replay_complete": replay_complete,
            "physical_yard_preflight_passed": (
                self.physical_yard.initial_clearance_preflight_passed
                and self.physical_yard.sequential_route_preflight_passed
                and self.physical_yard.all_robots_active_preflight_passed
                and self.physical_yard.continuous_world_clearance_preflight_passed
                and self.physical_yard.recovery_route_preflight_passed
                and self.physical_yard.transformed_plan_digest
                == _sha256_json(self.plan.model_dump(mode="json"))
            ),
            "two_base_physical_formations_used": bool(self.replay)
            and all(
                len(item.executed_robot_ids)
                == self.yard_config.phase5_team_size
                for item in self.replay
            ),
            "bases_clear_of_active_module_footprints": bool(self.replay)
            and all(
                item.minimum_pickup_base_clearance_m
                >= self.yard_config.robot_footprint_radius_m
                and item.minimum_install_base_clearance_m
                >= self.yard_config.robot_footprint_radius_m
                for item in self.replay
            ),
            "continuous_payload_clearance_passed": bool(self.replay)
            and all(
                item.minimum_planned_payload_clearance_m
                >= (
                    self.yard_config.robot_footprint_radius_m
                    + self.yard_config.route_clearance_m
                )
                and item.minimum_measured_payload_clearance_m
                >= (
                    self.yard_config.robot_footprint_radius_m
                    + self.yard_config.route_clearance_m
                )
                for item in self.replay
            ),
            "continuous_approach_world_clearance_passed": (
                phase_clearance_passed("approach")
            ),
            "continuous_carry_world_clearance_passed": (
                phase_clearance_passed("carry")
            ),
            "continuous_return_world_clearance_passed": (
                phase_clearance_passed("return")
            ),
            "final_target_contact_only": final_snap_only,
            "logical_transport_lifts_attested": logical_lifts_attested,
            "logical_transition_zero_motion_proven": (
                transition_stop_proofs_passed
            ),
            "full_rpy_envelope_preflight_passed": (
                self.physical_yard.continuous_world_clearance_preflight_passed
                and all(
                    item.logical_transport_height_m
                    <= self.yard_config.maximum_logical_transport_height_m
                    for item in self.replay
                )
            ),
            "rigid_synchronized_carry_routes": bool(self.replay)
            and all(
                len(item.carry_route) > 1
                and all(
                    len(route) == len(item.carry_route)
                    for route in item.assigned_carry_routes.values()
                )
                and item.maximum_synchronized_formation_error_m
                <= self.executor.config.formation_tolerance_m
                for item in self.replay
            ),
            "formation_offsets_within_configured_bound": bool(
                self.replay
            )
            and all(
                math.hypot(offset.x, offset.y)
                <= maximum_allowed_formation_offset
                for item in self.replay
                for offset in item.formation_offsets.values()
            ),
            "collision_queries_cover_every_physics_step": (
                self.executor.physics_steps > 0
                and self.executor.collision_query_rounds
                == self.executor.physics_steps
                and self.executor.collision_query_count
                == self.executor.physics_steps
                * len(self.executor.collision_pairs)
            ),
            "exclusive_wheel_command_ownership_proven": (
                self.executor.script_control_gate_passed
                and set(self.executor.script_control_by_robot)
                == {robot.robot_id for robot in self.plan.robots}
                and all(self.executor.script_control_by_robot.values())
            ),
            "site_obstacles_instantiated": (
                len(self.executor.obstacle_handles) == expected_obstacles
                and (
                    self.evidence_kind != "live_coppelia"
                    or expected_obstacles > 0
                )
            ),
            "prior_generated_scene_cleanup_verified": (
                self.executor.generated_scene_cleanup_verified
            ),
            "logical_payload_disclosed": True,
        }
        if scenario == "nominal":
            acceptance["zero_nominal_collision_stops"] = self.executor.collision_stops == 0
            acceptance["all_fleet_robots_active_nominal"] = active_robot_ids == {
                robot.robot_id for robot in self.plan.robots
            }
        else:
            recovery = self.recovery
            acceptance.update(
                {
                    "robot_disabled_after_25_percent": (
                        recovery is not None
                        and recovery.completion_fraction_at_disable >= 0.25
                    ),
                    "disabled_robot_not_commanded_after_stop": (
                        recovery is not None and recovery.commands_after_stop == 0
                    ),
                    "disabled_robot_settled_after_stop": (
                        recovery is not None
                        and recovery.settled
                        and recovery.settle_sample_count
                        >= self.executor.config.disabled_settle_consecutive_samples
                        + 1
                        and self.executor.disabled_settled.get(
                            recovery.robot_id
                        )
                        is True
                    ),
                    "remaining_work_reassigned": (
                        recovery is not None and bool(recovery.reassigned_job_ids)
                    ),
                    "disabled_robot_payload_clearance_passed": (
                        recovery is not None
                        and any(
                            record.phase == "carry"
                            and record.mover_kind == "logical_payload"
                            and record.obstacle_kind == "disabled_robot"
                            and record.evaluated_pair_count > 0
                            and record.minimum_surface_clearance_m is not None
                            and record.minimum_surface_clearance_m + 1e-9
                            >= record.required_clearance_m
                            for item in self.replay
                            if item.pickup_at_s >= recovery.disabled_at_s
                            for record in item.clearance_minima
                        )
                    ),
                }
            )
        return acceptance


def _require_live_runtime_session(
    *,
    result: Phase5RunResult,
    scenario: ScenarioManifest,
    executor: DynamicCoppeliaExecutor,
    simulator_version: str | None,
) -> _LiveRuntimeSession:
    session = _LIVE_RUNTIME_SESSIONS.get(executor)
    if session is None or not session.finalized:
        raise ValueError(
            "live Coppelia evidence lacks a finalized runtime-origin attestation"
        )
    if session.consumed:
        raise ValueError("live runtime attestation has already bound one bundle")
    if not simulator_version or simulator_version != session.simulator_version:
        raise ValueError(
            "caller simulator version differs from the runtime-origin identity"
        )
    physical_yard = Phase5PhysicalYardManifest.model_validate(
        _mapping_field(result.metrics, "physical_yard")
    )
    expected_configuration_origin_digest = _sha256_json(
        {
            "executor": executor.config.model_dump(mode="json"),
            "physical_yard": physical_yard.model_dump(mode="json"),
            "evidence_kind": "live_coppelia",
        }
    )
    command_stream_sha256 = _sha256_json(
        [item.model_dump(mode="json") for item in executor.commands]
    )
    telemetry_stream_sha256 = _sha256_json(
        [item.model_dump(mode="json") for item in executor.telemetry]
    )
    if (
        session.scenario_id != scenario.scenario_id
        or session.scenario_seed != scenario.seed
        or session.plan_digest
        != _sha256_json(scenario.plan.model_dump(mode="json"))
        or session.configuration_origin_digest
        != expected_configuration_origin_digest
        or session.physics_steps != executor.physics_steps
        or session.command_count != len(executor.commands)
        or session.telemetry_count != len(executor.telemetry)
        or session.command_stream_sha256 != command_stream_sha256
        or session.telemetry_stream_sha256 != telemetry_stream_sha256
        or session.installed_module_ids
        != sorted(executor.installed_modules)
    ):
        raise ValueError(
            "runtime-origin attestation no longer matches executor evidence"
        )
    _verify_runtime_attestation_trace(
        trace=result.trace,
        client_uuid=session.client_uuid,
        remote_api_info_sha256=session.remote_api_info_sha256,
        simulator_version=session.simulator_version,
        scene_root_uid=session.scene_root_uid,
        challenge_payload_sha256=session.challenge_payload_sha256,
        simulation_state_before=session.simulation_state_before,
        simulation_state_after=session.simulation_state_after,
        challenge_response_sha256=session.challenge_payload_sha256,
        physics_steps=session.physics_steps,
        command_count=session.command_count,
        telemetry_count=session.telemetry_count,
        client_send_count_after=session.client_send_count_after,
    )
    return session


def _build_phase5_runtime_attestation(
    *,
    run_dir: Path,
    run_id: str,
    result: Phase5RunResult,
    scenario: ScenarioManifest,
    session: _LiveRuntimeSession,
    source_commit: str,
    source_tree_digest: str,
    configuration_digest: str,
) -> Phase5RuntimeAttestation:
    required_final_values = (
        session.simulation_state_after,
        session.simulation_time_after_s,
        session.client_send_count_after,
        session.physics_steps,
        session.command_count,
        session.telemetry_count,
        session.command_stream_sha256,
        session.telemetry_stream_sha256,
        session.installed_module_ids,
    )
    if not session.finalized or any(
        value is None for value in required_final_values
    ):
        raise ValueError("live runtime attestation finalization is incomplete")
    simulation_state_after = session.simulation_state_after
    simulation_time_after_s = session.simulation_time_after_s
    client_send_count_after = session.client_send_count_after
    physics_steps = session.physics_steps
    command_count = session.command_count
    telemetry_count = session.telemetry_count
    command_stream_sha256 = session.command_stream_sha256
    telemetry_stream_sha256 = session.telemetry_stream_sha256
    installed_module_ids = session.installed_module_ids
    if (
        simulation_state_after is None
        or simulation_time_after_s is None
        or client_send_count_after is None
        or physics_steps is None
        or command_count is None
        or telemetry_count is None
        or command_stream_sha256 is None
        or telemetry_stream_sha256 is None
        or installed_module_ids is None
    ):
        raise ValueError("live runtime attestation finalization is incomplete")
    artifact_sha256 = {
        name: _sha256_file(run_dir / name)
        for name in sorted(_REQUIRED_ARTIFACTS)
    }
    payload: dict[str, object] = {
        "schema_version": (
            "construction_intelligence.coppelia_runtime_attestation.v1"
        ),
        "run_id": run_id,
        "scenario_id": scenario.scenario_id,
        "scenario_seed": scenario.seed,
        "source_commit": source_commit,
        "source_tree_digest": source_tree_digest,
        "plan_digest": session.plan_digest,
        "configuration_digest": configuration_digest,
        "configuration_origin_digest": session.configuration_origin_digest,
        "transport": "coppeliasim_zmq_remote_api",
        "client_implementation": (
            "coppeliasim_zmqremoteapi_client.RemoteAPIClient"
        ),
        "remote_api_client_version": session.package_version,
        "remote_api_protocol_version": session.protocol_version,
        "client_uuid": session.client_uuid,
        "client_send_count_before": session.client_send_count_before,
        "client_send_count_after": client_send_count_after,
        "endpoint_host": session.endpoint_host,
        "endpoint_port": session.endpoint_port,
        "remote_api_capabilities": list(
            _LIVE_REQUIRED_REMOTE_API_CAPABILITIES
        ),
        "remote_api_info_sha256": session.remote_api_info_sha256,
        "simulator_version": session.simulator_version,
        "simulator_identity_source": session.simulator_identity_source,
        "simulator_program_version": session.simulator_program_version,
        "simulator_program_revision": session.simulator_program_revision,
        "scene_root_handle": session.scene_root_handle,
        "scene_root_uid": session.scene_root_uid,
        "simulation_stopped_state": session.simulation_stopped_state,
        "simulation_state_before": session.simulation_state_before,
        "simulation_state_after": simulation_state_after,
        "simulation_time_before_s": session.simulation_time_before_s,
        "simulation_time_after_s": simulation_time_after_s,
        "session_nonce": session.session_nonce,
        "challenge_tag": session.challenge_tag,
        "challenge_payload_sha256": session.challenge_payload_sha256,
        "challenge_response_sha256": session.challenge_payload_sha256,
        "physics_steps": physics_steps,
        "command_count": command_count,
        "telemetry_count": telemetry_count,
        "command_stream_sha256": command_stream_sha256,
        "telemetry_stream_sha256": telemetry_stream_sha256,
        "installed_module_ids": installed_module_ids,
        "artifact_sha256": artifact_sha256,
    }
    payload["binding_digest"] = _sha256_json(payload)
    attestation = Phase5RuntimeAttestation.model_validate(payload)
    if attestation.scenario_id != result.scenario_id:
        raise ValueError("runtime attestation scenario differs from the result")
    return attestation


def _verify_runtime_attestation_trace(
    *,
    trace: Iterable[object],
    client_uuid: str,
    remote_api_info_sha256: str,
    simulator_version: str,
    scene_root_uid: int,
    challenge_payload_sha256: str,
    simulation_state_before: int,
    simulation_state_after: int | None,
    challenge_response_sha256: str,
    physics_steps: int | None,
    command_count: int | None,
    telemetry_count: int | None,
    client_send_count_after: int | None,
) -> None:
    events = [item for item in trace if isinstance(item, Mapping)]
    started = [
        item
        for item in events
        if item.get("event") == "live_runtime_attestation_started"
    ]
    finalized = [
        item
        for item in events
        if item.get("event") == "live_runtime_attestation_finalized"
    ]
    if len(started) != 1 or len(finalized) != 1:
        raise ValueError(
            "live evidence requires one start and one final runtime attestation event"
        )
    start = started[0]
    end = finalized[0]
    if (
        start.get("client_uuid") != client_uuid
        or start.get("remote_api_info_sha256")
        != remote_api_info_sha256
        or start.get("simulator_version") != simulator_version
        or start.get("scene_root_uid") != scene_root_uid
        or start.get("challenge_payload_sha256")
        != challenge_payload_sha256
        or start.get("simulation_state") != simulation_state_before
        or end.get("client_uuid") != client_uuid
        or end.get("scene_root_uid") != scene_root_uid
        or end.get("challenge_response_sha256")
        != challenge_response_sha256
        or end.get("simulation_state") != simulation_state_after
        or end.get("physics_steps") != physics_steps
        or end.get("command_count") != command_count
        or end.get("telemetry_count") != telemetry_count
        or end.get("client_send_count_after")
        != client_send_count_after
    ):
        raise ValueError(
            "runtime attestation events do not match the runtime-origin record"
        )


def write_phase5_artifact_bundle(
    run_dir: Path,
    *,
    run_id: str,
    result: Phase5RunResult,
    scenario: ScenarioManifest,
    executor: DynamicCoppeliaExecutor,
    source_commit: str,
    source_tree_digest: str,
    source_dirty: bool = False,
    approval_gate_confirmed: bool,
    simulator_version: str | None = None,
    save_scene: bool = True,
) -> Phase5ProvenanceManifest:
    if result.evidence_kind == "live_coppelia" and not approval_gate_confirmed:
        raise ValueError("live Coppelia evidence requires an explicit approval confirmation")
    if result.evidence_kind == "live_coppelia" and not simulator_version:
        raise ValueError("live Coppelia evidence requires a simulator version")
    if not re.fullmatch(_GIT_COMMIT_PATTERN, source_commit):
        raise ValueError("source commit must be a full hexadecimal Git object ID")
    if not re.fullmatch(_SHA256_PATTERN, source_tree_digest):
        raise ValueError("source tree digest must be a SHA-256 value")
    if not save_scene:
        raise ValueError("Phase 5 evidence always requires a reusable saved scene")
    live_runtime_session: _LiveRuntimeSession | None = None
    if result.evidence_kind == "live_coppelia":
        live_runtime_session = _require_live_runtime_session(
            result=result,
            scenario=scenario,
            executor=executor,
            simulator_version=simulator_version,
        )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "scenario.json", scenario.model_dump(mode="json"))
    _write_json(
        run_dir / "planned_jobs.json",
        [
            {
                "job_id": item.job_id,
                "module_id": item.module_id,
                "planned_order": item.planned_order,
                "planned_robot_ids": item.planned_robot_ids,
                "executed_robot_ids": item.executed_robot_ids,
                "planned_start_s": item.planned_start_s,
                "planned_pickup_s": item.planned_pickup_s,
                "planned_end_s": item.planned_end_s,
                "approach_routes": _jsonable(item.approach_routes),
                "carry_route": _jsonable(item.carry_route),
                "assigned_carry_routes": _jsonable(
                    item.assigned_carry_routes
                ),
                "return_routes": _jsonable(item.return_routes),
                "formation_offsets": _jsonable(
                    item.formation_offsets
                ),
                "logical_carrier_offset": _jsonable(
                    item.logical_carrier_offset
                ),
                "logical_transport_height_m": (
                    item.logical_transport_height_m
                ),
                "measured_carry_routes": _jsonable(item.measured_carry_routes),
                "approach_robot_positions": _jsonable(
                    item.approach_robot_positions
                ),
                "carry_robot_positions": _jsonable(
                    item.carry_robot_positions
                ),
                "return_robot_positions": _jsonable(
                    item.return_robot_positions
                ),
                "clearance_minima": _jsonable(item.clearance_minima),
                "installation_snap": _jsonable(item.installation_snap),
                "returned_at_s": item.returned_at_s,
                "minimum_planned_payload_clearance_m": (
                    item.minimum_planned_payload_clearance_m
                ),
                "minimum_measured_payload_clearance_m": (
                    item.minimum_measured_payload_clearance_m
                ),
            }
            for item in result.replay
        ],
    )
    _write_json(
        run_dir / "planned_vs_measured_replay.json",
        [item.model_dump(mode="json") for item in result.replay],
    )
    _write_json(run_dir / "trace.json", result.trace)
    _write_json(
        run_dir / "metrics.json",
        result.model_dump(mode="json", exclude={"replay", "trace"}),
    )
    _write_jsonl(
        run_dir / "wheel_commands.jsonl",
        [item.model_dump(mode="json") for item in executor.commands],
    )
    _write_jsonl(
        run_dir / "measured_telemetry.jsonl",
        [item.model_dump(mode="json") for item in executor.telemetry],
    )
    (run_dir / "report.md").write_text(_phase5_report(result), encoding="utf-8")
    executor.save_scene(run_dir / "construction_intelligence.ttt")

    configuration_payload = {
        "scenario": result.scenario,
        "executor": executor.config.model_dump(mode="json"),
        "physical_yard": result.metrics.get("physical_yard"),
        "evidence_kind": result.evidence_kind,
    }
    configuration_digest = _sha256_json(configuration_payload)
    plan_digest = _sha256_json(scenario.plan.model_dump(mode="json"))
    runtime_attestation_digest: str | None = None
    if live_runtime_session is not None:
        runtime_attestation = _build_phase5_runtime_attestation(
            run_dir=run_dir,
            run_id=run_id,
            result=result,
            scenario=scenario,
            session=live_runtime_session,
            source_commit=source_commit,
            source_tree_digest=source_tree_digest,
            configuration_digest=configuration_digest,
        )
        runtime_attestation_path = (
            run_dir / _LIVE_RUNTIME_ATTESTATION_ARTIFACT
        )
        _write_json(
            runtime_attestation_path,
            runtime_attestation.model_dump(mode="json"),
        )
        runtime_attestation_digest = _sha256_file(
            runtime_attestation_path
        )
    artifact_paths = sorted(
        path for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    artifacts = [
        Phase5ArtifactRecord(
            path=path.name,
            sha256=_sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in artifact_paths
    ]
    manifest = Phase5ProvenanceManifest(
        run_id=run_id,
        generated_at=datetime.now(UTC).isoformat(),
        evidence_kind=result.evidence_kind,
        live_evidence=result.evidence_kind == "live_coppelia",
        run_status=result.status,
        live_gate_passed=result.live_gate_passed,
        approval_gate_confirmed=approval_gate_confirmed,
        scenario=result.scenario,
        scenario_id=scenario.scenario_id,
        scenario_seed=scenario.seed,
        source_commit=source_commit,
        source_dirty=source_dirty,
        source_tree_digest=source_tree_digest,
        plan_digest=plan_digest,
        configuration_digest=configuration_digest,
        simulator_version=simulator_version,
        runtime_attestation_digest=runtime_attestation_digest,
        limitations=[
            "Payload transport is a bounded overhead logical carrier synchronized to measured robot centroids.",
            "Logical lift and descent are not physical motion; only the final target-pose snap may contact installed structure.",
            "Idle and disabled robots remain infinite-height XY no-overflight exclusions.",
            "No arm, gripper, grasp contact, cooperative contact, or payload dynamics are claimed.",
            "Only YouBot base wheel commands and measured base telemetry are physical simulator evidence.",
        ],
        artifacts=artifacts,
    )
    _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    verified = verify_phase5_artifact_bundle(run_dir)
    if verified != manifest:
        raise ValueError("written Phase 5 manifest did not round-trip verification")
    if live_runtime_session is not None:
        live_runtime_session.consumed = True
    return verified


def verify_phase5_artifact_bundle(run_dir: Path) -> Phase5ProvenanceManifest:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Phase 5 bundle is missing manifest.json")
    manifest = Phase5ProvenanceManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    generated_at = datetime.fromisoformat(manifest.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None:
        raise ValueError("Phase 5 manifest timestamp must include a timezone")
    listed_paths = [artifact.path for artifact in manifest.artifacts]
    if len(listed_paths) != len(set(listed_paths)):
        raise ValueError("manifest contains duplicate artifact paths")
    actual_paths = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if set(listed_paths) != actual_paths:
        raise ValueError("manifest artifact inventory does not match the bundle")
    for artifact in manifest.artifacts:
        if Path(artifact.path).name != artifact.path:
            raise ValueError(f"artifact path must be a top-level filename: {artifact.path}")
        path = (run_dir / artifact.path).resolve()
        if path.parent != run_dir:
            raise ValueError(f"artifact escapes bundle directory: {artifact.path}")
        if not path.is_file():
            raise ValueError(f"artifact is missing: {artifact.path}")
        if path.stat().st_size != artifact.bytes:
            raise ValueError(f"artifact size mismatch: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"artifact digest mismatch: {artifact.path}")
    required_artifacts = set(_REQUIRED_ARTIFACTS)
    if manifest.evidence_kind == "live_coppelia":
        required_artifacts.add(_LIVE_RUNTIME_ATTESTATION_ARTIFACT)
    if actual_paths != required_artifacts:
        missing = sorted(required_artifacts - actual_paths)
        extra = sorted(actual_paths - required_artifacts)
        raise ValueError(
            "Phase 5 bundle inventory is not canonical "
            f"(missing={missing}, extra={extra})"
        )
    _verify_phase5_bundle_contents(run_dir, manifest)
    return manifest


def _verify_phase5_bundle_contents(
    run_dir: Path,
    manifest: Phase5ProvenanceManifest,
) -> None:
    scenario = ScenarioManifest.model_validate_json(
        (run_dir / "scenario.json").read_text(encoding="utf-8")
    )
    replay = TypeAdapter(list[Phase5JobReplay]).validate_python(
        _load_json_list(run_dir / "planned_vs_measured_replay.json")
    )
    planned_jobs = TypeAdapter(list[Phase5PlannedJobRecord]).validate_python(
        _load_json_list(run_dir / "planned_jobs.json")
    )
    commands = TypeAdapter(list[RobotCommand]).validate_python(
        _load_jsonl(run_dir / "wheel_commands.jsonl")
    )
    telemetry = TypeAdapter(list[RobotTelemetry]).validate_python(
        _load_jsonl(run_dir / "measured_telemetry.jsonl")
    )
    trace = _load_json_list(run_dir / "trace.json")
    if not all(isinstance(item, dict) for item in trace):
        raise ValueError("Phase 5 trace entries must be JSON objects")
    metrics_payload = _load_json_object(run_dir / "metrics.json")
    result = Phase5RunResult.model_validate(
        {
            **metrics_payload,
            "replay": [item.model_dump(mode="json") for item in replay],
            "trace": trace,
        }
    )
    try:
        validate_coppelia_scene_buffer(
            (run_dir / "construction_intelligence.ttt").read_bytes()
        )
    except DynamicCoppeliaError as exc:
        raise ValueError(f"Phase 5 scene is invalid: {exc}") from exc
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    if (
        len(report.strip()) < 80
        or "logical" not in report.lower()
        or "gripper" not in report.lower()
    ):
        raise ValueError("Phase 5 report omits the required payload limitations")

    if (
        scenario.scenario_id != manifest.scenario_id
        or scenario.seed != manifest.scenario_seed
        or result.scenario != manifest.scenario
        or result.scenario_id != manifest.scenario_id
        or result.scenario_seed != manifest.scenario_seed
        or result.evidence_kind != manifest.evidence_kind
        or result.status != manifest.run_status
        or result.live_gate_passed != manifest.live_gate_passed
    ):
        raise ValueError("Phase 5 scenario, result, and manifest identities differ")
    if _sha256_json(scenario.plan.model_dump(mode="json")) != manifest.plan_digest:
        raise ValueError("Phase 5 plan digest does not match scenario.json")
    diagnostics = result.metrics
    executor_config = DynamicCoppeliaConfig.model_validate(
        _mapping_field(diagnostics, "executor_config")
    )
    physical_yard = Phase5PhysicalYardManifest.model_validate(
        _mapping_field(diagnostics, "physical_yard")
    )
    if (
        physical_yard.transformed_plan_digest
        != _sha256_json(scenario.plan.model_dump(mode="json"))
        or diagnostics.get("physical_yard_digest")
        != _sha256_json(physical_yard.model_dump(mode="json"))
    ):
        raise ValueError("Phase 5 physical-yard provenance is inconsistent")
    expected_configuration_digest = _sha256_json(
        {
            "scenario": result.scenario,
            "executor": executor_config.model_dump(mode="json"),
            "physical_yard": physical_yard.model_dump(mode="json"),
            "evidence_kind": result.evidence_kind,
        }
    )
    if expected_configuration_digest != manifest.configuration_digest:
        raise ValueError("Phase 5 configuration digest does not match metrics")

    if manifest.evidence_kind == "live_coppelia":
        _verify_phase5_runtime_attestation(
            run_dir=run_dir,
            manifest=manifest,
            scenario=scenario,
            result=result,
            commands=commands,
            telemetry=telemetry,
            trace=trace,
            physical_yard=physical_yard,
        )

    expected_planned = [
        Phase5PlannedJobRecord(
            job_id=item.job_id,
            module_id=item.module_id,
            planned_order=item.planned_order,
            planned_robot_ids=item.planned_robot_ids,
            executed_robot_ids=item.executed_robot_ids,
            planned_start_s=item.planned_start_s,
            planned_pickup_s=item.planned_pickup_s,
            planned_end_s=item.planned_end_s,
            approach_routes=item.approach_routes,
            carry_route=item.carry_route,
            assigned_carry_routes=item.assigned_carry_routes,
            return_routes=item.return_routes,
            formation_offsets=item.formation_offsets,
            logical_carrier_offset=item.logical_carrier_offset,
            logical_transport_height_m=item.logical_transport_height_m,
            measured_carry_routes=item.measured_carry_routes,
            approach_robot_positions=item.approach_robot_positions,
            carry_robot_positions=item.carry_robot_positions,
            return_robot_positions=item.return_robot_positions,
            clearance_minima=item.clearance_minima,
            installation_snap=item.installation_snap,
            returned_at_s=item.returned_at_s,
            minimum_planned_payload_clearance_m=(
                item.minimum_planned_payload_clearance_m
            ),
            minimum_measured_payload_clearance_m=(
                item.minimum_measured_payload_clearance_m
            ),
        )
        for item in replay
    ]
    if planned_jobs != expected_planned:
        raise ValueError("planned_jobs.json does not match the measured replay")

    if result.status == "completed":
        _verify_replay_semantics(
            scenario=scenario,
            replay=replay,
            commands=commands,
            telemetry=telemetry,
            trace=trace,
            config=executor_config,
            physical_yard=physical_yard,
        )
    if not manifest.live_gate_passed:
        return
    _verify_passing_phase5_evidence(
        scenario=scenario,
        result=result,
        replay=replay,
        commands=commands,
        telemetry=telemetry,
        trace=trace,
        config=executor_config,
        physical_yard=physical_yard,
    )


def _verify_phase5_runtime_attestation(
    *,
    run_dir: Path,
    manifest: Phase5ProvenanceManifest,
    scenario: ScenarioManifest,
    result: Phase5RunResult,
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    trace: list[object],
    physical_yard: Phase5PhysicalYardManifest,
) -> None:
    path = run_dir / _LIVE_RUNTIME_ATTESTATION_ARTIFACT
    if manifest.runtime_attestation_digest is None or not path.is_file():
        raise ValueError("live Phase 5 bundle lacks runtime attestation")
    if _sha256_file(path) != manifest.runtime_attestation_digest:
        raise ValueError("runtime attestation digest differs from manifest")
    attestation = Phase5RuntimeAttestation.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    expected_configuration_origin_digest = _sha256_json(
        {
            "executor": _mapping_field(
                result.metrics,
                "executor_config",
            ),
            "physical_yard": physical_yard.model_dump(mode="json"),
            "evidence_kind": "live_coppelia",
        }
    )
    if (
        attestation.run_id != manifest.run_id
        or attestation.scenario_id != scenario.scenario_id
        or attestation.scenario_seed != scenario.seed
        or attestation.source_commit != manifest.source_commit
        or attestation.source_tree_digest != manifest.source_tree_digest
        or attestation.plan_digest != manifest.plan_digest
        or attestation.configuration_digest
        != manifest.configuration_digest
        or attestation.configuration_origin_digest
        != expected_configuration_origin_digest
        or attestation.simulator_version != manifest.simulator_version
        or attestation.physics_steps
        != _integer_metric(result.metrics, "physics_steps")
        or attestation.command_count != len(commands)
        or attestation.telemetry_count != len(telemetry)
        or attestation.installed_module_ids
        != sorted(result.installed_module_ids)
    ):
        raise ValueError(
            "runtime attestation identities differ from bundle evidence"
        )
    expected_artifact_sha256 = {
        name: _sha256_file(run_dir / name)
        for name in sorted(_REQUIRED_ARTIFACTS)
    }
    if attestation.artifact_sha256 != expected_artifact_sha256:
        raise ValueError(
            "runtime attestation is not bound to the canonical evidence files"
        )
    if attestation.command_stream_sha256 != _sha256_json(
        [item.model_dump(mode="json") for item in commands]
    ):
        raise ValueError("runtime attestation command stream differs")
    if attestation.telemetry_stream_sha256 != _sha256_json(
        [item.model_dump(mode="json") for item in telemetry]
    ):
        raise ValueError("runtime attestation telemetry stream differs")
    _verify_runtime_attestation_trace(
        trace=trace,
        client_uuid=attestation.client_uuid,
        remote_api_info_sha256=attestation.remote_api_info_sha256,
        simulator_version=attestation.simulator_version,
        scene_root_uid=attestation.scene_root_uid,
        challenge_payload_sha256=(
            attestation.challenge_payload_sha256
        ),
        simulation_state_before=attestation.simulation_state_before,
        simulation_state_after=attestation.simulation_state_after,
        challenge_response_sha256=(
            attestation.challenge_response_sha256
        ),
        physics_steps=attestation.physics_steps,
        command_count=attestation.command_count,
        telemetry_count=attestation.telemetry_count,
        client_send_count_after=attestation.client_send_count_after,
    )
    if manifest.live_gate_passed and (
        attestation.physics_steps <= 0
        or attestation.command_count <= 0
        or attestation.telemetry_count <= 0
    ):
        raise ValueError(
            "passing live evidence lacks an executed measured runtime"
        )


def _verify_passing_phase5_evidence(
    *,
    scenario: ScenarioManifest,
    result: Phase5RunResult,
    replay: list[Phase5JobReplay],
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    trace: list[object],
    config: DynamicCoppeliaConfig,
    physical_yard: Phase5PhysicalYardManifest,
) -> None:
    if (
        result.evidence_kind != "live_coppelia"
        or result.status != "completed"
        or result.error is not None
        or result.error_type is not None
    ):
        raise ValueError("passing Phase 5 evidence is not a completed live run")
    module_ids = {module.module_id for module in scenario.plan.modules}
    robot_ids = {robot.robot_id for robot in scenario.plan.robots}
    active_robot_ids = set(result.active_robot_ids)
    if (
        result.expected_module_count != len(module_ids)
        or set(result.installed_module_ids) != module_ids
        or len(result.installed_module_ids) != len(module_ids)
        or len(replay) != len(module_ids)
        or {item.module_id for item in replay} != module_ids
        or sorted(item.planned_order for item in replay)
        != list(range(len(module_ids)))
    ):
        raise ValueError("Phase 5 replay does not cover every planned module exactly once")
    if (
        not physical_yard.initial_clearance_preflight_passed
        or not physical_yard.sequential_route_preflight_passed
        or not physical_yard.all_robots_active_preflight_passed
        or not physical_yard.continuous_world_clearance_preflight_passed
        or not physical_yard.recovery_route_preflight_passed
        or physical_yard.transformed_plan_digest
        != _sha256_json(scenario.plan.model_dump(mode="json"))
        or physical_yard.allocator_configuration_digest
        != _sha256_json(
            physical_yard.configuration.model_dump(mode="json")
        )
        or physical_yard.module_count != len(module_ids)
        or physical_yard.robot_count != len(robot_ids)
        or physical_yard.obstacle_count
        != len(scenario.plan.site_grid.obstacle_cells)
        or any(
            len(item.executed_robot_ids)
            != physical_yard.configuration.phase5_team_size
            or item.minimum_pickup_base_clearance_m
            < physical_yard.configuration.robot_footprint_radius_m
            or item.minimum_install_base_clearance_m
            < physical_yard.configuration.robot_footprint_radius_m
            or item.minimum_planned_payload_clearance_m
            < (
                physical_yard.configuration.robot_footprint_radius_m
                + physical_yard.configuration.route_clearance_m
            )
            or item.minimum_measured_payload_clearance_m
            < (
                physical_yard.configuration.robot_footprint_radius_m
                + physical_yard.configuration.route_clearance_m
            )
            for item in replay
        )
    ):
        raise ValueError("Phase 5 physical-yard evidence is incomplete")
    try:
        _validate_initial_physical_clearance(
            scenario.plan,
            physical_yard.configuration,
        )
        (
            reproduced_maximum_formation_offset_m,
            reproduced_maximum_transport_height_m,
        ) = (
            _preflight_phase5_sequential_routes(
                scenario.plan,
                physical_yard.configuration,
            )
        )
        if not math.isclose(
            reproduced_maximum_formation_offset_m,
            physical_yard.maximum_preflight_formation_offset_m,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Phase 5 maximum preflight formation offset differs"
            )
        if not math.isclose(
            reproduced_maximum_transport_height_m,
            physical_yard.maximum_preflight_transport_height_m,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Phase 5 maximum preflight transport height differs"
            )
    except (DynamicCoppeliaError, RoutingError, ValueError) as exc:
        raise ValueError(
            f"Phase 5 physical-yard preflight cannot be reproduced: {exc}"
        ) from exc
    if not scenario.plan.site_grid.obstacle_cells:
        raise ValueError("live Phase 5 evidence requires an instantiated site obstacle")
    if (
        not active_robot_ids
        or not active_robot_ids.issubset(robot_ids)
        or not commands
        or not telemetry
    ):
        raise ValueError("Phase 5 evidence lacks an active measured robot team")
    if not _timestamps_are_monotonic(commands) or not _timestamps_are_monotonic(
        telemetry
    ):
        raise ValueError("Phase 5 command or telemetry timestamps are not monotonic")
    if any(item.robot_id not in robot_ids for item in commands) or any(
        item.robot_id not in robot_ids for item in telemetry
    ):
        raise ValueError("Phase 5 command or telemetry references an unknown robot")
    for robot_id in active_robot_ids:
        robot_commands = [item for item in commands if item.robot_id == robot_id]
        robot_telemetry = [item for item in telemetry if item.robot_id == robot_id]
        nonzero_commands = [
            item
            for item in robot_commands
            if any(
                abs(value) > 1e-9
                for value in item.wheel_target_velocity_rad_s
            )
        ]
        if not nonzero_commands:
            raise ValueError(f"{robot_id} has no real wheel command activity")
        if len(robot_telemetry) < 2:
            raise ValueError(f"{robot_id} has insufficient measured telemetry")
        first_command = nonzero_commands[0]
        before_command = [
            item
            for item in robot_telemetry
            if item.timestamp_s <= first_command.timestamp_s
        ]
        after_command = [
            item
            for item in robot_telemetry
            if item.timestamp_s > first_command.timestamp_s
        ]
        if not before_command or not after_command:
            raise ValueError(
                f"{robot_id} lacks before/after telemetry for its first "
                "non-zero wheel command"
            )
        start = before_command[-1].measured_pose.position
        measured_displacement = max(
            math.sqrt(
                (item.measured_pose.position.x - start.x) ** 2
                + (item.measured_pose.position.y - start.y) ** 2
                + (item.measured_pose.position.z - start.z) ** 2
            )
            for item in after_command
        )
        if measured_displacement < config.command_response_min_displacement_m:
            raise ValueError(f"{robot_id} has no measured response to wheel commands")

    telemetry_rows = Counter(
        _canonical_json(item.model_dump(mode="json")) for item in telemetry
    )
    replay_rows: Counter[str] = Counter()
    for item in replay:
        job_rows = Counter(
            _canonical_json(sample.model_dump(mode="json"))
            for sample in item.measured_poses
        )
        carry_rows = Counter(
            _canonical_json(sample.model_dump(mode="json"))
            for samples in item.measured_carry_samples.values()
            for sample in samples
        )
        replay_rows.update(job_rows)
        if set(item.executed_robot_ids) != set(
            item.measured_carry_samples
        ) or not item.measured_poses or carry_rows - job_rows:
            raise ValueError(
                f"{item.job_id} replay is not backed by measured telemetry"
            )
    if replay_rows - telemetry_rows:
        raise ValueError(
            "Phase 5 replay contains samples absent from measured telemetry"
        )
    diagnostics = result.metrics
    replay_maximum_formation_offset = max(
        (
            math.hypot(offset.x, offset.y)
            for item in replay
            for offset in item.formation_offsets.values()
        ),
        default=0.0,
    )
    replay_maximum_transport_height = max(
        (item.logical_transport_height_m for item in replay),
        default=0.0,
    )
    if (
        not math.isclose(
            _number_metric(
                diagnostics,
                "maximum_formation_offset_m",
            ),
            replay_maximum_formation_offset,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _number_metric(
                diagnostics,
                "maximum_preflight_formation_offset_m",
            ),
            physical_yard.maximum_preflight_formation_offset_m,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _number_metric(
                diagnostics,
                "maximum_logical_transport_height_m",
            ),
            replay_maximum_transport_height,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _number_metric(
                diagnostics,
                "maximum_preflight_transport_height_m",
            ),
            physical_yard.maximum_preflight_transport_height_m,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "Phase 5 formation or transport diagnostics do not match replay"
        )
    physics_steps = _integer_metric(diagnostics, "physics_steps")
    query_rounds = _integer_metric(diagnostics, "collision_query_rounds")
    query_count = _integer_metric(diagnostics, "collision_query_count")
    expected_queries = _integer_metric(
        diagnostics,
        "expected_collision_queries_per_step",
    )
    if (
        physics_steps <= 0
        or query_rounds != physics_steps
        or expected_queries <= 0
        or query_count != physics_steps * expected_queries
        or diagnostics.get("collision_queries_cover_every_physics_step")
        is not True
    ):
        raise ValueError("physical collision queries do not cover every physics step")
    _verify_collision_trace(
        scenario=scenario,
        trace=trace,
        diagnostics=diagnostics,
        physics_steps=physics_steps,
        expected_queries=expected_queries,
    )
    _verify_runtime_inventory_trace(
        scenario=scenario,
        trace=trace,
        diagnostics=diagnostics,
    )
    if (
        _integer_metric(diagnostics, "wheel_command_count") != len(commands)
        or _integer_metric(diagnostics, "telemetry_sample_count") != len(telemetry)
        or _integer_metric(diagnostics, "post_start_robot_pose_writes") != 0
        or _integer_metric(diagnostics, "scene_robot_count") != len(robot_ids)
        or _integer_metric(diagnostics, "scene_module_count") != len(module_ids)
        or _integer_metric(diagnostics, "scene_obstacle_count")
        != len(scenario.plan.site_grid.obstacle_cells)
        or _integer_metric(diagnostics, "installed_modules") != len(module_ids)
        or _integer_metric(diagnostics, "logical_payload_lift_count")
        != len(replay)
        or _integer_metric(
            diagnostics,
            "logical_installation_snap_count",
        )
        != len(replay)
    ):
        raise ValueError("Phase 5 diagnostics do not match the recorded evidence")
    script_control = _mapping_field(diagnostics, "script_control_by_robot")
    wheel_writers = _mapping_field(
        diagnostics,
        "wheel_command_writer_count_by_robot",
    )
    retained_maintenance = _mapping_field(
        diagnostics,
        "retained_maintenance_count_by_robot",
    )
    discovered_scripts = _integer_metric(
        diagnostics,
        "bundled_motion_scripts_found",
    )
    disabled_scripts = _integer_metric(
        diagnostics,
        "disabled_bundled_motion_scripts",
    )
    retained_scripts = _integer_metric(
        diagnostics,
        "retained_bundled_maintenance_scripts",
    )
    if (
        diagnostics.get("script_control_gate_passed") is not True
        or diagnostics.get("script_inventory_classification_complete")
        is not True
        or set(script_control) != robot_ids
        or any(value is not True for value in script_control.values())
        or set(wheel_writers) != robot_ids
        or any(value != 1 for value in wheel_writers.values())
        or set(retained_maintenance) != robot_ids
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in retained_maintenance.values()
        )
        or discovered_scripts <= 0
        or disabled_scripts <= 0
        or retained_scripts <= 0
        or disabled_scripts + retained_scripts != discovered_scripts
        or _integer_metric(
            diagnostics,
            "verified_disabled_bundled_motion_scripts",
        )
        != disabled_scripts
        or _integer_metric(
            diagnostics,
            "verified_enabled_maintenance_scripts",
        )
        != retained_scripts
        or _integer_metric(diagnostics, "disabled_wheel_command_scripts")
        != len(robot_ids)
        or _integer_metric(diagnostics, "disabled_arm_gripper_scripts")
        < len(robot_ids)
    ):
        raise ValueError("exclusive Python wheel-command ownership is not proven")
    _verify_script_control_audit(
        diagnostics=diagnostics,
        robot_ids=robot_ids,
        discovered_scripts=discovered_scripts,
        disabled_scripts=disabled_scripts,
        retained_scripts=retained_scripts,
    )
    response = _mapping_field(diagnostics, "command_response_displacement_m")
    observed = _mapping_field(diagnostics, "command_response_observed_at_s")
    for robot_id in active_robot_ids:
        if (
            _number_value(response.get(robot_id), default=-1)
            < config.command_response_min_displacement_m
            or robot_id not in observed
        ):
            raise ValueError(f"{robot_id} command response diagnostics are incomplete")
    if (
        _number_metric(diagnostics, "maximum_formation_error_m")
        > config.formation_tolerance_m
        or _number_metric(
            diagnostics,
            "maximum_formation_assignment_error_m",
        )
        > config.formation_tolerance_m
        or _number_metric(
            diagnostics,
            "maximum_formation_spacing_error_m",
        )
        > config.formation_tolerance_m
        or _number_metric(diagnostics, "maximum_install_error_m")
        > config.install_tolerance_m
    ):
        raise ValueError("measured formation or installation exceeds tolerance")
    if (
        diagnostics.get("payload_transport") != "logical_carrier"
        or diagnostics.get("logical_payload_model") is None
        or diagnostics.get("live_evidence") is not True
        or diagnostics.get("live_gate_passed") is not True
    ):
        raise ValueError("Phase 5 payload boundary is not explicitly logical")
    cleanup_roots = _integer_metric(
        diagnostics,
        "prior_generated_scene_root_count",
    )
    cleanup_objects = _integer_metric(
        diagnostics,
        "prior_generated_scene_object_count",
    )
    cleanup_events = [
        item
        for item in trace
        if isinstance(item, Mapping)
        and item.get("event") == "prior_generated_scene_cleanup"
    ]
    if (
        diagnostics.get("generated_scene_cleanup_verified") is not True
        or cleanup_roots < 0
        or cleanup_objects < cleanup_roots
        or len(cleanup_events) != 1
        or cleanup_events[0].get("verified") is not True
        or cleanup_events[0].get("removed_root_count") != cleanup_roots
        or cleanup_events[0].get("removed_object_count") != cleanup_objects
    ):
        raise ValueError("project-owned Coppelia scene cleanup is not verified")

    required_acceptance = {
        "every_module_installed",
        "zero_post_start_robot_pose_writes",
        "active_robots_received_wheel_commands",
        "active_robots_have_measured_telemetry",
        "active_robots_show_measured_command_response",
        "formation_tolerance_passed",
        "per_robot_formation_tolerance_passed",
        "install_tolerance_passed",
        "planned_measured_replay_complete",
        "physical_yard_preflight_passed",
        "two_base_physical_formations_used",
        "bases_clear_of_active_module_footprints",
        "continuous_payload_clearance_passed",
        "continuous_approach_world_clearance_passed",
        "continuous_carry_world_clearance_passed",
        "continuous_return_world_clearance_passed",
        "final_target_contact_only",
        "logical_transport_lifts_attested",
        "logical_transition_zero_motion_proven",
        "full_rpy_envelope_preflight_passed",
        "rigid_synchronized_carry_routes",
        "formation_offsets_within_configured_bound",
        "collision_queries_cover_every_physics_step",
        "exclusive_wheel_command_ownership_proven",
        "site_obstacles_instantiated",
        "prior_generated_scene_cleanup_verified",
        "logical_payload_disclosed",
    }
    if result.scenario == "nominal":
        required_acceptance.update(
            {
                "zero_nominal_collision_stops",
                "all_fleet_robots_active_nominal",
            }
        )
        if (
            _integer_metric(diagnostics, "collision_stops") != 0
            or _integer_metric(diagnostics, "physical_collision_stops") != 0
            or _integer_metric(diagnostics, "physical_collision_event_count") != 0
        ):
            raise ValueError("nominal Phase 5 run contains a collision stop")
    else:
        required_acceptance.update(
            {
                "robot_disabled_after_25_percent",
                "disabled_robot_not_commanded_after_stop",
                "disabled_robot_settled_after_stop",
                "remaining_work_reassigned",
                "disabled_robot_payload_clearance_passed",
            }
        )
        _verify_recovery_evidence(result, replay, commands, telemetry, trace, config)
    if not required_acceptance.issubset(result.acceptance) or any(
        result.acceptance.get(name) is not True for name in required_acceptance
    ):
        raise ValueError("Phase 5 required acceptance gates did not all pass")


def _verify_replay_semantics(
    *,
    scenario: ScenarioManifest,
    replay: list[Phase5JobReplay],
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    trace: list[object],
    config: DynamicCoppeliaConfig,
    physical_yard: Phase5PhysicalYardManifest,
) -> None:
    schedule = schedule_build(scenario.plan, "sequential")
    modules = {
        module.module_id: module for module in scenario.plan.modules
    }
    if len(schedule.jobs) != len(replay):
        raise ValueError("Phase 5 replay length differs from the frozen schedule")
    trace_records = [item for item in trace if isinstance(item, Mapping)]
    installed_before: set[str] = set()
    for index, (item, planned) in enumerate(
        zip(replay, schedule.jobs, strict=True)
    ):
        expected_job_id = f"job-{index:03d}-{planned.module_id}"
        if (
            item.planned_order != index
            or item.job_id != expected_job_id
            or item.module_id != planned.module_id
            or item.planned_robot_ids != planned.robot_ids
            or item.planned_start_s != planned.start_s
            or item.planned_pickup_s != planned.pickup_s
            or item.planned_end_s != planned.end_s
        ):
            raise ValueError(
                f"{item.job_id} does not match the deterministic schedule"
            )
        if (
            len(item.executed_robot_ids)
            != physical_yard.configuration.phase5_team_size
            or any(not route for route in item.approach_routes.values())
            or any(not route for route in item.assigned_carry_routes.values())
            or not item.carry_route
        ):
            raise ValueError(f"{item.job_id} has incomplete planned routes")
        if any(
            sample.timestamp_s < item.started_at_s
            or sample.timestamp_s > item.returned_at_s
            for sample in item.measured_poses
        ):
            raise ValueError(
                f"{item.job_id} contains telemetry outside its measured interval"
            )
        module = modules[item.module_id]
        pickup_positions: dict[str, Vec2] = {}
        install_positions: dict[str, Vec2] = {}
        for robot_id in item.executed_robot_ids:
            samples = item.measured_carry_samples[robot_id]
            if not _timestamps_are_monotonic(samples):
                raise ValueError(
                    f"{item.job_id}/{robot_id} carry telemetry is not monotonic"
                )
            if any(
                sample.attached_module_id != item.module_id
                for sample in samples
            ):
                raise ValueError(
                    f"{item.job_id}/{robot_id} lacks logical-carrier identity"
                )
            pickup = Vec2.model_validate(samples[0].measured_pose.position)
            install = Vec2.model_validate(samples[-1].measured_pose.position)
            pickup_positions[robot_id] = pickup
            install_positions[robot_id] = install
            pickup_target = item.approach_routes[robot_id][-1]
            install_target = item.assigned_carry_routes[robot_id][-1]
            if (
                _distance(pickup, pickup_target)
                > config.formation_tolerance_m
                or _distance(install, install_target)
                > config.formation_tolerance_m
            ):
                raise ValueError(
                    f"{item.job_id}/{robot_id} measured route misses its "
                    "assigned formation endpoint"
                )
        pickup_center = _centroid(pickup_positions.values())
        install_center = _centroid(install_positions.values())
        staging = Vec2.model_validate(module.staging_pose.position)
        target = Vec2.model_validate(module.target_pose.position)
        carrier_offset = Vec2(
            x=staging.x - pickup_center.x,
            y=staging.y - pickup_center.y,
        )
        measured_carrier_install = Vec2(
            x=install_center.x + carrier_offset.x,
            y=install_center.y + carrier_offset.y,
        )
        if (
            _distance(pickup_center, staging)
            > config.formation_tolerance_m
            or _distance(measured_carrier_install, target)
            > config.install_tolerance_m
            or _distance(
                carrier_offset,
                item.logical_carrier_offset,
            )
            > 1e-4
            or _distance(item.carry_route[0], staging) > 1e-6
            or _distance(item.carry_route[-1], target) > 1e-6
        ):
            raise ValueError(
                f"{item.job_id} measured carrier endpoints exceed tolerance"
            )
        if any(
            _distance(left, right)
            > physical_yard.configuration.carry_sample_spacing_m
            + 1e-9
            for left, right in zip(
                item.carry_route,
                item.carry_route[1:],
                strict=False,
            )
        ):
            raise ValueError(
                f"{item.job_id} carrier route is not continuously sampled"
            )
        planned_clearances = _payload_clearances_for_routes(
            module=module,
            carrier_route=item.carry_route,
            base_routes=item.assigned_carry_routes,
        )
        measured_clearances: list[float] = []
        measured_sync_errors: list[float] = []
        measured_horizon = len(
            item.measured_carry_routes[item.executed_robot_ids[0]]
        )
        first_id, second_id = sorted(item.executed_robot_ids)
        yaws = {
            module.staging_pose.rotation_rpy_degrees.z,
            module.target_pose.rotation_rpy_degrees.z,
        }
        for sample_index in range(measured_horizon):
            measured_points = {
                robot_id: item.measured_carry_routes[robot_id][
                    sample_index
                ]
                for robot_id in item.executed_robot_ids
            }
            measured_center = _centroid(measured_points.values())
            measured_carrier = Vec2(
                x=(
                    measured_center.x
                    + item.logical_carrier_offset.x
                ),
                y=(
                    measured_center.y
                    + item.logical_carrier_offset.y
                ),
            )
            measured_clearances.append(
                min(
                    _oriented_module_clearance(
                        point,
                        measured_carrier,
                        module,
                        yaw,
                    )
                    for point in measured_points.values()
                    for yaw in yaws
                )
            )
            measured_sync_errors.append(
                math.hypot(
                    (
                        measured_points[second_id].x
                        - measured_points[first_id].x
                    )
                    - (
                        item.formation_offsets[second_id].x
                        - item.formation_offsets[first_id].x
                    ),
                    (
                        measured_points[second_id].y
                        - measured_points[first_id].y
                    )
                    - (
                        item.formation_offsets[second_id].y
                        - item.formation_offsets[first_id].y
                    ),
                )
            )
        minimum_required_clearance = (
            physical_yard.configuration.robot_footprint_radius_m
            + physical_yard.configuration.route_clearance_m
        )
        if (
            min(planned_clearances) < minimum_required_clearance
            or min(measured_clearances) < minimum_required_clearance
            or not math.isclose(
                min(planned_clearances),
                item.minimum_planned_payload_clearance_m,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
            or not math.isclose(
                min(measured_clearances),
                item.minimum_measured_payload_clearance_m,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
            or not math.isclose(
                max(measured_sync_errors, default=0.0),
                item.maximum_synchronized_formation_error_m,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
            or item.maximum_synchronized_formation_error_m
            > config.formation_tolerance_m
        ):
            raise ValueError(
                f"{item.job_id} rigid payload-clearance evidence is invalid"
            )
        pickup_spacing_error = _pair_spacing_error(
            pickup_positions,
            {
                robot_id: item.approach_routes[robot_id][-1]
                for robot_id in item.executed_robot_ids
            },
        )
        install_spacing_error = _pair_spacing_error(
            install_positions,
            {
                robot_id: item.assigned_carry_routes[robot_id][-1]
                for robot_id in item.executed_robot_ids
            },
        )
        if max(pickup_spacing_error, install_spacing_error) > (
            config.formation_tolerance_m
        ):
            raise ValueError(f"{item.job_id} measured spacing exceeds tolerance")
        pickup_clearance = min(
            _point_clearance_to_bounds(
                point,
                _module_footprint(module, installed=False),
            )
            for point in pickup_positions.values()
        )
        install_clearance = min(
            _point_clearance_to_bounds(
                point,
                _module_footprint(module, installed=True),
            )
            for point in install_positions.values()
        )
        if (
            not math.isclose(
                pickup_clearance,
                item.minimum_pickup_base_clearance_m,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
            or not math.isclose(
                install_clearance,
                item.minimum_install_base_clearance_m,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
        ):
            raise ValueError(
                f"{item.job_id} base-clearance record does not match telemetry"
            )
        disabled_robot_ids = {
            str(record.get("robot_id"))
            for record in trace_records
            if record.get("event") == "robot_unavailable"
            and _number_value(record.get("timestamp_s")) <= item.started_at_s
        }
        active_ids = set(item.executed_robot_ids)
        approach_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in item.approach_robot_positions.items()
        }
        carry_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in item.carry_robot_positions.items()
        }
        return_positions = {
            robot_id: Vec2.model_validate(position)
            for robot_id, position in item.return_robot_positions.items()
        }
        try:
            recomputed: list[Phase5ClearanceMinimum] = (
                _base_world_clearance_minima(
                    scenario.plan,
                    phase="approach",
                    source="planned",
                    routes=item.approach_routes,
                    active_module_id=item.module_id,
                    installed_module_ids=installed_before,
                    robot_positions=approach_positions,
                    active_robot_ids=active_ids,
                    disabled_robot_ids=disabled_robot_ids,
                    active_module_is_payload=False,
                    config=physical_yard.configuration,
                )
            )
            recomputed.extend(
                _validate_rigid_carrier_route(
                    scenario.plan,
                    module=module,
                    installed_module_ids=installed_before,
                    robot_positions=carry_positions,
                    active_robot_ids=active_ids,
                    carrier_route=item.carry_route,
                    base_routes=item.assigned_carry_routes,
                    transport_height_m=item.logical_transport_height_m,
                    disabled_robot_ids=disabled_robot_ids,
                    source="planned",
                    config=physical_yard.configuration,
                )
            )
            recomputed.extend(
                _base_world_clearance_minima(
                    scenario.plan,
                    phase="return",
                    source="planned",
                    routes=item.return_routes,
                    active_module_id=item.module_id,
                    installed_module_ids={*installed_before, item.module_id},
                    robot_positions=return_positions,
                    active_robot_ids=active_ids,
                    disabled_robot_ids=disabled_robot_ids,
                    active_module_is_payload=False,
                    config=physical_yard.configuration,
                )
            )
            measured_approach_routes = _telemetry_xy_routes(
                telemetry,
                robot_ids=item.executed_robot_ids,
                start_s=item.started_at_s,
                end_s=item.pickup_at_s,
            )
            measured_return_routes = _telemetry_xy_routes(
                telemetry,
                robot_ids=item.executed_robot_ids,
                start_s=item.installed_at_s,
                end_s=item.returned_at_s,
            )
            recomputed.extend(
                _base_world_clearance_minima(
                    scenario.plan,
                    phase="approach",
                    source="measured",
                    routes=measured_approach_routes,
                    active_module_id=item.module_id,
                    installed_module_ids=installed_before,
                    robot_positions=approach_positions,
                    active_robot_ids=active_ids,
                    disabled_robot_ids=disabled_robot_ids,
                    active_module_is_payload=False,
                    config=physical_yard.configuration,
                )
            )
            measured_carrier_route = []
            measured_horizon = len(
                item.measured_carry_routes[item.executed_robot_ids[0]]
            )
            for sample_index in range(measured_horizon):
                center = _centroid(
                    item.measured_carry_routes[robot_id][sample_index]
                    for robot_id in item.executed_robot_ids
                )
                measured_carrier_route.append(
                    Vec2(
                        x=center.x + item.logical_carrier_offset.x,
                        y=center.y + item.logical_carrier_offset.y,
                    )
                )
            recomputed.extend(
                _validate_rigid_carrier_route(
                    scenario.plan,
                    module=module,
                    installed_module_ids=installed_before,
                    robot_positions=carry_positions,
                    active_robot_ids=active_ids,
                    carrier_route=measured_carrier_route,
                    base_routes=item.measured_carry_routes,
                    transport_height_m=item.logical_transport_height_m,
                    disabled_robot_ids=disabled_robot_ids,
                    source="measured",
                    config=physical_yard.configuration,
                )
            )
            recomputed.extend(
                _base_world_clearance_minima(
                    scenario.plan,
                    phase="return",
                    source="measured",
                    routes=measured_return_routes,
                    active_module_id=item.module_id,
                    installed_module_ids={*installed_before, item.module_id},
                    robot_positions=return_positions,
                    active_robot_ids=active_ids,
                    disabled_robot_ids=disabled_robot_ids,
                    active_module_is_payload=False,
                    config=physical_yard.configuration,
                )
            )
        except (DynamicCoppeliaError, RoutingError, ValueError) as exc:
            raise ValueError(
                f"{item.job_id} continuous world clearance cannot be reproduced: {exc}"
            ) from exc
        if recomputed != item.clearance_minima:
            raise ValueError(
                f"{item.job_id} continuous world-clearance records differ"
            )
        expected_height = _logical_transport_height(
            scenario.plan,
            module=module,
            installed_module_ids=installed_before,
            config=physical_yard.configuration,
        )
        expected_contacts = _installation_contact_module_ids(
            scenario.plan,
            module=module,
            installed_module_ids=installed_before,
            config=physical_yard.configuration,
        )
        snap_events = [
            record
            for record in trace_records
            if record.get("event") == "logical_installation_snap"
            and record.get("module_id") == item.module_id
        ]
        lift_events = [
            record
            for record in trace_records
            if record.get("event") == "logical_payload_lifted"
            and record.get("module_id") == item.module_id
        ]
        if len(lift_events) != 1:
            raise ValueError(
                f"{item.job_id} requires exactly one logical payload lift"
            )
        lift_event = lift_events[0]
        lift_from_pose = Pose3D.model_validate(
            lift_event.get("module_from_pose")
        )
        lifted_pose = Pose3D.model_validate(
            lift_event.get("module_lifted_pose")
        )
        carrier_pose = Pose3D.model_validate(lift_event.get("carrier_pose"))
        measured_centroid = Vec3.model_validate(
            lift_event.get("measured_robot_centroid")
        )
        recorded_offset = Vec3.model_validate(
            lift_event.get("logical_centroid_offset")
        )
        expected_lifted_pose = Pose3D(
            position=Vec3(
                x=module.staging_pose.position.x,
                y=module.staging_pose.position.y,
                z=expected_height,
            ),
            rotation_rpy_degrees=(
                module.staging_pose.rotation_rpy_degrees.model_copy(deep=True)
            ),
        )
        if (
            not _pose3d_is_close(lift_from_pose, module.staging_pose)
            or not _pose3d_is_close(lifted_pose, expected_lifted_pose)
            or not _vec3_is_close(carrier_pose.position, expected_lifted_pose.position)
            or not _vec3_is_close(recorded_offset, item.logical_carrier_offset)
            or not _vec3_is_close(
                Vec3(
                    x=measured_centroid.x + recorded_offset.x,
                    y=measured_centroid.y + recorded_offset.y,
                    z=measured_centroid.z + recorded_offset.z,
                ),
                expected_lifted_pose.position,
            )
            or lift_event.get("robot_ids") != sorted(item.executed_robot_ids)
            or not math.isclose(
                _number_value(lift_event.get("timestamp_s")),
                item.pickup_at_s,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _number_value(lift_event.get("transport_height_m")),
                expected_height,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _number_value(lift_event.get("lift_delta_m")),
                expected_height - module.staging_pose.position.z,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or lift_event.get("scope") != "logical_transport_only"
            or lift_event.get("physical_lift_claimed") is not False
            or lift_event.get("physical_descent_claimed") is not False
        ):
            raise ValueError(
                f"{item.job_id} logical transport lift is invalid"
            )
        _verify_logical_transition_stop_proof(
            event=lift_event,
            module_id=item.module_id,
            robot_ids=item.executed_robot_ids,
            transition="logical_payload_lift",
            timestamp_s=item.pickup_at_s,
            commands=commands,
            telemetry=telemetry,
            trace_records=trace_records,
            config=config,
        )
        if (
            not math.isclose(
                item.logical_transport_height_m,
                expected_height,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or item.logical_transport_height_m
            > physical_yard.configuration.maximum_logical_transport_height_m
            or not math.isclose(
                item.installation_snap.from_pose.position.z,
                expected_height,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not _rpy_is_close(
                item.installation_snap.from_pose.rotation_rpy_degrees,
                module.staging_pose.rotation_rpy_degrees,
            )
            or item.installation_snap.target_pose != module.target_pose
            or item.installation_snap.target_pose_digest
            != _sha256_json(module.target_pose.model_dump(mode="json"))
            or item.installation_snap.contact_module_ids != expected_contacts
            or item.installation_snap.scope != "final_target_pose_only"
            or item.installation_snap.at_s != item.installed_at_s
            or len(snap_events) != 1
            or snap_events[0].get("from_pose")
            != item.installation_snap.from_pose.model_dump(mode="json")
            or snap_events[0].get("target_pose")
            != item.installation_snap.target_pose.model_dump(mode="json")
            or snap_events[0].get("target_pose_sha256")
            != item.installation_snap.target_pose_digest
            or snap_events[0].get("contact_module_ids") != expected_contacts
            or snap_events[0].get("scope") != "final_target_pose_only"
            or snap_events[0].get("timestamp_s") != item.installed_at_s
            or snap_events[0].get("transport_height_m")
            != item.logical_transport_height_m
            or snap_events[0].get("physical_lift_claimed") is not False
            or snap_events[0].get("physical_descent_claimed") is not False
        ):
            raise ValueError(
                f"{item.job_id} final-target-only logical snap is invalid"
            )
        _verify_logical_transition_stop_proof(
            event=snap_events[0],
            module_id=item.module_id,
            robot_ids=item.executed_robot_ids,
            transition="logical_installation_snap",
            timestamp_s=item.installed_at_s,
            commands=commands,
            telemetry=telemetry,
            trace_records=trace_records,
            config=config,
        )
        required_pair_separation = (
            2 * physical_yard.configuration.robot_footprint_radius_m
            + physical_yard.configuration.route_clearance_m
        )
        for phase_name, routes in (
            ("approach", item.approach_routes),
            ("carry", item.assigned_carry_routes),
            ("return", item.return_routes),
        ):
            first_route = routes[first_id]
            second_route = routes[second_id]
            if len(first_route) != len(second_route) or any(
                _segment_segment_distance(
                    first_route[route_index],
                    first_route[route_index + 1],
                    second_route[route_index],
                    second_route[route_index + 1],
                )
                + 1e-9
                < required_pair_separation
                for route_index in range(len(first_route) - 1)
            ):
                raise ValueError(
                    f"{item.job_id} {phase_name} route-time sweep is unsafe"
                )
        expected_trace_times = {
            "job_started": item.started_at_s,
            "logical_payload_attached": item.pickup_at_s,
            "module_installed": item.installed_at_s,
            "team_returned_to_dispatch": item.returned_at_s,
        }
        for event_name, timestamp_s in expected_trace_times.items():
            matching = [
                record
                for record in trace_records
                if record.get("event") == event_name
                and record.get("job_id") == item.job_id
                and record.get("module_id") == item.module_id
            ]
            if len(matching) != 1 or not math.isclose(
                _number_value(matching[0].get("timestamp_s")),
                timestamp_s,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{item.job_id} trace does not match {event_name}"
                )
        installed_before.add(item.module_id)


def _vec3_is_close(
    left: Vec3,
    right: Vec3,
    *,
    absolute_tolerance: float = 1e-6,
) -> bool:
    return all(
        math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
        for left_value, right_value in (
            (left.x, right.x),
            (left.y, right.y),
            (left.z, right.z),
        )
    )


def _rpy_is_close(
    left: Vec3,
    right: Vec3,
    *,
    absolute_tolerance_degrees: float = 1e-6,
) -> bool:
    return all(
        abs((left_value - right_value + 180.0) % 360.0 - 180.0)
        <= absolute_tolerance_degrees
        for left_value, right_value in (
            (left.x, right.x),
            (left.y, right.y),
            (left.z, right.z),
        )
    )


def _pose3d_is_close(left: Pose3D, right: Pose3D) -> bool:
    return _vec3_is_close(left.position, right.position) and _rpy_is_close(
        left.rotation_rpy_degrees,
        right.rotation_rpy_degrees,
    )


def _verify_logical_transition_stop_proof(
    *,
    event: Mapping[str, object],
    module_id: str,
    robot_ids: list[str],
    transition: str,
    timestamp_s: float,
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    trace_records: list[Mapping[str, object]],
    config: DynamicCoppeliaConfig,
) -> None:
    raw_proof = event.get("team_zero_motion_proof")
    if not isinstance(raw_proof, Mapping):
        raise ValueError(
            f"{module_id} {transition} lacks a zero-motion proof"
        )
    expected_ids = sorted(robot_ids)
    command_records = raw_proof.get("latest_commands_by_robot")
    measured_records = raw_proof.get("measured_motion_by_robot")
    minimum_scene_separation = _number_value(
        raw_proof.get("minimum_measured_team_to_scene_separation_m")
    )
    limiting_scene_pair = raw_proof.get(
        "minimum_measured_team_to_scene_pair"
    )
    if (
        raw_proof.get("robot_ids") != expected_ids
        or raw_proof.get("latest_commands_zero") is not True
        or raw_proof.get("measured_team_still") is not True
        or raw_proof.get("no_nonzero_team_command_after_hold") is not True
        or raw_proof.get("required_consecutive_still_samples")
        != config.logical_transition_settle_consecutive_samples
        or raw_proof.get("observed_consecutive_still_samples")
        != config.logical_transition_settle_consecutive_samples
        or not math.isclose(
            _number_value(raw_proof.get("settled_linear_speed_mps")),
            config.settled_linear_speed_mps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _number_value(raw_proof.get("settled_angular_speed_rps")),
            config.settled_angular_speed_rps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or minimum_scene_separation + 1e-9 < config.safety_distance_m
        or not math.isclose(
            _number_value(raw_proof.get("safety_distance_m")),
            config.safety_distance_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(limiting_scene_pair, list)
        or len(limiting_scene_pair) != 2
        or limiting_scene_pair != sorted(limiting_scene_pair)
        or not set(limiting_scene_pair).intersection(expected_ids)
        or not isinstance(command_records, Mapping)
        or set(command_records) != set(expected_ids)
        or not isinstance(measured_records, Mapping)
        or set(measured_records) != set(expected_ids)
    ):
        raise ValueError(
            f"{module_id} {transition} zero-motion proof is incomplete"
        )
    hold_command_index = _integer_value(
        raw_proof.get("hold_command_index")
    )
    command_count_at_settle = _integer_value(
        raw_proof.get("command_count_at_settle")
    )
    if (
        hold_command_index < 0
        or command_count_at_settle < hold_command_index + len(expected_ids)
        or command_count_at_settle > len(commands)
        or not math.isclose(
            _number_value(raw_proof.get("settled_at_s")),
            timestamp_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or _number_value(raw_proof.get("started_at_s")) > timestamp_s
        or _integer_value(raw_proof.get("physics_steps_elapsed")) < 0
    ):
        raise ValueError(
            f"{module_id} {transition} settle boundary is invalid"
        )
    settle_commands = commands[
        hold_command_index:command_count_at_settle
    ]
    team_settle_commands = [
        command
        for command in settle_commands
        if command.robot_id in expected_ids
    ]
    if (
        {command.robot_id for command in team_settle_commands}
        != set(expected_ids)
        or any(
            abs(command.linear_velocity_mps) > 1e-9
            or abs(command.angular_velocity_rps) > 1e-9
            or any(
                abs(value) > 1e-9
                for value in command.wheel_target_velocity_rad_s
            )
            for command in team_settle_commands
        )
    ):
        raise ValueError(
            f"{module_id} {transition} command slice is not a zero team hold"
        )
    by_robot: dict[str, RobotCommand] = {}
    for command in team_settle_commands:
        by_robot[command.robot_id] = command
    for robot_id in expected_ids:
        command = by_robot[robot_id]
        raw_command = command_records.get(robot_id)
        if not isinstance(raw_command, Mapping):
            raise ValueError(
                f"{module_id} {transition} has malformed command proof"
            )
        if (
            command.source != "formation_hold"
            or abs(command.linear_velocity_mps) > 1e-9
            or abs(command.angular_velocity_rps) > 1e-9
            or any(
                abs(value) > 1e-9
                for value in command.wheel_target_velocity_rad_s
            )
            or raw_command.get("present") is not True
            or raw_command.get("zero") is not True
            or raw_command.get("timestamp_s") != command.timestamp_s
            or raw_command.get("source") != command.source
            or raw_command.get("linear_velocity_mps")
            != command.linear_velocity_mps
            or raw_command.get("angular_velocity_rps")
            != command.angular_velocity_rps
            or raw_command.get("wheel_target_velocity_rad_s")
            != list(command.wheel_target_velocity_rad_s)
        ):
            raise ValueError(
                f"{module_id} {transition} hold command proof differs"
            )
        raw_measured = measured_records.get(robot_id)
        if not isinstance(raw_measured, Mapping):
            raise ValueError(
                f"{module_id} {transition} has malformed telemetry proof"
            )
        measured_timestamp = _number_value(
            raw_measured.get("timestamp_s")
        )
        measured_linear = _number_value(
            raw_measured.get("linear_velocity_mps")
        )
        measured_angular = _number_value(
            raw_measured.get("angular_velocity_rps")
        )
        if (
            raw_measured.get("still") is not True
            or abs(measured_linear) > config.settled_linear_speed_mps
            or abs(measured_angular) > config.settled_angular_speed_rps
            or not any(
                sample.robot_id == robot_id
                and math.isclose(
                    sample.timestamp_s,
                    measured_timestamp,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    sample.linear_velocity_mps,
                    measured_linear,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    sample.angular_velocity_rps,
                    measured_angular,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for sample in telemetry
            )
        ):
            raise ValueError(
                f"{module_id} {transition} measured-stop proof differs"
            )
    settled_events = [
        record
        for record in trace_records
        if record.get("event") == "logical_transition_settled"
        and record.get("module_id") == module_id
        and record.get("transition") == transition
    ]
    if (
        len(settled_events) != 1
        or settled_events[0].get("team_zero_motion_proof") != raw_proof
        or not math.isclose(
            _number_value(settled_events[0].get("timestamp_s")),
            timestamp_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            f"{module_id} {transition} settle event is missing or different"
        )
    settle_samples = settled_events[0].get("settle_samples")
    required_count = config.logical_transition_settle_consecutive_samples
    if not isinstance(settle_samples, list) or len(settle_samples) < required_count:
        raise ValueError(
            f"{module_id} {transition} settle samples are missing"
        )
    recomputed_separation: tuple[float, str, str] | None = None
    for raw_sample in settle_samples:
        if not isinstance(raw_sample, Mapping):
            raise ValueError(
                f"{module_id} {transition} settle sample is malformed"
            )
        sample_timestamp_s = _number_value(raw_sample.get("timestamp_s"))
        poses_at_timestamp: dict[str, Vec2] = {}
        for sample in telemetry:
            if math.isclose(
                sample.timestamp_s,
                sample_timestamp_s,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                poses_at_timestamp[sample.robot_id] = Vec2.model_validate(
                    sample.measured_pose.position
                )
        if (
            not set(expected_ids).issubset(poses_at_timestamp)
            or len(poses_at_timestamp) < 2
        ):
            raise ValueError(
                f"{module_id} {transition} scene separation lacks telemetry"
            )
        pose_ids = sorted(poses_at_timestamp)
        for left_index, left_id in enumerate(pose_ids):
            for right_id in pose_ids[left_index + 1 :]:
                if left_id not in expected_ids and right_id not in expected_ids:
                    continue
                candidate = (
                    _distance(
                        poses_at_timestamp[left_id],
                        poses_at_timestamp[right_id],
                    ),
                    left_id,
                    right_id,
                )
                if (
                    recomputed_separation is None
                    or candidate < recomputed_separation
                ):
                    recomputed_separation = candidate
    if (
        recomputed_separation is None
        or not math.isclose(
            recomputed_separation[0],
            minimum_scene_separation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or list(recomputed_separation[1:]) != limiting_scene_pair
    ):
        raise ValueError(
            f"{module_id} {transition} scene separation proof differs"
        )
    stable_tail = settle_samples[-required_count:]
    previous_timestamp_s = -math.inf
    previous_physics_step = -1
    control_period_s = 1.0 / config.control_hz
    for stable_index, raw_sample in enumerate(stable_tail, start=1):
        if (
            not isinstance(raw_sample, Mapping)
            or raw_sample.get("team_still") is not True
            or raw_sample.get("consecutive_still_samples") != stable_index
        ):
            raise ValueError(
                f"{module_id} {transition} stable settle tail is invalid"
            )
        sample_timestamp_s = _number_value(raw_sample.get("timestamp_s"))
        physics_step = _integer_value(raw_sample.get("physics_step"))
        raw_robots = raw_sample.get("robots")
        if (
            sample_timestamp_s < previous_timestamp_s
            or sample_timestamp_s > timestamp_s + 1e-9
            or physics_step <= previous_physics_step
            or (
                stable_index > 1
                and (
                    physics_step != previous_physics_step + 1
                    or not math.isclose(
                        sample_timestamp_s,
                        previous_timestamp_s + control_period_s,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )
            )
            or not isinstance(raw_robots, Mapping)
            or set(raw_robots) != set(expected_ids)
        ):
            raise ValueError(
                f"{module_id} {transition} settle sample ordering is invalid"
            )
        previous_timestamp_s = sample_timestamp_s
        previous_physics_step = physics_step
        for robot_id in expected_ids:
            raw_robot = raw_robots.get(robot_id)
            if not isinstance(raw_robot, Mapping):
                raise ValueError(
                    f"{module_id} {transition} stable robot sample is invalid"
                )
            robot_timestamp_s = _number_value(
                raw_robot.get("timestamp_s")
            )
            linear_velocity_mps = _number_value(
                raw_robot.get("linear_velocity_mps")
            )
            angular_velocity_rps = _number_value(
                raw_robot.get("angular_velocity_rps")
            )
            if (
                raw_robot.get("still") is not True
                or not math.isclose(
                    robot_timestamp_s,
                    sample_timestamp_s,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or abs(linear_velocity_mps)
                > config.settled_linear_speed_mps
                or abs(angular_velocity_rps)
                > config.settled_angular_speed_rps
                or not any(
                    sample.robot_id == robot_id
                    and math.isclose(
                        sample.timestamp_s,
                        robot_timestamp_s,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        sample.linear_velocity_mps,
                        linear_velocity_mps,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        sample.angular_velocity_rps,
                        angular_velocity_rps,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    for sample in telemetry
                )
            ):
                raise ValueError(
                    f"{module_id} {transition} stable telemetry was fabricated"
                )
    if not math.isclose(
        previous_timestamp_s,
        timestamp_s,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{module_id} {transition} stable settle tail does not end at the transition"
        )


def _verify_collision_trace(
    *,
    scenario: ScenarioManifest,
    trace: list[object],
    diagnostics: Mapping[str, object],
    physics_steps: int,
    expected_queries: int,
) -> None:
    inventory = _expected_collision_inventory(scenario.plan)
    category_counts = Counter(
        str(item["category"]) for item in inventory
    )
    expected_categories = dict(sorted(category_counts.items()))
    if expected_queries != len(inventory):
        raise ValueError(
            "collision-query count does not match the scenario inventory"
        )
    metric_categories = _mapping_field(
        diagnostics,
        "collision_pair_category_counts",
    )
    if metric_categories != expected_categories:
        raise ValueError(
            "collision-query categories do not match the scenario inventory"
        )
    records = [item for item in trace if isinstance(item, Mapping)]
    ready = [
        item for item in records if item.get("event") == "collision_monitor_ready"
    ]
    robot_ids = sorted(robot.robot_id for robot in scenario.plan.robots)
    module_ids = sorted(module.module_id for module in scenario.plan.modules)
    obstacle_ids = sorted(
        f"{x},{y}" for x, y in scenario.plan.site_grid.obstacle_cells
    )
    if (
        len(ready) != 1
        or ready[0].get("expected_queries_per_step") != len(inventory)
        or ready[0].get("pair_category_counts") != expected_categories
        or ready[0].get("pair_inventory_sha256")
        != _sha256_json(inventory)
        or ready[0].get("robot_ids") != robot_ids
        or ready[0].get("module_ids") != module_ids
        or ready[0].get("obstacle_ids") != obstacle_ids
    ):
        raise ValueError("collision-monitor inventory trace is incomplete")
    rounds = [
        item for item in records if item.get("event") == "collision_query_round"
    ]
    by_step: dict[int, Mapping[str, object]] = {}
    allowed_pairs = {
        (str(item["entity_1"]), str(item["entity_2"])): item
        for item in inventory
    }
    detected_event_count = 0
    physical_stop_count = 0
    for record in rounds:
        step = _integer_value(record.get("physics_step"))
        if step in by_step:
            raise ValueError("collision trace repeats a physics step")
        by_step[step] = record
        if (
            record.get("query_count") != len(inventory)
            or record.get("pair_category_counts") != expected_categories
        ):
            raise ValueError(
                f"collision trace step {step} has incomplete pair coverage"
            )
        detected = record.get("detected_collisions")
        colliding = record.get("colliding_robot_ids")
        if not isinstance(detected, list) or not isinstance(colliding, list):
            raise ValueError(
                f"collision trace step {step} has malformed results"
            )
        derived_robots: set[str] = set()
        for raw_event in detected:
            if not isinstance(raw_event, Mapping):
                raise ValueError("collision trace contains a malformed event")
            pair = (
                str(raw_event.get("entity_1")),
                str(raw_event.get("entity_2")),
            )
            expected = allowed_pairs.get(pair)
            expected_robot_ids = (
                expected.get("robot_ids")
                if expected is not None
                else None
            )
            if (
                expected is None
                or not isinstance(expected_robot_ids, list)
                or raw_event.get("category") != expected["category"]
                or raw_event.get("robot_ids") != expected_robot_ids
                or raw_event.get("permitted_logical_payload_overlap")
                is not False
            ):
                raise ValueError(
                    "collision trace contains an unknown or exempted pair"
                )
            derived_robots.update(
                str(item) for item in expected_robot_ids
            )
            detected_event_count += 1
        if sorted(derived_robots) != colliding:
            raise ValueError(
                f"collision trace step {step} has inconsistent robot stops"
            )
        physical_stop_count += len(derived_robots)
    if set(by_step) != set(range(1, physics_steps + 1)):
        raise ValueError(
            "collision trace does not contain exactly one round per physics step"
        )
    if (
        _integer_metric(diagnostics, "physical_collision_event_count")
        != detected_event_count
        or _integer_metric(diagnostics, "physical_collision_stops")
        != physical_stop_count
        or _integer_metric(diagnostics, "permitted_logical_payload_overlaps")
        != 0
    ):
        raise ValueError(
            "collision diagnostics do not match raw per-step query results"
        )


def _verify_runtime_inventory_trace(
    *,
    scenario: ScenarioManifest,
    trace: list[object],
    diagnostics: Mapping[str, object],
) -> None:
    records = [item for item in trace if isinstance(item, Mapping)]
    scene = [
        item for item in records if item.get("event") == "scene_inventory_built"
    ]
    robot_ids = sorted(robot.robot_id for robot in scenario.plan.robots)
    module_ids = sorted(module.module_id for module in scenario.plan.modules)
    obstacle_ids = sorted(
        f"{x},{y}" for x, y in scenario.plan.site_grid.obstacle_cells
    )
    if (
        len(scene) != 1
        or scene[0].get("robot_ids") != robot_ids
        or scene[0].get("module_ids") != module_ids
        or scene[0].get("obstacle_ids") != obstacle_ids
        or scene[0].get("robot_count") != len(robot_ids)
        or scene[0].get("module_count") != len(module_ids)
        or scene[0].get("obstacle_count") != len(obstacle_ids)
    ):
        raise ValueError("scene-inventory trace does not match the scenario")
    initial_writes = [
        item
        for item in records
        if item.get("event") == "initial_robot_pose_write"
    ]
    if (
        len(initial_writes) != len(robot_ids)
        or len(
            {
                _integer_value(item.get("robot_handle"))
                for item in initial_writes
            }
        )
        != len(robot_ids)
        or any(
            item.get("simulation_started") is not False
            or item.get("timestamp_s") != 0.0
            or not _is_xyz_list(item.get("position"))
            for item in initial_writes
        )
        or _integer_metric(diagnostics, "initial_robot_pose_writes")
        != len(robot_ids)
        or _integer_metric(diagnostics, "post_start_robot_pose_writes") != 0
    ):
        raise ValueError("robot pose-write audit is incomplete or post-start")


def _expected_collision_inventory(plan: BuildPlan) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    robot_ids = sorted(robot.robot_id for robot in plan.robots)
    module_ids = sorted(module.module_id for module in plan.modules)
    obstacle_ids = sorted(f"{x},{y}" for x, y in plan.site_grid.obstacle_cells)
    for index, robot_id in enumerate(robot_ids):
        for other_id in robot_ids[index + 1 :]:
            inventory.append(
                {
                    "entity_1": f"robot:{robot_id}",
                    "entity_2": f"robot:{other_id}",
                    "category": "robot_robot",
                    "robot_ids": [robot_id, other_id],
                }
            )
        for module_id in module_ids:
            inventory.append(
                {
                    "entity_1": f"robot:{robot_id}",
                    "entity_2": f"module:{module_id}",
                    "category": "robot_module",
                    "robot_ids": [robot_id],
                }
            )
        for obstacle_id in obstacle_ids:
            inventory.append(
                {
                    "entity_1": f"robot:{robot_id}",
                    "entity_2": f"obstacle:{obstacle_id}",
                    "category": "robot_obstacle",
                    "robot_ids": [robot_id],
                }
            )
    return inventory


def _distance(left: Vec2, right: Vec2) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _centroid(points: Iterable[Vec2]) -> Vec2:
    values = list(points)
    if not values:
        raise ValueError("centroid requires measured two-dimensional points")
    return Vec2(
        x=sum(item.x for item in values) / len(values),
        y=sum(item.y for item in values) / len(values),
    )


def _pair_spacing_error(
    measured: Mapping[str, Vec2],
    planned: Mapping[str, Vec2],
) -> float:
    robot_ids = sorted(measured)
    if len(robot_ids) != 2 or set(planned) != set(robot_ids):
        raise ValueError("spacing evidence requires the same two-base team")
    return abs(
        _distance(measured[robot_ids[0]], measured[robot_ids[1]])
        - _distance(planned[robot_ids[0]], planned[robot_ids[1]])
    )


def _integer_value(value: object) -> int:
    parsed = _number_value(value)
    if not parsed.is_integer():
        raise ValueError("value must be an integer")
    return int(parsed)


def _is_xyz_list(value: object) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def _verify_script_control_audit(
    *,
    diagnostics: Mapping[str, object],
    robot_ids: set[str],
    discovered_scripts: int,
    disabled_scripts: int,
    retained_scripts: int,
) -> None:
    audit = diagnostics.get("script_control_audit")
    if not isinstance(audit, list) or len(audit) != len(robot_ids):
        raise ValueError("YouBot script-control audit is incomplete")
    audited_robots: set[str] = set()
    audited_scripts = 0
    audited_disabled = 0
    audited_retained = 0
    for raw_robot_audit in audit:
        if not isinstance(raw_robot_audit, Mapping):
            raise ValueError("YouBot script-control audit entry must be an object")
        robot_id = raw_robot_audit.get("robot_id")
        scripts = raw_robot_audit.get("scripts")
        if (
            not isinstance(robot_id, str)
            or robot_id not in robot_ids
            or robot_id in audited_robots
            or raw_robot_audit.get("exclusive_control_verified") is not True
            or not isinstance(scripts, list)
            or not scripts
        ):
            raise ValueError("YouBot script-control robot audit is invalid")
        audited_robots.add(robot_id)
        wheel_writer_count = 0
        maintenance_count = 0
        for raw_script in scripts:
            if not isinstance(raw_script, Mapping):
                raise ValueError("YouBot script audit record must be an object")
            role = raw_script.get("role")
            action = raw_script.get("action")
            source_sha256 = raw_script.get("source_sha256")
            disabled_after = raw_script.get("disabled_after")
            if (
                not isinstance(source_sha256, str)
                or re.fullmatch(_SHA256_PATTERN, source_sha256) is None
            ):
                raise ValueError("YouBot script audit lacks a source digest")
            if role == "wheel_command_writer":
                wheel_writer_count += 1
                expected_action = "disabled"
                expected_disabled = True
            elif role == "arm_gripper_writer":
                expected_action = "disabled"
                expected_disabled = True
            elif role == "passive_omniwheel_maintenance":
                maintenance_count += 1
                expected_action = "retained_enabled"
                expected_disabled = False
            else:
                raise ValueError("YouBot script audit contains an unknown role")
            if (
                action != expected_action
                or disabled_after is not expected_disabled
            ):
                raise ValueError("YouBot script ownership action is inconsistent")
            audited_scripts += 1
            if expected_disabled:
                audited_disabled += 1
            else:
                audited_retained += 1
        if wheel_writer_count != 1 or maintenance_count < 1:
            raise ValueError("YouBot per-robot script roles are incomplete")
    if (
        audited_robots != robot_ids
        or audited_scripts != discovered_scripts
        or audited_disabled != disabled_scripts
        or audited_retained != retained_scripts
    ):
        raise ValueError("YouBot script-control counts do not match the audit")


def _verify_recovery_evidence(
    result: Phase5RunResult,
    replay: list[Phase5JobReplay],
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    trace: list[object],
    config: DynamicCoppeliaConfig,
) -> None:
    recovery = result.recovery
    if (
        recovery is None
        or recovery.completion_fraction_at_disable < 0.25
        or recovery.commands_after_stop != 0
        or not recovery.settled
        or recovery.settle_sample_count
        < config.disabled_settle_consecutive_samples + 1
        or not recovery.reassigned_job_ids
    ):
        raise ValueError("unavailable-robot recovery record is incomplete")
    if any(
        command.robot_id == recovery.robot_id
        for command in commands[recovery.stop_command_index :]
    ):
        raise ValueError("disabled robot received commands after its stop cutoff")
    disabled_samples = [
        item
        for item in telemetry
        if item.robot_id == recovery.robot_id
        and recovery.disabled_at_s <= item.timestamp_s <= recovery.settled_at_s
    ]
    if len(disabled_samples) < recovery.settle_sample_count:
        raise ValueError("disabled-robot settle evidence is missing telemetry")
    stable_tail = disabled_samples[-config.disabled_settle_consecutive_samples :]
    if any(
        item.linear_velocity_mps > config.settled_linear_speed_mps
        or item.angular_velocity_rps > config.settled_angular_speed_rps
        for item in stable_tail
    ):
        raise ValueError("disabled robot did not measurably settle")
    first_position = disabled_samples[0].measured_pose.position
    last_position = disabled_samples[-1].measured_pose.position
    measured_settle_displacement = math.sqrt(
        (last_position.x - first_position.x) ** 2
        + (last_position.y - first_position.y) ** 2
        + (last_position.z - first_position.z) ** 2
    )
    if not math.isclose(
        measured_settle_displacement,
        recovery.settle_displacement_m,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("recovery settle displacement does not match telemetry")
    reassigned = {
        item.job_id
        for item in replay
        if recovery.robot_id in item.planned_robot_ids
        and recovery.robot_id not in item.executed_robot_ids
        and item.reassigned_after_unavailability
    }
    if reassigned != set(recovery.reassigned_job_ids):
        raise ValueError("recovery reassignment list does not match replay")
    post_disable_jobs = [
        item for item in replay if item.pickup_at_s >= recovery.disabled_at_s
    ]
    if not post_disable_jobs:
        raise ValueError("recovery evidence has no post-disable payload routes")
    for item in post_disable_jobs:
        disabled_payload_records = [
            record
            for record in item.clearance_minima
            if record.phase == "carry"
            and record.mover_kind == "logical_payload"
            and record.obstacle_kind == "disabled_robot"
        ]
        if (
            {record.source for record in disabled_payload_records}
            != {"planned", "measured"}
            or any(
                record.evaluated_pair_count < 1
                or record.minimum_surface_clearance_m is None
                or record.minimum_surface_clearance_m + 1e-9
                < record.required_clearance_m
                for record in disabled_payload_records
            )
        ):
            raise ValueError(
                f"{item.job_id} lacks planned/measured payload clearance "
                "to the disabled robot"
            )
    trace_events = {
        str(item.get("event"))
        for item in trace
        if isinstance(item, dict)
    }
    if not {"robot_unavailable", "disabled_robot_settled"}.issubset(trace_events):
        raise ValueError("recovery trace lacks disable or settle events")
    installed_before_disable = {
        str(item.get("module_id"))
        for item in trace
        if isinstance(item, dict)
        and item.get("event") == "module_installed"
        and _number_value(item.get("timestamp_s"), default=math.inf)
        <= recovery.disabled_at_s
    }
    measured_completion = len(installed_before_disable) / result.expected_module_count
    if (
        measured_completion < 0.25
        or not math.isclose(
            measured_completion,
            recovery.completion_fraction_at_disable,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or recovery.remaining_module_count
        != result.expected_module_count - len(installed_before_disable)
    ):
        raise ValueError("recovery disable point does not match the measured trace")


def _formation_point(center: Vec2, index: int, team_size: int) -> Vec2:
    offset = 0.0 if team_size == 1 else (-0.35 if index == 0 else 0.35)
    return Vec2(x=center.x, y=center.y + offset)


def _with_exact_endpoint(path: list[Vec2], endpoint: Vec2) -> list[Vec2]:
    result = [point.model_copy(deep=True) for point in path]
    if not result or math.hypot(
        result[-1].x - endpoint.x,
        result[-1].y - endpoint.y,
    ) > 1e-9:
        result.append(endpoint.model_copy(deep=True))
    return result


def _event_sort_key(item: dict[str, object]) -> tuple[float, str]:
    return (
        _number_value(item.get("timestamp_s"), default=0.0),
        str(item.get("event", "")),
    )


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _load_json_list(path: Path) -> list[object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return payload


def _load_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number} is blank")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name}:{line_number} is not valid JSON"
            ) from exc
    return rows


def _timestamps_are_monotonic(
    rows: list[RobotCommand] | list[RobotTelemetry],
) -> bool:
    return all(
        rows[index].timestamp_s <= rows[index + 1].timestamp_s
        for index in range(len(rows) - 1)
    )


def _mapping_field(
    payload: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _number_value(value: object, *, default: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric metric")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if default is not None:
        return default
    raise ValueError("metric must be a finite number")


def _number_metric(payload: Mapping[str, object], name: str) -> float:
    try:
        return _number_value(payload[name])
    except KeyError as exc:
        raise ValueError(f"missing numeric metric: {name}") from exc


def _integer_metric(payload: Mapping[str, object], name: str) -> int:
    value = _number_metric(payload, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase5_report(result: Phase5RunResult) -> str:
    def minimum_world_clearance(
        phase: ClearancePhase,
        source: ClearanceSource,
    ) -> float:
        return min(
            (
                record.minimum_surface_clearance_m
                for item in result.replay
                for record in item.clearance_minima
                if record.phase == phase
                and record.source == source
                and record.minimum_surface_clearance_m is not None
            ),
            default=0.0,
        )

    acceptance = "\n".join(
        f"- [{'x' if passed else ' '}] `{gate}`"
        for gate, passed in result.acceptance.items()
    )
    recovery = (
        "\n".join(
            [
                "## Unavailable-robot recovery",
                "",
                f"- Robot: `{result.recovery.robot_id}`",
                (
                    "- Completion at disable: "
                    f"`{result.recovery.completion_fraction_at_disable:.3f}`"
                ),
                f"- Reassigned jobs: `{len(result.recovery.reassigned_job_ids)}`",
                f"- Commands after stop: `{result.recovery.commands_after_stop}`",
                "",
            ]
        )
        if result.recovery is not None
        else ""
    )
    return "\n".join(
        [
            "# Construction Intelligence v1 — Dynamic Coppelia Evidence",
            "",
            f"- Status: `{result.status}`",
            f"- Evidence kind: `{result.evidence_kind}`",
            f"- Scenario: `{result.scenario}`",
            f"- Scenario ID: `{result.scenario_id}`",
            (
                "- Modules installed: "
                f"`{len(result.installed_module_ids)}/{result.expected_module_count}`"
            ),
            f"- Live gate passed: `{str(result.live_gate_passed).lower()}`",
            (
                "- Physical-yard digest: "
                f"`{result.metrics.get('physical_yard_digest')}`"
            ),
            (
                "- Physical carrier formation: `two bases minimum`; planned "
                "and executed teams are recorded separately"
            ),
            (
                "- Maximum measured formation offset: "
                f"`{result.metrics.get('maximum_formation_offset_m')} m`"
            ),
            (
                "- Maximum preflight formation offset: "
                f"`{result.metrics.get('maximum_preflight_formation_offset_m')} m`"
            ),
            (
                "- Maximum logical transport center height: "
                f"`{result.metrics.get('maximum_logical_transport_height_m')} m`"
            ),
            (
                "- Maximum preflight transport center height: "
                f"`{result.metrics.get('maximum_preflight_transport_height_m')} m`"
            ),
            (
                "- Minimum planned payload clearance: "
                f"`{min((item.minimum_planned_payload_clearance_m for item in result.replay), default=0.0):.4f} m`"
            ),
            (
                "- Minimum measured payload clearance: "
                f"`{min((item.minimum_measured_payload_clearance_m for item in result.replay), default=0.0):.4f} m`"
            ),
            (
                "- Continuous approach world clearance (planned/measured): "
                f"`{minimum_world_clearance('approach', 'planned'):.4f} / "
                f"{minimum_world_clearance('approach', 'measured'):.4f} m`"
            ),
            (
                "- Continuous carry world clearance (planned/measured): "
                f"`{minimum_world_clearance('carry', 'planned'):.4f} / "
                f"{minimum_world_clearance('carry', 'measured'):.4f} m`"
            ),
            (
                "- Continuous return world clearance (planned/measured): "
                f"`{minimum_world_clearance('return', 'planned'):.4f} / "
                f"{minimum_world_clearance('return', 'measured'):.4f} m`"
            ),
            (
                "- Logical lift / final snap records: "
                f"`{result.metrics.get('logical_payload_lift_count')} / "
                f"{result.metrics.get('logical_installation_snap_count')}`"
            ),
            "",
            "## Acceptance gates",
            "",
            acceptance,
            "",
            recovery,
            "## Payload model and limitations",
            "",
            (
                "Payload transport is explicitly logical and overhead. A bounded "
                "transport center height is derived from conservative full-RPY module "
                "AABBs. Finite installed, staged, and site-obstacle geometry may be "
                "overflown only with the recorded vertical separation; idle and disabled "
                "robots are infinite-height XY exclusion columns. Modules are parented "
                "to a carrier dummy synchronized to measured two-base centroids, and "
                "equal-horizon base routes are fixed offsets from one densified carrier "
                "route."
            ),
            "",
            (
                "The only permitted structure contact is one attested logical snap at the "
                "frozen final target pose after the measured team is stopped. Logical "
                "lifting and descent are not physical motions. This evidence does not "
                "claim arm motion, gripper actuation, grasp contact, cooperative contact "
                "dynamics, or physical payload dynamics."
            ),
            "",
            f"Failure: `{result.error}`" if result.error else "",
            "",
        ]
    )
