from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

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
)
from embodied_skill_composer.construction.scheduler import schedule_build


Phase5Scenario = Literal["nominal", "unavailable_robot_recovery"]
EvidenceKind = Literal["offline_harness", "live_coppelia"]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40,64}$"
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
    measured_carry_routes: dict[str, list[Vec2]]
    measured_carry_samples: dict[str, list[RobotTelemetry]]
    started_at_s: float = Field(ge=0)
    pickup_at_s: float = Field(ge=0)
    installed_at_s: float = Field(ge=0)
    measured_poses: list[RobotTelemetry]
    maximum_formation_error_m: float = Field(ge=0)
    maximum_assignment_error_m: float = Field(ge=0)
    maximum_spacing_error_m: float = Field(ge=0)
    maximum_install_error_m: float = Field(ge=0)
    minimum_pickup_base_clearance_m: float = Field(ge=0)
    minimum_install_base_clearance_m: float = Field(ge=0)
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
            self.started_at_s <= self.pickup_at_s <= self.installed_at_s
        ):
            raise ValueError("measured job timestamps are not monotonic")
        expected = set(self.executed_robot_ids)
        if (
            not expected
            or set(self.approach_routes) != expected
            or set(self.assigned_carry_routes) != expected
            or set(self.measured_carry_routes) != expected
            or set(self.measured_carry_samples) != expected
        ):
            raise ValueError("replay route identities must match the executed team")
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
    measured_carry_routes: dict[str, list[Vec2]]


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
        ):
            raise ValueError("live gate lacks a complete simulator attestation")
        return self


class Phase5PhysicalYardConfig(_StrictModel):
    schema_version: Literal["construction_intelligence.phase5_yard_config.v1"] = (
        "construction_intelligence.phase5_yard_config.v1"
    )
    allocator: Literal["deterministic_shelf_yard"] = (
        "deterministic_shelf_yard"
    )
    grid_resolution_m: float = Field(default=0.5, gt=0)
    yard_shelf_width_m: float = Field(default=14.0, ge=8.0)
    module_gap_m: float = Field(default=1.2, ge=0.8)
    yard_house_aisle_m: float = Field(default=2.0, ge=1.0)
    boundary_margin_m: float = Field(default=2.0, ge=1.0)
    dispatch_lane_gap_m: float = Field(default=1.5, ge=0.8)
    dispatch_robot_spacing_m: float = Field(default=1.0, ge=0.6)
    robot_footprint_radius_m: float = Field(default=0.12, ge=0.08)
    route_clearance_m: float = Field(default=0.16, ge=0.1)
    formation_clearance_m: float = Field(default=0.5, ge=0.3)
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
        module.staging_pose.position = Vec3(
            x=yard_right - local_x,
            y=yard_bottom + local_y,
            z=module.dimensions.height / 2,
        )
        module.staging_pose.rotation_rpy_degrees = Vec3(x=0, y=0, z=0)

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
    _preflight_phase5_sequential_routes(plan, config)
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
            -item.dimensions.depth,
            -item.dimensions.width,
            item.module_id,
        ),
    )
    for module in ordered:
        width = module.dimensions.width
        depth = module.dimensions.depth
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
    yaw = math.radians(pose.rotation_rpy_degrees.z)
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
    return (
        pose.position.x - projected_x,
        pose.position.x + projected_x,
        pose.position.y - projected_y,
        pose.position.y + projected_y,
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
    for cell in plan.site_grid.obstacle_cells:
        obstacles.update(
            _rasterize_bounds(
                grid,
                _inflate_bounds(
                    _obstacle_bounds(cell, grid),
                    config.route_clearance_m,
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
                _inflate_bounds(bounds, config.route_clearance_m),
            )
        )
    idle_clearance = (
        config.robot_footprint_radius_m + config.route_clearance_m
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


def _formation_axes(
    module: BuildModule,
    *,
    yaw_degrees: float,
) -> list[tuple[float, float, float]]:
    yaw = math.radians(yaw_degrees)
    local_x = (math.cos(yaw), math.sin(yaw))
    local_y = (-math.sin(yaw), math.cos(yaw))
    axes = [
        (module.dimensions.width / 2, local_x[0], local_x[1]),
        (module.dimensions.depth / 2, local_y[0], local_y[1]),
    ]
    return sorted(axes, key=lambda item: (item[0], item[1], item[2]))


def _route_to_clear_symmetric_formation(
    *,
    router: RoutingAdapter,
    grid: SiteGrid,
    starts: Mapping[str, Vec2],
    module: BuildModule,
    center: Vec2,
    yaw_degrees: float,
    config: Phase5PhysicalYardConfig,
) -> tuple[RoutePlan, dict[str, Vec2]]:
    robot_ids = sorted(starts)
    if len(robot_ids) != config.phase5_team_size:
        raise DynamicCoppeliaError(
            "Phase 5 physical formations require exactly two bases"
        )
    maximum_span = max(
        grid.width * grid.resolution_m,
        grid.height * grid.resolution_m,
    )
    for half_extent, axis_x, axis_y in _formation_axes(
        module,
        yaw_degrees=yaw_degrees,
    ):
        minimum_offset = half_extent + config.formation_clearance_m
        increment_count = math.ceil(
            (maximum_span / 2 - minimum_offset) / grid.resolution_m
        )
        for increment in range(max(0, increment_count) + 1):
            offset = minimum_offset + increment * grid.resolution_m
            negative = Vec2(
                x=center.x - axis_x * offset,
                y=center.y - axis_y * offset,
            )
            positive = Vec2(
                x=center.x + axis_x * offset,
                y=center.y + axis_y * offset,
            )
            for goals in (
                {
                    robot_ids[0]: negative,
                    robot_ids[1]: positive,
                },
                {
                    robot_ids[0]: positive,
                    robot_ids[1]: negative,
                },
            ):
                try:
                    route = router.route_many(grid, starts, goals)
                except (RoutingError, RuntimeError, ValueError):
                    continue
                return route, goals
    raise DynamicCoppeliaError(
        f"no collision-clear symmetric formation is routeable for "
        f"{module.module_id}"
    )


def _centroid_route(
    paths: Mapping[str, list[Vec2]],
    *,
    carrier_offset: Vec3 | None = None,
) -> list[Vec2]:
    if len(paths) != 2 or any(not path for path in paths.values()):
        raise DynamicCoppeliaError(
            "Phase 5 carrier routes require two non-empty base paths"
        )
    ordered = [paths[robot_id] for robot_id in sorted(paths)]
    horizon = max(len(path) for path in ordered)
    offset = carrier_offset or Vec3(x=0, y=0, z=0)
    route: list[Vec2] = []
    for index in range(horizon):
        first = ordered[0][min(index, len(ordered[0]) - 1)]
        second = ordered[1][min(index, len(ordered[1]) - 1)]
        route.append(
            Vec2(
                x=(first.x + second.x) / 2 + offset.x,
                y=(first.y + second.y) / 2 + offset.y,
            )
        )
    return route


def _preflight_phase5_sequential_routes(
    plan: BuildPlan,
    config: Phase5PhysicalYardConfig,
) -> None:
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
    for job in schedule.jobs:
        module = modules[job.module_id]
        team = _phase5_team(
            plan,
            module,
            unavailable=set(),
            usage=usage,
            config=config,
        )
        active_across_run.update(team)
        starts = {robot_id: positions[robot_id] for robot_id in team}
        pickup_grid = _phase5_route_grid(
            plan,
            active_module_id=module.module_id,
            active_module_is_logical_payload=False,
            installed_module_ids=installed,
            robot_positions=positions,
            active_robot_ids=set(team),
            config=config,
        )
        pickup_route, pickup_goals = _route_to_clear_symmetric_formation(
            router=router,
            grid=pickup_grid,
            starts=starts,
            module=module,
            center=Vec2.model_validate(module.staging_pose.position),
            yaw_degrees=module.staging_pose.rotation_rpy_degrees.z,
            config=config,
        )
        if not pickup_route.world_paths:
            raise ValueError(f"{module.module_id} staging bay is unreachable")
        positions.update(pickup_goals)
        carry_grid = _phase5_route_grid(
            plan,
            active_module_id=module.module_id,
            active_module_is_logical_payload=True,
            installed_module_ids=installed,
            robot_positions=positions,
            active_robot_ids=set(team),
            config=config,
        )
        carry_route, install_goals = _route_to_clear_symmetric_formation(
            router=router,
            grid=carry_grid,
            starts={robot_id: positions[robot_id] for robot_id in team},
            module=module,
            center=Vec2.model_validate(module.target_pose.position),
            yaw_degrees=module.target_pose.rotation_rpy_degrees.z,
            config=config,
        )
        if not carry_route.world_paths:
            raise ValueError(f"{module.module_id} install formation is unreachable")
        positions.update(install_goals)
        installed.add(module.module_id)
        for robot_id in team:
            usage[robot_id] += 1
    expected_robots = {robot.robot_id for robot in plan.robots}
    if active_across_run != expected_robots:
        raise ValueError(
            "Phase 5 balanced two-base allocation did not activate every robot"
        )


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
            self.executor.stop()

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
        started_at_s = self.executor.simulation_time_s
        telemetry_start = len(self.executor.telemetry)
        formation_error_start = len(self.executor.formation_errors_m)
        assignment_error_start = len(
            self.executor.formation_assignment_errors_m
        )
        spacing_error_start = len(self.executor.formation_spacing_errors_m)
        install_error_start = len(self.executor.install_errors_m)
        measured_positions = self._measured_robot_positions()
        starts = {
            robot_id: measured_positions[robot_id]
            for robot_id in executed_robot_ids
        }
        approach_grid = _phase5_route_grid(
            self.plan,
            active_module_id=module.module_id,
            active_module_is_logical_payload=False,
            installed_module_ids=set(self.executor.installed_modules),
            robot_positions=measured_positions,
            active_robot_ids=set(executed_robot_ids),
            config=self.yard_config,
        )
        approach, pickup_goals = _route_to_clear_symmetric_formation(
            router=self.router,
            grid=approach_grid,
            starts=starts,
            module=module,
            center=Vec2.model_validate(module.staging_pose.position),
            yaw_degrees=module.staging_pose.rotation_rpy_degrees.z,
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
        self.executor.follow_routes(approach_routes)
        pickup_at_s = self.executor.simulation_time_s
        pickup_clearance = self._minimum_active_module_clearance(
            module,
            executed_robot_ids,
            installed=False,
        )
        self.executor.attach_logical_payload(
            module.module_id,
            executed_robot_ids,
            assigned_targets={
                robot_id: approach_routes[robot_id][-1]
                for robot_id in executed_robot_ids
            },
        )
        carry_telemetry_start = len(self.executor.telemetry)
        self.trace.append(
            {
                "timestamp_s": pickup_at_s,
                "event": "logical_payload_attached",
                "job_id": job_id,
                "module_id": module.module_id,
            }
        )
        measured_positions = self._measured_robot_positions()
        carry_grid = _phase5_route_grid(
            self.plan,
            active_module_id=module.module_id,
            active_module_is_logical_payload=True,
            installed_module_ids=set(self.executor.installed_modules),
            robot_positions=measured_positions,
            active_robot_ids=set(executed_robot_ids),
            config=self.yard_config,
        )
        carry_plan, install_goals = _route_to_clear_symmetric_formation(
            router=self.router,
            grid=carry_grid,
            starts={
                robot_id: measured_positions[robot_id]
                for robot_id in executed_robot_ids
            },
            module=module,
            center=Vec2(
                x=(
                    module.target_pose.position.x
                    - self.executor.logical_carrier_offsets[
                        module.module_id
                    ].x
                ),
                y=(
                    module.target_pose.position.y
                    - self.executor.logical_carrier_offsets[
                        module.module_id
                    ].y
                ),
            ),
            yaw_degrees=module.target_pose.rotation_rpy_degrees.z,
            config=self.yard_config,
        )
        assigned_carry_routes = {
            robot_id: _with_exact_endpoint(
                carry_plan.world_paths[robot_id],
                install_goals[robot_id],
            )
            for robot_id in executed_robot_ids
        }
        carry_route = _centroid_route(
            assigned_carry_routes,
            carrier_offset=self.executor.logical_carrier_offsets[
                module.module_id
            ],
        )
        self.trace.append(
            {
                "timestamp_s": pickup_at_s,
                "event": "carry_route_planned",
                "job_id": job_id,
                "module_id": module.module_id,
                "carry_routing_backend": carry_plan.backend,
                "carry_dynamic_obstacle_cells": len(
                    carry_grid.obstacle_cells
                ),
            }
        )
        self.executor.follow_routes(assigned_carry_routes)
        install_clearance = self._minimum_active_module_clearance(
            module,
            executed_robot_ids,
            installed=True,
        )
        self.executor.install_logical_payload(
            module.module_id,
            assigned_targets={
                robot_id: assigned_carry_routes[robot_id][-1]
                for robot_id in executed_robot_ids
            },
        )
        installed_at_s = self.executor.simulation_time_s
        self.trace.append(
            {
                "timestamp_s": installed_at_s,
                "event": "module_installed",
                "job_id": job_id,
                "module_id": module.module_id,
            }
        )
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
            measured_carry_routes=measured_carry_routes,
            measured_carry_samples=measured_carry_samples,
            started_at_s=started_at_s,
            pickup_at_s=pickup_at_s,
            installed_at_s=installed_at_s,
            measured_poses=measured,
            maximum_formation_error_m=max(formation_errors, default=0.0),
            maximum_assignment_error_m=max(assignment_errors, default=0.0),
            maximum_spacing_error_m=max(spacing_errors, default=0.0),
            maximum_install_error_m=max(install_errors, default=0.0),
            minimum_pickup_base_clearance_m=pickup_clearance,
            minimum_install_base_clearance_m=install_clearance,
            reassigned_after_unavailability=(
                self.recovery is not None
                and self.recovery.robot_id in planned_robot_ids
                and self.recovery.robot_id not in executed_robot_ids
            ),
        )

    def _measured_robot_positions(self) -> dict[str, Vec2]:
        return {
            robot.robot_id: Vec2.model_validate(
                self.executor.sample_telemetry(
                    robot.robot_id
                ).measured_pose.position
            )
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
            ],
            default=0.0,
        )
        expected_obstacles = len(self.plan.site_grid.obstacle_cells)
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
                }
            )
        return acceptance


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
                "measured_carry_routes": _jsonable(item.measured_carry_routes),
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
        limitations=[
            "Payload transport is a logical carrier synchronized to measured robot centroids.",
            "No arm, gripper, grasp contact, cooperative contact, or payload dynamics are claimed.",
            "Only YouBot base wheel commands and measured base telemetry are physical simulator evidence.",
        ],
        artifacts=artifacts,
    )
    _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    verified = verify_phase5_artifact_bundle(run_dir)
    if verified != manifest:
        raise ValueError("written Phase 5 manifest did not round-trip verification")
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
    if actual_paths != _REQUIRED_ARTIFACTS:
        missing = sorted(_REQUIRED_ARTIFACTS - actual_paths)
        extra = sorted(actual_paths - _REQUIRED_ARTIFACTS)
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
            measured_carry_routes=item.measured_carry_routes,
        )
        for item in replay
    ]
    if planned_jobs != expected_planned:
        raise ValueError("planned_jobs.json does not match the measured replay")

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
            for item in replay
        )
    ):
        raise ValueError("Phase 5 physical-yard evidence is incomplete")
    try:
        _validate_initial_physical_clearance(
            scenario.plan,
            physical_yard.configuration,
        )
        _preflight_phase5_sequential_routes(
            scenario.plan,
            physical_yard.configuration,
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
    _verify_replay_semantics(
        scenario=scenario,
        replay=replay,
        trace=trace,
        config=config,
        physical_yard=physical_yard,
    )

    diagnostics = result.metrics
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
        "collision_queries_cover_every_physics_step",
        "exclusive_wheel_command_ownership_proven",
        "site_obstacles_instantiated",
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
            or sample.timestamp_s > item.installed_at_s
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
            or _distance(item.carry_route[-1], target) > 1e-6
        ):
            raise ValueError(
                f"{item.job_id} measured carrier endpoints exceed tolerance"
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
        expected_trace_times = {
            "job_started": item.started_at_s,
            "logical_payload_attached": item.pickup_at_s,
            "module_installed": item.installed_at_s,
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
            "",
            "## Acceptance gates",
            "",
            acceptance,
            "",
            recovery,
            "## Payload model and limitations",
            "",
            (
                "Payload transport is explicitly logical. Modules are parented to a carrier "
                "dummy synchronized to measured two-base centroids. The bases remain "
                "outside active-module footprints; no physical grasp is implied."
            ),
            "",
            (
                "This evidence does not claim arm motion, gripper actuation, grasp contact, "
                "cooperative contact dynamics, or physical payload dynamics."
            ),
            "",
            f"Failure: `{result.error}`" if result.error else "",
            "",
        ]
    )
