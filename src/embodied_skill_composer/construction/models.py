from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Vec2(BaseModel):
    x: float
    y: float


class Vec3(Vec2):
    z: float = 0.0


class Pose3D(BaseModel):
    position: Vec3
    rotation_rpy_degrees: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))


class Dimensions3D(BaseModel):
    width: float = Field(gt=0)
    depth: float = Field(gt=0)
    height: float = Field(gt=0)


class ArchitecturalIntent(BaseModel):
    project_id: str
    floor_plan_path: str | None = None
    facade_image_path: str | None = None
    known_dimension_m: float | None = Field(default=None, gt=0)
    roof_style: Literal["gable", "hip", "flat"] = "gable"
    material_style: str = "white_plaster_and_timber"
    notes: str = ""


class WallSegment(BaseModel):
    wall_id: str
    start: Vec2
    end: Vec2
    thickness_m: float = Field(default=0.2, gt=0)
    height_m: float = Field(default=2.8, gt=0)


class Opening(BaseModel):
    opening_id: str
    wall_id: str
    kind: Literal["door", "window"]
    offset_m: float = Field(ge=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    sill_height_m: float = Field(default=0, ge=0)


class Room(BaseModel):
    room_id: str
    name: str
    polygon: list[Vec2] = Field(min_length=3)


class VectorFloorPlan(BaseModel):
    walls: list[WallSegment] = Field(min_length=4)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    approved: bool = False


class RoofSpec(BaseModel):
    style: Literal["gable", "hip", "flat"] = "gable"
    pitch_degrees: float = Field(default=28, ge=0, le=60)
    overhang_m: float = Field(default=0.35, ge=0)


class HouseDesign(BaseModel):
    design_id: str
    title: str
    footprint_width_m: float = Field(gt=0)
    footprint_depth_m: float = Field(gt=0)
    floor_plan: VectorFloorPlan
    roof: RoofSpec = Field(default_factory=RoofSpec)
    wall_material: str = "plaster_white"
    roof_material: str = "standing_seam_charcoal"
    level_count: int = Field(default=1, ge=1, le=1)

    @model_validator(mode="after")
    def validate_openings(self) -> "HouseDesign":
        wall_ids = {wall.wall_id for wall in self.floor_plan.walls}
        missing = sorted({item.wall_id for item in self.floor_plan.openings} - wall_ids)
        if missing:
            raise ValueError(f"openings reference unknown walls: {missing}")
        return self


class ModuleType(StrEnum):
    FOUNDATION = "foundation"
    WALL = "wall_panel"
    DOOR = "door_panel"
    WINDOW = "window_panel"
    INTERIOR = "interior_panel"
    ROOF = "roof_panel"


class BuildModule(BaseModel):
    module_id: str
    module_type: ModuleType
    mesh_node: str
    target_pose: Pose3D
    staging_pose: Pose3D
    dimensions: Dimensions3D
    mass_kg: float = Field(gt=0)
    grip_points: list[Vec3] = Field(min_length=1)
    required_team_size: int = Field(ge=1, le=2)
    install_duration_s: int = Field(gt=0)
    dependencies: list[str] = Field(default_factory=list)
    material: str


class RobotSpec(BaseModel):
    robot_id: str
    role: Literal["general", "heavy", "precision"] = "general"
    payload_capacity_kg: float = Field(gt=0)
    speed_mps: float = Field(gt=0)
    battery_capacity_wh: float = Field(gt=0)
    start_pose: Pose3D


class SiteGrid(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    resolution_m: float = Field(default=0.5, gt=0)
    origin: Vec2 = Field(default_factory=lambda: Vec2(x=-8, y=-6))
    obstacle_cells: list[tuple[int, int]] = Field(default_factory=list)


class BuildPlan(BaseModel):
    plan_id: str
    design: HouseDesign
    modules: list[BuildModule] = Field(min_length=1)
    robots: list[RobotSpec] = Field(min_length=2)
    site_grid: SiteGrid

    @model_validator(mode="after")
    def validate_plan(self) -> "BuildPlan":
        module_ids = [item.module_id for item in self.modules]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("module IDs must be unique")
        known = set(module_ids)
        for module in self.modules:
            missing = sorted(set(module.dependencies) - known)
            if missing:
                raise ValueError(f"{module.module_id} has unknown dependencies: {missing}")
            if module.module_id in module.dependencies:
                raise ValueError(f"{module.module_id} depends on itself")
            if module.required_team_size > len(self.robots):
                raise ValueError(f"{module.module_id} requires too many robots")
            eligible = sorted(
                robot.payload_capacity_kg for robot in self.robots
            )[-module.required_team_size :]
            if sum(eligible) < module.mass_kg:
                raise ValueError(f"no robot team can carry {module.module_id}")
        _topological_order(self.modules)
        return self


class ScheduledJob(BaseModel):
    module_id: str
    robot_ids: list[str]
    start_s: int
    pickup_s: int
    end_s: int
    travel_distance_m: float
    route: list[Vec2]
    critical: bool = False


class ConstructionSchedule(BaseModel):
    controller: Literal["sequential", "greedy", "optimized"]
    jobs: list[ScheduledJob]
    makespan_s: int
    total_travel_m: float
    total_energy_wh: float
    idle_robot_seconds: int
    solver_status: str
    critical_path: list[str]


class BrainEvent(BaseModel):
    timestamp_s: int
    event_type: Literal["assignment", "rejection", "recovery", "completion"]
    module_id: str | None = None
    robot_ids: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    reason: str
    predicted_remaining_s: int


class RobotTraceState(BaseModel):
    robot_id: str
    position: Vec3
    status: Literal["idle", "moving", "carrying", "installing", "unavailable"]
    module_id: str | None = None


class ModuleTraceState(BaseModel):
    module_id: str
    status: Literal["staged", "in_transit", "installed", "blocked"]
    position: Vec3


class ExecutionFrame(BaseModel):
    timestamp_s: int
    robots: list[RobotTraceState]
    modules: list[ModuleTraceState]
    completed_module_ids: list[str]


class ConstructionMetrics(BaseModel):
    controller: str
    structure_completion_rate: float
    makespan_s: int
    total_travel_m: float
    total_energy_wh: float
    idle_robot_seconds: int
    robot_utilization: dict[str, float]
    collision_count: int = 0
    wasted_work_s: int = 0
    recovery_cost_s: int = 0


class ExecutionTrace(BaseModel):
    plan_id: str
    schedule: ConstructionSchedule
    frames: list[ExecutionFrame]
    brain_events: list[BrainEvent]
    metrics: ConstructionMetrics


def _topological_order(modules: list[BuildModule]) -> list[str]:
    remaining = {item.module_id: set(item.dependencies) for item in modules}
    order: list[str] = []
    while remaining:
        ready = sorted(key for key, value in remaining.items() if not value)
        if not ready:
            raise ValueError(f"module dependency graph contains a cycle: {sorted(remaining)}")
        for key in ready:
            order.append(key)
            remaining.pop(key)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return order
