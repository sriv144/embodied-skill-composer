from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


Vector3 = tuple[float, float, float]


class TaskType(str, Enum):
    PICK_AND_PLACE = "pick_and_place"
    SORT_TO_ZONE = "sort_to_zone"
    STACK_BLOCKS = "stack_blocks"
    MULTI_OBJECT_COLLECTION = "multi_object_collection"


class Pose(BaseModel):
    position: Vector3
    yaw: float = 0.0


class ObjectState(BaseModel):
    name: str
    color_name: str
    position: Vector3
    size: Vector3
    held: bool = False
    collected: bool = False
    station_name: str | None = None


class ZoneState(BaseModel):
    name: str
    center: Vector3
    size: Vector3


class RobotState(BaseModel):
    end_effector_position: Vector3
    gripper_opening: float
    holding_object: str | None = None
    base_position: Vector3 | None = None
    navigation_node: str | None = None


class StationState(BaseModel):
    name: str
    position: Vector3
    kind: str = "pickup"
    capacity: int = 1


class WorldState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: RobotState
    objects: dict[str, ObjectState]
    zones: dict[str, ZoneState]
    stations: dict[str, StationState] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    name: str
    task_type: TaskType
    source_object: str
    target_zone: str | None = None
    target_object: str | None = None
    color_routing: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    target_objects: list[str] = Field(default_factory=list)
    drop_zone: str | None = None
    environment: str = "tabletop"
    use_perception: bool = False
    perception_mode: str = "oracle"
    policy_mode: str = "scripted"


class SkillStep(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 0


class SkillResult(BaseModel):
    success: bool
    message: str
    error_code: str | None = None


class ExecutionEvent(BaseModel):
    step_name: str
    attempt: int
    success: bool
    message: str
    params: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class ExecutionReport(BaseModel):
    task_name: str
    success: bool
    plan: list[SkillStep]
    events: list[ExecutionEvent]
    failure_step: str | None = None
    final_world: WorldState
    log_path: str | None = None


class SensorObservation(BaseModel):
    camera_name: str
    rgb: list[list[list[int]]]
    station_slots: dict[str, tuple[int, int]] = Field(default_factory=dict)
    resolution: tuple[int, int] = (0, 0)


class PerceptionReport(BaseModel):
    mode: str
    detected_objects: list[str] = Field(default_factory=list)
    missed_targets: list[str] = Field(default_factory=list)
    station_predictions: dict[str, str | None] = Field(default_factory=dict)
    confidence_by_object: dict[str, float] = Field(default_factory=dict)
