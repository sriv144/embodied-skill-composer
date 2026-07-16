from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.marl_env_v1 import (
    TemporalConstructionCoordinationEnv,
)
from embodied_skill_composer.construction.models import BuildPlan, Pose3D, Vec2, Vec3


WHEEL_NAMES = ("fl", "rl", "rr", "fr")
RobotCommandSource = Literal[
    "settling",
    "path_follower",
    "collision_stop",
    "formation_hold",
    "recovery",
]


class _DynamicAssignment(BaseModel):
    module_id: str
    robot_ids: list[str]
    approach_routes: dict[str, list[Vec2]]
    carry_route: list[Vec2]


class DynamicCoppeliaConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=23000, ge=1, le=65535)
    robot_model_path: str = (
        "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/mobile/"
        "KUKA YouBot.ttm"
    )
    robot_model_scale: float = Field(default=0.42, gt=0, le=2)
    robot_spawn_height_m: float = Field(default=0.12, gt=0, le=1)
    control_hz: int = Field(default=20, ge=5, le=100)
    settle_steps: int = Field(default=10, ge=0, le=200)
    maximum_wheel_speed: float = Field(default=5.0, gt=0)
    effective_wheel_radius_m: float = Field(default=0.021, gt=0, le=0.2)
    position_gain: float = Field(default=1.8, gt=0)
    waypoint_tolerance_m: float = Field(default=0.12, gt=0)
    formation_tolerance_m: float = Field(default=0.45, gt=0)
    install_tolerance_m: float = Field(default=0.3, gt=0)
    safety_distance_m: float = Field(default=0.34, gt=0)
    max_steps_per_waypoint: int = Field(default=500, ge=10)


class DynamicCoppeliaError(RuntimeError):
    pass


class DynamicCoppeliaExecutor:
    """Measured-pose YouBot wheel control with explicitly logical payloads."""

    controller_name = "dynamic_base_logical_payload"

    def __init__(
        self,
        plan: BuildPlan,
        *,
        config: DynamicCoppeliaConfig | None = None,
        client_factory: Callable[[DynamicCoppeliaConfig], Any] | None = None,
    ) -> None:
        self.plan = plan.model_copy(deep=True)
        self.config = config or DynamicCoppeliaConfig()
        self.client_factory = client_factory
        self.client: Any = None
        self.sim: Any = None
        self.root_handle: int | None = None
        self.robot_handles: dict[str, int] = {}
        self.wheel_handles: dict[str, dict[str, int]] = {}
        self.module_handles: dict[str, int] = {}
        self.payload_carriers: dict[str, int] = {}
        self.logical_attachments: dict[str, list[str]] = {}
        self.commands: list[RobotCommand] = []
        self.telemetry: list[RobotTelemetry] = []
        self.installed_modules: set[str] = set()
        self.physics_steps = 0
        self.collision_stops = 0
        self.initial_robot_pose_writes = 0
        self.post_start_robot_pose_writes = 0
        self.disabled_bundled_motion_scripts = 0
        self.started = False
        self.is_ready = False

    @property
    def simulation_time_s(self) -> float:
        return self.physics_steps / self.config.control_hz

    def connect(self) -> None:
        self.client = (self.client_factory or _connect_client)(self.config)
        self.sim = self.client.require("sim")
        self._ensure_stopped()
        self._build_scene()
        self.is_ready = True

    def start(self) -> None:
        self._require_ready()
        if self.started:
            return
        if hasattr(self.sim, "setFloatParam") and hasattr(
            self.sim,
            "floatparam_simulation_time_step",
        ):
            self.sim.setFloatParam(
                self.sim.floatparam_simulation_time_step,
                1.0 / self.config.control_hz,
            )
        self.client.setStepping(True)
        self.sim.startSimulation()
        self.started = True
        for robot_id in self.robot_handles:
            self.command_body_velocity(robot_id, 0.0, 0.0, 0.0, source="settling")
        for _ in range(self.config.settle_steps):
            self.client.step()
            self.physics_steps += 1
        invalid_heights = {
            robot_id: self.sample_telemetry(robot_id).measured_pose.position.z
            for robot_id in self.robot_handles
        }
        invalid_heights = {
            robot_id: height
            for robot_id, height in invalid_heights.items()
            if height < -0.05 or height > 0.75
        }
        if invalid_heights:
            self.stop()
            raise DynamicCoppeliaError(
                f"robots did not settle onto the construction floor: {invalid_heights}"
            )

    def stop(self) -> None:
        if not self.is_ready:
            return
        for robot_id in self.robot_handles:
            self.command_body_velocity(robot_id, 0.0, 0.0, 0.0, source="collision_stop")
        if self.started:
            self.sim.stopSimulation()
        self.started = False

    def command_body_velocity(
        self,
        robot_id: str,
        forward_velocity: float,
        lateral_velocity: float,
        angular_velocity: float,
        *,
        source: RobotCommandSource = "path_follower",
        target: Vec2 | None = None,
    ) -> RobotCommand:
        self._require_started()
        wheel_values = youbot_wheel_targets(
            forward_velocity / self.config.effective_wheel_radius_m,
            lateral_velocity / self.config.effective_wheel_radius_m,
            angular_velocity,
            maximum=self.config.maximum_wheel_speed,
        )
        for wheel_name, velocity in zip(WHEEL_NAMES, wheel_values, strict=True):
            self.sim.setJointTargetVelocity(
                self.wheel_handles[robot_id][wheel_name],
                velocity,
            )
        command = RobotCommand(
            timestamp_s=self.simulation_time_s,
            robot_id=robot_id,
            linear_velocity_mps=math.hypot(forward_velocity, lateral_velocity),
            angular_velocity_rps=angular_velocity,
            wheel_target_velocity_rad_s=wheel_values,
            target_position=target,
            source=source,
        )
        self.commands.append(command)
        return command

    def sample_telemetry(self, robot_id: str) -> RobotTelemetry:
        position = self.sim.getObjectPosition(self.robot_handles[robot_id])
        orientation = self.sim.getObjectOrientation(self.robot_handles[robot_id])
        linear_speed = 0.0
        angular_speed = 0.0
        if hasattr(self.sim, "getObjectVelocity"):
            linear_velocity, angular_velocity = self.sim.getObjectVelocity(
                self.robot_handles[robot_id]
            )
            linear_speed = math.sqrt(sum(float(value) ** 2 for value in linear_velocity))
            angular_speed = math.sqrt(sum(float(value) ** 2 for value in angular_velocity))
        robot = next(item for item in self.plan.robots if item.robot_id == robot_id)
        telemetry = RobotTelemetry(
            timestamp_s=self.simulation_time_s,
            robot_id=robot_id,
            measured_pose=Pose3D(
                position=Vec3(x=position[0], y=position[1], z=position[2]),
                rotation_rpy_degrees=Vec3(
                    x=math.degrees(orientation[0]),
                    y=math.degrees(orientation[1]),
                    z=math.degrees(orientation[2]),
                ),
            ),
            linear_velocity_mps=linear_speed,
            angular_velocity_rps=angular_speed,
            battery_remaining_wh=robot.battery_capacity_wh,
            attached_module_id=next(
                (
                    module_id
                    for module_id, robot_ids in self.logical_attachments.items()
                    if robot_id in robot_ids
                ),
                None,
            ),
        )
        self.telemetry.append(telemetry)
        return telemetry

    def execute_online(
        self,
        env: TemporalConstructionCoordinationEnv,
        action_provider,
        *,
        max_decisions: int | None = None,
    ) -> dict[str, object]:
        self.start()
        observations, _ = env.reset(seed=env.scenario.seed if env.scenario else 0)
        try:
            while env.agents and (
                max_decisions is None or env.decision_count < max_decisions
            ):
                provided = action_provider(env, observations)
                if isinstance(provided, tuple):
                    actions, diagnostics = provided
                else:
                    actions, diagnostics = provided, None
                observations, _, _, _, infos = env.step(actions)
                if diagnostics:
                    env.annotate_latest_decisions("online_policy", diagnostics)
                assignments = next(iter(infos.values()))["assignments"]
                if assignments:
                    self._execute_assignments(assignments)
        finally:
            self.stop()
        return self.diagnostics(logical_metrics=env.metrics())

    def follow_routes(self, routes: dict[str, list[Vec2]]) -> None:
        waypoint_indices = {robot_id: 0 for robot_id in routes}
        steps_at_waypoint = {robot_id: 0 for robot_id in routes}
        while any(waypoint_indices[robot_id] < len(path) for robot_id, path in routes.items()):
            measured = {
                robot_id: self.sample_telemetry(robot_id).measured_pose
                for robot_id in routes
            }
            for robot_id, path in routes.items():
                index = waypoint_indices[robot_id]
                if index >= len(path):
                    self.command_body_velocity(
                        robot_id,
                        0.0,
                        0.0,
                        0.0,
                        source="formation_hold",
                    )
                    continue
                target = path[index]
                pose = measured[robot_id]
                dx = target.x - pose.position.x
                dy = target.y - pose.position.y
                if math.hypot(dx, dy) <= self.config.waypoint_tolerance_m:
                    waypoint_indices[robot_id] += 1
                    steps_at_waypoint[robot_id] = 0
                    self.command_body_velocity(
                        robot_id,
                        0.0,
                        0.0,
                        0.0,
                        source="formation_hold",
                        target=target,
                    )
                    continue
                if self._unsafe_proximity(robot_id, measured):
                    self.collision_stops += 1
                    self.command_body_velocity(
                        robot_id,
                        0.0,
                        0.0,
                        0.0,
                        source="collision_stop",
                        target=target,
                    )
                    steps_at_waypoint[robot_id] += 1
                    if steps_at_waypoint[robot_id] > self.config.max_steps_per_waypoint:
                        raise DynamicCoppeliaError(
                            f"{robot_id} remained inside the safety stop zone"
                        )
                    continue
                yaw = math.radians(pose.rotation_rpy_degrees.z)
                body_forward, body_lateral = world_error_to_youbot_body(dx, dy, yaw)
                scale = self.config.position_gain
                self.command_body_velocity(
                    robot_id,
                    _clamp(body_forward * scale, -1.0, 1.0),
                    _clamp(body_lateral * scale, -1.0, 1.0),
                    0.0,
                    target=target,
                )
                steps_at_waypoint[robot_id] += 1
                if steps_at_waypoint[robot_id] > self.config.max_steps_per_waypoint:
                    raise DynamicCoppeliaError(
                        f"{robot_id} failed to reach waypoint ({target.x:.2f}, {target.y:.2f})"
                    )
            self._update_logical_payloads()
            self.client.step()
            self.physics_steps += 1
        for robot_id in routes:
            self.command_body_velocity(
                robot_id,
                0.0,
                0.0,
                0.0,
                source="formation_hold",
            )

    def attach_logical_payload(self, module_id: str, robot_ids: list[str]) -> None:
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        positions = [self.sample_telemetry(item).measured_pose.position for item in robot_ids]
        center_x = sum(item.x for item in positions) / len(positions)
        center_y = sum(item.y for item in positions) / len(positions)
        staging = module.staging_pose.position
        if math.hypot(center_x - staging.x, center_y - staging.y) > self.config.formation_tolerance_m:
            raise DynamicCoppeliaError(f"{module_id} pickup formation is outside tolerance")
        carrier = self.payload_carriers[module_id]
        self.sim.setObjectPosition(carrier, [center_x, center_y, staging.z])
        self.sim.setObjectParent(self.module_handles[module_id], carrier, True)
        self.logical_attachments[module_id] = list(robot_ids)

    def install_logical_payload(self, module_id: str) -> None:
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        robot_ids = self.logical_attachments[module_id]
        positions = [self.sample_telemetry(item).measured_pose.position for item in robot_ids]
        center_x = sum(item.x for item in positions) / len(positions)
        center_y = sum(item.y for item in positions) / len(positions)
        target = module.target_pose.position
        if math.hypot(center_x - target.x, center_y - target.y) > self.config.install_tolerance_m:
            raise DynamicCoppeliaError(f"{module_id} install formation is outside tolerance")
        handle = self.module_handles[module_id]
        self.sim.setObjectParent(handle, self.root_handle, True)
        self.sim.setObjectPosition(handle, [target.x, target.y, target.z])
        rotation = module.target_pose.rotation_rpy_degrees
        self.sim.setObjectOrientation(
            handle,
            [math.radians(rotation.x), math.radians(rotation.y), math.radians(rotation.z)],
        )
        self.logical_attachments.pop(module_id)
        self.installed_modules.add(module_id)

    def diagnostics(
        self,
        *,
        logical_metrics: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "backend": "coppelia_sim",
            "controller": self.controller_name,
            "connected": self.is_ready,
            "control_hz": self.config.control_hz,
            "physics_steps": self.physics_steps,
            "measured_duration_s": self.simulation_time_s,
            "wheel_command_count": len(self.commands),
            "telemetry_sample_count": len(self.telemetry),
            "collision_stops": self.collision_stops,
            "initial_robot_pose_writes": self.initial_robot_pose_writes,
            "post_start_robot_pose_writes": self.post_start_robot_pose_writes,
            "disabled_bundled_motion_scripts": self.disabled_bundled_motion_scripts,
            "installed_modules": len(self.installed_modules),
            "logical_payload_model": "parented carrier dummy synchronized to measured robot centroid",
            "logical_metrics": logical_metrics,
            "backend_limitations": [
                "YouBot bases are wheel-driven from measured poses at deterministic 20 Hz stepping.",
                "Payload attachment is logical; arms, grippers, and cooperative contact dynamics are not modeled.",
                "Module placement is snapped only after measured carrier formation enters target tolerance.",
            ],
        }

    def _execute_assignments(self, assignments: list[dict[str, object]]) -> None:
        parsed = [_DynamicAssignment.model_validate(assignment) for assignment in assignments]
        approach_routes = {
            robot_id: route
            for assignment in parsed
            for robot_id, route in assignment.approach_routes.items()
        }
        self.follow_routes(approach_routes)
        for assignment in parsed:
            self.attach_logical_payload(
                assignment.module_id,
                assignment.robot_ids,
            )
        carry_routes: dict[str, list[Vec2]] = {}
        for assignment in parsed:
            base_route = assignment.carry_route
            robot_ids = assignment.robot_ids
            for index, robot_id in enumerate(robot_ids):
                offset = 0.0 if len(robot_ids) == 1 else (-0.35 if index == 0 else 0.35)
                carry_routes[robot_id] = [Vec2(x=point.x, y=point.y + offset) for point in base_route]
        self.follow_routes(carry_routes)
        for assignment in parsed:
            self.install_logical_payload(assignment.module_id)

    def _build_scene(self) -> None:
        self.root_handle = self.sim.createDummy(0.01)
        self.sim.setObjectAlias(self.root_handle, "ESCConstructionIntelligenceV1")
        floor = self._create_box(
            "construction_intelligence_floor",
            (0.0, 0.0, -0.08),
            (22.0, 18.0, 0.12),
            (0.12, 0.15, 0.14),
        )
        self.sim.setObjectInt32Param(floor, self.sim.shapeintparam_static, 1)
        self.sim.setObjectInt32Param(floor, self.sim.shapeintparam_respondable, 1)
        for module in self.plan.modules:
            position = module.staging_pose.position
            handle = self._create_box(
                f"construction_intelligence_module_{module.module_id}",
                (position.x, position.y, max(position.z, 0.04)),
                (
                    module.dimensions.width,
                    module.dimensions.depth,
                    module.dimensions.height,
                ),
                _module_color(module.module_type.value),
            )
            self.module_handles[module.module_id] = handle
            carrier = self.sim.createDummy(0.03)
            self.sim.setObjectAlias(carrier, f"logical_carrier_{module.module_id}")
            self.payload_carriers[module.module_id] = carrier
        for robot in self.plan.robots:
            handle = self.sim.loadModel(str(Path(self.config.robot_model_path).resolve()))
            self.sim.setObjectAlias(handle, f"construction_intelligence_{robot.robot_id}")
            self._disable_bundled_motion_script(handle)
            descendants = self.sim.getObjectsInTree(handle)
            self.sim.scaleObjects(descendants, self.config.robot_model_scale, True)
            self.sim.setObjectParent(handle, self.root_handle, True)
            self._set_initial_robot_position(
                handle,
                [
                    robot.start_pose.position.x,
                    robot.start_pose.position.y,
                    self.config.robot_spawn_height_m,
                ],
            )
            self.robot_handles[robot.robot_id] = handle
            self.wheel_handles[robot.robot_id] = self._resolve_wheels(handle)

    def _disable_bundled_motion_script(self, robot_handle: int) -> None:
        for handle in self.sim.getObjectsInTree(robot_handle):
            alias = str(self.sim.getObjectAlias(handle, 1)).strip("/").lower()
            if alias.count("/") != 1 or not alias.endswith("/script"):
                continue
            self.sim.setObjectInt32Param(handle, self.sim.scriptintparam_enabled, 0)
            self.disabled_bundled_motion_scripts += 1

    def _create_box(
        self,
        alias: str,
        position: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> int:
        handle = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cuboid,
            list(dimensions),
            2,
        )
        self.sim.setObjectAlias(handle, alias)
        self.sim.setObjectParent(handle, self.root_handle, True)
        self.sim.setObjectPosition(handle, list(position))
        self.sim.setShapeColor(
            handle,
            None,
            self.sim.colorcomponent_ambient_diffuse,
            list(color),
        )
        self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_static, 1)
        return int(handle)

    def _resolve_wheels(self, robot_handle: int) -> dict[str, int]:
        descendants = self.sim.getObjectsInTree(robot_handle)
        aliases = {
            handle: str(self.sim.getObjectAlias(handle, 1)).lower() for handle in descendants
        }
        resolved = {}
        for wheel_name in WHEEL_NAMES:
            matches = [
                handle
                for handle, alias in aliases.items()
                if alias.rsplit("/", 1)[-1]
                in {f"rollingjoint_{wheel_name}", f"wheel_{wheel_name}"}
            ]
            if len(matches) != 1:
                raise DynamicCoppeliaError(
                    f"could not uniquely resolve YouBot wheel {wheel_name}; "
                    f"discovered aliases: {sorted(aliases.values())}"
                )
            resolved[wheel_name] = matches[0]
        return resolved

    def _set_initial_robot_position(self, handle: int, position: list[float]) -> None:
        if self.started:
            self.post_start_robot_pose_writes += 1
            raise DynamicCoppeliaError("direct robot pose writes are forbidden after simulation start")
        self.sim.setObjectPosition(handle, position)
        self.initial_robot_pose_writes += 1

    def _update_logical_payloads(self) -> None:
        for module_id, robot_ids in self.logical_attachments.items():
            positions = [self.sim.getObjectPosition(self.robot_handles[item]) for item in robot_ids]
            center = [
                sum(item[axis] for item in positions) / len(positions) for axis in range(3)
            ]
            self.sim.setObjectPosition(self.payload_carriers[module_id], center)

    def _unsafe_proximity(self, robot_id: str, measured: dict[str, Pose3D]) -> bool:
        position = measured[robot_id].position
        for other_id, pose in measured.items():
            if other_id == robot_id:
                continue
            distance = math.hypot(position.x - pose.position.x, position.y - pose.position.y)
            if distance < self.config.safety_distance_m and robot_id > other_id:
                return True
        return False

    def _ensure_stopped(self) -> None:
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise DynamicCoppeliaError("dynamic Coppelia executor is not connected")

    def _require_started(self) -> None:
        self._require_ready()
        if not self.started:
            raise DynamicCoppeliaError("dynamic Coppelia simulation is not started")


def youbot_wheel_targets(
    forward_velocity: float,
    lateral_velocity: float,
    angular_velocity: float,
    *,
    maximum: float,
) -> tuple[float, float, float, float]:
    raw = (
        -forward_velocity - lateral_velocity - angular_velocity,
        -forward_velocity + lateral_velocity - angular_velocity,
        -forward_velocity - lateral_velocity + angular_velocity,
        -forward_velocity + lateral_velocity + angular_velocity,
    )
    largest = max(max(abs(value) for value in raw), maximum)
    scale = maximum / largest
    return (
        float(raw[0] * scale),
        float(raw[1] * scale),
        float(raw[2] * scale),
        float(raw[3] * scale),
    )


def world_error_to_youbot_body(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    """Map world error into the bundled model's reversed-X base frame."""
    forward = -(math.cos(yaw) * dx + math.sin(yaw) * dy)
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return forward, lateral


def _module_color(module_type: str) -> tuple[float, float, float]:
    return {
        "foundation": (0.28, 0.31, 0.31),
        "roof_panel": (0.10, 0.14, 0.15),
        "door_panel": (0.45, 0.23, 0.10),
        "window_panel": (0.08, 0.45, 0.52),
    }.get(module_type, (0.84, 0.85, 0.81))


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _connect_client(config: DynamicCoppeliaConfig):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    return RemoteAPIClient(host=config.host, port=config.port)
