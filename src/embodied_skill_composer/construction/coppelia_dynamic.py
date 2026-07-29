from __future__ import annotations

import hashlib
import json
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
COPPELIA_SCENE_MAGIC = b"VREP"
MINIMUM_COPPELIA_SCENE_BYTES = 128
GENERATED_SCENE_ROOT_ALIAS = "ESCConstructionIntelligenceV1"
RobotCommandSource = Literal[
    "settling",
    "path_follower",
    "collision_stop",
    "formation_hold",
    "recovery",
]
YouBotScriptRole = Literal[
    "wheel_command_writer",
    "arm_gripper_writer",
    "passive_omniwheel_maintenance",
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
    disabled_settle_max_steps: int = Field(default=100, ge=3, le=2_000)
    disabled_settle_consecutive_samples: int = Field(default=3, ge=2, le=20)
    settled_linear_speed_mps: float = Field(default=0.02, ge=0, le=0.5)
    settled_angular_speed_rps: float = Field(default=0.05, ge=0, le=1.0)
    command_response_min_displacement_m: float = Field(default=0.002, gt=0, le=0.1)


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
        self.obstacle_handles: dict[str, int] = {}
        self.robot_collision_entities: dict[str, int] = {}
        self.collision_pairs: list[
            tuple[str, int, str, int, str, tuple[str, ...]]
        ] = []
        self.collision_pair_category_counts: dict[str, int] = {}
        self.payload_carriers: dict[str, int] = {}
        self.logical_attachments: dict[str, list[str]] = {}
        self.logical_carrier_offsets: dict[str, Vec3] = {}
        self.commands: list[RobotCommand] = []
        self.telemetry: list[RobotTelemetry] = []
        self.installed_modules: set[str] = set()
        self.disabled_robots: set[str] = set()
        self.disabled_command_cutoffs: dict[str, int] = {}
        self.formation_errors_m: list[float] = []
        self.formation_assignment_errors_m: list[float] = []
        self.formation_spacing_errors_m: list[float] = []
        self.install_errors_m: list[float] = []
        self.physics_steps = 0
        self.collision_stops = 0
        self.proximity_safety_stops = 0
        self.physical_collision_stops = 0
        self.collision_query_count = 0
        self.collision_query_rounds = 0
        self.physical_collision_events: list[dict[str, object]] = []
        self.permitted_logical_payload_overlaps = 0
        self._latest_collision_robots: set[str] = set()
        self.initial_robot_pose_writes = 0
        self.post_start_robot_pose_writes = 0
        self.bundled_motion_scripts_found = 0
        self.disabled_bundled_motion_scripts = 0
        self.verified_disabled_bundled_motion_scripts = 0
        self.retained_bundled_maintenance_scripts = 0
        self.verified_enabled_maintenance_scripts = 0
        self.disabled_wheel_command_scripts = 0
        self.disabled_arm_gripper_scripts = 0
        self.bundled_script_absence_proven = False
        self.script_inventory_classification_complete = False
        self.script_control_gate_passed = False
        self.script_control_by_robot: dict[str, bool] = {}
        self.wheel_command_writer_count_by_robot: dict[str, int] = {}
        self.retained_maintenance_count_by_robot: dict[str, int] = {}
        self.script_control_audit: list[dict[str, object]] = []
        self.disabled_settled: dict[str, bool] = {}
        self.disabled_settle_samples: dict[str, list[RobotTelemetry]] = {}
        self.disabled_settle_displacement_m: dict[str, float] = {}
        self.nonzero_wheel_command_count: dict[str, int] = {}
        self.command_response_anchor: dict[str, tuple[float, Vec3]] = {}
        self.command_response_displacement_m: dict[str, float] = {}
        self.command_response_observed_at_s: dict[str, float] = {}
        self.prior_generated_scene_root_count = 0
        self.prior_generated_scene_object_count = 0
        self.generated_scene_cleanup_verified = False
        self.runtime_events: list[dict[str, object]] = []
        self.started = False
        self.is_ready = False

    @property
    def simulation_time_s(self) -> float:
        return self.physics_steps / self.config.control_hz

    def connect(self) -> None:
        self.client = (self.client_factory or _connect_client)(self.config)
        self.sim = self.client.require("sim")
        self._ensure_stopped()
        self._remove_prior_generated_scenes()
        self._build_scene()
        self._prepare_collision_monitoring()
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
            self._step_physics()
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
        if not self.is_ready or not self.started:
            return
        for robot_id in self.robot_handles:
            if robot_id in self.disabled_robots:
                continue
            self.command_body_velocity(robot_id, 0.0, 0.0, 0.0, source="formation_hold")
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
        if robot_id in self.disabled_robots:
            raise DynamicCoppeliaError(
                f"{robot_id} is unavailable; wheel commands are forbidden after disable"
            )
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
        if any(abs(value) > 1e-9 for value in wheel_values):
            self.nonzero_wheel_command_count[robot_id] = (
                self.nonzero_wheel_command_count.get(robot_id, 0) + 1
            )
            if robot_id not in self.command_response_anchor:
                position = self.sim.getObjectPosition(self.robot_handles[robot_id])
                self.command_response_anchor[robot_id] = (
                    self.simulation_time_s,
                    Vec3(x=position[0], y=position[1], z=position[2]),
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

    def disable_robot(self, robot_id: str) -> RobotCommand:
        """Stop a robot once, then permanently reject commands for this run."""
        if robot_id not in self.robot_handles:
            raise DynamicCoppeliaError(f"unknown robot: {robot_id}")
        if robot_id in self.disabled_robots:
            raise DynamicCoppeliaError(f"{robot_id} is already unavailable")
        command = self.command_body_velocity(
            robot_id,
            0.0,
            0.0,
            0.0,
            source="recovery",
        )
        self.disabled_robots.add(robot_id)
        self.disabled_command_cutoffs[robot_id] = len(self.commands)
        self.disabled_settled[robot_id] = False
        return command

    def settle_disabled_robot(self, robot_id: str) -> list[RobotTelemetry]:
        """Step and measure an unavailable robot until it is demonstrably stationary."""
        if robot_id not in self.disabled_robots:
            raise DynamicCoppeliaError(f"{robot_id} is not unavailable")
        command_cutoff = self.disabled_command_cutoffs[robot_id]
        samples = [self.sample_telemetry(robot_id)]
        stable_samples = 0
        for _ in range(self.config.disabled_settle_max_steps):
            self._step_physics()
            sample = self.sample_telemetry(robot_id)
            samples.append(sample)
            if (
                sample.linear_velocity_mps <= self.config.settled_linear_speed_mps
                and sample.angular_velocity_rps
                <= self.config.settled_angular_speed_rps
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            if stable_samples >= self.config.disabled_settle_consecutive_samples:
                break
        if any(
            command.robot_id == robot_id
            for command in self.commands[command_cutoff:]
        ):
            raise DynamicCoppeliaError(
                f"{robot_id} received a wheel command while settling after disable"
            )
        if stable_samples < self.config.disabled_settle_consecutive_samples:
            raise DynamicCoppeliaError(
                f"{robot_id} did not settle after disable within "
                f"{self.config.disabled_settle_max_steps} physics steps"
            )
        first = samples[0].measured_pose.position
        last = samples[-1].measured_pose.position
        displacement = math.sqrt(
            (last.x - first.x) ** 2
            + (last.y - first.y) ** 2
            + (last.z - first.z) ** 2
        )
        self.disabled_settle_samples[robot_id] = samples
        self.disabled_settle_displacement_m[robot_id] = displacement
        self.disabled_settled[robot_id] = True
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "disabled_robot_settled",
                "robot_id": robot_id,
                "sample_count": len(samples),
                "displacement_m": displacement,
                "commands_after_disable": 0,
            }
        )
        return samples

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
            collision_stop=robot_id in self._latest_collision_robots,
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
        unavailable = sorted(set(routes) & self.disabled_robots)
        if unavailable:
            raise DynamicCoppeliaError(
                f"cannot route unavailable robots: {', '.join(unavailable)}"
            )
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
                    self.proximity_safety_stops += 1
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
            self._step_physics()
        for robot_id in routes:
            self.command_body_velocity(
                robot_id,
                0.0,
                0.0,
                0.0,
                source="formation_hold",
            )

    def attach_logical_payload(
        self,
        module_id: str,
        robot_ids: list[str],
        *,
        assigned_targets: Mapping[str, Vec2] | None = None,
    ) -> None:
        unavailable = sorted(set(robot_ids) & self.disabled_robots)
        if unavailable:
            raise DynamicCoppeliaError(
                f"cannot attach {module_id} to unavailable robots: {', '.join(unavailable)}"
            )
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        positions = [self.sample_telemetry(item).measured_pose.position for item in robot_ids]
        center_x = sum(item.x for item in positions) / len(positions)
        center_y = sum(item.y for item in positions) / len(positions)
        staging = module.staging_pose.position
        formation_error = math.hypot(center_x - staging.x, center_y - staging.y)
        self.formation_errors_m.append(formation_error)
        targets = assigned_targets or {
            robot_id: _formation_point(
                Vec2(x=staging.x, y=staging.y),
                index,
                len(robot_ids),
            )
            for index, robot_id in enumerate(robot_ids)
        }
        self._record_assigned_formation_errors(robot_ids, positions, targets)
        if formation_error > self.config.formation_tolerance_m:
            raise DynamicCoppeliaError(f"{module_id} pickup formation is outside tolerance")
        carrier = self.payload_carriers[module_id]
        self.sim.setObjectPosition(carrier, [staging.x, staging.y, staging.z])
        self.sim.setObjectParent(self.module_handles[module_id], carrier, True)
        center_z = sum(item.z for item in positions) / len(positions)
        self.logical_carrier_offsets[module_id] = Vec3(
            x=staging.x - center_x,
            y=staging.y - center_y,
            z=staging.z - center_z,
        )
        self.logical_attachments[module_id] = list(robot_ids)

    def install_logical_payload(
        self,
        module_id: str,
        *,
        assigned_targets: Mapping[str, Vec2] | None = None,
    ) -> None:
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        robot_ids = self.logical_attachments[module_id]
        positions = [self.sample_telemetry(item).measured_pose.position for item in robot_ids]
        target = module.target_pose.position
        carrier_position = self.sim.getObjectPosition(
            self.payload_carriers[module_id]
        )
        install_error = math.hypot(
            carrier_position[0] - target.x,
            carrier_position[1] - target.y,
        )
        self.install_errors_m.append(install_error)
        targets = assigned_targets or {
            robot_id: _formation_point(
                Vec2(x=target.x, y=target.y),
                index,
                len(robot_ids),
            )
            for index, robot_id in enumerate(robot_ids)
        }
        self._record_assigned_formation_errors(robot_ids, positions, targets)
        if install_error > self.config.install_tolerance_m:
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
        self.logical_carrier_offsets.pop(module_id)
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
            "proximity_safety_stops": self.proximity_safety_stops,
            "physical_collision_stops": self.physical_collision_stops,
            "physical_collision_event_count": len(self.physical_collision_events),
            "permitted_logical_payload_overlaps": (
                self.permitted_logical_payload_overlaps
            ),
            "collision_query_count": self.collision_query_count,
            "collision_query_rounds": self.collision_query_rounds,
            "expected_collision_queries_per_step": len(self.collision_pairs),
            "collision_pair_category_counts": dict(
                self.collision_pair_category_counts
            ),
            "collision_queries_cover_every_physics_step": (
                self.physics_steps > 0
                and self.collision_query_rounds == self.physics_steps
                and self.collision_query_count
                == self.physics_steps * len(self.collision_pairs)
            ),
            "initial_robot_pose_writes": self.initial_robot_pose_writes,
            "post_start_robot_pose_writes": self.post_start_robot_pose_writes,
            "bundled_motion_scripts_found": self.bundled_motion_scripts_found,
            "disabled_bundled_motion_scripts": self.disabled_bundled_motion_scripts,
            "verified_disabled_bundled_motion_scripts": (
                self.verified_disabled_bundled_motion_scripts
            ),
            "retained_bundled_maintenance_scripts": (
                self.retained_bundled_maintenance_scripts
            ),
            "verified_enabled_maintenance_scripts": (
                self.verified_enabled_maintenance_scripts
            ),
            "disabled_wheel_command_scripts": (
                self.disabled_wheel_command_scripts
            ),
            "disabled_arm_gripper_scripts": (
                self.disabled_arm_gripper_scripts
            ),
            "bundled_script_absence_proven": self.bundled_script_absence_proven,
            "script_inventory_classification_complete": (
                self.script_inventory_classification_complete
            ),
            "script_control_gate_passed": self.script_control_gate_passed,
            "script_control_by_robot": dict(self.script_control_by_robot),
            "wheel_command_writer_count_by_robot": dict(
                self.wheel_command_writer_count_by_robot
            ),
            "retained_maintenance_count_by_robot": dict(
                self.retained_maintenance_count_by_robot
            ),
            "script_control_audit": list(self.script_control_audit),
            "scene_robot_count": len(self.robot_handles),
            "prior_generated_scene_root_count": (
                self.prior_generated_scene_root_count
            ),
            "prior_generated_scene_object_count": (
                self.prior_generated_scene_object_count
            ),
            "generated_scene_cleanup_verified": (
                self.generated_scene_cleanup_verified
            ),
            "scene_module_count": len(self.module_handles),
            "scene_obstacle_count": len(self.obstacle_handles),
            "disabled_robots": sorted(self.disabled_robots),
            "disabled_command_cutoffs": dict(self.disabled_command_cutoffs),
            "disabled_settled": dict(self.disabled_settled),
            "disabled_settle_displacement_m": dict(
                self.disabled_settle_displacement_m
            ),
            "nonzero_wheel_command_count": dict(
                self.nonzero_wheel_command_count
            ),
            "command_response_displacement_m": dict(
                self.command_response_displacement_m
            ),
            "command_response_observed_at_s": dict(
                self.command_response_observed_at_s
            ),
            "command_response_min_displacement_m": (
                self.config.command_response_min_displacement_m
            ),
            "maximum_formation_error_m": max(self.formation_errors_m, default=0.0),
            "maximum_formation_assignment_error_m": max(
                self.formation_assignment_errors_m,
                default=0.0,
            ),
            "maximum_formation_spacing_error_m": max(
                self.formation_spacing_errors_m,
                default=0.0,
            ),
            "maximum_install_error_m": max(self.install_errors_m, default=0.0),
            "installed_modules": len(self.installed_modules),
            "executor_config": self.config.model_dump(mode="json"),
            "logical_payload_model": (
                "parented carrier dummy synchronized to the measured two-base "
                "centroid with an attested pickup offset"
            ),
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
                assigned_targets={
                    robot_id: route[-1]
                    for robot_id, route in assignment.approach_routes.items()
                    if route
                },
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
            self.install_logical_payload(
                assignment.module_id,
                assigned_targets={
                    robot_id: carry_routes[robot_id][-1]
                    for robot_id in assignment.robot_ids
                    if carry_routes[robot_id]
                },
            )

    def execute_assignments(self, assignments: list[dict[str, object]]) -> None:
        """Execute already-planned assignments through measured wheel control."""
        self._require_started()
        self._execute_assignments(assignments)

    def _remove_prior_generated_scenes(self) -> None:
        required_api = (
            "getObjectsInTree",
            "getObjectAlias",
            "removeObjects",
        )
        missing = [
            name for name in required_api if not callable(getattr(self.sim, name, None))
        ]
        handle_scene = getattr(self.sim, "handle_scene", None)
        handle_all = getattr(self.sim, "handle_all", None)
        if missing or handle_scene is None or handle_all is None:
            detail = ", ".join(
                [
                    *missing,
                    *(
                        ["handle_scene"]
                        if handle_scene is None
                        else []
                    ),
                    *(
                        ["handle_all"]
                        if handle_all is None
                        else []
                    ),
                ]
            )
            raise DynamicCoppeliaError(
                "project-owned Coppelia scene cleanup is unavailable: "
                f"{detail}"
            )
        try:
            scene_objects = [
                int(handle)
                for handle in self.sim.getObjectsInTree(
                    handle_scene,
                    handle_all,
                    0,
                )
            ]
            prior_roots = [
                handle
                for handle in scene_objects
                if str(self.sim.getObjectAlias(handle, -1))
                == GENERATED_SCENE_ROOT_ALIAS
            ]
            removal_handles: set[int] = set()
            for root_handle in prior_roots:
                removal_handles.update(
                    int(handle)
                    for handle in self.sim.getObjectsInTree(
                        root_handle,
                        handle_all,
                        0,
                    )
                )
            if removal_handles:
                self.sim.removeObjects(sorted(removal_handles), False)
            remaining_roots = [
                int(handle)
                for handle in self.sim.getObjectsInTree(
                    handle_scene,
                    handle_all,
                    0,
                )
                if str(self.sim.getObjectAlias(handle, -1))
                == GENERATED_SCENE_ROOT_ALIAS
            ]
        except Exception as exc:
            raise DynamicCoppeliaError(
                "project-owned Coppelia scene cleanup failed"
            ) from exc
        if remaining_roots:
            raise DynamicCoppeliaError(
                "project-owned Coppelia scene cleanup could not remove every "
                "prior generated root"
            )
        self.prior_generated_scene_root_count = len(prior_roots)
        self.prior_generated_scene_object_count = len(removal_handles)
        self.generated_scene_cleanup_verified = True
        self.runtime_events.append(
            {
                "timestamp_s": 0.0,
                "event": "prior_generated_scene_cleanup",
                "removed_root_count": len(prior_roots),
                "removed_object_count": len(removal_handles),
                "verified": True,
            }
        )

    def _build_scene(self) -> None:
        self.root_handle = self.sim.createDummy(0.01)
        self.sim.setObjectAlias(self.root_handle, GENERATED_SCENE_ROOT_ALIAS)
        grid = self.plan.site_grid
        grid_min_x = grid.origin.x - grid.resolution_m
        grid_max_x = (
            grid.origin.x
            + (grid.width - 1) * grid.resolution_m
            + grid.resolution_m
        )
        grid_min_y = grid.origin.y - grid.resolution_m
        grid_max_y = (
            grid.origin.y
            + (grid.height - 1) * grid.resolution_m
            + grid.resolution_m
        )
        floor = self._create_box(
            "construction_intelligence_floor",
            (
                (grid_min_x + grid_max_x) / 2,
                (grid_min_y + grid_max_y) / 2,
                -0.08,
            ),
            (
                grid_max_x - grid_min_x,
                grid_max_y - grid_min_y,
                0.12,
            ),
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
            self.sim.setObjectInt32Param(
                handle,
                self.sim.shapeintparam_respondable,
                1,
            )
            rotation = module.staging_pose.rotation_rpy_degrees
            self.sim.setObjectOrientation(
                handle,
                [
                    math.radians(rotation.x),
                    math.radians(rotation.y),
                    math.radians(rotation.z),
                ],
            )
            self.module_handles[module.module_id] = handle
            carrier = self.sim.createDummy(0.03)
            self.sim.setObjectAlias(carrier, f"logical_carrier_{module.module_id}")
            self.sim.setObjectParent(carrier, self.root_handle, True)
            self.payload_carriers[module.module_id] = carrier
        for robot in self.plan.robots:
            handle = self.sim.loadModel(str(Path(self.config.robot_model_path).resolve()))
            self.sim.setObjectAlias(handle, f"construction_intelligence_{robot.robot_id}")
            self._disable_bundled_motion_script(robot.robot_id, handle)
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
        for x, y in sorted(grid.obstacle_cells):
            alias = f"construction_intelligence_obstacle_{x}_{y}"
            handle = self._create_box(
                alias,
                (
                    grid.origin.x + x * grid.resolution_m,
                    grid.origin.y + y * grid.resolution_m,
                    0.3,
                ),
                (
                    grid.resolution_m * 0.8,
                    grid.resolution_m * 0.8,
                    0.6,
                ),
                (0.86, 0.27, 0.12),
            )
            self.sim.setObjectInt32Param(
                handle,
                self.sim.shapeintparam_respondable,
                1,
            )
            self.obstacle_handles[f"{x},{y}"] = handle
        if len(self.obstacle_handles) != len(grid.obstacle_cells):
            raise DynamicCoppeliaError(
                "not every planned site obstacle was instantiated in CoppeliaSim"
            )
        self.runtime_events.append(
            {
                "timestamp_s": 0.0,
                "event": "scene_inventory_built",
                "robot_ids": sorted(self.robot_handles),
                "module_ids": sorted(self.module_handles),
                "obstacle_ids": sorted(self.obstacle_handles),
                "robot_count": len(self.robot_handles),
                "module_count": len(self.module_handles),
                "obstacle_count": len(self.obstacle_handles),
            }
        )

    def _disable_bundled_motion_script(
        self,
        robot_id: str,
        robot_handle: int,
    ) -> None:
        descendants = list(self.sim.getObjectsInTree(robot_handle))
        script_handles: set[int] = set()
        script_type = getattr(self.sim, "object_script_type", None)
        if script_type is not None:
            try:
                script_handles.update(
                    int(handle)
                    for handle in self.sim.getObjectsInTree(
                        robot_handle,
                        script_type,
                        0,
                    )
                )
            except Exception as exc:
                raise DynamicCoppeliaError(
                    "could not exhaustively enumerate bundled YouBot scripts"
                ) from exc
        for handle in descendants:
            alias = str(self.sim.getObjectAlias(handle, 1)).strip("/").lower()
            if alias.endswith("/script") or alias.rsplit("/", 1)[-1] == "script":
                script_handles.add(int(handle))
        self.bundled_motion_scripts_found += len(script_handles)
        if not script_handles:
            raise DynamicCoppeliaError(
                f"no bundled YouBot script was found for {robot_id}; exclusive "
                "wheel-command ownership cannot be proven"
            )
        classified: list[tuple[int, str, str, YouBotScriptRole]] = []
        for handle in sorted(script_handles):
            alias = str(self.sim.getObjectAlias(handle, 1))
            source = self._read_script_source(handle)
            role = _classify_youbot_script(alias, source)
            classified.append((handle, alias, source, role))

        role_counts = {
            role: sum(item[3] == role for item in classified)
            for role in (
                "wheel_command_writer",
                "arm_gripper_writer",
                "passive_omniwheel_maintenance",
            )
        }
        if role_counts["wheel_command_writer"] != 1:
            raise DynamicCoppeliaError(
                f"{robot_id} must contain exactly one bundled wheel-command "
                "writer"
            )
        if role_counts["passive_omniwheel_maintenance"] < 1:
            raise DynamicCoppeliaError(
                f"{robot_id} has no allowlisted passive omni-wheel maintenance "
                "scripts"
            )

        script_records: list[dict[str, object]] = []
        for handle, alias, source, role in classified:
            should_disable = role != "passive_omniwheel_maintenance"
            disabled_before, disabled_after, control_api = (
                self._set_script_disabled(
                    handle,
                    disabled=should_disable,
                )
            )
            if should_disable:
                self.disabled_bundled_motion_scripts += 1
                self.verified_disabled_bundled_motion_scripts += 1
                if role == "wheel_command_writer":
                    self.disabled_wheel_command_scripts += 1
                else:
                    self.disabled_arm_gripper_scripts += 1
                action = "disabled"
            else:
                self.retained_bundled_maintenance_scripts += 1
                self.verified_enabled_maintenance_scripts += 1
                action = "retained_enabled"
            script_records.append(
                {
                    "handle": handle,
                    "alias": alias,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "role": role,
                    "action": action,
                    "disabled_before": disabled_before,
                    "disabled_after": disabled_after,
                    "enabled_before": int(not disabled_before),
                    "enabled_after": int(not disabled_after),
                    "control_api": control_api,
                }
            )
        self.wheel_command_writer_count_by_robot[robot_id] = role_counts[
            "wheel_command_writer"
        ]
        self.retained_maintenance_count_by_robot[robot_id] = role_counts[
            "passive_omniwheel_maintenance"
        ]
        self.script_control_by_robot[robot_id] = bool(script_records) and all(
            (
                record["disabled_after"] is True
                if record["action"] == "disabled"
                else record["disabled_after"] is False
            )
            for record in script_records
        )
        self.script_control_audit.append(
            {
                "robot_id": robot_id,
                "role_counts": role_counts,
                "scripts": script_records,
                "exclusive_control_verified": self.script_control_by_robot[
                    robot_id
                ],
            }
        )
        self.script_inventory_classification_complete = (
            self.bundled_motion_scripts_found
            == self.disabled_bundled_motion_scripts
            + self.retained_bundled_maintenance_scripts
        )
        self.script_control_gate_passed = (
            len(self.script_control_by_robot) == len(self.robot_handles) + 1
            and all(self.script_control_by_robot.values())
            and self.script_inventory_classification_complete
            and self.disabled_bundled_motion_scripts
            == self.verified_disabled_bundled_motion_scripts
            and self.retained_bundled_maintenance_scripts
            == self.verified_enabled_maintenance_scripts
            and self.disabled_wheel_command_scripts
            == len(self.script_control_by_robot)
            and all(
                count == 1
                for count in self.wheel_command_writer_count_by_robot.values()
            )
            and all(
                count >= 1
                for count in self.retained_maintenance_count_by_robot.values()
            )
        )

    def _read_script_source(self, handle: int) -> str:
        get_source_property = getattr(self.sim, "getStringProperty", None)
        if callable(get_source_property):
            source = get_source_property(handle, "code")
        else:
            get_legacy_source = getattr(self.sim, "getScriptStringParam", None)
            source_parameter = getattr(
                self.sim,
                "scriptstringparam_text",
                None,
            )
            if not callable(get_legacy_source) or source_parameter is None:
                raise DynamicCoppeliaError(
                    "bundled YouBot script source cannot be audited"
                )
            source = get_legacy_source(handle, source_parameter)
        if not isinstance(source, str) or not source.strip():
            raise DynamicCoppeliaError(
                f"bundled YouBot script {handle} returned indeterminate source"
            )
        return source

    def _set_script_disabled(
        self,
        handle: int,
        *,
        disabled: bool,
    ) -> tuple[bool, bool, str]:
        get_disabled_property = getattr(self.sim, "getBoolProperty", None)
        set_disabled_property = getattr(self.sim, "setBoolProperty", None)
        if callable(get_disabled_property) and callable(
            set_disabled_property
        ):
            disabled_before = get_disabled_property(
                handle,
                "scriptDisabled",
            )
            if not isinstance(disabled_before, bool):
                raise DynamicCoppeliaError(
                    f"bundled YouBot script {handle} returned an "
                    "indeterminate disabled state"
                )
            set_disabled_property(handle, "scriptDisabled", disabled)
            disabled_after = get_disabled_property(
                handle,
                "scriptDisabled",
            )
            if disabled_after is not disabled:
                raise DynamicCoppeliaError(
                    f"bundled YouBot script {handle} state readback did not "
                    "match the requested ownership policy"
                )
            return disabled_before, disabled_after, "scriptDisabled_property"

        get_int_parameter = getattr(self.sim, "getObjectInt32Param", None)
        set_int_parameter = getattr(self.sim, "setObjectInt32Param", None)
        if not callable(get_int_parameter) or not callable(set_int_parameter):
            raise DynamicCoppeliaError(
                "bundled YouBot script state cannot be verified"
            )
        enabled_before = get_int_parameter(
            handle,
            self.sim.scriptintparam_enabled,
        )
        if not isinstance(enabled_before, int) or isinstance(
            enabled_before,
            bool,
        ):
            raise DynamicCoppeliaError(
                f"bundled YouBot script {handle} returned an indeterminate "
                "enabled state"
            )
        set_int_parameter(
            handle,
            self.sim.scriptintparam_enabled,
            int(not disabled),
        )
        enabled_after = get_int_parameter(
            handle,
            self.sim.scriptintparam_enabled,
        )
        if (
            not isinstance(enabled_after, int)
            or isinstance(enabled_after, bool)
            or enabled_after != int(not disabled)
        ):
            raise DynamicCoppeliaError(
                f"bundled YouBot script {handle} state readback did not match "
                "the requested ownership policy"
            )
        return (
            not bool(enabled_before),
            not bool(enabled_after),
            "scriptintparam_enabled_legacy",
        )

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
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "initial_robot_pose_write",
                "robot_handle": handle,
                "position": list(position),
                "simulation_started": False,
            }
        )

    def _update_logical_payloads(self) -> None:
        for module_id, robot_ids in self.logical_attachments.items():
            positions = [self.sim.getObjectPosition(self.robot_handles[item]) for item in robot_ids]
            offset = self.logical_carrier_offsets[module_id]
            center = [
                sum(item[axis] for item in positions) / len(positions)
                + (offset.x, offset.y, offset.z)[axis]
                for axis in range(3)
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

    def _prepare_collision_monitoring(self) -> None:
        required_api = (
            "createCollection",
            "addItemToCollection",
            "checkCollision",
        )
        missing = [
            name for name in required_api if not callable(getattr(self.sim, name, None))
        ]
        handle_tree = getattr(self.sim, "handle_tree", None)
        if missing or handle_tree is None:
            detail = ", ".join([*missing, *(["handle_tree"] if handle_tree is None else [])])
            raise DynamicCoppeliaError(
                "Coppelia physical collision monitoring is unavailable: "
                f"{detail}"
            )
        for robot_id, robot_handle in sorted(self.robot_handles.items()):
            collection = int(self.sim.createCollection(0))
            self.sim.addItemToCollection(collection, handle_tree, robot_handle, 0)
            self.robot_collision_entities[robot_id] = collection

        pairs: list[tuple[str, int, str, int, str, tuple[str, ...]]] = []
        robot_ids = sorted(self.robot_collision_entities)
        for index, robot_id in enumerate(robot_ids):
            for other_id in robot_ids[index + 1 :]:
                pairs.append(
                    (
                        f"robot:{robot_id}",
                        self.robot_collision_entities[robot_id],
                        f"robot:{other_id}",
                        self.robot_collision_entities[other_id],
                        "robot_robot",
                        (robot_id, other_id),
                    )
                )
            for module_id, module_handle in sorted(self.module_handles.items()):
                pairs.append(
                    (
                        f"robot:{robot_id}",
                        self.robot_collision_entities[robot_id],
                        f"module:{module_id}",
                        module_handle,
                        "robot_module",
                        (robot_id,),
                    )
                )
            for obstacle_id, obstacle_handle in sorted(
                self.obstacle_handles.items()
            ):
                pairs.append(
                    (
                        f"robot:{robot_id}",
                        self.robot_collision_entities[robot_id],
                        f"obstacle:{obstacle_id}",
                        obstacle_handle,
                        "robot_obstacle",
                        (robot_id,),
                    )
                )
        if not pairs:
            raise DynamicCoppeliaError(
                "collision monitoring has no robot/entity pairs to query"
            )
        self.collision_pairs = pairs
        category_counts: dict[str, int] = {}
        for pair in pairs:
            category_counts[pair[4]] = category_counts.get(pair[4], 0) + 1
        self.collision_pair_category_counts = category_counts
        inventory = [
            {
                "entity_1": pair[0],
                "entity_2": pair[2],
                "category": pair[4],
                "robot_ids": list(pair[5]),
            }
            for pair in pairs
        ]
        self.runtime_events.append(
            {
                "timestamp_s": 0.0,
                "event": "collision_monitor_ready",
                "expected_queries_per_step": len(pairs),
                "pair_category_counts": dict(category_counts),
                "pair_inventory_sha256": _sha256_json(inventory),
                "robot_ids": robot_ids,
                "module_ids": sorted(self.module_handles),
                "obstacle_ids": sorted(self.obstacle_handles),
            }
        )

    def _step_physics(self) -> None:
        self.client.step()
        self.physics_steps += 1
        self._query_physical_collisions()
        self._update_command_responses()

    def _query_physical_collisions(self) -> None:
        detected: list[dict[str, object]] = []
        colliding_robots: set[str] = set()
        for (
            first_label,
            first_handle,
            second_label,
            second_handle,
            category,
            robot_ids,
        ) in self.collision_pairs:
            try:
                raw = self.sim.checkCollision(first_handle, second_handle)
            except Exception as exc:
                raise DynamicCoppeliaError(
                    "a required physical collision query failed for "
                    f"{first_label} versus {second_label}"
                ) from exc
            self.collision_query_count += 1
            result, object_handles = _parse_collision_result(raw)
            if result < 0:
                raise DynamicCoppeliaError(
                    "Coppelia returned an invalid collision-query result for "
                    f"{first_label} versus {second_label}"
                )
            if result == 0:
                continue
            event = {
                "timestamp_s": self.simulation_time_s,
                "physics_step": self.physics_steps,
                "entity_1": first_label,
                "entity_2": second_label,
                "category": category,
                "robot_ids": list(robot_ids),
                "colliding_object_handles": object_handles,
                "permitted_logical_payload_overlap": False,
            }
            detected.append(event)
            colliding_robots.update(robot_ids)
            self.physical_collision_events.append(event)
        self.collision_query_rounds += 1
        self._latest_collision_robots = colliding_robots
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "collision_query_round",
                "physics_step": self.physics_steps,
                "query_count": len(self.collision_pairs),
                "pair_category_counts": dict(
                    self.collision_pair_category_counts
                ),
                "detected_collisions": detected,
                "colliding_robot_ids": sorted(colliding_robots),
            }
        )
        for robot_id in sorted(colliding_robots):
            self.physical_collision_stops += 1
            self.collision_stops += 1
            if robot_id not in self.disabled_robots:
                self.command_body_velocity(
                    robot_id,
                    0.0,
                    0.0,
                    0.0,
                    source="collision_stop",
                )

    def _update_command_responses(self) -> None:
        for robot_id, (commanded_at_s, anchor) in sorted(
            self.command_response_anchor.items()
        ):
            position = self.sim.getObjectPosition(self.robot_handles[robot_id])
            displacement = math.sqrt(
                (position[0] - anchor.x) ** 2
                + (position[1] - anchor.y) ** 2
                + (position[2] - anchor.z) ** 2
            )
            previous = self.command_response_displacement_m.get(robot_id, 0.0)
            self.command_response_displacement_m[robot_id] = max(
                previous,
                displacement,
            )
            if (
                robot_id not in self.command_response_observed_at_s
                and displacement
                >= self.config.command_response_min_displacement_m
            ):
                self.command_response_observed_at_s[robot_id] = (
                    self.simulation_time_s
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "wheel_command_response_observed",
                        "robot_id": robot_id,
                        "commanded_at_s": commanded_at_s,
                        "displacement_m": displacement,
                    }
                )

    def _record_assigned_formation_errors(
        self,
        robot_ids: list[str],
        positions: list[Vec3],
        targets: Mapping[str, Vec2],
    ) -> None:
        if set(robot_ids) != set(targets):
            raise DynamicCoppeliaError(
                "formation targets do not match the assigned robot team"
            )
        position_by_robot = dict(zip(robot_ids, positions, strict=True))
        for robot_id in robot_ids:
            position = position_by_robot[robot_id]
            target = targets[robot_id]
            self.formation_assignment_errors_m.append(
                math.hypot(position.x - target.x, position.y - target.y)
            )
        for index, robot_id in enumerate(robot_ids):
            for other_id in robot_ids[index + 1 :]:
                first_position = position_by_robot[robot_id]
                second_position = position_by_robot[other_id]
                first_target = targets[robot_id]
                second_target = targets[other_id]
                measured_spacing = math.hypot(
                    first_position.x - second_position.x,
                    first_position.y - second_position.y,
                )
                planned_spacing = math.hypot(
                    first_target.x - second_target.x,
                    first_target.y - second_target.y,
                )
                self.formation_spacing_errors_m.append(
                    abs(measured_spacing - planned_spacing)
                )

    def save_scene(self, path: Path) -> Path:
        """Serialize the current Coppelia scene and validate the native buffer."""
        if self.started:
            raise DynamicCoppeliaError(
                "the reusable Coppelia scene must be saved after simulation stops"
            )
        save_scene = getattr(self.sim, "saveScene", None)
        if not callable(save_scene):
            raise DynamicCoppeliaError("Coppelia scene serialization is unavailable")
        try:
            raw = save_scene()
        except Exception as exc:
            raise DynamicCoppeliaError(
                "Coppelia failed to serialize the evidence scene"
            ) from exc
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise DynamicCoppeliaError(
                "Coppelia scene serialization did not return a binary scene buffer"
            )
        payload = bytes(raw)
        validate_coppelia_scene_buffer(payload)
        path = path.resolve()
        path.write_bytes(payload)
        if path.read_bytes() != payload:
            raise DynamicCoppeliaError(
                "the saved Coppelia scene does not match the serialized buffer"
            )
        return path

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


def validate_coppelia_scene_buffer(payload: bytes) -> None:
    if len(payload) < MINIMUM_COPPELIA_SCENE_BYTES:
        raise DynamicCoppeliaError(
            "serialized Coppelia scene is too small to be a reusable native scene"
        )
    if not payload.startswith(COPPELIA_SCENE_MAGIC):
        raise DynamicCoppeliaError(
            "serialized scene is missing the native Coppelia VREP signature"
        )


def _classify_youbot_script(alias: str, source: str) -> YouBotScriptRole:
    compact_source = "".join(source.casefold().split())
    compact_alias = alias.casefold()
    writes_target_velocity = "setjointtargetvelocity" in compact_source
    writes_target_position = "setjointtargetposition" in compact_source
    references_wheels = (
        "wheeljoints" in compact_source
        or "rollingjoint_" in compact_source
    )
    references_arm_or_gripper = any(
        marker in compact_source or marker in compact_alias
        for marker in ("armjoint", "gripperjoint", "gripper")
    )
    if writes_target_velocity and references_wheels:
        return "wheel_command_writer"
    if (
        writes_target_velocity or writes_target_position
    ) and references_arm_or_gripper:
        return "arm_gripper_writer"
    passive_omniwheel_maintenance = (
        "setobjectorientation" in compact_source
        and "slippingjoint_" in compact_source
        and not writes_target_velocity
        and not writes_target_position
    )
    if passive_omniwheel_maintenance:
        return "passive_omniwheel_maintenance"
    raise DynamicCoppeliaError(
        f"bundled YouBot script {alias!r} has an unrecognized control role"
    )


def _parse_collision_result(raw: object) -> tuple[int, list[int]]:
    if isinstance(raw, bool):
        return int(raw), []
    if isinstance(raw, int):
        return raw, []
    if isinstance(raw, (list, tuple)) and raw:
        result = int(raw[0])
        handles: list[int] = []
        if len(raw) > 1 and isinstance(raw[1], (list, tuple)):
            handles = [int(item) for item in raw[1]]
        return result, handles
    raise DynamicCoppeliaError(
        f"Coppelia returned a malformed collision-query result: {raw!r}"
    )


def _formation_point(center: Vec2, index: int, team_size: int) -> Vec2:
    offset = 0.0 if team_size == 1 else (-0.35 if index == 0 else 0.35)
    return Vec2(x=center.x, y=center.y + offset)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
