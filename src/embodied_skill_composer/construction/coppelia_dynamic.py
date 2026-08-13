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
# Match Coppelia's default ground plane and module-bottom convention so the
# generated site cannot create an unmonitored step at the default-floor seam.
CONSTRUCTION_FLOOR_TOP_Z_M = 0.0
CONSTRUCTION_FLOOR_THICKNESS_M = 0.12
_REMOTE_STEP_EXECUTED_HANDSHAKE_ERROR = "No such function: _*executed*_"
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
        "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/mobile/KUKA YouBot.ttm"
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
    planned_robot_footprint_radius_m: float = Field(default=0.12, gt=0, le=1.0)
    max_steps_per_waypoint: int = Field(default=500, ge=10)
    disabled_settle_max_steps: int = Field(default=100, ge=3, le=2_000)
    disabled_settle_consecutive_samples: int = Field(default=3, ge=2, le=20)
    settled_linear_speed_mps: float = Field(default=0.02, ge=0, le=0.5)
    settled_angular_speed_rps: float = Field(default=0.05, ge=0, le=1.0)
    logical_transition_settle_max_steps: int = Field(
        default=100,
        ge=1,
        le=2_000,
    )
    logical_transition_settle_consecutive_samples: int = Field(
        default=3,
        ge=2,
        le=20,
    )
    command_response_min_displacement_m: float = Field(default=0.002, gt=0, le=0.1)
    maximum_remote_step_handshake_reconciliations: int = Field(
        default=8,
        ge=0,
        le=100,
    )
    maximum_remote_step_handshake_retries: int = Field(
        default=8,
        ge=0,
        le=100,
    )


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
        self.robot_world_collision_entity: int | None = None
        self.world_collision_entity: int | None = None
        self.robot_base_shape_handles: dict[str, set[int]] = {}
        self.robot_excluded_shape_handles: dict[str, set[int]] = {}
        self.robot_base_physics_audit: list[dict[str, object]] = []
        self.robot_collision_self_tests: list[dict[str, object]] = []
        self.robot_base_contact_gate_passed = False
        self.collision_pairs: list[tuple[str, int, str, int, str, tuple[str, ...]]] = []
        self.collision_pair_category_counts: dict[str, int] = {}
        self.payload_carriers: dict[str, int] = {}
        self.logical_attachments: dict[str, list[str]] = {}
        self.logical_carrier_offsets: dict[str, Vec3] = {}
        self.logical_transport_heights_m: dict[str, float] = {}
        self.logical_payload_lift_records: list[dict[str, object]] = []
        self.logical_installation_snap_records: list[dict[str, object]] = []
        self.commands: list[RobotCommand] = []
        self.telemetry: list[RobotTelemetry] = []
        self._route_telemetry_cache_step: int | None = None
        self._route_telemetry_cache: dict[str, RobotTelemetry] = {}
        self.installed_modules: set[str] = set()
        self.disabled_robots: set[str] = set()
        self.disabled_command_cutoffs: dict[str, int] = {}
        self.formation_errors_m: list[float] = []
        self.formation_assignment_errors_m: list[float] = []
        self.formation_spacing_errors_m: list[float] = []
        self.synchronized_formation_errors_m: list[float] = []
        self.synchronized_route_separations_m: list[float] = []
        self.install_errors_m: list[float] = []
        self.physics_steps = 0
        self.remote_step_handshake_reconciliations = 0
        self.remote_step_handshake_retries = 0
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
                and sample.angular_velocity_rps <= self.config.settled_angular_speed_rps
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            if stable_samples >= self.config.disabled_settle_consecutive_samples:
                break
        if any(command.robot_id == robot_id for command in self.commands[command_cutoff:]):
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
            (last.x - first.x) ** 2 + (last.y - first.y) ** 2 + (last.z - first.z) ** 2
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

    def _sample_route_telemetry_tick(self) -> dict[str, RobotTelemetry]:
        """Return one time-aligned sample per scene robot for this physics tick."""
        robot_ids = sorted(self.robot_handles)
        if getattr(self, "_route_telemetry_cache_step", None) == self.physics_steps and set(
            getattr(self, "_route_telemetry_cache", {})
        ) == set(robot_ids):
            return self._route_telemetry_cache

        timestamp_s = self.simulation_time_s
        samples: dict[str, RobotTelemetry] = {}
        expected = set(robot_ids)
        for sample in reversed(getattr(self, "telemetry", [])):
            if sample.timestamp_s < timestamp_s:
                break
            if (
                sample.timestamp_s == timestamp_s
                and sample.robot_id in expected
                and sample.robot_id not in samples
            ):
                samples[sample.robot_id] = sample
        for robot_id in robot_ids:
            if robot_id not in samples:
                samples[robot_id] = self.sample_telemetry(robot_id)
        self._route_telemetry_cache_step = self.physics_steps
        self._route_telemetry_cache = samples
        return samples

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
            while env.agents and (max_decisions is None or env.decision_count < max_decisions):
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

    def follow_routes(
        self,
        routes: dict[str, list[Vec2]],
        *,
        waypoint_tolerance_m: float | None = None,
    ) -> None:
        unavailable = sorted(set(routes) & self.disabled_robots)
        if unavailable:
            raise DynamicCoppeliaError(f"cannot route unavailable robots: {', '.join(unavailable)}")
        tolerance = (
            self.config.waypoint_tolerance_m
            if waypoint_tolerance_m is None
            else waypoint_tolerance_m
        )
        if tolerance <= 0:
            raise DynamicCoppeliaError("route waypoint tolerance must be positive")
        if len(routes) > 1:
            self._follow_shared_route_time(
                routes,
                waypoint_tolerance_m=tolerance,
                enforce_relative_formation=False,
            )
            return
        if len(routes) == 1:
            robot_id, route = next(iter(routes.items()))
            if route:
                if robot_id not in self.robot_handles:
                    raise DynamicCoppeliaError(f"cannot route unknown robot: {robot_id}")
                initial_samples = self._sample_route_telemetry_tick()
                initial_points = {
                    item: Vec2(
                        x=sample.measured_pose.position.x,
                        y=sample.measured_pose.position.y,
                    )
                    for item, sample in initial_samples.items()
                }
                minimum_separation, limiting_separation = self._validate_route_time_separation(
                    routes,
                    horizon=len(route),
                    measured_route_starts={robot_id: initial_points[robot_id]},
                    stationary_robot_points={
                        item: point for item, point in initial_points.items() if item != robot_id
                    },
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "route_preflight_completed",
                        "robot_ids": [robot_id],
                        "minimum_planned_separation_m": minimum_separation,
                        "limiting_planned_separation": limiting_separation,
                        "measured_start_to_first_target_included": True,
                    }
                )
        waypoint_indices = {robot_id: 0 for robot_id in routes}
        steps_at_waypoint = {robot_id: 0 for robot_id in routes}
        while any(waypoint_indices[robot_id] < len(path) for robot_id, path in routes.items()):
            measured_all = {
                robot_id: sample.measured_pose
                for robot_id, sample in self._sample_route_telemetry_tick().items()
            }
            measured = {robot_id: measured_all[robot_id] for robot_id in routes}
            measured_enabled = {
                robot_id: pose
                for robot_id, pose in measured_all.items()
                if robot_id not in self.disabled_robots
            }
            active_targets = {
                robot_id: path[waypoint_indices[robot_id]]
                for robot_id, path in routes.items()
                if waypoint_indices[robot_id] < len(path)
            }
            separation = _minimum_pose_separation(measured_enabled)
            if separation is not None:
                distance, left_id, right_id = separation
                if distance < self.config.safety_distance_m:
                    self.proximity_safety_stops += 1
                    self.collision_stops += 1
                    self._stop_enabled_route_motion(
                        active_targets,
                        source="collision_stop",
                    )
                    self.runtime_events.append(
                        {
                            "timestamp_s": self.simulation_time_s,
                            "event": "route_safety_stop",
                            "reason": "measured_base_separation",
                            "robot_ids": [left_id, right_id],
                            "measured_separation_m": distance,
                            "safety_distance_m": (self.config.safety_distance_m),
                        }
                    )
                    raise DynamicCoppeliaError(
                        "measured enabled-base separation breached the route "
                        f"safety gate: {left_id} and {right_id} are "
                        f"{distance:.3f} m apart"
                    )
            colliding_ids = sorted(self._latest_collision_robots)
            if colliding_ids:
                self._stop_enabled_route_motion(
                    active_targets,
                    source="collision_stop",
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "route_safety_stop",
                        "reason": "physical_collision_monitor",
                        "robot_ids": colliding_ids,
                    }
                )
                raise DynamicCoppeliaError(
                    "physical collision monitor stopped route: " + ", ".join(colliding_ids)
                )
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
                if math.hypot(dx, dy) <= tolerance:
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

    def follow_synchronized_routes(
        self,
        routes: dict[str, list[Vec2]],
    ) -> None:
        """Follow one equal-horizon formation route with a shared waypoint index."""
        unavailable = sorted(set(routes) & self.disabled_robots)
        if unavailable:
            raise DynamicCoppeliaError(f"cannot route unavailable robots: {', '.join(unavailable)}")
        if len(routes) != 2 or any(not route for route in routes.values()):
            raise DynamicCoppeliaError("synchronized transport requires two non-empty base routes")
        horizons = {len(route) for route in routes.values()}
        if len(horizons) != 1:
            raise DynamicCoppeliaError("synchronized transport routes must have equal horizons")
        self._follow_shared_route_time(
            routes,
            waypoint_tolerance_m=self.config.waypoint_tolerance_m,
            enforce_relative_formation=True,
        )

    def _follow_shared_route_time(
        self,
        routes: dict[str, list[Vec2]],
        *,
        waypoint_tolerance_m: float,
        enforce_relative_formation: bool,
    ) -> None:
        """Execute MAPF paths against one shared, measured route-time index."""
        robot_ids = sorted(routes)
        empty_routes = [robot_id for robot_id in robot_ids if not routes[robot_id]]
        if empty_routes:
            raise DynamicCoppeliaError(
                "synchronized routes cannot be empty: " + ", ".join(empty_routes)
            )
        unknown = sorted(set(robot_ids) - set(self.robot_handles))
        if unknown:
            raise DynamicCoppeliaError("cannot route unknown robots: " + ", ".join(unknown))
        horizon = max(len(route) for route in routes.values())
        initial_samples = self._sample_route_telemetry_tick()
        initial_points = {
            robot_id: Vec2(
                x=sample.measured_pose.position.x,
                y=sample.measured_pose.position.y,
            )
            for robot_id, sample in initial_samples.items()
        }
        stationary_points = {
            robot_id: point for robot_id, point in initial_points.items() if robot_id not in routes
        }
        (
            minimum_planned_separation_m,
            limiting_planned_separation,
        ) = self._validate_route_time_separation(
            routes,
            horizon=horizon,
            measured_route_starts={robot_id: initial_points[robot_id] for robot_id in robot_ids},
            stationary_robot_points=stationary_points,
        )
        separation_sample_start = len(self.synchronized_route_separations_m)
        formation_sample_start = len(self.synchronized_formation_errors_m)
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "route_time_synchronization_started",
                "robot_ids": robot_ids,
                "route_lengths": {robot_id: len(routes[robot_id]) for robot_id in robot_ids},
                "shared_horizon": horizon,
                "shorter_routes_hold_endpoint": True,
                "minimum_planned_separation_m": (minimum_planned_separation_m),
                "limiting_planned_separation": limiting_planned_separation,
                "safety_distance_m": self.config.safety_distance_m,
                "relative_formation_enforced": enforce_relative_formation,
                "measured_start_to_first_target_included": True,
                "stationary_robot_ids": sorted(stationary_points),
            }
        )
        for waypoint_index in range(horizon):
            targets = {
                robot_id: routes[robot_id][min(waypoint_index, len(routes[robot_id]) - 1)]
                for robot_id in robot_ids
            }
            self.runtime_events.append(
                {
                    "timestamp_s": self.simulation_time_s,
                    "event": "route_time_index_started",
                    "route_time_index": waypoint_index,
                    "targets": {
                        robot_id: targets[robot_id].model_dump(mode="json")
                        for robot_id in robot_ids
                    },
                    "endpoint_hold_robot_ids": [
                        robot_id
                        for robot_id in robot_ids
                        if waypoint_index >= len(routes[robot_id])
                    ],
                }
            )
            steps_at_waypoint = 0
            while True:
                measured_all = {
                    robot_id: sample.measured_pose
                    for robot_id, sample in self._sample_route_telemetry_tick().items()
                }
                measured_enabled = {
                    robot_id: pose
                    for robot_id, pose in measured_all.items()
                    if robot_id not in self.disabled_robots
                }
                measured = {robot_id: measured_enabled[robot_id] for robot_id in robot_ids}
                separation = _minimum_pose_separation(measured_enabled)
                if separation is not None:
                    distance, left_id, right_id = separation
                    self.synchronized_route_separations_m.append(distance)
                    if distance < self.config.safety_distance_m:
                        self.proximity_safety_stops += 1
                        self.collision_stops += 1
                        self._stop_enabled_route_motion(
                            targets,
                            source="collision_stop",
                        )
                        self.runtime_events.append(
                            {
                                "timestamp_s": self.simulation_time_s,
                                "event": "route_time_safety_stop",
                                "reason": "measured_base_separation",
                                "route_time_index": waypoint_index,
                                "robot_ids": [left_id, right_id],
                                "measured_separation_m": distance,
                                "safety_distance_m": (self.config.safety_distance_m),
                            }
                        )
                        raise DynamicCoppeliaError(
                            "measured enabled-base separation breached the "
                            "synchronized route safety gate at time index "
                            f"{waypoint_index}: {left_id} and {right_id} are "
                            f"{distance:.3f} m apart"
                        )
                colliding_ids = sorted(self._latest_collision_robots)
                if colliding_ids:
                    self._stop_enabled_route_motion(
                        targets,
                        source="collision_stop",
                    )
                    self.runtime_events.append(
                        {
                            "timestamp_s": self.simulation_time_s,
                            "event": "route_time_safety_stop",
                            "reason": "physical_collision_monitor",
                            "route_time_index": waypoint_index,
                            "robot_ids": colliding_ids,
                        }
                    )
                    raise DynamicCoppeliaError(
                        "physical collision monitor stopped synchronized route "
                        f"at time index {waypoint_index}: " + ", ".join(colliding_ids)
                    )
                if enforce_relative_formation:
                    formation_error = _relative_formation_error(
                        robot_ids,
                        measured,
                        targets,
                    )
                    self.synchronized_formation_errors_m.append(formation_error)
                    if formation_error > self.config.formation_tolerance_m:
                        self._stop_enabled_route_motion(
                            targets,
                            source="formation_hold",
                        )
                        self.runtime_events.append(
                            {
                                "timestamp_s": self.simulation_time_s,
                                "event": "route_time_safety_stop",
                                "reason": "relative_formation_divergence",
                                "route_time_index": waypoint_index,
                                "formation_error_m": formation_error,
                                "formation_tolerance_m": (self.config.formation_tolerance_m),
                            }
                        )
                        raise DynamicCoppeliaError(
                            "synchronized carry formation diverged beyond tolerance"
                        )

                reached = {
                    robot_id: math.hypot(
                        targets[robot_id].x - measured[robot_id].position.x,
                        targets[robot_id].y - measured[robot_id].position.y,
                    )
                    <= waypoint_tolerance_m
                    for robot_id in robot_ids
                }
                if all(reached.values()):
                    for robot_id in robot_ids:
                        self.command_body_velocity(
                            robot_id,
                            0.0,
                            0.0,
                            0.0,
                            source="formation_hold",
                            target=targets[robot_id],
                        )
                    self.runtime_events.append(
                        {
                            "timestamp_s": self.simulation_time_s,
                            "event": "route_time_index_completed",
                            "route_time_index": waypoint_index,
                        }
                    )
                    break

                for robot_id in robot_ids:
                    target = targets[robot_id]
                    pose = measured[robot_id]
                    if reached[robot_id]:
                        self.command_body_velocity(
                            robot_id,
                            0.0,
                            0.0,
                            0.0,
                            source="formation_hold",
                            target=target,
                        )
                        continue
                    dx = target.x - pose.position.x
                    dy = target.y - pose.position.y
                    yaw = math.radians(pose.rotation_rpy_degrees.z)
                    body_forward, body_lateral = world_error_to_youbot_body(
                        dx,
                        dy,
                        yaw,
                    )
                    scale = self.config.position_gain
                    self.command_body_velocity(
                        robot_id,
                        _clamp(body_forward * scale, -1.0, 1.0),
                        _clamp(body_lateral * scale, -1.0, 1.0),
                        0.0,
                        target=target,
                    )
                steps_at_waypoint += 1
                if steps_at_waypoint > self.config.max_steps_per_waypoint:
                    self._stop_enabled_route_motion(
                        targets,
                        source="formation_hold",
                    )
                    raise DynamicCoppeliaError(
                        f"synchronized route failed to reach shared time index {waypoint_index}"
                    )
                self._update_logical_payloads()
                self._step_physics()
        self._update_logical_payloads()
        for robot_id in robot_ids:
            self.command_body_velocity(
                robot_id,
                0.0,
                0.0,
                0.0,
                source="formation_hold",
            )
        route_separation_samples = self.synchronized_route_separations_m[separation_sample_start:]
        route_formation_samples = self.synchronized_formation_errors_m[formation_sample_start:]
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "route_time_synchronization_completed",
                "robot_ids": robot_ids,
                "shared_horizon": horizon,
                "minimum_measured_separation_m": (
                    min(route_separation_samples) if route_separation_samples else None
                ),
                "maximum_formation_error_m": (
                    max(route_formation_samples) if route_formation_samples else None
                ),
            }
        )

    def _validate_route_time_separation(
        self,
        routes: Mapping[str, list[Vec2]],
        *,
        horizon: int,
        measured_route_starts: Mapping[str, Vec2],
        stationary_robot_points: Mapping[str, Vec2],
    ) -> tuple[float, dict[str, object]]:
        robot_ids = sorted(routes)
        routed_ids = set(robot_ids)
        minimum_separation = math.inf
        limiting_separation: dict[str, object] = {}
        previous_targets = dict(measured_route_starts)
        for waypoint_index in range(horizon):
            targets = {
                robot_id: routes[robot_id][min(waypoint_index, len(routes[robot_id]) - 1)]
                for robot_id in robot_ids
            }
            all_targets = {**targets, **stationary_robot_points}
            separation = _minimum_point_separation(
                all_targets,
                relevant_ids=routed_ids,
            )
            if separation is None:
                distance = math.inf
            else:
                distance, left_id, right_id = separation
                if distance < minimum_separation:
                    minimum_separation = distance
                    limiting_separation = {
                        "scope": "route_time_target",
                        "route_time_index": waypoint_index,
                        "robot_ids": [left_id, right_id],
                        "planned_separation_m": distance,
                    }
                if distance < self.config.safety_distance_m:
                    self.runtime_events.append(
                        {
                            "timestamp_s": self.simulation_time_s,
                            "event": "route_time_preflight_rejected",
                            "reason": "planned_target_separation",
                            "route_time_index": waypoint_index,
                            "robot_ids": [left_id, right_id],
                            "planned_separation_m": distance,
                            "safety_distance_m": self.config.safety_distance_m,
                        }
                    )
                    raise DynamicCoppeliaError(
                        "synchronized route time index "
                        f"{waypoint_index} places {left_id} and {right_id} "
                        f"{distance:.3f} m apart, below the configured safety "
                        f"separation of {self.config.safety_distance_m:.3f} m"
                    )
            interval_start = {
                **previous_targets,
                **stationary_robot_points,
            }
            interval_end = {**targets, **stationary_robot_points}
            interval_separation = _minimum_independent_interval_separation(
                interval_start,
                interval_end,
                relevant_ids=routed_ids,
            )
            if interval_separation is None:
                previous_targets = targets
                continue
            (
                interval_distance,
                left_interval_fraction,
                right_interval_fraction,
                interval_left_id,
                interval_right_id,
            ) = interval_separation
            route_time_interval = (
                [-1, 0] if waypoint_index == 0 else [waypoint_index - 1, waypoint_index]
            )
            if interval_distance < minimum_separation:
                minimum_separation = interval_distance
                limiting_separation = {
                    "scope": "route_time_interval",
                    "route_time_interval": route_time_interval,
                    "limiting_interval_fractions": {
                        interval_left_id: left_interval_fraction,
                        interval_right_id: right_interval_fraction,
                    },
                    "robot_ids": [
                        interval_left_id,
                        interval_right_id,
                    ],
                    "planned_separation_m": interval_distance,
                }
            if interval_distance < self.config.safety_distance_m:
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "route_time_preflight_rejected",
                        "reason": "planned_swept_separation",
                        "route_time_interval": route_time_interval,
                        "limiting_interval_fractions": {
                            interval_left_id: left_interval_fraction,
                            interval_right_id: right_interval_fraction,
                        },
                        "robot_ids": [
                            interval_left_id,
                            interval_right_id,
                        ],
                        "planned_separation_m": interval_distance,
                        "safety_distance_m": self.config.safety_distance_m,
                    }
                )
                raise DynamicCoppeliaError(
                    "synchronized route interval "
                    f"{route_time_interval[0]}->{route_time_interval[1]} brings "
                    f"{interval_left_id} and {interval_right_id} within "
                    f"{interval_distance:.3f} m at independent interval "
                    f"fractions {left_interval_fraction:.6f} and "
                    f"{right_interval_fraction:.6f}, below the configured safety "
                    f"separation of {self.config.safety_distance_m:.3f} m"
                )
            previous_targets = targets
        return minimum_separation, limiting_separation

    def _stop_enabled_route_motion(
        self,
        targets: Mapping[str, Vec2],
        *,
        source: RobotCommandSource,
    ) -> None:
        for robot_id in sorted(set(self.robot_handles) - self.disabled_robots):
            self.command_body_velocity(
                robot_id,
                0.0,
                0.0,
                0.0,
                source=source,
                target=targets.get(robot_id),
            )

    def attach_logical_payload(
        self,
        module_id: str,
        robot_ids: list[str],
        *,
        assigned_targets: Mapping[str, Vec2] | None = None,
        transport_height_m: float | None = None,
    ) -> None:
        if not robot_ids:
            raise DynamicCoppeliaError(f"cannot attach {module_id} without a robot team")
        unavailable = sorted(set(robot_ids) & self.disabled_robots)
        if unavailable:
            raise DynamicCoppeliaError(
                f"cannot attach {module_id} to unavailable robots: {', '.join(unavailable)}"
            )
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        staging = module.staging_pose.position
        resolved_transport_height_m = (
            staging.z if transport_height_m is None else float(transport_height_m)
        )
        if (
            not math.isfinite(resolved_transport_height_m)
            or resolved_transport_height_m < staging.z
        ):
            raise DynamicCoppeliaError(
                f"{module_id} logical transport height must be finite and "
                f"at least its staging height ({staging.z:.3f} m)"
            )
        stop_proof = self._settle_logical_transition(
            module_id,
            robot_ids,
            transition="logical_payload_lift",
        )
        samples = self._sample_route_telemetry_tick()
        positions = [samples[item].measured_pose.position for item in robot_ids]
        center_x = sum(item.x for item in positions) / len(positions)
        center_y = sum(item.y for item in positions) / len(positions)
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
        module_handle = self.module_handles[module_id]
        module_from_pose = self._read_object_pose(module_handle)
        self.sim.setObjectPosition(carrier, [staging.x, staging.y, staging.z])
        self.sim.setObjectParent(module_handle, carrier, True)
        self.sim.setObjectPosition(
            carrier,
            [staging.x, staging.y, resolved_transport_height_m],
        )
        center_z = sum(item.z for item in positions) / len(positions)
        centroid = Vec3(x=center_x, y=center_y, z=center_z)
        centroid_offset = Vec3(
            x=staging.x - center_x,
            y=staging.y - center_y,
            z=resolved_transport_height_m - center_z,
        )
        self.logical_carrier_offsets[module_id] = centroid_offset
        self.logical_transport_heights_m[module_id] = resolved_transport_height_m
        self.logical_attachments[module_id] = list(robot_ids)
        if transport_height_m is not None:
            module_lifted_pose = self._read_object_pose(module_handle)
            record: dict[str, object] = {
                "timestamp_s": self.simulation_time_s,
                "event": "logical_payload_lifted",
                "module_id": module_id,
                "robot_ids": sorted(robot_ids),
                "transport_height_m": resolved_transport_height_m,
                "module_from_pose": module_from_pose.model_dump(mode="json"),
                "module_lifted_pose": module_lifted_pose.model_dump(mode="json"),
                "measured_robot_centroid": centroid.model_dump(mode="json"),
                "logical_centroid_offset": centroid_offset.model_dump(mode="json"),
                "carrier_pose": self._read_object_pose(carrier).model_dump(mode="json"),
                "lift_delta_m": (module_lifted_pose.position.z - module_from_pose.position.z),
                "scope": "logical_transport_only",
                "physical_lift_claimed": False,
                "physical_descent_claimed": False,
                "team_zero_motion_proof": stop_proof,
            }
            self.logical_payload_lift_records.append(record)
            self.runtime_events.append(dict(record))

    def install_logical_payload(
        self,
        module_id: str,
        *,
        assigned_targets: Mapping[str, Vec2] | None = None,
        contact_module_ids: list[str] | None = None,
    ) -> None:
        module = next(item for item in self.plan.modules if item.module_id == module_id)
        robot_ids = self.logical_attachments[module_id]
        stop_proof = self._settle_logical_transition(
            module_id,
            robot_ids,
            transition="logical_installation_snap",
        )
        samples = self._sample_route_telemetry_tick()
        positions = [samples[item].measured_pose.position for item in robot_ids]
        target = module.target_pose.position
        carrier_position = self.sim.getObjectPosition(self.payload_carriers[module_id])
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
        from_pose = self._read_object_pose(handle)
        transport_height = self.logical_transport_heights_m.get(module_id)
        if transport_height is not None and not math.isclose(
            from_pose.position.z,
            transport_height,
            abs_tol=1e-6,
        ):
            raise DynamicCoppeliaError(
                f"{module_id} left its attested logical transport height before installation"
            )
        exact_target_pose = module.target_pose.model_dump(mode="json")
        target_pose_sha256 = _sha256_json(exact_target_pose)
        snap_timestamp_s = self.simulation_time_s
        self.sim.setObjectParent(handle, self.root_handle, True)
        self.sim.setObjectPosition(handle, [target.x, target.y, target.z])
        rotation = module.target_pose.rotation_rpy_degrees
        self.sim.setObjectOrientation(
            handle,
            [math.radians(rotation.x), math.radians(rotation.y), math.radians(rotation.z)],
        )
        self.logical_attachments.pop(module_id)
        self.logical_carrier_offsets.pop(module_id)
        self.logical_transport_heights_m.pop(module_id, None)
        self.installed_modules.add(module_id)
        record = {
            "timestamp_s": snap_timestamp_s,
            "event": "logical_installation_snap",
            "module_id": module_id,
            "from_pose": from_pose.model_dump(mode="json"),
            "target_pose": exact_target_pose,
            "target_pose_sha256": target_pose_sha256,
            "contact_module_ids": sorted(set(contact_module_ids or [])),
            "scope": "final_target_pose_only",
            "team_zero_motion_proof": stop_proof,
            "transport_height_m": transport_height,
            "physical_lift_claimed": False,
            "physical_descent_claimed": False,
        }
        self.logical_installation_snap_records.append(record)
        self.runtime_events.append(dict(record))

    def _settle_logical_transition(
        self,
        module_id: str,
        robot_ids: list[str],
        *,
        transition: str,
    ) -> dict[str, object]:
        hold_command_index = len(self.commands)
        for robot_id in sorted(robot_ids):
            self.command_body_velocity(
                robot_id,
                0.0,
                0.0,
                0.0,
                source="formation_hold",
            )
        started_at_s = self.simulation_time_s
        started_at_step = self.physics_steps
        consecutive_still_samples = 0
        sample_records: list[dict[str, object]] = []
        latest_samples: Mapping[str, RobotTelemetry] = {}
        minimum_measured_separation: tuple[float, str, str] | None = None
        for step_offset in range(self.config.logical_transition_settle_max_steps + 1):
            all_samples = self._sample_route_telemetry_tick()
            latest_samples = {robot_id: all_samples[robot_id] for robot_id in robot_ids}
            all_poses = {robot_id: sample.measured_pose for robot_id, sample in all_samples.items()}
            separation = _minimum_pose_separation(
                all_poses,
                relevant_ids=set(robot_ids),
            )
            if separation is not None:
                distance, left_id, right_id = separation
                if minimum_measured_separation is None or distance < minimum_measured_separation[0]:
                    minimum_measured_separation = (distance, left_id, right_id)
                if distance < self.config.safety_distance_m:
                    self.proximity_safety_stops += 1
                    self.collision_stops += 1
                    self._stop_enabled_route_motion(
                        {},
                        source="collision_stop",
                    )
                    self.runtime_events.append(
                        {
                            "timestamp_s": self.simulation_time_s,
                            "event": "logical_transition_settle_failed",
                            "module_id": module_id,
                            "transition": transition,
                            "reason": "measured_base_separation",
                            "robot_ids": [left_id, right_id],
                            "measured_separation_m": distance,
                            "safety_distance_m": (self.config.safety_distance_m),
                        }
                    )
                    raise DynamicCoppeliaError(
                        f"{module_id} {transition} breached scene-base "
                        f"safety separation: {left_id} and {right_id} are "
                        f"{distance:.3f} m apart"
                    )
            per_robot: dict[str, object] = {}
            team_still = True
            for robot_id in sorted(robot_ids):
                sample = latest_samples[robot_id]
                still = (
                    abs(sample.linear_velocity_mps) <= self.config.settled_linear_speed_mps
                    and abs(sample.angular_velocity_rps) <= self.config.settled_angular_speed_rps
                )
                team_still = team_still and still
                per_robot[robot_id] = {
                    "timestamp_s": sample.timestamp_s,
                    "linear_velocity_mps": sample.linear_velocity_mps,
                    "angular_velocity_rps": sample.angular_velocity_rps,
                    "still": still,
                }
            consecutive_still_samples = consecutive_still_samples + 1 if team_still else 0
            sample_records.append(
                {
                    "timestamp_s": self.simulation_time_s,
                    "physics_step": self.physics_steps,
                    "team_still": team_still,
                    "consecutive_still_samples": (consecutive_still_samples),
                    "robots": per_robot,
                }
            )
            nonzero_after_hold = [
                command
                for command in self.commands[hold_command_index:]
                if command.robot_id in robot_ids and not _robot_command_is_zero(command)
            ]
            if nonzero_after_hold:
                self._stop_enabled_route_motion(
                    {},
                    source="formation_hold",
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "logical_transition_settle_failed",
                        "module_id": module_id,
                        "transition": transition,
                        "reason": "nonzero_command_after_hold",
                        "nonzero_command_count": len(nonzero_after_hold),
                    }
                )
                raise DynamicCoppeliaError(
                    f"{module_id} {transition} received a nonzero team "
                    "command after its settle hold"
                )
            if self._latest_collision_robots:
                collision_ids = sorted(self._latest_collision_robots)
                self._stop_enabled_route_motion(
                    {},
                    source="collision_stop",
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "logical_transition_settle_failed",
                        "module_id": module_id,
                        "transition": transition,
                        "reason": "physical_collision_monitor",
                        "robot_ids": collision_ids,
                    }
                )
                raise DynamicCoppeliaError(
                    f"physical collision monitor stopped {module_id} "
                    f"{transition}: " + ", ".join(collision_ids)
                )
            if (
                consecutive_still_samples
                >= self.config.logical_transition_settle_consecutive_samples
            ):
                self._update_logical_payloads()
                proof = self._logical_transition_stop_proof(
                    module_id,
                    robot_ids,
                    latest_samples,
                    transition=transition,
                )
                proof.update(
                    {
                        "hold_command_index": hold_command_index,
                        "command_count_at_settle": len(self.commands),
                        "started_at_s": started_at_s,
                        "settled_at_s": self.simulation_time_s,
                        "physics_steps_elapsed": (self.physics_steps - started_at_step),
                        "required_consecutive_still_samples": (
                            self.config.logical_transition_settle_consecutive_samples
                        ),
                        "observed_consecutive_still_samples": (consecutive_still_samples),
                        "no_nonzero_team_command_after_hold": True,
                        "minimum_measured_team_to_scene_separation_m": (
                            minimum_measured_separation[0]
                            if minimum_measured_separation is not None
                            else None
                        ),
                        "minimum_measured_team_to_scene_pair": (
                            list(minimum_measured_separation[1:])
                            if minimum_measured_separation is not None
                            else None
                        ),
                        "safety_distance_m": self.config.safety_distance_m,
                    }
                )
                self.runtime_events.append(
                    {
                        "timestamp_s": self.simulation_time_s,
                        "event": "logical_transition_settled",
                        "module_id": module_id,
                        "transition": transition,
                        "team_zero_motion_proof": proof,
                        "settle_samples": sample_records,
                    }
                )
                return proof
            if step_offset >= self.config.logical_transition_settle_max_steps:
                break
            self._update_logical_payloads()
            self._step_physics()
        self.runtime_events.append(
            {
                "timestamp_s": self.simulation_time_s,
                "event": "logical_transition_settle_failed",
                "module_id": module_id,
                "transition": transition,
                "reason": "timeout",
                "maximum_physics_steps": (self.config.logical_transition_settle_max_steps),
                "settle_samples": sample_records,
            }
        )
        raise DynamicCoppeliaError(
            f"{module_id} {transition} did not settle within "
            f"{self.config.logical_transition_settle_max_steps} physics steps"
        )

    def _logical_installation_stop_proof(
        self,
        module_id: str,
        robot_ids: list[str],
        samples: Mapping[str, RobotTelemetry],
    ) -> dict[str, object]:
        return self._logical_transition_stop_proof(
            module_id,
            robot_ids,
            samples,
            transition="logical_installation_snap",
        )

    def _logical_transition_stop_proof(
        self,
        module_id: str,
        robot_ids: list[str],
        samples: Mapping[str, RobotTelemetry],
        *,
        transition: str,
    ) -> dict[str, object]:
        latest_commands: dict[str, RobotCommand] = {}
        required = set(robot_ids)
        for candidate in reversed(self.commands):
            if candidate.robot_id in required and candidate.robot_id not in latest_commands:
                latest_commands[candidate.robot_id] = candidate
            if set(latest_commands) == required:
                break
        command_records: dict[str, object] = {}
        latest_commands_zero = set(latest_commands) == required
        for robot_id in sorted(required):
            latest_command = latest_commands.get(robot_id)
            if latest_command is None:
                command_records[robot_id] = {"present": False, "zero": False}
                continue
            zero = (
                abs(latest_command.linear_velocity_mps) <= 1e-9
                and abs(latest_command.angular_velocity_rps) <= 1e-9
                and all(abs(value) <= 1e-9 for value in latest_command.wheel_target_velocity_rad_s)
            )
            latest_commands_zero = latest_commands_zero and zero
            command_records[robot_id] = {
                "present": True,
                "zero": zero,
                "timestamp_s": latest_command.timestamp_s,
                "source": latest_command.source,
                "linear_velocity_mps": latest_command.linear_velocity_mps,
                "angular_velocity_rps": latest_command.angular_velocity_rps,
                "wheel_target_velocity_rad_s": list(latest_command.wheel_target_velocity_rad_s),
            }
        measured_records: dict[str, object] = {}
        measured_team_still = True
        for robot_id in sorted(required):
            sample = samples[robot_id]
            still = (
                abs(sample.linear_velocity_mps) <= self.config.settled_linear_speed_mps
                and abs(sample.angular_velocity_rps) <= self.config.settled_angular_speed_rps
            )
            measured_team_still = measured_team_still and still
            measured_records[robot_id] = {
                "timestamp_s": sample.timestamp_s,
                "still": still,
                "linear_velocity_mps": sample.linear_velocity_mps,
                "angular_velocity_rps": sample.angular_velocity_rps,
            }
        proof = {
            "robot_ids": sorted(required),
            "latest_commands_zero": latest_commands_zero,
            "latest_commands_by_robot": command_records,
            "measured_team_still": measured_team_still,
            "measured_motion_by_robot": measured_records,
            "settled_linear_speed_mps": (self.config.settled_linear_speed_mps),
            "settled_angular_speed_rps": (self.config.settled_angular_speed_rps),
        }
        if not latest_commands_zero or not measured_team_still:
            self.runtime_events.append(
                {
                    "timestamp_s": self.simulation_time_s,
                    "event": "logical_transition_stop_proof_rejected",
                    "module_id": module_id,
                    "transition": transition,
                    "reason": "team_not_proven_stopped",
                    "team_zero_motion_proof": proof,
                }
            )
            raise DynamicCoppeliaError(
                f"{module_id} logical installation snap requires zero latest "
                "commands and a measured-still robot team"
            )
        return proof

    def _read_object_pose(self, handle: int) -> Pose3D:
        position = self.sim.getObjectPosition(handle)
        orientation = self.sim.getObjectOrientation(handle)
        return Pose3D(
            position=Vec3(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            rotation_rpy_degrees=Vec3(
                x=math.degrees(float(orientation[0])),
                y=math.degrees(float(orientation[1])),
                z=math.degrees(float(orientation[2])),
            ),
        )

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
            "remote_step_handshake_reconciliations": (
                self.remote_step_handshake_reconciliations
            ),
            "remote_step_handshake_retries": (
                self.remote_step_handshake_retries
            ),
            "measured_duration_s": self.simulation_time_s,
            "wheel_command_count": len(self.commands),
            "telemetry_sample_count": len(self.telemetry),
            "collision_stops": self.collision_stops,
            "proximity_safety_stops": self.proximity_safety_stops,
            "physical_collision_stops": self.physical_collision_stops,
            "physical_collision_event_count": len(self.physical_collision_events),
            "permitted_logical_payload_overlaps": (self.permitted_logical_payload_overlaps),
            "collision_query_count": self.collision_query_count,
            "collision_query_rounds": self.collision_query_rounds,
            "expected_collision_queries_per_step": len(self.collision_pairs),
            "collision_pair_category_counts": dict(self.collision_pair_category_counts),
            "robot_base_contact_gate_passed": self.robot_base_contact_gate_passed,
            "robot_base_physics_audit": list(self.robot_base_physics_audit),
            "robot_base_physics_audit_sha256": _sha256_json(
                self.robot_base_physics_audit
            ),
            "robot_collision_self_tests": list(self.robot_collision_self_tests),
            "measured_base_footprint_radius_m_by_robot": {
                str(item["robot_id"]): item["measured_footprint_radius_m"]
                for item in self.robot_base_physics_audit
            },
            "collision_queries_cover_every_physics_step": (
                self.physics_steps > 0
                and self.collision_query_rounds == self.physics_steps
                and self.collision_query_count == self.physics_steps * len(self.collision_pairs)
            ),
            "initial_robot_pose_writes": self.initial_robot_pose_writes,
            "post_start_robot_pose_writes": self.post_start_robot_pose_writes,
            "bundled_motion_scripts_found": self.bundled_motion_scripts_found,
            "disabled_bundled_motion_scripts": self.disabled_bundled_motion_scripts,
            "verified_disabled_bundled_motion_scripts": (
                self.verified_disabled_bundled_motion_scripts
            ),
            "retained_bundled_maintenance_scripts": (self.retained_bundled_maintenance_scripts),
            "verified_enabled_maintenance_scripts": (self.verified_enabled_maintenance_scripts),
            "disabled_wheel_command_scripts": (self.disabled_wheel_command_scripts),
            "disabled_arm_gripper_scripts": (self.disabled_arm_gripper_scripts),
            "bundled_script_absence_proven": self.bundled_script_absence_proven,
            "script_inventory_classification_complete": (
                self.script_inventory_classification_complete
            ),
            "script_control_gate_passed": self.script_control_gate_passed,
            "script_control_by_robot": dict(self.script_control_by_robot),
            "wheel_command_writer_count_by_robot": dict(self.wheel_command_writer_count_by_robot),
            "retained_maintenance_count_by_robot": dict(self.retained_maintenance_count_by_robot),
            "script_control_audit": list(self.script_control_audit),
            "scene_robot_count": len(self.robot_handles),
            "prior_generated_scene_root_count": (self.prior_generated_scene_root_count),
            "prior_generated_scene_object_count": (self.prior_generated_scene_object_count),
            "generated_scene_cleanup_verified": (self.generated_scene_cleanup_verified),
            "scene_module_count": len(self.module_handles),
            "scene_obstacle_count": len(self.obstacle_handles),
            "disabled_robots": sorted(self.disabled_robots),
            "disabled_command_cutoffs": dict(self.disabled_command_cutoffs),
            "disabled_settled": dict(self.disabled_settled),
            "disabled_settle_displacement_m": dict(self.disabled_settle_displacement_m),
            "nonzero_wheel_command_count": dict(self.nonzero_wheel_command_count),
            "command_response_displacement_m": dict(self.command_response_displacement_m),
            "command_response_observed_at_s": dict(self.command_response_observed_at_s),
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
            "maximum_synchronized_formation_error_m": max(
                self.synchronized_formation_errors_m,
                default=0.0,
            ),
            "minimum_synchronized_route_separation_m": (
                min(self.synchronized_route_separations_m)
                if self.synchronized_route_separations_m
                else None
            ),
            "logical_payload_lift_count": len(self.logical_payload_lift_records),
            "logical_payload_lifts": list(self.logical_payload_lift_records),
            "logical_installation_snap_count": len(self.logical_installation_snap_records),
            "logical_installation_snaps": list(self.logical_installation_snap_records),
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
                "Vertical lift, transport height, and final descent are logical pose operations; no physical lifting or descent is claimed.",
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
                carry_routes[robot_id] = [
                    Vec2(x=point.x, y=point.y + offset) for point in base_route
                ]
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
        missing = [name for name in required_api if not callable(getattr(self.sim, name, None))]
        handle_scene = getattr(self.sim, "handle_scene", None)
        handle_all = getattr(self.sim, "handle_all", None)
        if missing or handle_scene is None or handle_all is None:
            detail = ", ".join(
                [
                    *missing,
                    *(["handle_scene"] if handle_scene is None else []),
                    *(["handle_all"] if handle_all is None else []),
                ]
            )
            raise DynamicCoppeliaError(
                f"project-owned Coppelia scene cleanup is unavailable: {detail}"
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
                if str(self.sim.getObjectAlias(handle, -1)) == GENERATED_SCENE_ROOT_ALIAS
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
                if str(self.sim.getObjectAlias(handle, -1)) == GENERATED_SCENE_ROOT_ALIAS
            ]
        except Exception as exc:
            raise DynamicCoppeliaError("project-owned Coppelia scene cleanup failed") from exc
        if remaining_roots:
            raise DynamicCoppeliaError(
                "project-owned Coppelia scene cleanup could not remove every prior generated root"
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
        grid_max_x = grid.origin.x + (grid.width - 1) * grid.resolution_m + grid.resolution_m
        grid_min_y = grid.origin.y - grid.resolution_m
        grid_max_y = grid.origin.y + (grid.height - 1) * grid.resolution_m + grid.resolution_m
        floor = self._create_box(
            "construction_intelligence_floor",
            (
                (grid_min_x + grid_max_x) / 2,
                (grid_min_y + grid_max_y) / 2,
                CONSTRUCTION_FLOOR_TOP_Z_M - CONSTRUCTION_FLOOR_THICKNESS_M / 2,
            ),
            (
                grid_max_x - grid_min_x,
                grid_max_y - grid_min_y,
                CONSTRUCTION_FLOOR_THICKNESS_M,
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
            self._configure_youbot_base_physics(robot.robot_id, handle)
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

    def _configure_youbot_base_physics(
        self,
        robot_id: str,
        robot_handle: int,
    ) -> None:
        """Constrain the imported YouBot to the v1 mobile-base physics contract."""
        required_api = (
            "getObjectsInTree",
            "getObjectAlias",
            "getObjectInt32Param",
            "setObjectInt32Param",
            "getBoolProperty",
            "setBoolProperty",
            "getShapeBB",
            "getObjectMatrix",
        )
        missing = [
            name
            for name in required_api
            if not callable(getattr(self.sim, name, None))
        ]
        shape_type = getattr(self.sim, "object_shape_type", None)
        if missing or shape_type is None:
            detail = ", ".join(
                [*missing, *(["object_shape_type"] if shape_type is None else [])]
            )
            raise DynamicCoppeliaError(
                f"YouBot base-physics attestation is unavailable: {detail}"
            )
        shape_handles = {
            int(handle)
            for handle in self.sim.getObjectsInTree(
                robot_handle,
                shape_type,
                0,
            )
        }
        if robot_handle not in shape_handles:
            raise DynamicCoppeliaError(
                f"{robot_id} model root is not an auditable shape"
            )
        aliases = {
            handle: str(self.sim.getObjectAlias(handle, 1))
            for handle in shape_handles
        }
        required_wheel_aliases = {
            f"wheel_respondable_{wheel_name}" for wheel_name in WHEEL_NAMES
        }
        allowed_handles = {robot_handle}
        for required_alias in sorted(required_wheel_aliases):
            matches = {
                handle
                for handle, alias in aliases.items()
                if alias.strip("/").rsplit("/", 1)[-1].casefold()
                == required_alias
            }
            if len(matches) != 1:
                raise DynamicCoppeliaError(
                    f"{robot_id} must contain exactly one {required_alias} base shape"
                )
            allowed_handles.update(matches)
        if len(allowed_handles) != 5:
            raise DynamicCoppeliaError(
                f"{robot_id} mobile-base physics must contain five unique shapes"
            )
        excluded_handles = shape_handles - allowed_handles
        if not excluded_handles:
            raise DynamicCoppeliaError(
                f"{robot_id} has no excluded arm, gripper, or visual shapes to attest"
            )

        records: list[dict[str, object]] = []
        for handle in sorted(shape_handles):
            allowed = handle in allowed_handles
            before = {
                "respondable": int(
                    self.sim.getObjectInt32Param(
                        handle,
                        self.sim.shapeintparam_respondable,
                    )
                ),
                "static": int(
                    self.sim.getObjectInt32Param(
                        handle,
                        self.sim.shapeintparam_static,
                    )
                ),
                "collidable": self.sim.getBoolProperty(handle, "collidable"),
            }
            requested = {
                "respondable": int(allowed),
                "static": 0 if allowed else before["static"],
                "collidable": allowed,
            }
            self.sim.setObjectInt32Param(
                handle,
                self.sim.shapeintparam_respondable,
                requested["respondable"],
            )
            self.sim.setObjectInt32Param(
                handle,
                self.sim.shapeintparam_static,
                requested["static"],
            )
            self.sim.setBoolProperty(
                handle,
                "collidable",
                requested["collidable"],
            )
            after = {
                "respondable": int(
                    self.sim.getObjectInt32Param(
                        handle,
                        self.sim.shapeintparam_respondable,
                    )
                ),
                "static": int(
                    self.sim.getObjectInt32Param(
                        handle,
                        self.sim.shapeintparam_static,
                    )
                ),
                "collidable": self.sim.getBoolProperty(handle, "collidable"),
            }
            if (
                not isinstance(before["collidable"], bool)
                or after != requested
            ):
                raise DynamicCoppeliaError(
                    f"{robot_id} shape {aliases[handle]} did not read back the "
                    "requested base-only physics policy"
                )
            records.append(
                {
                    "handle": handle,
                    "alias": aliases[handle],
                    "role": "mobile_base" if allowed else "excluded_v1_geometry",
                    "before": before,
                    "after": after,
                }
            )

        measured_radius = max(
            _shape_xy_footprint_radius_m(
                self.sim,
                shape_handle=handle,
                robot_handle=robot_handle,
            )
            for handle in allowed_handles
        )
        planned_radius = self.config.planned_robot_footprint_radius_m
        footprint_gate_passed = measured_radius <= planned_radius + 1e-9
        if not footprint_gate_passed:
            raise DynamicCoppeliaError(
                f"{robot_id} measured base footprint radius {measured_radius:.6f} m "
                f"exceeds planned radius {planned_radius:.6f} m"
            )
        audit = {
            "robot_id": robot_id,
            "robot_handle": robot_handle,
            "allowed_shape_handles": sorted(allowed_handles),
            "excluded_shape_handles": sorted(excluded_handles),
            "shape_records": records,
            "measured_footprint_radius_m": measured_radius,
            "planned_footprint_radius_m": planned_radius,
            "footprint_gate_passed": footprint_gate_passed,
            "base_only_policy_verified": True,
        }
        self.robot_base_shape_handles[robot_id] = allowed_handles
        self.robot_excluded_shape_handles[robot_id] = excluded_handles
        self.robot_base_physics_audit.append(audit)

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
                f"{robot_id} must contain exactly one bundled wheel-command writer"
            )
        if role_counts["passive_omniwheel_maintenance"] < 1:
            raise DynamicCoppeliaError(
                f"{robot_id} has no allowlisted passive omni-wheel maintenance scripts"
            )

        script_records: list[dict[str, object]] = []
        for handle, alias, source, role in classified:
            should_disable = role != "passive_omniwheel_maintenance"
            disabled_before, disabled_after, control_api = self._set_script_disabled(
                handle,
                disabled=should_disable,
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
        self.wheel_command_writer_count_by_robot[robot_id] = role_counts["wheel_command_writer"]
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
                "exclusive_control_verified": self.script_control_by_robot[robot_id],
            }
        )
        self.script_inventory_classification_complete = (
            self.bundled_motion_scripts_found
            == self.disabled_bundled_motion_scripts + self.retained_bundled_maintenance_scripts
        )
        self.script_control_gate_passed = (
            len(self.script_control_by_robot) == len(self.robot_handles) + 1
            and all(self.script_control_by_robot.values())
            and self.script_inventory_classification_complete
            and self.disabled_bundled_motion_scripts
            == self.verified_disabled_bundled_motion_scripts
            and self.retained_bundled_maintenance_scripts
            == self.verified_enabled_maintenance_scripts
            and self.disabled_wheel_command_scripts == len(self.script_control_by_robot)
            and all(count == 1 for count in self.wheel_command_writer_count_by_robot.values())
            and all(count >= 1 for count in self.retained_maintenance_count_by_robot.values())
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
                raise DynamicCoppeliaError("bundled YouBot script source cannot be audited")
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
        if callable(get_disabled_property) and callable(set_disabled_property):
            disabled_before = get_disabled_property(
                handle,
                "scriptDisabled",
            )
            if not isinstance(disabled_before, bool):
                raise DynamicCoppeliaError(
                    f"bundled YouBot script {handle} returned an indeterminate disabled state"
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
            raise DynamicCoppeliaError("bundled YouBot script state cannot be verified")
        enabled_before = get_int_parameter(
            handle,
            self.sim.scriptintparam_enabled,
        )
        if not isinstance(enabled_before, int) or isinstance(
            enabled_before,
            bool,
        ):
            raise DynamicCoppeliaError(
                f"bundled YouBot script {handle} returned an indeterminate enabled state"
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
                if alias.rsplit("/", 1)[-1] in {f"rollingjoint_{wheel_name}", f"wheel_{wheel_name}"}
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
            raise DynamicCoppeliaError(
                "direct robot pose writes are forbidden after simulation start"
            )
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
            if module_id in self.logical_transport_heights_m:
                center[2] = self.logical_transport_heights_m[module_id]
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
            "getCollectionObjects",
            "checkCollision",
        )
        missing = [name for name in required_api if not callable(getattr(self.sim, name, None))]
        handle_single = getattr(self.sim, "handle_single", None)
        if missing or handle_single is None:
            detail = ", ".join(
                [*missing, *(["handle_single"] if handle_single is None else [])]
            )
            raise DynamicCoppeliaError(
                f"Coppelia physical collision monitoring is unavailable: {detail}"
            )
        for robot_id in sorted(self.robot_handles):
            base_handles = self.robot_base_shape_handles.get(robot_id)
            if not base_handles:
                raise DynamicCoppeliaError(
                    f"{robot_id} has no attested mobile-base collision shapes"
                )
            collection = int(self.sim.createCollection(0))
            for shape_handle in sorted(base_handles):
                self.sim.addItemToCollection(
                    collection,
                    handle_single,
                    shape_handle,
                    0,
                )
            self.robot_collision_entities[robot_id] = collection

        self._self_test_collision_monitoring()

        robot_world = int(self.sim.createCollection(0))
        for robot_id in sorted(self.robot_base_shape_handles):
            for shape_handle in sorted(self.robot_base_shape_handles[robot_id]):
                self.sim.addItemToCollection(
                    robot_world,
                    handle_single,
                    shape_handle,
                    0,
                )
        world = int(self.sim.createCollection(0))
        for object_handle in sorted(
            [*self.module_handles.values(), *self.obstacle_handles.values()]
        ):
            self.sim.addItemToCollection(
                world,
                handle_single,
                object_handle,
                0,
            )
        expected_robot_world_handles = set().union(
            *self.robot_base_shape_handles.values()
        )
        expected_world_handles = {
            *self.module_handles.values(),
            *self.obstacle_handles.values(),
        }
        if (
            set(self.sim.getCollectionObjects(robot_world))
            != expected_robot_world_handles
            or set(self.sim.getCollectionObjects(world)) != expected_world_handles
        ):
            raise DynamicCoppeliaError(
                "aggregate collision collection membership readback failed"
            )
        self.robot_world_collision_entity = robot_world
        self.world_collision_entity = world

        query_groups: list[
            tuple[str, int, str, int, str, tuple[str, ...]]
        ] = []
        robot_ids = sorted(self.robot_collision_entities)
        for index, robot_id in enumerate(robot_ids):
            for other_id in robot_ids[index + 1 :]:
                query_groups.append(
                    (
                        f"robot:{robot_id}",
                        self.robot_collision_entities[robot_id],
                        f"robot:{other_id}",
                        self.robot_collision_entities[other_id],
                        "robot_robot",
                        (robot_id, other_id),
                    )
                )
        query_groups.append(
            (
                "all_robot_bases",
                robot_world,
                "modules_and_obstacles",
                world,
                "robot_world",
                (),
            )
        )
        inventory: list[dict[str, object]] = []
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
            for module_id, module_handle in sorted(self.module_handles.items()):
                inventory.append(
                    {
                        "entity_1": f"robot:{robot_id}",
                        "entity_2": f"module:{module_id}",
                        "category": "robot_module",
                        "robot_ids": [robot_id],
                    }
                )
            for obstacle_id, obstacle_handle in sorted(self.obstacle_handles.items()):
                inventory.append(
                    {
                        "entity_1": f"robot:{robot_id}",
                        "entity_2": f"obstacle:{obstacle_id}",
                        "category": "robot_obstacle",
                        "robot_ids": [robot_id],
                    }
                )
        if not query_groups or not inventory:
            raise DynamicCoppeliaError(
                "collision monitoring has no query groups or entity pairs"
            )
        self.collision_pairs = query_groups
        category_counts: dict[str, int] = {}
        for pair in inventory:
            category = str(pair["category"])
            category_counts[category] = category_counts.get(category, 0) + 1
        self.collision_pair_category_counts = category_counts
        query_group_inventory = [
            {
                "entity_1": pair[0],
                "entity_2": pair[2],
                "category": pair[4],
                "robot_ids": list(pair[5]),
            }
            for pair in query_groups
        ]
        self.runtime_events.append(
            {
                "timestamp_s": 0.0,
                "event": "collision_monitor_ready",
                "expected_queries_per_step": len(query_groups),
                "pair_category_counts": dict(category_counts),
                "pair_inventory_sha256": _sha256_json(inventory),
                "query_group_inventory": query_group_inventory,
                "query_group_inventory_sha256": _sha256_json(
                    query_group_inventory
                ),
                "aggregate_collection_membership_sha256": _sha256_json(
                    {
                        "robot_base_shape_handles": sorted(
                            expected_robot_world_handles
                        ),
                        "world_object_handles": sorted(expected_world_handles),
                    }
                ),
                "aggregate_robot_base_membership": {
                    robot_id: sorted(handles)
                    for robot_id, handles in sorted(
                        self.robot_base_shape_handles.items()
                    )
                },
                "aggregate_world_membership": [
                    {
                        "entity": f"module:{module_id}",
                        "object_handle": handle,
                    }
                    for module_id, handle in sorted(self.module_handles.items())
                ]
                + [
                    {
                        "entity": f"obstacle:{obstacle_id}",
                        "object_handle": handle,
                    }
                    for obstacle_id, handle in sorted(
                        self.obstacle_handles.items()
                    )
                ],
                "robot_ids": robot_ids,
                "module_ids": sorted(self.module_handles),
                "obstacle_ids": sorted(self.obstacle_handles),
                "base_physics_audit_sha256": _sha256_json(
                    self.robot_base_physics_audit
                ),
                "collision_self_tests": list(self.robot_collision_self_tests),
            }
        )
        self.robot_base_contact_gate_passed = True

    def _self_test_collision_monitoring(self) -> None:
        remove_objects = getattr(self.sim, "removeObjects", None)
        if not callable(remove_objects):
            raise DynamicCoppeliaError(
                "Coppelia collision-monitor self-test cleanup is unavailable"
            )
        for robot_id in sorted(self.robot_handles):
            base_handles = self.robot_base_shape_handles[robot_id]
            robot_position = self.sim.getObjectPosition(
                self.robot_handles[robot_id]
            )
            probe = self._create_box(
                f"collision_monitor_self_test_{robot_id}",
                (
                    float(robot_position[0]),
                    float(robot_position[1]),
                    float(robot_position[2]),
                ),
                (0.02, 0.02, 0.02),
                (1.0, 0.0, 1.0),
            )
            self.sim.setObjectInt32Param(
                probe,
                self.sim.shapeintparam_respondable,
                0,
            )
            self.sim.setBoolProperty(probe, "collidable", True)
            try:
                raw = self.sim.checkCollision(
                    self.robot_collision_entities[robot_id],
                    probe,
                )
                result, handles = _parse_collision_result(raw)
            finally:
                remove_objects([probe], False)
            verified = (
                result == 1
                and probe in handles
                and bool(base_handles.intersection(handles))
                and not bool(
                    self.robot_excluded_shape_handles[robot_id].intersection(
                        handles
                    )
                )
            )
            if not verified:
                raise DynamicCoppeliaError(
                    f"{robot_id} mobile-base collision collection failed its "
                    "positive self-test"
                )
            self.robot_collision_self_tests.append(
                {
                    "robot_id": robot_id,
                    "collection_handle": self.robot_collision_entities[robot_id],
                    "probe_handle": probe,
                    "colliding_object_handles": handles,
                    "verified": True,
                }
            )

    def _step_physics(self) -> None:
        while True:
            try:
                self.client.step()
            except Exception as exc:
                if str(exc).strip() != _REMOTE_STEP_EXECUTED_HANDSHAKE_ERROR:
                    raise
                outcome, observed_time = self._classify_remote_step_handshake(exc)
                expected_time = (self.physics_steps + 1) / self.config.control_hz
                if outcome == "completed":
                    if (
                        self.remote_step_handshake_reconciliations
                        >= self.config.maximum_remote_step_handshake_reconciliations
                    ):
                        raise DynamicCoppeliaError(
                            "remote step handshake reconciliation limit was exceeded"
                        ) from exc
                    self.runtime_events.append(
                        {
                            "timestamp_s": observed_time,
                            "event": "remote_step_handshake_reconciled",
                            "physics_step": self.physics_steps + 1,
                            "observed_simulation_time_s": observed_time,
                            "expected_simulation_time_s": expected_time,
                            "error": _REMOTE_STEP_EXECUTED_HANDSHAKE_ERROR,
                        }
                    )
                    self.remote_step_handshake_reconciliations += 1
                    break
                if (
                    self.remote_step_handshake_retries
                    >= self.config.maximum_remote_step_handshake_retries
                ):
                    raise DynamicCoppeliaError(
                        "remote step handshake failed without exactly one measured "
                        "simulator step and retry limit was exceeded"
                    ) from exc
                self.remote_step_handshake_retries += 1
                self.runtime_events.append(
                    {
                        "timestamp_s": observed_time,
                        "event": "remote_step_handshake_retried",
                        "physics_step": self.physics_steps + 1,
                        "retry_index": self.remote_step_handshake_retries,
                        "observed_simulation_time_s": observed_time,
                        "previous_simulation_time_s": (
                            self.physics_steps / self.config.control_hz
                        ),
                        "expected_simulation_time_s": expected_time,
                        "error": _REMOTE_STEP_EXECUTED_HANDSHAKE_ERROR,
                    }
                )
                continue
            break
        self.physics_steps += 1
        self._query_physical_collisions()
        self._update_command_responses()

    def _classify_remote_step_handshake(
        self,
        exc: Exception,
    ) -> tuple[Literal["completed", "not_started"], float]:
        get_simulation_time = getattr(self.sim, "getSimulationTime", None)
        if not callable(get_simulation_time):
            raise DynamicCoppeliaError(
                "remote step handshake failed without measurable simulator time"
            ) from exc
        try:
            observed_time = float(get_simulation_time())
        except Exception as time_exc:
            raise DynamicCoppeliaError(
                "remote step handshake failed without measurable simulator time"
            ) from time_exc
        previous_time = self.physics_steps / self.config.control_hz
        expected_time = (self.physics_steps + 1) / self.config.control_hz
        tolerance = max(1e-9, (1.0 / self.config.control_hz) * 1e-6)
        if not math.isfinite(observed_time):
            raise DynamicCoppeliaError(
                "remote step handshake failed without exactly one measured "
                "simulator step"
            ) from exc
        if math.isclose(
            observed_time,
            expected_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return "completed", observed_time
        if math.isclose(
            observed_time,
            previous_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return "not_started", observed_time
        raise DynamicCoppeliaError(
            "remote step handshake failed without exactly one measured simulator step"
        ) from exc

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
            if category == "robot_world":
                (
                    first_label,
                    second_label,
                    category,
                    robot_ids,
                ) = self._resolve_robot_world_collision(object_handles)
            self._validate_collision_object_handles(
                first_label=first_label,
                second_label=second_label,
                robot_ids=robot_ids,
                object_handles=object_handles,
            )
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
                "pair_category_counts": dict(self.collision_pair_category_counts),
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

    def _resolve_robot_world_collision(
        self,
        object_handles: list[int],
    ) -> tuple[str, str, str, tuple[str, ...]]:
        if len(object_handles) != 2:
            raise DynamicCoppeliaError(
                "aggregate collision query did not identify one exact pair"
            )
        excluded = set().union(*self.robot_excluded_shape_handles.values())
        if set(object_handles).intersection(excluded):
            raise DynamicCoppeliaError(
                "Coppelia collision evidence referenced excluded YouBot geometry"
            )
        base_owner = {
            handle: robot_id
            for robot_id, handles in self.robot_base_shape_handles.items()
            for handle in handles
        }
        robot_matches = [
            (handle, base_owner[handle])
            for handle in object_handles
            if handle in base_owner
        ]
        if len(robot_matches) != 1:
            raise DynamicCoppeliaError(
                "aggregate world collision is not bound to exactly one mobile base"
            )
        robot_handle, robot_id = robot_matches[0]
        other_handle = next(
            handle for handle in object_handles if handle != robot_handle
        )
        module_by_handle = {
            handle: module_id for module_id, handle in self.module_handles.items()
        }
        obstacle_by_handle = {
            handle: obstacle_id for obstacle_id, handle in self.obstacle_handles.items()
        }
        if other_handle in module_by_handle:
            return (
                f"robot:{robot_id}",
                f"module:{module_by_handle[other_handle]}",
                "robot_module",
                (robot_id,),
            )
        if other_handle in obstacle_by_handle:
            return (
                f"robot:{robot_id}",
                f"obstacle:{obstacle_by_handle[other_handle]}",
                "robot_obstacle",
                (robot_id,),
            )
        raise DynamicCoppeliaError(
            "aggregate world collision referenced an unattested object"
        )

    def _validate_collision_object_handles(
        self,
        *,
        first_label: str,
        second_label: str,
        robot_ids: tuple[str, ...],
        object_handles: list[int],
    ) -> None:
        if len(object_handles) != 2:
            raise DynamicCoppeliaError(
                "Coppelia did not identify the exact colliding object pair for "
                f"{first_label} versus {second_label}"
            )
        reported = set(object_handles)
        excluded = set().union(*self.robot_excluded_shape_handles.values())
        if reported.intersection(excluded):
            raise DynamicCoppeliaError(
                "Coppelia collision evidence referenced excluded YouBot geometry"
            )
        if any(
            not reported.intersection(self.robot_base_shape_handles[robot_id])
            for robot_id in robot_ids
        ):
            raise DynamicCoppeliaError(
                "Coppelia collision evidence is not bound to every implicated "
                "mobile base"
            )

    def _update_command_responses(self) -> None:
        for robot_id, (commanded_at_s, anchor) in sorted(self.command_response_anchor.items()):
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
                and displacement >= self.config.command_response_min_displacement_m
            ):
                self.command_response_observed_at_s[robot_id] = self.simulation_time_s
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
            raise DynamicCoppeliaError("formation targets do not match the assigned robot team")
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
                self.formation_spacing_errors_m.append(abs(measured_spacing - planned_spacing))

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
            raise DynamicCoppeliaError("Coppelia failed to serialize the evidence scene") from exc
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
        raise DynamicCoppeliaError("serialized scene is missing the native Coppelia VREP signature")


def _classify_youbot_script(alias: str, source: str) -> YouBotScriptRole:
    compact_source = "".join(source.casefold().split())
    compact_alias = alias.casefold()
    writes_target_velocity = "setjointtargetvelocity" in compact_source
    writes_target_position = "setjointtargetposition" in compact_source
    references_wheels = "wheeljoints" in compact_source or "rollingjoint_" in compact_source
    references_arm_or_gripper = any(
        marker in compact_source or marker in compact_alias
        for marker in ("armjoint", "gripperjoint", "gripper")
    )
    if writes_target_velocity and references_wheels:
        return "wheel_command_writer"
    if (writes_target_velocity or writes_target_position) and references_arm_or_gripper:
        return "arm_gripper_writer"
    passive_omniwheel_maintenance = (
        "setobjectorientation" in compact_source
        and "slippingjoint_" in compact_source
        and not writes_target_velocity
        and not writes_target_position
    )
    if passive_omniwheel_maintenance:
        return "passive_omniwheel_maintenance"
    raise DynamicCoppeliaError(f"bundled YouBot script {alias!r} has an unrecognized control role")


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
    raise DynamicCoppeliaError(f"Coppelia returned a malformed collision-query result: {raw!r}")


def _shape_xy_footprint_radius_m(
    sim: Any,
    *,
    shape_handle: int,
    robot_handle: int,
) -> float:
    raw_bounds = sim.getShapeBB(shape_handle)
    if (
        not isinstance(raw_bounds, (list, tuple))
        or len(raw_bounds) != 2
        or not isinstance(raw_bounds[0], (list, tuple))
        or len(raw_bounds[0]) != 3
        or not isinstance(raw_bounds[1], (list, tuple))
        or len(raw_bounds[1]) != 7
    ):
        raise DynamicCoppeliaError(
            f"shape {shape_handle} returned malformed bounding-box evidence"
        )
    dimensions = [float(value) for value in raw_bounds[0]]
    bounding_pose = [float(value) for value in raw_bounds[1]]
    object_matrix = [
        float(value)
        for value in sim.getObjectMatrix(shape_handle, robot_handle)
    ]
    if (
        len(object_matrix) != 12
        or any(not math.isfinite(value) for value in [*dimensions, *bounding_pose, *object_matrix])
        or any(value <= 0 for value in dimensions)
    ):
        raise DynamicCoppeliaError(
            f"shape {shape_handle} returned invalid footprint geometry"
        )
    quaternion = bounding_pose[3:]
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    if quaternion_norm <= 1e-12:
        raise DynamicCoppeliaError(
            f"shape {shape_handle} returned a zero bounding-box quaternion"
        )
    qx, qy, qz, qw = (value / quaternion_norm for value in quaternion)
    rotation = (
        (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )
    maximum_radius = 0.0
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                local_corner = (
                    x_sign * dimensions[0] / 2,
                    y_sign * dimensions[1] / 2,
                    z_sign * dimensions[2] / 2,
                )

                shape_corner = [
                    bounding_pose[axis]
                    + sum(rotation[axis][item] * local_corner[item] for item in range(3))
                    for axis in range(3)
                ]
                robot_corner = [
                    object_matrix[axis * 4 + 3]
                    + sum(
                        object_matrix[axis * 4 + item] * shape_corner[item]
                        for item in range(3)
                    )
                    for axis in range(3)
                ]
                maximum_radius = max(
                    maximum_radius,
                    math.hypot(robot_corner[0], robot_corner[1]),
                )
    return maximum_radius


def _minimum_point_separation(
    points: Mapping[str, Vec2],
    *,
    relevant_ids: set[str] | None = None,
) -> tuple[float, str, str] | None:
    identifiers = sorted(points)
    closest: tuple[float, str, str] | None = None
    for index, left_id in enumerate(identifiers):
        left = points[left_id]
        for right_id in identifiers[index + 1 :]:
            if (
                relevant_ids is not None
                and left_id not in relevant_ids
                and right_id not in relevant_ids
            ):
                continue
            right = points[right_id]
            candidate = (
                math.hypot(left.x - right.x, left.y - right.y),
                left_id,
                right_id,
            )
            if closest is None or candidate < closest:
                closest = candidate
    return closest


def _minimum_pose_separation(
    poses: Mapping[str, Pose3D],
    *,
    relevant_ids: set[str] | None = None,
) -> tuple[float, str, str] | None:
    return _minimum_point_separation(
        {
            robot_id: Vec2(
                x=pose.position.x,
                y=pose.position.y,
            )
            for robot_id, pose in poses.items()
        },
        relevant_ids=relevant_ids,
    )


def _minimum_independent_interval_separation(
    starts: Mapping[str, Vec2],
    ends: Mapping[str, Vec2],
    *,
    relevant_ids: set[str] | None = None,
) -> tuple[float, float, float, str, str] | None:
    identifiers = sorted(starts)
    if set(identifiers) != set(ends):
        raise DynamicCoppeliaError("synchronized interval endpoints have different robot IDs")
    closest: tuple[float, float, float, str, str] | None = None
    for index, left_id in enumerate(identifiers):
        left_start = starts[left_id]
        left_end = ends[left_id]
        for right_id in identifiers[index + 1 :]:
            if (
                relevant_ids is not None
                and left_id not in relevant_ids
                and right_id not in relevant_ids
            ):
                continue
            right_start = starts[right_id]
            right_end = ends[right_id]
            distance, left_fraction, right_fraction = _segment_to_segment_distance(
                left_start,
                left_end,
                right_start,
                right_end,
            )
            candidate = (
                distance,
                left_fraction,
                right_fraction,
                left_id,
                right_id,
            )
            if closest is None or candidate < closest:
                closest = candidate
    return closest


def _segment_to_segment_distance(
    left_start: Vec2,
    left_end: Vec2,
    right_start: Vec2,
    right_end: Vec2,
) -> tuple[float, float, float]:
    left_dx = left_end.x - left_start.x
    left_dy = left_end.y - left_start.y
    right_dx = right_end.x - right_start.x
    right_dy = right_end.y - right_start.y
    between_x = right_start.x - left_start.x
    between_y = right_start.y - left_start.y
    denominator = left_dx * right_dy - left_dy * right_dx
    if abs(denominator) > 1e-12:
        left_fraction = (between_x * right_dy - between_y * right_dx) / denominator
        right_fraction = (between_x * left_dy - between_y * left_dx) / denominator
        if -1e-12 <= left_fraction <= 1.0 + 1e-12 and -1e-12 <= right_fraction <= 1.0 + 1e-12:
            return (
                0.0,
                _clamp(left_fraction, 0.0, 1.0),
                _clamp(right_fraction, 0.0, 1.0),
            )

    candidates: list[tuple[float, float, float]] = []
    distance, right_fraction = _point_to_segment_distance(
        left_start,
        right_start,
        right_end,
    )
    candidates.append((distance, 0.0, right_fraction))
    distance, right_fraction = _point_to_segment_distance(
        left_end,
        right_start,
        right_end,
    )
    candidates.append((distance, 1.0, right_fraction))
    distance, left_fraction = _point_to_segment_distance(
        right_start,
        left_start,
        left_end,
    )
    candidates.append((distance, left_fraction, 0.0))
    distance, left_fraction = _point_to_segment_distance(
        right_end,
        left_start,
        left_end,
    )
    candidates.append((distance, left_fraction, 1.0))
    return min(candidates)


def _point_to_segment_distance(
    point: Vec2,
    start: Vec2,
    end: Vec2,
) -> tuple[float, float]:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx**2 + dy**2
    if length_squared <= 1e-18:
        fraction = 0.0
    else:
        fraction = _clamp(
            ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared,
            0.0,
            1.0,
        )
    closest_x = start.x + fraction * dx
    closest_y = start.y + fraction * dy
    return math.hypot(point.x - closest_x, point.y - closest_y), fraction


def _relative_formation_error(
    robot_ids: list[str],
    measured: Mapping[str, Pose3D],
    targets: Mapping[str, Vec2],
) -> float:
    if len(robot_ids) != 2:
        raise DynamicCoppeliaError("relative formation enforcement requires exactly two robots")
    first_id, second_id = robot_ids
    measured_delta = Vec2(
        x=(measured[second_id].position.x - measured[first_id].position.x),
        y=(measured[second_id].position.y - measured[first_id].position.y),
    )
    target_delta = Vec2(
        x=targets[second_id].x - targets[first_id].x,
        y=targets[second_id].y - targets[first_id].y,
    )
    return math.hypot(
        measured_delta.x - target_delta.x,
        measured_delta.y - target_delta.y,
    )


def _robot_command_is_zero(command: RobotCommand) -> bool:
    return (
        abs(command.linear_velocity_mps) <= 1e-9
        and abs(command.angular_velocity_rps) <= 1e-9
        and all(abs(value) <= 1e-9 for value in command.wheel_target_velocity_rad_s)
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
