from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MethodType
from typing import Callable

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
    DynamicCoppeliaExecutor,
    RobotCommandSource,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.models import BuildPlan, Pose3D, Vec2, Vec3
from embodied_skill_composer.construction.runtime import load_house_design


WORKSPACE = Path(__file__).resolve().parents[1]


class _LogicalTransportSim:
    def __init__(self) -> None:
        self.positions: dict[int, list[float]] = {}
        self.orientations: dict[int, list[float]] = {}
        self.velocities: dict[int, tuple[list[float], list[float]]] = {}
        self.parents: dict[int, int] = {}
        self.physics_step = 0
        self.velocity_decay = 0.0
        self.position_writes: list[tuple[int, int, list[float]]] = []
        self.parent_writes: list[tuple[int, int, int]] = []
        self.after_step: Callable[[DynamicCoppeliaExecutor], None] | None = None

    def getObjectPosition(self, handle: int) -> list[float]:
        return list(self.positions[handle])

    def setObjectPosition(self, handle: int, position: list[float]) -> None:
        previous = self.positions.get(handle, list(position))
        current = [float(value) for value in position]
        self.positions[handle] = current
        self.position_writes.append((self.physics_step, handle, list(current)))
        delta = [current[index] - previous[index] for index in range(3)]
        for child, parent in list(self.parents.items()):
            if parent != handle:
                continue
            child_position = self.positions[child]
            self.setObjectPosition(
                child,
                [child_position[index] + delta[index] for index in range(3)],
            )

    def getObjectOrientation(self, handle: int) -> list[float]:
        return list(self.orientations.get(handle, [0.0, 0.0, 0.0]))

    def setObjectOrientation(self, handle: int, orientation: list[float]) -> None:
        self.orientations[handle] = [float(value) for value in orientation]

    def setObjectParent(self, handle: int, parent: int, _keep: bool) -> None:
        self.parents[handle] = parent
        self.parent_writes.append((self.physics_step, handle, parent))

    def getObjectVelocity(
        self,
        handle: int,
    ) -> tuple[list[float], list[float]]:
        return self.velocities.get(
            handle,
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        )


@pytest.fixture(scope="module")
def plan() -> BuildPlan:
    design = load_house_design(WORKSPACE / "configs" / "construction" / "cottage_v1.yaml")
    return compile_house_design(design)


def _executor_for_first_module(
    plan: BuildPlan,
) -> tuple[DynamicCoppeliaExecutor, _LogicalTransportSim, str, list[str]]:
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(
            formation_tolerance_m=0.5,
            install_tolerance_m=0.3,
            settled_linear_speed_mps=0.02,
            settled_angular_speed_rps=0.05,
            logical_transition_settle_max_steps=12,
            logical_transition_settle_consecutive_samples=3,
        ),
    )
    sim = _LogicalTransportSim()
    executor.sim = sim
    executor.physics_steps = 0
    executor.root_handle = 1
    sim.positions[1] = [0.0, 0.0, 0.0]
    sim.orientations[1] = [0.0, 0.0, 0.0]

    module = plan.modules[0]
    module_handle = 20
    carrier_handle = 21
    executor.module_handles = {module.module_id: module_handle}
    executor.payload_carriers = {module.module_id: carrier_handle}
    staging = module.staging_pose
    sim.positions[module_handle] = [
        staging.position.x,
        staging.position.y,
        staging.position.z,
    ]
    sim.orientations[module_handle] = [
        math.radians(staging.rotation_rpy_degrees.x),
        math.radians(staging.rotation_rpy_degrees.y),
        math.radians(staging.rotation_rpy_degrees.z),
    ]
    sim.positions[carrier_handle] = [0.0, 0.0, 0.0]
    sim.orientations[carrier_handle] = [0.0, 0.0, 0.0]

    robot_ids = ["robot_1", "robot_2"]
    executor.robot_handles = {"robot_1": 10, "robot_2": 11}
    for index, robot_id in enumerate(robot_ids):
        handle = executor.robot_handles[robot_id]
        sim.positions[handle] = [
            staging.position.x,
            staging.position.y + (-0.35 if index == 0 else 0.35),
            0.12,
        ]
        sim.orientations[handle] = [0.0, 0.0, 0.0]

    def command_body_velocity(
        _executor: DynamicCoppeliaExecutor,
        robot_id: str,
        forward_velocity: float,
        lateral_velocity: float,
        angular_velocity: float,
        *,
        source: RobotCommandSource = "path_follower",
        target: Vec2 | None = None,
    ) -> RobotCommand:
        zero = all(
            abs(value) <= 1e-12
            for value in (
                forward_velocity,
                lateral_velocity,
                angular_velocity,
            )
        )
        command = RobotCommand(
            timestamp_s=executor.simulation_time_s,
            robot_id=robot_id,
            linear_velocity_mps=math.hypot(
                forward_velocity,
                lateral_velocity,
            ),
            angular_velocity_rps=angular_velocity,
            wheel_target_velocity_rad_s=((0.0, 0.0, 0.0, 0.0) if zero else (1.0, 1.0, 1.0, 1.0)),
            target_position=target,
            source=source,
        )
        executor.commands.append(command)
        return command

    def step_physics(_executor: DynamicCoppeliaExecutor) -> None:
        for handle, (linear, angular) in list(sim.velocities.items()):
            position = sim.positions[handle]
            sim.positions[handle] = [position[index] + linear[index] * 0.1 for index in range(3)]
            sim.velocities[handle] = (
                [value * sim.velocity_decay for value in linear],
                [value * sim.velocity_decay for value in angular],
            )
        executor.physics_steps += 1
        sim.physics_step = executor.physics_steps
        if sim.after_step is not None:
            sim.after_step(executor)

    executor.command_body_velocity = MethodType(  # type: ignore[method-assign]
        command_body_velocity,
        executor,
    )
    executor._step_physics = MethodType(  # type: ignore[method-assign]
        step_physics,
        executor,
    )
    return executor, sim, module.module_id, robot_ids


def _targets(center: Vec3, robot_ids: list[str]) -> dict[str, Vec2]:
    return {
        robot_id: Vec2(
            x=center.x,
            y=center.y + (-0.35 if index == 0 else 0.35),
        )
        for index, robot_id in enumerate(robot_ids)
    }


def _zero_command(executor: DynamicCoppeliaExecutor, robot_id: str) -> RobotCommand:
    return RobotCommand(
        timestamp_s=executor.simulation_time_s,
        robot_id=robot_id,
        linear_velocity_mps=0.0,
        angular_velocity_rps=0.0,
        wheel_target_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
        target_position=None,
        source="formation_hold",
    )


def _move_team_to_target(
    executor: DynamicCoppeliaExecutor,
    sim: _LogicalTransportSim,
    robot_ids: list[str],
    target: Vec3,
) -> None:
    for index, robot_id in enumerate(robot_ids):
        sim.setObjectPosition(
            executor.robot_handles[robot_id],
            [
                target.x,
                target.y + (-0.35 if index == 0 else 0.35),
                0.12,
            ],
        )
    executor.physics_steps += 1
    sim.physics_step = executor.physics_steps
    executor._update_logical_payloads()


def test_logical_transport_lifts_and_holds_exact_height(plan: BuildPlan) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    transport_height_m = 1.25

    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=transport_height_m,
    )

    carrier = executor.payload_carriers[module_id]
    assert sim.getObjectPosition(carrier)[2] == pytest.approx(transport_height_m)
    assert sim.getObjectPosition(executor.module_handles[module_id])[2] == pytest.approx(
        transport_height_m
    )
    offset = executor.logical_carrier_offsets[module_id]
    assert offset.x == pytest.approx(0.0)
    assert offset.y == pytest.approx(0.0)
    assert offset.z == pytest.approx(transport_height_m - 0.12)
    lifted = [
        event for event in executor.runtime_events if event["event"] == "logical_payload_lifted"
    ]
    assert len(lifted) == 1
    assert lifted[0]["scope"] == "logical_transport_only"
    assert lifted[0]["physical_lift_claimed"] is False
    assert lifted[0]["physical_descent_claimed"] is False
    assert lifted[0]["logical_centroid_offset"] == offset.model_dump(mode="json")
    proof = lifted[0]["team_zero_motion_proof"]
    assert proof["latest_commands_zero"] is True  # type: ignore[index]
    assert proof["measured_team_still"] is True  # type: ignore[index]
    assert proof["no_nonzero_team_command_after_hold"] is True  # type: ignore[index]
    assert proof["command_count_at_settle"] == len(executor.commands)  # type: ignore[index]
    assert proof["minimum_measured_team_to_scene_separation_m"] == pytest.approx(  # type: ignore[index]
        0.7
    )
    assert proof["minimum_measured_team_to_scene_pair"] == robot_ids  # type: ignore[index]
    settled = next(
        event for event in executor.runtime_events if event["event"] == "logical_transition_settled"
    )
    latest_settle_sample = settled["settle_samples"][-1]  # type: ignore[index]
    assert latest_settle_sample["physics_step"] == executor.physics_steps
    for robot_id in robot_ids:
        assert (
            latest_settle_sample["robots"][robot_id]["timestamp_s"]  # type: ignore[index]
            == latest_settle_sample["timestamp_s"]
        )

    _move_team_to_target(
        executor,
        sim,
        robot_ids,
        module.target_pose.position,
    )
    assert sim.getObjectPosition(carrier) == pytest.approx(
        [
            module.target_pose.position.x,
            module.target_pose.position.y,
            transport_height_m,
        ]
    )
    assert sim.getObjectPosition(executor.module_handles[module_id])[2] == pytest.approx(
        transport_height_m
    )


def test_logical_installation_snap_records_exact_provenance(plan: BuildPlan) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    transport_height_m = 1.25
    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=transport_height_m,
    )
    _move_team_to_target(
        executor,
        sim,
        robot_ids,
        module.target_pose.position,
    )
    executor.commands.extend(_zero_command(executor, robot_id) for robot_id in robot_ids)

    executor.install_logical_payload(
        module_id,
        assigned_targets=_targets(module.target_pose.position, robot_ids),
        contact_module_ids=["foundation_0_2", "foundation_0_1", "foundation_0_1"],
    )

    snaps = [
        event for event in executor.runtime_events if event["event"] == "logical_installation_snap"
    ]
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap["timestamp_s"] == pytest.approx(executor.simulation_time_s)
    assert snap["scope"] == "final_target_pose_only"
    assert snap["from_pose"]["position"]["z"] == pytest.approx(  # type: ignore[index]
        transport_height_m
    )
    exact_target = module.target_pose.model_dump(mode="json")
    assert snap["target_pose"] == exact_target
    encoded = json.dumps(
        exact_target,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert snap["target_pose_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert snap["contact_module_ids"] == [
        "foundation_0_1",
        "foundation_0_2",
    ]
    proof = snap["team_zero_motion_proof"]
    assert proof["robot_ids"] == robot_ids  # type: ignore[index]
    assert proof["latest_commands_zero"] is True  # type: ignore[index]
    assert proof["measured_team_still"] is True  # type: ignore[index]
    assert snap["physical_lift_claimed"] is False
    assert snap["physical_descent_claimed"] is False
    assert executor.installed_modules == {module_id}
    assert sim.parents[executor.module_handles[module_id]] == executor.root_handle
    assert sim.getObjectPosition(executor.module_handles[module_id]) == pytest.approx(
        [
            module.target_pose.position.x,
            module.target_pose.position.y,
            module.target_pose.position.z,
        ]
    )
    diagnostics = executor.diagnostics()
    assert diagnostics["logical_payload_lift_count"] == 1
    assert diagnostics["logical_installation_snap_count"] == 1


def test_install_settle_refreshes_carrier_from_final_measured_centroid(
    plan: BuildPlan,
) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=1.25,
    )
    _move_team_to_target(
        executor,
        sim,
        robot_ids,
        module.target_pose.position,
    )
    sim.velocity_decay = 0.25
    for robot_id in robot_ids:
        sim.velocities[executor.robot_handles[robot_id]] = (
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )

    executor.install_logical_payload(
        module_id,
        assigned_targets=_targets(module.target_pose.position, robot_ids),
        contact_module_ids=[],
    )

    snap = [
        event for event in executor.runtime_events if event["event"] == "logical_installation_snap"
    ][0]
    centroid_x = sum(
        sim.positions[executor.robot_handles[robot_id]][0] for robot_id in robot_ids
    ) / len(robot_ids)
    assert snap["from_pose"]["position"]["x"] == pytest.approx(  # type: ignore[index]
        centroid_x
    )


def test_logical_installation_snap_rejects_nonzero_command_after_hold(
    plan: BuildPlan,
) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=1.25,
    )
    _move_team_to_target(
        executor,
        sim,
        robot_ids,
        module.target_pose.position,
    )

    def inject_nonzero_after_hold(
        active_executor: DynamicCoppeliaExecutor,
    ) -> None:
        if not any(command.source == "path_follower" for command in active_executor.commands):
            active_executor.commands.append(
                RobotCommand(
                    timestamp_s=active_executor.simulation_time_s,
                    robot_id=robot_ids[0],
                    linear_velocity_mps=0.1,
                    angular_velocity_rps=0.0,
                    wheel_target_velocity_rad_s=(1.0, 1.0, 1.0, 1.0),
                    target_position=None,
                    source="path_follower",
                )
            )

    sim.after_step = inject_nonzero_after_hold
    with pytest.raises(
        DynamicCoppeliaError,
        match="nonzero team command after its settle hold",
    ):
        executor.install_logical_payload(
            module_id,
            assigned_targets=_targets(module.target_pose.position, robot_ids),
            contact_module_ids=[],
        )

    assert not any(
        event["event"] == "logical_installation_snap" for event in executor.runtime_events
    )
    assert module_id not in executor.installed_modules


def test_logical_lift_waits_for_residual_velocity_to_decay(
    plan: BuildPlan,
) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    sim.velocity_decay = 0.25
    for robot_id in robot_ids:
        sim.velocities[executor.robot_handles[robot_id]] = (
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )

    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=1.25,
    )

    carrier = executor.payload_carriers[module_id]
    carrier_writes = [
        physics_step for physics_step, handle, _position in sim.position_writes if handle == carrier
    ]
    assert carrier_writes
    assert min(carrier_writes) >= 4
    settled = [
        event
        for event in executor.runtime_events
        if event["event"] == "logical_transition_settled"
        and event["transition"] == "logical_payload_lift"
    ]
    assert len(settled) == 1
    proof = settled[0]["team_zero_motion_proof"]
    assert proof["physics_steps_elapsed"] >= 4  # type: ignore[index]


def test_logical_lift_fails_closed_on_settle_timeout(plan: BuildPlan) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    executor.config.logical_transition_settle_max_steps = 3
    sim.velocity_decay = 1.0
    for robot_id in robot_ids:
        sim.velocities[executor.robot_handles[robot_id]] = (
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )

    with pytest.raises(
        DynamicCoppeliaError,
        match="did not settle within 3 physics steps",
    ):
        executor.attach_logical_payload(
            module_id,
            robot_ids,
            assigned_targets=_targets(module.staging_pose.position, robot_ids),
            transport_height_m=1.25,
        )

    assert module_id not in executor.logical_attachments
    assert sim.parent_writes == []
    carrier = executor.payload_carriers[module_id]
    assert not any(handle == carrier for _physics_step, handle, _position in sim.position_writes)
    assert any(
        event["event"] == "logical_transition_settle_failed" and event["reason"] == "timeout"
        for event in executor.runtime_events
    )


@pytest.mark.parametrize("disabled_neighbor", [False, True])
def test_logical_lift_settle_rejects_coast_toward_scene_robot(
    plan: BuildPlan,
    disabled_neighbor: bool,
) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    neighbor_id = next(robot.robot_id for robot in plan.robots if robot.robot_id not in robot_ids)
    neighbor_handle = 12
    executor.robot_handles[neighbor_id] = neighbor_handle
    first_handle = executor.robot_handles[robot_ids[0]]
    first_position = sim.positions[first_handle]
    sim.positions[neighbor_handle] = [
        first_position[0] + 0.4,
        first_position[1],
        first_position[2],
    ]
    sim.orientations[neighbor_handle] = [0.0, 0.0, 0.0]
    sim.velocities[first_handle] = ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    executor.config.safety_distance_m = 0.3
    if disabled_neighbor:
        executor.disabled_robots.add(neighbor_id)

    with pytest.raises(
        DynamicCoppeliaError,
        match="breached scene-base safety separation",
    ):
        executor.attach_logical_payload(
            module_id,
            robot_ids,
            assigned_targets=_targets(module.staging_pose.position, robot_ids),
            transport_height_m=1.25,
        )

    assert executor.physics_steps == 1
    assert sim.parent_writes == []
    carrier = executor.payload_carriers[module_id]
    assert not any(handle == carrier for _step, handle, _position in sim.position_writes)
    assert module_id not in executor.logical_attachments
    assert not any(event["event"] == "logical_payload_lifted" for event in executor.runtime_events)
    failures = [
        event
        for event in executor.runtime_events
        if event["event"] == "logical_transition_settle_failed"
    ]
    assert failures[-1]["reason"] == "measured_base_separation"
    assert set(failures[-1]["robot_ids"]) == {robot_ids[0], neighbor_id}
    if disabled_neighbor:
        assert not any(command.robot_id == neighbor_id for command in executor.commands)


def test_logical_snap_settle_rejects_coast_toward_disabled_scene_robot(
    plan: BuildPlan,
) -> None:
    executor, sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)
    executor.attach_logical_payload(
        module_id,
        robot_ids,
        assigned_targets=_targets(module.staging_pose.position, robot_ids),
        transport_height_m=1.25,
    )
    _move_team_to_target(executor, sim, robot_ids, module.target_pose.position)
    neighbor_id = next(robot.robot_id for robot in plan.robots if robot.robot_id not in robot_ids)
    neighbor_handle = 12
    executor.robot_handles[neighbor_id] = neighbor_handle
    first_handle = executor.robot_handles[robot_ids[0]]
    first_position = sim.positions[first_handle]
    sim.positions[neighbor_handle] = [
        first_position[0] + 0.4,
        first_position[1],
        first_position[2],
    ]
    sim.orientations[neighbor_handle] = [0.0, 0.0, 0.0]
    sim.velocities[first_handle] = ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    executor.config.safety_distance_m = 0.3
    executor.disabled_robots.add(neighbor_id)
    parent_write_count = len(sim.parent_writes)

    with pytest.raises(
        DynamicCoppeliaError,
        match="breached scene-base safety separation",
    ):
        executor.install_logical_payload(
            module_id,
            assigned_targets=_targets(module.target_pose.position, robot_ids),
            contact_module_ids=[],
        )

    assert len(sim.parent_writes) == parent_write_count
    assert module_id in executor.logical_attachments
    assert module_id not in executor.installed_modules
    assert not any(
        event["event"] == "logical_installation_snap" for event in executor.runtime_events
    )
    assert not any(command.robot_id == neighbor_id for command in executor.commands)


def test_stop_proof_rejects_negative_signed_motion(plan: BuildPlan) -> None:
    executor, _sim, module_id, robot_ids = _executor_for_first_module(plan)
    executor.commands.extend(_zero_command(executor, robot_id) for robot_id in robot_ids)
    samples = {
        robot_id: RobotTelemetry(
            timestamp_s=executor.simulation_time_s,
            robot_id=robot_id,
            measured_pose=Pose3D(
                position=Vec3(x=0.0, y=0.0, z=0.12),
                rotation_rpy_degrees=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            linear_velocity_mps=-0.1 if index == 0 else 0.0,
            angular_velocity_rps=-0.1 if index == 1 else 0.0,
            battery_remaining_wh=900.0,
        )
        for index, robot_id in enumerate(robot_ids)
    }

    with pytest.raises(DynamicCoppeliaError, match="measured-still robot team"):
        executor._logical_installation_stop_proof(
            module_id,
            robot_ids,
            samples,
        )


def test_transport_height_below_staging_is_rejected(plan: BuildPlan) -> None:
    executor, _sim, module_id, robot_ids = _executor_for_first_module(plan)
    module = next(item for item in plan.modules if item.module_id == module_id)

    with pytest.raises(
        DynamicCoppeliaError,
        match="at least its staging height",
    ):
        executor.attach_logical_payload(
            module_id,
            robot_ids,
            assigned_targets=_targets(module.staging_pose.position, robot_ids),
            transport_height_m=module.staging_pose.position.z - 0.01,
        )

    assert module_id not in executor.logical_attachments
    assert not any(event["event"] == "logical_payload_lifted" for event in executor.runtime_events)
