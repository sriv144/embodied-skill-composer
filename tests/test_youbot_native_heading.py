"""Native-frame control regression tests; all physics here is a local fake."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
    DynamicCoppeliaExecutor,
    _shape_xy_footprint_radius_m,
    world_error_to_youbot_body,
    world_error_to_youbot_body_abg,
)
from embodied_skill_composer.construction.models import BuildPlan, Vec2
from embodied_skill_composer.construction.runtime import load_house_design
from test_construction_dynamic_coppelia import FakeDynamicClient, FakeDynamicSim


def _rotate_axis(
    axis: tuple[float, float, float],
    abg: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Independent Rz→Ry→Rx vector rotation, not the production projection."""
    alpha, beta, gamma = abg
    x, y, z = axis
    x, y = (
        math.cos(gamma) * x - math.sin(gamma) * y,
        math.sin(gamma) * x + math.cos(gamma) * y,
    )
    x, z = (
        math.cos(beta) * x + math.sin(beta) * z,
        -math.sin(beta) * x + math.cos(beta) * z,
    )
    y, z = (
        math.cos(alpha) * y - math.sin(alpha) * z,
        math.sin(alpha) * y + math.cos(alpha) * z,
    )
    return x, y, z


@pytest.mark.parametrize("beta_degrees", [-179, -90, -77.052, -45, 0, 45, 90, 179])
def test_native_projection_matches_independent_rotation_not_gamma(
    beta_degrees: float,
) -> None:
    abg = (-math.pi / 2, math.radians(beta_degrees), -math.pi / 2)
    forward_axis = _rotate_axis((0, 0, 1), abg)
    lateral_axis = _rotate_axis((0, 1, 0), abg)
    for dx, dy in ((1, 0), (0, 1), (-0.4, 0.7)):
        assert world_error_to_youbot_body_abg(dx, dy, *abg) == pytest.approx(
            (
                dx * forward_axis[0] + dy * forward_axis[1],
                dx * lateral_axis[0] + dy * lateral_axis[1],
            )
        )


def test_recorded_robot3_frame_reverses_the_erroneous_forward_correction() -> None:
    abg = tuple(
        math.radians(value)
        for value in (-90.05415032290226, -77.0517454262027, -90.04778160200463)
    )
    dx, dy = -0.044383722467422615, -0.027472742936272
    forward, lateral = world_error_to_youbot_body_abg(dx, dy, *abg)
    assert forward == pytest.approx(0.03709932814522485)
    assert lateral == pytest.approx(-0.036719289943970015)
    old_forward, _ = world_error_to_youbot_body(dx, dy, abg[2])
    assert old_forward < 0 < forward


@pytest.mark.parametrize(
    "abg_degrees",
    [(-89, -77, -91), (-91, 89.99, -89), (-90, -90, -90)],
)
def test_projection_uses_full_abg_even_with_small_chassis_tilt(
    abg_degrees: tuple[float, float, float],
) -> None:
    abg = tuple(math.radians(value) for value in abg_degrees)
    forward_axis = _rotate_axis((0, 0, 1), abg)
    lateral_axis = _rotate_axis((0, 1, 0), abg)
    dx, dy = 0.2, -0.35
    assert world_error_to_youbot_body_abg(dx, dy, *abg) == pytest.approx(
        (
            dx * forward_axis[0] + dy * forward_axis[1],
            dx * lateral_axis[0] + dy * lateral_axis[1],
        )
    )


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_world_errors_or_angles_fail_closed(index: int, invalid: float) -> None:
    values = [0.2, -0.3, -math.pi / 2, 0.0, -math.pi / 2]
    values[index] = invalid
    with pytest.raises(DynamicCoppeliaError, match="inputs must be finite"):
        world_error_to_youbot_body_abg(*values)


@pytest.mark.parametrize("abg", [(0, 0, 0), (0, math.pi / 4, math.pi / 2)])
def test_degenerate_native_ground_axes_fail_closed(abg: tuple[float, float, float]) -> None:
    with pytest.raises(DynamicCoppeliaError, match="degenerate planar axes"):
        world_error_to_youbot_body_abg(0.2, -0.3, *abg)


def test_finite_inputs_with_overflowing_projection_fail_closed() -> None:
    with pytest.raises(DynamicCoppeliaError, match="projection is nonfinite"):
        world_error_to_youbot_body_abg(1.7e308, 1.7e308, -math.pi / 2, math.pi / 4, -math.pi / 2)


def test_conservative_ground_footprint_includes_long_forward_and_vertical_dimensions() -> None:
    sim = FakeDynamicSim()
    handle = sim.loadModel("local fixture, never a live simulator")
    measured = _shape_xy_footprint_radius_m(sim, shape_handle=handle, robot_handle=handle)
    assert measured == pytest.approx(math.hypot(0.07 / 2, 0.15 / 2, 0.24 / 2))
    old_local_xy_radius = math.hypot(0.07 / 2, 0.15 / 2)
    assert old_local_xy_radius < 0.12 < measured < 0.16
    assert measured > math.hypot(0.15 / 2, 0.24 / 2)


def test_corner_sphere_bounds_world_ground_projection_even_when_chassis_tilts() -> None:
    sim = FakeDynamicSim()
    root = sim.loadModel("local fixture, never a live simulator")
    bound = _shape_xy_footprint_radius_m(sim, shape_handle=root, robot_handle=root)
    for abg in ((0, 0, 0), (1.2, -0.9, 2.7), (-math.pi / 2, -1.34, -math.pi / 2)):
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                for z_sign in (-1, 1):
                    corner = _rotate_axis(
                        (x_sign * 0.07 / 2, y_sign * 0.15 / 2, z_sign * 0.24 / 2),
                        abg,
                    )
                    assert math.hypot(corner[0], corner[1]) <= bound + 1e-12


def test_native_footprint_projects_shape_transform_and_bounding_box_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = FakeDynamicSim()
    root = sim.loadModel("local fixture, never a live simulator")
    wheel = next(handle for handle, (owner, _) in sim.wheel_owner.items() if owner == root)
    sim.shape_dimensions[wheel] = [0.02, 0.04, 0.06]
    sim.object_matrices[(wheel, root)] = [
        1, 0, 0, 100,  # Vertical extent contributes to the orientation-independent bound.
        0, 1, 0, 0.2,
        0, 0, 1, 0.3,
    ]
    monkeypatch.setattr(
        sim,
        "getShapeBB",
        lambda _handle: [
            sim.shape_dimensions[wheel],
            [0, 0.01, 0.02, math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
        ],
    )
    assert _shape_xy_footprint_radius_m(
        sim, shape_handle=wheel, robot_handle=root
    ) == pytest.approx(math.hypot(100.01, 0.24, 0.34))


@pytest.fixture(scope="module")
def native_plan() -> BuildPlan:
    workspace = Path(__file__).resolve().parents[1]
    return compile_house_design(load_house_design(workspace / "configs/construction/cottage_v1.yaml"))


def _executor(plan: BuildPlan) -> tuple[DynamicCoppeliaExecutor, FakeDynamicClient]:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(
            settle_steps=0,
            effective_wheel_radius_m=0.2,
            planned_robot_footprint_radius_m=0.16,
            waypoint_tolerance_m=0.015,
            max_steps_per_waypoint=1000,
        ),
        client_factory=lambda _config: fake,
    )
    executor.connect()
    return executor, fake


@pytest.mark.parametrize("beta_degrees", [-170, -90, -77.052, -30, 0, 60, 90, 170])
@pytest.mark.parametrize("synchronized", [False, True])
def test_closed_loop_native_heading_reaches_routes_without_robot_pose_writes(
    native_plan: BuildPlan,
    beta_degrees: float,
    synchronized: bool,
) -> None:
    executor, fake = _executor(native_plan)
    ids = [robot.robot_id for robot in native_plan.robots[:2 if synchronized else 1]]
    for handle in executor.robot_handles.values():
        fake.sim.orientations[handle] = [-math.pi / 2, math.radians(beta_degrees), -math.pi / 2]
    routes = {
        robot_id: [
            Vec2(x=fake.sim.positions[executor.robot_handles[robot_id]][0] + 0.2,
                 y=fake.sim.positions[executor.robot_handles[robot_id]][1] + 0.1),
        ]
        for robot_id in ids
    }
    executor.start()
    if synchronized:
        executor.follow_synchronized_routes(routes)
    else:
        executor.follow_routes(routes)
    executor.stop()
    for robot_id in ids:
        position = fake.sim.positions[executor.robot_handles[robot_id]]
        assert math.hypot(position[0] - routes[robot_id][-1].x,
                          position[1] - routes[robot_id][-1].y) <= 0.015
    assert executor.post_start_robot_pose_writes == 0
    assert not [started for started, handle, _ in fake.sim.position_writes
                if started and handle in executor.robot_handles.values()]


def test_native_heading_corrects_idle_base_position_hold_after_measured_drift(
    native_plan: BuildPlan,
) -> None:
    executor, fake = _executor(native_plan)
    for handle in executor.robot_handles.values():
        fake.sim.orientations[handle] = [-math.pi / 2, math.radians(-77), -math.pi / 2]
    active_ids = [robot.robot_id for robot in native_plan.robots[:2]]
    idle_id = native_plan.robots[2].robot_id
    idle_handle = executor.robot_handles[idle_id]
    idle_start = list(fake.sim.positions[idle_handle])
    original_integrate = fake.sim.integrate
    perturbed = False

    def integrate_with_drift(dt: float, wheel_radius_m: float = 0.2) -> None:
        nonlocal perturbed
        original_integrate(dt, wheel_radius_m)
        if not perturbed:
            # Synthetic measured disturbance in physics, not a robot-pose API write.
            fake.sim.positions[idle_handle][0] += 0.04
            fake.sim.positions[idle_handle][1] -= 0.03
            perturbed = True

    fake.sim.integrate = integrate_with_drift
    routes = {
        robot_id: [Vec2(x=fake.sim.positions[executor.robot_handles[robot_id]][0] + 0.3,
                       y=fake.sim.positions[executor.robot_handles[robot_id]][1])]
        for robot_id in active_ids
    }
    executor.start()
    executor.follow_synchronized_routes(routes)
    executor.stop()
    idle_final = fake.sim.positions[idle_handle]
    assert math.hypot(idle_final[0] - idle_start[0], idle_final[1] - idle_start[1]) <= 0.015
    assert any(command.robot_id == idle_id and command.source == "path_follower"
               and command.linear_velocity_mps > 0 for command in executor.commands)
    assert executor.post_start_robot_pose_writes == 0


def test_native_heading_tracks_yaw_drift_while_gamma_stays_fixed(native_plan: BuildPlan) -> None:
    executor, fake = _executor(native_plan)
    robot_id = native_plan.robots[0].robot_id
    handle = executor.robot_handles[robot_id]
    start = list(fake.sim.positions[handle])
    target = Vec2(x=start[0] + 0.5, y=start[1] + 0.2)
    original_integrate = fake.sim.integrate
    terminal_beta = math.radians(-77.052)

    def integrate_with_native_yaw_drift(dt: float, wheel_radius_m: float = 0.2) -> None:
        original_integrate(dt, wheel_radius_m)
        fake.sim.orientations[handle][1] = max(
            terminal_beta,
            fake.sim.orientations[handle][1] - math.radians(2),
        )

    fake.sim.integrate = integrate_with_native_yaw_drift
    executor.start()
    executor.follow_routes({robot_id: [target]})
    executor.stop()
    final = fake.sim.positions[handle]
    assert math.hypot(final[0] - target.x, final[1] - target.y) <= 0.015
    assert fake.sim.orientations[handle][1] == pytest.approx(terminal_beta)
    assert fake.sim.orientations[handle][2] == -math.pi / 2
    assert executor.post_start_robot_pose_writes == 0


def test_telemetry_retains_raw_abg_in_legacy_rotation_wire_field(native_plan: BuildPlan) -> None:
    executor, fake = _executor(native_plan)
    robot_id = native_plan.robots[0].robot_id
    raw_degrees = (-90.05415032290226, -77.0517454262027, -90.04778160200463)
    fake.sim.orientations[executor.robot_handles[robot_id]] = [math.radians(value) for value in raw_degrees]
    rotation = executor.sample_telemetry(robot_id).measured_pose.rotation_rpy_degrees
    assert (rotation.x, rotation.y, rotation.z) == pytest.approx(raw_degrees)
    assert executor.post_start_robot_pose_writes == 0
