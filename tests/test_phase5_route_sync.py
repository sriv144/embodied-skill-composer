from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Callable, cast

import pytest

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
    DynamicCoppeliaExecutor,
    RobotCommandSource,
)
from embodied_skill_composer.construction.models import Pose3D, Vec2, Vec3


@dataclass(frozen=True)
class _RecordedCommand:
    physics_step: int
    robot_id: str
    source: RobotCommandSource
    target: Vec2 | None
    moving: bool


@dataclass
class _RouteHarness:
    executor: DynamicCoppeliaExecutor
    positions: dict[str, Vec2]
    commands: list[_RecordedCommand]
    sample_calls: list[tuple[int, str]]
    after_step: Callable[[DynamicCoppeliaExecutor, dict[str, Vec2]], None] | None = None


def _route_harness(
    positions: dict[str, Vec2],
    *,
    speeds: dict[str, float] | None = None,
    safety_distance_m: float = 0.3,
    waypoint_tolerance_m: float = 0.02,
) -> _RouteHarness:
    executor = cast(
        DynamicCoppeliaExecutor,
        object.__new__(DynamicCoppeliaExecutor),
    )
    executor.config = DynamicCoppeliaConfig(
        settle_steps=0,
        safety_distance_m=safety_distance_m,
        waypoint_tolerance_m=waypoint_tolerance_m,
        formation_tolerance_m=0.2,
        max_steps_per_waypoint=1_000,
    )
    executor.disabled_robots = set()
    executor.robot_handles = {
        robot_id: index for index, robot_id in enumerate(sorted(positions), start=1)
    }
    executor.physics_steps = 0
    executor.runtime_events = []
    executor._latest_collision_robots = set()
    executor.proximity_safety_stops = 0
    executor.collision_stops = 0
    executor.synchronized_formation_errors_m = []
    executor.synchronized_route_separations_m = []
    executor.telemetry = []
    executor._route_telemetry_cache_step = None
    executor._route_telemetry_cache = {}

    current_positions = dict(positions)
    route_speeds = speeds or {robot_id: 0.2 for robot_id in positions}
    pending_targets: dict[str, Vec2 | None] = {robot_id: None for robot_id in positions}
    commands: list[_RecordedCommand] = []
    sample_calls: list[tuple[int, str]] = []
    harness = _RouteHarness(
        executor=executor,
        positions=current_positions,
        commands=commands,
        sample_calls=sample_calls,
    )

    def sample_telemetry(
        _executor: DynamicCoppeliaExecutor,
        robot_id: str,
    ) -> Any:
        point = current_positions[robot_id]
        sample_calls.append((executor.physics_steps, robot_id))
        sample = SimpleNamespace(
            timestamp_s=executor.simulation_time_s,
            robot_id=robot_id,
            measured_pose=Pose3D(
                position=Vec3(x=point.x, y=point.y, z=0.0),
                rotation_rpy_degrees=Vec3(x=0.0, y=0.0, z=0.0),
            ),
        )
        executor.telemetry.append(sample)
        return sample

    def command_body_velocity(
        _executor: DynamicCoppeliaExecutor,
        robot_id: str,
        forward_velocity: float,
        lateral_velocity: float,
        angular_velocity: float,
        *,
        source: RobotCommandSource = "path_follower",
        target: Vec2 | None = None,
    ) -> _RecordedCommand:
        moving = any(
            abs(value) > 1e-12
            for value in (
                forward_velocity,
                lateral_velocity,
                angular_velocity,
            )
        )
        pending_targets[robot_id] = target if moving else None
        command = _RecordedCommand(
            physics_step=executor.physics_steps,
            robot_id=robot_id,
            source=source,
            target=target,
            moving=moving,
        )
        commands.append(command)
        return command

    def step_physics(_executor: DynamicCoppeliaExecutor) -> None:
        for robot_id, target in pending_targets.items():
            if target is None:
                continue
            current = current_positions[robot_id]
            dx = target.x - current.x
            dy = target.y - current.y
            distance = math.hypot(dx, dy)
            if distance <= 1e-12:
                continue
            displacement = min(route_speeds[robot_id], distance)
            current_positions[robot_id] = Vec2(
                x=current.x + dx / distance * displacement,
                y=current.y + dy / distance * displacement,
            )
        executor.physics_steps += 1
        if harness.after_step is not None:
            harness.after_step(executor, current_positions)

    def update_logical_payloads(_executor: DynamicCoppeliaExecutor) -> None:
        return None

    executor.sample_telemetry = MethodType(  # type: ignore[method-assign]
        sample_telemetry,
        executor,
    )
    executor.command_body_velocity = MethodType(  # type: ignore[method-assign]
        command_body_velocity,
        executor,
    )
    executor._step_physics = MethodType(  # type: ignore[method-assign]
        step_physics,
        executor,
    )
    executor._update_logical_payloads = MethodType(  # type: ignore[method-assign]
        update_logical_payloads,
        executor,
    )
    return harness


def _events(
    executor: DynamicCoppeliaExecutor,
    name: str,
) -> list[dict[str, object]]:
    return [event for event in executor.runtime_events if event["event"] == name]


def test_multi_route_uses_shared_index_and_holds_fast_endpoint() -> None:
    harness = _route_harness(
        {
            "robot_fast": Vec2(x=0.0, y=0.0),
            "robot_slow": Vec2(x=0.0, y=2.0),
        },
        speeds={"robot_fast": 0.5, "robot_slow": 0.1},
    )
    routes = {
        "robot_fast": [
            Vec2(x=0.0, y=0.0),
            Vec2(x=1.0, y=0.0),
            Vec2(x=2.0, y=0.0),
        ],
        "robot_slow": [
            Vec2(x=0.0, y=2.0),
            Vec2(x=0.5, y=2.0),
            Vec2(x=1.0, y=2.0),
            Vec2(x=1.5, y=2.0),
            Vec2(x=2.0, y=2.0),
        ],
    }

    harness.executor.follow_routes(routes)

    completed = _events(
        harness.executor,
        "route_time_index_completed",
    )
    assert [event["route_time_index"] for event in completed] == list(range(5))
    started = _events(harness.executor, "route_time_index_started")
    assert started[3]["endpoint_hold_robot_ids"] == ["robot_fast"]
    assert started[4]["endpoint_hold_robot_ids"] == ["robot_fast"]
    assert any(
        fast.physics_step == slow.physics_step
        and fast.robot_id == "robot_fast"
        and fast.source == "formation_hold"
        and not fast.moving
        and slow.robot_id == "robot_slow"
        and slow.source == "path_follower"
        and slow.moving
        for fast in harness.commands
        for slow in harness.commands
    )
    for robot_id, route in routes.items():
        final = harness.positions[robot_id]
        assert (
            math.hypot(
                final.x - route[-1].x,
                final.y - route[-1].y,
            )
            <= harness.executor.config.waypoint_tolerance_m
        )
    assert harness.executor.synchronized_formation_errors_m == []


def test_route_time_allows_same_cell_at_different_indices() -> None:
    common = Vec2(x=0.0, y=0.0)
    harness = _route_harness(
        {
            "robot_1": Vec2(x=-1.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        },
        speeds={"robot_1": 0.25, "robot_2": 0.25},
        safety_distance_m=0.2,
    )
    routes = {
        "robot_1": [
            Vec2(x=-1.0, y=0.0),
            common,
            Vec2(x=1.0, y=0.0),
            Vec2(x=2.0, y=0.0),
        ],
        "robot_2": [
            Vec2(x=0.0, y=3.0),
            Vec2(x=0.0, y=2.0),
            Vec2(x=0.0, y=1.0),
            common,
        ],
    }

    harness.executor.follow_routes(routes)

    started = _events(harness.executor, "route_time_index_started")
    targets_at_one = cast(dict[str, dict[str, float]], started[1]["targets"])
    targets_at_three = cast(dict[str, dict[str, float]], started[3]["targets"])
    assert targets_at_one["robot_1"] == common.model_dump(mode="json")
    assert targets_at_three["robot_2"] == common.model_dump(mode="json")
    assert len(_events(harness.executor, "route_time_index_completed")) == 4
    assert harness.executor.proximity_safety_stops == 0


def test_multi_route_rejects_unsafe_same_time_targets_before_motion() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=1.1, y=0.0)],
    }

    with pytest.raises(
        DynamicCoppeliaError,
        match="time index 1.*below the configured safety separation",
    ):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    rejected = _events(harness.executor, "route_time_preflight_rejected")
    assert rejected[0]["route_time_index"] == 1


def test_multi_route_rejects_swept_head_on_swap_before_motion() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=-1.0, y=0.0),
            "robot_2": Vec2(x=1.0, y=0.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_1": [Vec2(x=-1.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=1.0, y=0.0), Vec2(x=-1.0, y=0.0)],
    }

    with pytest.raises(
        DynamicCoppeliaError,
        match="route interval 0->1.*below the configured safety separation",
    ):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    rejected = _events(harness.executor, "route_time_preflight_rejected")
    assert rejected == [
        {
            "timestamp_s": 0.0,
            "event": "route_time_preflight_rejected",
            "reason": "planned_swept_separation",
            "route_time_interval": [0, 1],
            "limiting_interval_fractions": {
                "robot_1": pytest.approx(0.0),
                "robot_2": pytest.approx(1.0),
            },
            "robot_ids": ["robot_1", "robot_2"],
            "planned_separation_m": pytest.approx(0.0),
            "safety_distance_m": 0.3,
        }
    ]


def test_multi_route_rejects_swept_conflict_after_endpoint_padding() -> None:
    harness = _route_harness(
        {
            "robot_hold": Vec2(x=0.0, y=0.0),
            "robot_move": Vec2(x=-1.0, y=1.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_hold": [Vec2(x=0.0, y=0.0)],
        "robot_move": [
            Vec2(x=-1.0, y=1.0),
            Vec2(x=1.0, y=-1.0),
        ],
    }

    with pytest.raises(DynamicCoppeliaError, match="route interval 0->1"):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []


def test_multi_route_rejects_unequal_progress_segment_crossing() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=-1.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=-2.0),
        },
        speeds={"robot_1": 0.5, "robot_2": 1.0},
        safety_distance_m=0.25,
    )
    routes = {
        "robot_1": [Vec2(x=-1.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=-2.0), Vec2(x=0.0, y=1.0)],
    }

    with pytest.raises(DynamicCoppeliaError, match="route interval 0->1"):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    rejected = _events(harness.executor, "route_time_preflight_rejected")[-1]
    assert rejected["planned_separation_m"] == pytest.approx(0.0)
    assert rejected["limiting_interval_fractions"] == {
        "robot_1": pytest.approx(0.5),
        "robot_2": pytest.approx(2.0 / 3.0),
    }


def test_multi_route_rejects_measured_start_to_first_target_swap() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=-1.0, y=0.0),
            "robot_2": Vec2(x=1.0, y=0.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_1": [Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=-1.0, y=0.0)],
    }

    with pytest.raises(DynamicCoppeliaError, match="route interval -1->0"):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    assert _events(harness.executor, "route_time_preflight_rejected")[-1][
        "route_time_interval"
    ] == [-1, 0]


@pytest.mark.parametrize("disabled", [False, True])
def test_multi_route_rejects_sweep_through_non_route_robot(
    disabled: bool,
) -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
            "robot_idle": Vec2(x=1.0, y=0.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=2.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=2.0, y=2.0)],
    }
    if disabled:
        harness.executor.disabled_robots = {"robot_idle"}

    with pytest.raises(DynamicCoppeliaError, match="route interval 0->1"):
        harness.executor.follow_routes(routes)

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    rejected = _events(harness.executor, "route_time_preflight_rejected")[-1]
    assert rejected["robot_ids"] == ["robot_1", "robot_idle"]


def test_multi_route_fails_closed_on_measured_separation_breach() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        },
        safety_distance_m=0.3,
    )
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=1.0, y=2.0)],
    }

    def breach_after_first_step(
        executor: DynamicCoppeliaExecutor,
        positions: dict[str, Vec2],
    ) -> None:
        if executor.physics_steps == 1:
            first = positions["robot_1"]
            positions["robot_2"] = Vec2(x=first.x + 0.1, y=first.y)

    harness.after_step = breach_after_first_step
    with pytest.raises(
        DynamicCoppeliaError,
        match="measured enabled-base separation breached",
    ):
        harness.executor.follow_routes(routes)

    collision_stops = [
        command for command in harness.commands if command.source == "collision_stop"
    ]
    assert {command.robot_id for command in collision_stops} == {
        "robot_1",
        "robot_2",
    }
    assert harness.executor.proximity_safety_stops == 1
    assert (
        _events(harness.executor, "route_time_safety_stop")[-1]["reason"]
        == "measured_base_separation"
    )


def test_multi_route_fails_closed_on_physical_collision_monitor() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        }
    )
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=1.0, y=2.0)],
    }

    def collide_after_first_step(
        executor: DynamicCoppeliaExecutor,
        _positions: dict[str, Vec2],
    ) -> None:
        if executor.physics_steps == 1:
            executor._latest_collision_robots = {"robot_1"}

    harness.after_step = collide_after_first_step
    with pytest.raises(
        DynamicCoppeliaError,
        match="physical collision monitor stopped synchronized route",
    ):
        harness.executor.follow_routes(routes)

    assert (
        _events(harness.executor, "route_time_safety_stop")[-1]["reason"]
        == "physical_collision_monitor"
    )
    assert {
        command.robot_id for command in harness.commands if command.source == "collision_stop"
    } == {"robot_1", "robot_2"}


def test_disabled_only_collision_aborts_without_commanding_disabled_robot() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
            "robot_disabled": Vec2(x=4.0, y=4.0),
        }
    )
    harness.executor.disabled_robots = {"robot_disabled"}
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=1.0, y=2.0)],
    }

    def disabled_collision_after_first_step(
        executor: DynamicCoppeliaExecutor,
        _positions: dict[str, Vec2],
    ) -> None:
        if executor.physics_steps == 1:
            executor._latest_collision_robots = {"robot_disabled"}

    harness.after_step = disabled_collision_after_first_step
    with pytest.raises(
        DynamicCoppeliaError,
        match="physical collision monitor stopped synchronized route.*robot_disabled",
    ):
        harness.executor.follow_routes(routes)

    assert not any(command.robot_id == "robot_disabled" for command in harness.commands)
    assert {
        command.robot_id for command in harness.commands if command.source == "collision_stop"
    } == {"robot_1", "robot_2"}


def test_route_telemetry_samples_full_fleet_once_per_physics_tick() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
            "robot_disabled": Vec2(x=4.0, y=4.0),
        },
        safety_distance_m=0.3,
    )
    harness.executor.disabled_robots = {"robot_disabled"}
    routes = {
        "robot_1": [Vec2(x=0.0, y=0.0), Vec2(x=1.0, y=0.0)],
        "robot_2": [Vec2(x=0.0, y=2.0), Vec2(x=1.0, y=2.0)],
    }

    harness.executor.follow_routes(routes)

    expected_ids = set(harness.positions)
    by_step: dict[int, list[str]] = {}
    for physics_step, robot_id in harness.sample_calls:
        by_step.setdefault(physics_step, []).append(robot_id)
    assert by_step
    assert all(set(robot_ids) == expected_ids for robot_ids in by_step.values())
    assert all(len(robot_ids) == len(expected_ids) for robot_ids in by_step.values())
    assert harness.executor.proximity_safety_stops == 0


def test_dedicated_synchronized_routes_report_local_metrics() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        },
        speeds={"robot_1": 0.1, "robot_2": 0.1},
    )
    routes = {
        "robot_1": [
            Vec2(x=0.0, y=0.0),
            Vec2(x=0.5, y=0.0),
            Vec2(x=1.0, y=0.0),
        ],
        "robot_2": [
            Vec2(x=0.0, y=2.0),
            Vec2(x=0.5, y=2.0),
            Vec2(x=1.0, y=2.0),
        ],
    }

    harness.executor.follow_synchronized_routes(routes)

    completed = _events(
        harness.executor,
        "route_time_synchronization_completed",
    )[-1]
    assert completed["minimum_measured_separation_m"] == pytest.approx(2.0)
    assert completed["maximum_formation_error_m"] == pytest.approx(0.0)


def test_dedicated_synchronized_routes_reject_unequal_horizons() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        }
    )

    with pytest.raises(DynamicCoppeliaError, match="must have equal horizons"):
        harness.executor.follow_synchronized_routes(
            {
                "robot_1": [
                    Vec2(x=0.0, y=0.0),
                    Vec2(x=1.0, y=0.0),
                ],
                "robot_2": [Vec2(x=0.0, y=2.0)],
            }
        )

    assert harness.commands == []


def test_dedicated_synchronized_routes_fail_on_formation_divergence() -> None:
    harness = _route_harness(
        {
            "robot_1": Vec2(x=0.0, y=0.0),
            "robot_2": Vec2(x=0.0, y=2.0),
        },
        safety_distance_m=0.3,
    )

    with pytest.raises(
        DynamicCoppeliaError,
        match="carry formation diverged beyond tolerance",
    ):
        harness.executor.follow_synchronized_routes(
            {
                "robot_1": [Vec2(x=0.0, y=0.0)],
                "robot_2": [Vec2(x=0.0, y=1.0)],
            }
        )

    assert {
        command.robot_id for command in harness.commands if command.source == "formation_hold"
    } == {"robot_1", "robot_2"}


def test_zero_routes_noop_and_empty_singleton_keeps_legacy_hold() -> None:
    zero = _route_harness({"robot_1": Vec2(x=0.0, y=0.0)})
    zero.executor.follow_routes({})
    assert zero.commands == []
    assert zero.sample_calls == []

    singleton = _route_harness({"robot_1": Vec2(x=0.0, y=0.0)})
    singleton.executor.follow_routes({"robot_1": []})
    assert len(singleton.commands) == 1
    assert singleton.commands[0].source == "formation_hold"
    assert not singleton.commands[0].moving
    assert singleton.sample_calls == []


def test_single_route_keeps_independent_follower_behavior() -> None:
    harness = _route_harness({"robot_1": Vec2(x=0.0, y=0.0)})
    route = [Vec2(x=0.0, y=0.0), Vec2(x=0.5, y=0.0)]

    harness.executor.follow_routes({"robot_1": route})

    final = harness.positions["robot_1"]
    assert math.hypot(final.x - route[-1].x, final.y - route[-1].y) <= (
        harness.executor.config.waypoint_tolerance_m
    )
    assert _events(harness.executor, "route_time_synchronization_started") == []
    assert any(command.moving for command in harness.commands)


@pytest.mark.parametrize(
    ("active_id", "idle_id"),
    [
        ("robot_1", "robot_2"),
        ("robot_2", "robot_1"),
    ],
)
def test_single_route_fails_for_any_enabled_neighbor_id_order(
    active_id: str,
    idle_id: str,
) -> None:
    harness = _route_harness(
        {
            active_id: Vec2(x=0.0, y=0.0),
            idle_id: Vec2(x=0.1, y=0.0),
        },
        safety_distance_m=0.3,
    )

    with pytest.raises(
        DynamicCoppeliaError,
        match="below the configured safety separation",
    ):
        harness.executor.follow_routes(
            {
                active_id: [
                    Vec2(x=0.0, y=0.0),
                    Vec2(x=0.5, y=0.0),
                ]
            }
        )

    assert harness.executor.physics_steps == 0
    assert harness.commands == []


def test_single_route_aborts_on_non_route_collision_monitor_hit() -> None:
    harness = _route_harness(
        {
            "robot_active": Vec2(x=0.0, y=0.0),
            "robot_idle": Vec2(x=0.0, y=2.0),
        }
    )

    def idle_collision_after_first_step(
        executor: DynamicCoppeliaExecutor,
        _positions: dict[str, Vec2],
    ) -> None:
        if executor.physics_steps == 1:
            executor._latest_collision_robots = {"robot_idle"}

    harness.after_step = idle_collision_after_first_step
    with pytest.raises(
        DynamicCoppeliaError,
        match="physical collision monitor stopped route: robot_idle",
    ):
        harness.executor.follow_routes(
            {
                "robot_active": [
                    Vec2(x=0.0, y=0.0),
                    Vec2(x=1.0, y=0.0),
                ]
            }
        )

    assert {
        command.robot_id for command in harness.commands if command.source == "collision_stop"
    } == {"robot_active", "robot_idle"}


@pytest.mark.parametrize("disabled", [False, True])
def test_single_route_rejects_sweep_through_idle_robot_before_motion(
    disabled: bool,
) -> None:
    harness = _route_harness(
        {
            "robot_active": Vec2(x=0.0, y=0.0),
            "robot_idle": Vec2(x=0.5, y=0.0),
        },
        safety_distance_m=0.3,
    )
    if disabled:
        harness.executor.disabled_robots = {"robot_idle"}

    with pytest.raises(DynamicCoppeliaError, match="route interval 0->1"):
        harness.executor.follow_routes(
            {
                "robot_active": [
                    Vec2(x=0.0, y=0.0),
                    Vec2(x=1.0, y=0.0),
                ]
            }
        )

    assert harness.executor.physics_steps == 0
    assert harness.commands == []
    rejected = _events(harness.executor, "route_time_preflight_rejected")[-1]
    assert rejected["robot_ids"] == ["robot_active", "robot_idle"]
