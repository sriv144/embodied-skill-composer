"""Cheap differential checks; these synthetic records are not simulator evidence."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import math
import random
from typing import Any

import pytest

import embodied_skill_composer.construction.coppelia_phase5 as phase5
from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.models import Pose3D, Vec2, Vec3


def _sample(
    timestamp_s: float,
    robot_id: str = "r0",
    *,
    x: float = 0,
    linear: float = 0,
    angular: float = 0,
) -> RobotTelemetry:
    # model_construct deliberately permits nonfinite boundary cases as well.
    return RobotTelemetry.model_construct(
        timestamp_s=timestamp_s,
        robot_id=robot_id,
        measured_pose=Pose3D(position=Vec3(x=x, y=0, z=0)),
        linear_velocity_mps=linear,
        angular_velocity_rps=angular,
        battery_remaining_wh=100,
    )


def _reference_routes(
    telemetry: Iterable[RobotTelemetry],
    *,
    robot_ids: Iterable[str],
    start_s: float,
    end_s: float,
) -> dict[str, list[Vec2]]:
    # This is the original full-scan contract, independent of the index.
    routes: dict[str, list[Vec2]] = {
        robot_id: [] for robot_id in sorted(set(robot_ids))
    }
    for sample in telemetry:
        if sample.robot_id in routes and start_s <= sample.timestamp_s <= end_s:
            point = Vec2.model_validate(sample.measured_pose.position)
            if not routes[sample.robot_id] or math.hypot(
                routes[sample.robot_id][-1].x - point.x,
                routes[sample.robot_id][-1].y - point.y,
            ) > 1e-12:
                routes[sample.robot_id].append(point)
    missing = sorted(robot_id for robot_id, route in routes.items() if not route)
    if missing:
        raise DynamicCoppeliaError(
            "measured phase route is empty for " + ", ".join(missing)
        )
    return routes


def _reference_close(
    telemetry: Iterable[RobotTelemetry],
    timestamp_s: float,
    *,
    robot_id: str | None = None,
) -> list[RobotTelemetry]:
    return [
        sample
        for sample in telemetry
        if (robot_id is None or sample.robot_id == robot_id)
        and math.isclose(
            sample.timestamp_s, timestamp_s, rel_tol=1e-9, abs_tol=1e-9
        )
    ]


def test_routes_preserve_closed_bounds_file_order_and_adjacent_deduplication() -> None:
    rows = [
        _sample(math.nextafter(2, math.inf), x=99),
        _sample(2, "r1", x=2),
        _sample(2, x=0),
        _sample(1, x=1),
        _sample(1.5, "ignored", x=99),
        _sample(1.5, x=1 + 5e-13),
        _sample(1.5, x=0),
        _sample(1, "r1", x=1),
        _sample(math.nextafter(1, -math.inf), x=99),
    ]
    kwargs = {"robot_ids": ["r1", "r0", "r1"], "start_s": 1, "end_s": 2}
    expected = _reference_routes(rows, **kwargs)
    assert phase5._telemetry_xy_routes(
        phase5._Phase5TelemetryLookup(iter(rows)), **kwargs
    ) == expected
    assert list(expected) == ["r0", "r1"]
    assert [(point.x, point.y) for point in expected["r0"]] == [(0, 0), (1, 0), (0, 0)]
    assert [(point.x, point.y) for point in expected["r1"]] == [(2, 0), (1, 0)]


@pytest.mark.parametrize("robot_ids", [[], ["z", "a", "z"], ["r0", "missing"]])
def test_route_missing_robot_errors_match_original_scan(robot_ids: list[str]) -> None:
    rows = [_sample(1)]
    kwargs = {"robot_ids": robot_ids, "start_s": 1, "end_s": 1}
    if not robot_ids:
        assert phase5._telemetry_xy_routes(
            phase5._Phase5TelemetryLookup(rows), **kwargs
        ) == _reference_routes(rows, **kwargs)
        return
    with pytest.raises(DynamicCoppeliaError) as original:
        _reference_routes(rows, **kwargs)
    with pytest.raises(DynamicCoppeliaError) as indexed:
        phase5._telemetry_xy_routes(phase5._Phase5TelemetryLookup(rows), **kwargs)
    assert str(indexed.value) == str(original.value)


@pytest.mark.parametrize("query", [0.0, 1e-9, 1.0, 1e9, 1e15, 1e300])
def test_close_time_differential_keeps_relative_tolerance_and_duplicates(
    query: float,
) -> None:
    width = max(1e-9, 1e-9 * abs(query))
    values = [
        query,
        query - width,
        query + width,
        math.nextafter(query - width, -math.inf),
        math.nextafter(query + width, math.inf),
        query - 2 * width,
        query + 2 * width,
    ]
    rows = [
        _sample(value, robot_id, x=index)
        for index, value in enumerate(values)
        for robot_id in ("r0", "r1")
    ]
    rows.extend([rows[0], rows[0]])
    random.Random(87).shuffle(rows)
    lookup = phase5._Phase5TelemetryLookup(rows)
    for robot_id in (None, "r0", "r1", "missing"):
        assert [id(row) for row in lookup.close_to(query, robot_id=robot_id)] == [
            id(row) for row in _reference_close(rows, query, robot_id=robot_id)
        ]
    if query == 1e9:
        half_second_later = _sample(query + 0.5)
        assert phase5._Phase5TelemetryLookup([half_second_later]).close_to(query) == [
            half_second_later
        ]


@pytest.mark.parametrize("query", [math.inf, -math.inf, math.nan, 0, 2])
def test_nonfinite_data_and_query_fallback_matches_original(query: float) -> None:
    rows = [_sample(2), _sample(math.inf), _sample(math.nan), _sample(-math.inf)]
    lookup = phase5._Phase5TelemetryLookup(rows)
    assert [id(row) for row in lookup.close_to(query)] == [
        id(row) for row in _reference_close(rows, query)
    ]
    for start, end in ((0, math.inf), (-math.inf, math.inf), (math.nan, 2), (2, 0)):
        assert [id(row) for row in lookup.in_interval(start, end)] == [
            id(row) for row in rows if start <= row.timestamp_s <= end
        ]


def test_randomized_interval_and_close_queries_match_full_scan() -> None:
    rng = random.Random(1207)
    rows = [
        _sample(rng.randrange(30) / 10, f"r{rng.randrange(4)}", x=rng.randrange(4))
        for _ in range(180)
    ]
    rows.extend([rows[5], rows[5]])
    rng.shuffle(rows)
    lookup = phase5._Phase5TelemetryLookup(rows)
    for _ in range(70):
        start, end = (rng.randrange(35) / 10 for _ in range(2))
        robot_id = rng.choice([None, "r0", "r1", "r3", "missing"])
        expected = [
            sample
            for sample in rows
            if (robot_id is None or sample.robot_id == robot_id)
            and start <= sample.timestamp_s <= end
        ]
        assert [id(row) for row in lookup.in_interval(start, end, robot_id=robot_id)] == [
            id(row) for row in expected
        ]
        assert [id(row) for row in lookup.close_to(start, robot_id=robot_id)] == [
            id(row) for row in _reference_close(rows, start, robot_id=robot_id)
        ]


class _LinearTelemetryLookup(phase5._Phase5TelemetryLookup):
    """Original scans substituted into the unchanged stop-proof predicates."""

    def __init__(self, telemetry: Iterable[RobotTelemetry]) -> None:
        self._samples = list(telemetry)

    def close_to(
        self, timestamp_s: float, *, robot_id: str | None = None
    ) -> list[RobotTelemetry]:
        return _reference_close(self._samples, timestamp_s, robot_id=robot_id)


def _stop_fixture(offset: float = 0) -> dict[str, Any]:
    config = DynamicCoppeliaConfig(logical_transition_settle_consecutive_samples=2)
    final_s = offset + 1.1
    times = [final_s - 1 / config.control_hz, final_s]
    commands = [
        RobotCommand(
            timestamp_s=final_s,
            robot_id=robot_id,
            linear_velocity_mps=0,
            angular_velocity_rps=0,
            wheel_target_velocity_rad_s=(0, 0, 0, 0),
            source="formation_hold",
        )
        for robot_id in ("r0", "r1")
    ]

    def motion(at: float) -> dict[str, Any]:
        return {
            "timestamp_s": at,
            "linear_velocity_mps": 0,
            "angular_velocity_rps": 0,
            "still": True,
        }

    proof = {
        "robot_ids": ["r0", "r1"],
        "latest_commands_zero": True,
        "measured_team_still": True,
        "no_nonzero_team_command_after_hold": True,
        "required_consecutive_still_samples": 2,
        "observed_consecutive_still_samples": 2,
        "settled_linear_speed_mps": config.settled_linear_speed_mps,
        "settled_angular_speed_rps": config.settled_angular_speed_rps,
        "minimum_measured_team_to_scene_separation_m": 1.0,
        "minimum_measured_team_to_scene_pair": ["r0", "r2"],
        "safety_distance_m": config.safety_distance_m,
        "latest_commands_by_robot": {
            command.robot_id: {
                **command.model_dump(mode="json"), "present": True, "zero": True
            }
            for command in commands
        },
        "measured_motion_by_robot": {robot_id: motion(final_s) for robot_id in ("r0", "r1")},
        "hold_command_index": 0,
        "command_count_at_settle": 2,
        "started_at_s": times[0],
        "settled_at_s": final_s,
        "physics_steps_elapsed": 1,
    }
    event = {"team_zero_motion_proof": proof}
    trace = [{
        "event": "logical_transition_settled",
        "module_id": "m0",
        "transition": "logical_payload_lift",
        "timestamp_s": final_s,
        "team_zero_motion_proof": proof,
        "settle_samples": [
            {
                "timestamp_s": at,
                "physics_step": index + 10,
                "team_still": True,
                "consecutive_still_samples": index + 1,
                "robots": {robot_id: motion(at) for robot_id in ("r0", "r1")},
            }
            for index, at in enumerate(times)
        ],
    }]
    return {
        "event": event,
        "module_id": "m0",
        "robot_ids": ["r0", "r1"],
        "transition": "logical_payload_lift",
        "timestamp_s": final_s,
        "commands": commands,
        "telemetry": [
            _sample(at, robot_id, x=x)
            for at in times
            for robot_id, x in (("r0", 0), ("r1", 2), ("r2", 1))
        ],
        "trace_records": trace,
        "config": config,
    }


def _stop_result(fixture: dict[str, Any], *, linear: bool) -> str | None:
    fixture = deepcopy(fixture)
    if linear:
        fixture["telemetry"] = _LinearTelemetryLookup(fixture["telemetry"])
    try:
        phase5._verify_logical_transition_stop_proof(**fixture)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(
    "variant",
    ["valid", "relative_time", "duplicate_motion", "last_pose", "missing_idle", "fake_stop", "fake_tail"],
)
def test_stop_proof_differential_preserves_any_match_last_pose_and_scene_robots(
    variant: str,
) -> None:
    fixture = _stop_fixture(1e9 if variant == "relative_time" else 0)
    proof = fixture["event"]["team_zero_motion_proof"]
    if variant == "relative_time":
        for row in fixture["telemetry"]:
            row.timestamp_s += 0.5
    elif variant == "duplicate_motion":
        fixture["telemetry"].append(_sample(1.1, linear=0.4))
    elif variant == "last_pose":
        # Smaller time, later original row: last file-order pose must win.
        fixture["telemetry"].append(_sample(1.1 - 5e-10, x=0.25))
        proof["minimum_measured_team_to_scene_separation_m"] = 0.75
    elif variant == "missing_idle":
        fixture["telemetry"] = [row for row in fixture["telemetry"] if row.robot_id != "r2"]
    elif variant == "fake_stop":
        proof["measured_motion_by_robot"]["r0"]["linear_velocity_mps"] = 0.01
    elif variant == "fake_tail":
        fixture["trace_records"][0]["settle_samples"][-1]["robots"]["r0"]["linear_velocity_mps"] = 0.01
    original = _stop_result(fixture, linear=True)
    indexed = _stop_result(fixture, linear=False)
    assert indexed == original
    if variant in {"missing_idle", "fake_stop", "fake_tail"}:
        assert indexed is not None
    else:
        assert indexed is None


def test_new_verification_observes_tampered_telemetry_instead_of_cached_lookup() -> None:
    fixture = _stop_fixture()
    assert _stop_result(fixture, linear=False) is None
    for row in fixture["telemetry"]:
        if row.robot_id == "r0":
            row.linear_velocity_mps = 0.1
    assert "measured-stop proof differs" in str(_stop_result(fixture, linear=False))


def test_synthetic_scaling_reduces_exact_time_predicates_without_timing_assumptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _sample(float(at), f"r{robot}")
        for at in range(1000)
        for robot in range(6)
    ]
    lookup = phase5._Phase5TelemetryLookup(rows)
    original_isclose = math.isclose
    calls = 0

    def counted_isclose(*args: float, **kwargs: float) -> bool:
        nonlocal calls
        calls += 1
        return original_isclose(*args, **kwargs)

    monkeypatch.setattr(phase5.math, "isclose", counted_isclose)
    queries = [(float(at), f"r{at % 6}") for at in range(0, 1000, 10)]
    expected = [_reference_close(rows, at, robot_id=robot_id) for at, robot_id in queries]
    scan_calls = calls
    calls = 0
    actual = [lookup.close_to(at, robot_id=robot_id) for at, robot_id in queries]
    assert actual == expected
    assert scan_calls == 100_000
    assert calls == 100
