from __future__ import annotations

import math

import pytest

from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5ClearanceMinimum,
    Phase5PhysicalYardConfig,
    _Aabb3,
    _minimum_record,
    _oriented_half_extents_xyz,
    _planned_robot_separation_m,
    _require_clearance_records,
    _route_bounds_clearance,
    _segment_segment_distance,
    _swept_payload_aabb3_clearance,
    _swept_payload_robot_clearance,
    _synchronize_safe_route_pair,
    _with_exact_route_endpoints,
)
from embodied_skill_composer.construction.models import (
    BuildModule,
    Dimensions3D,
    ModuleType,
    Pose3D,
    Vec2,
    Vec3,
)
from embodied_skill_composer.construction.routing import RoutingError


def _roof_module() -> BuildModule:
    pose = Pose3D(
        position=Vec3(x=0, y=0, z=0),
        rotation_rpy_degrees=Vec3(x=0, y=28, z=0),
    )
    return BuildModule(
        module_id="roof_probe",
        module_type=ModuleType.ROOF,
        mesh_node="roof_probe",
        target_pose=pose,
        staging_pose=pose.model_copy(deep=True),
        dimensions=Dimensions3D(width=4, depth=0.2, height=0.2),
        mass_kg=10,
        grip_points=[Vec3(x=0, y=0, z=0)],
        required_team_size=2,
        install_duration_s=1,
        material="probe",
    )


def test_full_rpy_envelope_accounts_for_roof_pitch_in_staging_height() -> None:
    module = _roof_module()

    half_x, half_y, half_z = _oriented_half_extents_xyz(
        module,
        module.target_pose.rotation_rpy_degrees,
    )

    pitch = math.radians(28)
    assert half_x == pytest.approx(abs(math.cos(pitch)) * 2 + abs(math.sin(pitch)) * 0.1)
    assert half_y == pytest.approx(0.1)
    assert half_z == pytest.approx(abs(math.sin(pitch)) * 2 + abs(math.cos(pitch)) * 0.1)
    assert half_z > module.dimensions.height / 2


def test_analytic_segment_clearance_detects_between_waypoint_intersection() -> None:
    bounds = (-0.5, 0.5, -0.5, 0.5)
    route = [Vec2(x=-2, y=0), Vec2(x=2, y=0)]

    assert all(
        _route_bounds_clearance([endpoint], bounds) == pytest.approx(1.5)
        for endpoint in route
    )
    assert _route_bounds_clearance(route, bounds) == pytest.approx(0.0)
    assert _segment_segment_distance(
        Vec2(x=-1, y=0),
        Vec2(x=1, y=0),
        Vec2(x=0, y=-1),
        Vec2(x=0, y=1),
    ) == pytest.approx(0.0)


def test_route_time_scheduler_inserts_wait_for_same_cell_handoff() -> None:
    routes = {
        "robot_0": [Vec2(x=0, y=0), Vec2(x=1, y=0)],
        "robot_1": [Vec2(x=-1, y=0), Vec2(x=0, y=0)],
    }

    synchronized = _synchronize_safe_route_pair(
        routes,
        minimum_separation_m=0.4,
    )

    assert synchronized["robot_0"] == [
        Vec2(x=0, y=0),
        Vec2(x=1, y=0),
        Vec2(x=1, y=0),
    ]
    assert synchronized["robot_1"] == [
        Vec2(x=-1, y=0),
        Vec2(x=-1, y=0),
        Vec2(x=0, y=0),
    ]
    assert all(
        _segment_segment_distance(
            synchronized["robot_0"][index],
            synchronized["robot_0"][index + 1],
            synchronized["robot_1"][index],
            synchronized["robot_1"][index + 1],
        )
        >= 0.4
        for index in range(len(synchronized["robot_0"]) - 1)
    )


def test_route_time_scheduler_rejects_an_impossible_position_swap() -> None:
    with pytest.raises(RoutingError, match="no collision-clear"):
        _synchronize_safe_route_pair(
            {
                "robot_0": [Vec2(x=-1, y=0), Vec2(x=1, y=0)],
                "robot_1": [Vec2(x=1, y=0), Vec2(x=-1, y=0)],
            },
            minimum_separation_m=0.4,
        )


def test_exact_measured_starts_resolve_drifted_dispatch_cell_handoff() -> None:
    measured_starts = {
        "robot_3": Vec2(x=-23.511230728099562, y=0.8520755257775265),
        "robot_4": Vec2(x=-23.4936447544956, y=1.5501889714899248),
    }
    raster_routes = {
        "robot_3": [
            Vec2(x=-23.5, y=1.0),
            Vec2(x=-23.0, y=1.0),
            Vec2(x=-22.5, y=1.0),
            Vec2(x=-22.5, y=0.5),
        ],
        "robot_4": [
            Vec2(x=-23.5, y=1.5),
            Vec2(x=-23.0, y=1.5),
            Vec2(x=-22.5, y=1.5),
            Vec2(x=-22.5, y=2.0),
        ],
    }

    with pytest.raises(RoutingError, match="start below"):
        _synchronize_safe_route_pair(
            raster_routes,
            minimum_separation_m=0.64,
        )

    exact_routes = {
        robot_id: _with_exact_route_endpoints(
            route,
            start=measured_starts[robot_id],
            endpoint=route[-1],
        )
        for robot_id, route in raster_routes.items()
    }
    synchronized = _synchronize_safe_route_pair(
        exact_routes,
        minimum_separation_m=0.64,
    )

    assert synchronized["robot_3"][0] == measured_starts["robot_3"]
    assert synchronized["robot_4"][0] == measured_starts["robot_4"]
    assert synchronized["robot_3"][-1] == raster_routes["robot_3"][-1]
    assert synchronized["robot_4"][-1] == raster_routes["robot_4"][-1]
    assert len(synchronized["robot_3"]) == len(synchronized["robot_4"])


def test_planned_robot_separation_absorbs_both_bases_tracking_error() -> None:
    config = Phase5PhysicalYardConfig(
        robot_footprint_radius_m=0.12,
        route_clearance_m=0.16,
        route_tracking_error_m=0.12,
    )

    assert _planned_robot_separation_m(config) == pytest.approx(0.64)
    assert _planned_robot_separation_m(config) > 0.34 + 2 * 0.12

    with pytest.raises(RoutingError, match="start below"):
        _synchronize_safe_route_pair(
            {
                "robot_0": [Vec2(x=0, y=0)],
                "robot_1": [Vec2(x=0.455, y=0)],
            },
            minimum_separation_m=_planned_robot_separation_m(config),
        )

    planned_idle_clearance = _minimum_record(
        phase="return",
        source="planned",
        mover_kind="robot_base",
        obstacle_kind="idle_robot",
        values=[(0.4, "robot_0", "robot_1")],
        config=config,
    )
    measured_idle_clearance = _minimum_record(
        phase="return",
        source="measured",
        mover_kind="robot_base",
        obstacle_kind="idle_robot",
        values=[(0.16, "robot_0", "robot_1")],
        config=config,
    )

    assert planned_idle_clearance.required_clearance_m == pytest.approx(0.4)
    assert measured_idle_clearance.required_clearance_m == pytest.approx(0.16)


def test_payload_may_clear_finite_structure_vertically_but_not_a_robot() -> None:
    module = _roof_module().model_copy(
        update={
            "dimensions": Dimensions3D(width=1, depth=1, height=1),
            "target_pose": Pose3D(
                position=Vec3(x=0, y=0, z=0.5),
                rotation_rpy_degrees=Vec3(x=0, y=0, z=0),
            ),
            "staging_pose": Pose3D(
                position=Vec3(x=0, y=0, z=0.5),
                rotation_rpy_degrees=Vec3(x=0, y=0, z=0),
            ),
        }
    )
    route = [Vec2(x=-2, y=0), Vec2(x=2, y=0)]
    obstacle = _Aabb3(
        minimum_x=-0.5,
        maximum_x=0.5,
        minimum_y=-0.5,
        maximum_y=0.5,
        minimum_z=0,
        maximum_z=1,
    )

    assert _swept_payload_aabb3_clearance(
        module,
        carrier_route=route,
        transport_height_m=0.5,
        obstacle=obstacle,
    ) == pytest.approx(0.0)
    assert _swept_payload_aabb3_clearance(
        module,
        carrier_route=route,
        transport_height_m=3,
        obstacle=obstacle,
    ) == pytest.approx(1.5)
    assert _swept_payload_robot_clearance(
        module,
        carrier_route=route,
        robot_position=Vec2(x=0, y=0),
        robot_radius_m=0.12,
    ) == pytest.approx(0.0)


def test_clearance_tampering_below_the_recorded_gate_is_rejected() -> None:
    record = Phase5ClearanceMinimum(
        phase="carry",
        source="measured",
        mover_kind="logical_payload",
        obstacle_kind="disabled_robot",
        minimum_surface_clearance_m=0.159,
        required_clearance_m=0.16,
        evaluated_pair_count=1,
        limiting_mover_id="roof_probe",
        limiting_obstacle_id="robot_disabled",
    )

    with pytest.raises(RoutingError, match="continuous carry"):
        _require_clearance_records([record])
