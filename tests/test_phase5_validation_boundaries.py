"""Small evidence fixtures test offline verification, never a live simulator.

Each negative case changes one attested fact in otherwise valid evidence. These
tests intentionally avoid cottage compilation, routing and expensive run fixtures.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
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
    ScenarioManifest,
    ScenarioSplit,
)
from embodied_skill_composer.construction.models import (
    BuildModule,
    BuildPlan,
    Dimensions3D,
    HouseDesign,
    ModuleType,
    Pose3D,
    RobotSpec,
    SiteGrid,
    Vec2,
    Vec3,
    VectorFloorPlan,
)
from embodied_skill_composer.construction.routing import RoutingError


def _changed(payload: Any, path: tuple[Any, ...], value: Any) -> Any:
    changed = deepcopy(payload)
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return changed


def _pose(x: float = 0, y: float = 0) -> Pose3D:
    return Pose3D(position=Vec3(x=x, y=y, z=0))


def _telemetry(robot_id: str, at: float, x: float = 0) -> RobotTelemetry:
    return RobotTelemetry(
        robot_id=robot_id,
        timestamp_s=at,
        measured_pose=_pose(x),
        linear_velocity_mps=0,
        angular_velocity_rps=0,
        battery_remaining_wh=100,
    )


def _scenario() -> ScenarioManifest:
    design = HouseDesign(
        design_id="boundary-fixture",
        title="Verifier fixture, not simulated evidence",
        footprint_width_m=2,
        footprint_depth_m=2,
        floor_plan=VectorFloorPlan(walls=[], approved=True),
    )
    module = BuildModule(
        module_id="m0",
        module_type=ModuleType.FOUNDATION,
        mesh_node="m0",
        target_pose=_pose(),
        staging_pose=_pose(2),
        dimensions=Dimensions3D(width=1, depth=1, height=0.2),
        mass_kg=1,
        grip_points=[Vec3(x=0, y=0, z=0)],
        required_team_size=2,
        install_duration_s=1,
        material="fixture",
    )
    return ScenarioManifest(
        scenario_id="boundary-fixture",
        seed=900,
        split=ScenarioSplit.TEST,
        plan=BuildPlan(
            plan_id="boundary-fixture",
            design=design,
            modules=[module],
            robots=[
                RobotSpec(
                    robot_id=robot_id,
                    payload_capacity_kg=10,
                    speed_mps=1,
                    battery_capacity_wh=100,
                    start_pose=_pose(index * 2),
                )
                for index, robot_id in enumerate(("r0", "r1"))
            ],
            site_grid=SiteGrid(width=10, height=10, obstacle_cells=[(4, 4)]),
        ),
    )


def _script_diagnostics() -> dict[str, Any]:
    return {
        "script_control_audit": [
            {
                "robot_id": "r0",
                "exclusive_control_verified": True,
                "scripts": [
                    {
                        "role": role,
                        "action": "disabled" if disabled else "retained_enabled",
                        "disabled_after": disabled,
                        "source_sha256": "a" * 64,
                    }
                    for role, disabled in (
                        ("wheel_command_writer", True),
                        ("arm_gripper_writer", True),
                        ("passive_omniwheel_maintenance", False),
                    )
                ],
            }
        ]
    }


def _verify_scripts(diagnostics: dict[str, Any], **counts: int) -> None:
    phase5._verify_script_control_audit(
        diagnostics=diagnostics,
        robot_ids={"r0"},
        discovered_scripts=counts.get("discovered", 3),
        disabled_scripts=counts.get("disabled", 2),
        retained_scripts=counts.get("retained", 1),
    )


def test_script_ownership_requires_disabled_writers_and_enabled_maintenance() -> None:
    _verify_scripts(_script_diagnostics())


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("script_control_audit",), None, "incomplete"),
        (("script_control_audit", 0), None, "entry must be an object"),
        (("script_control_audit", 0, "robot_id"), "other", "robot audit"),
        (("script_control_audit", 0, "exclusive_control_verified"), False, "robot audit"),
        (("script_control_audit", 0, "scripts"), [], "robot audit"),
        (("script_control_audit", 0, "scripts", 0), None, "record must be an object"),
        (("script_control_audit", 0, "scripts", 0, "source_sha256"), 1, "source digest"),
        (("script_control_audit", 0, "scripts", 0, "source_sha256"), "Z" * 64, "source digest"),
        (("script_control_audit", 0, "scripts", 0, "role"), "unknown", "unknown role"),
        (("script_control_audit", 0, "scripts", 0, "action"), "retained_enabled", "ownership action"),
        (("script_control_audit", 0, "scripts", 2, "disabled_after"), True, "ownership action"),
        (("script_control_audit", 0, "scripts", 0, "role"), "arm_gripper_writer", "roles are incomplete"),
        (("script_control_audit", 0, "scripts", 2), {
            "role": "arm_gripper_writer", "action": "disabled",
            "disabled_after": True, "source_sha256": "a" * 64,
        }, "roles are incomplete"),
    ],
)
def test_script_audit_rejects_incomplete_or_contradictory_ownership(
    path: tuple[Any, ...], value: Any, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _verify_scripts(_changed(_script_diagnostics(), path, value))


@pytest.mark.parametrize("counter", ["discovered", "disabled", "retained"])
def test_script_audit_counts_must_be_derived_from_records(counter: str) -> None:
    with pytest.raises(ValueError, match="counts do not match"):
        _verify_scripts(_script_diagnostics(), **{counter: 0})


def _base_physics_diagnostics() -> tuple[dict[str, Any], list[Any]]:
    aliases = ["construction_intelligence_r0", *(
        f"wheel_respondable_{name}" for name in ("fl", "rl", "rr", "fr")
    ), "arm_visual"]
    audit = [{
        "robot_id": "r0",
        "allowed_shape_handles": [1, 2, 3, 4, 5],
        "excluded_shape_handles": [6],
        "footprint_gate_passed": True,
        "base_only_policy_verified": True,
        "planned_footprint_radius_m": 0.4,
        "measured_footprint_radius_m": 0.3,
        "shape_records": [{
            "handle": index,
            "alias": alias,
            "role": "mobile_base" if index <= 5 else "excluded_v1_geometry",
            "before": {"respondable": 1, "static": 0, "collidable": True},
            "after": {
                "respondable": 1 if index <= 5 else 0,
                "static": 0,
                "collidable": index <= 5,
            },
        } for index, alias in enumerate(aliases, start=1)],
    }]
    tests = [{
        "robot_id": "r0", "verified": True,
        "colliding_object_handles": [1, 99], "probe_handle": 99,
    }]
    diagnostics = {
        "robot_base_contact_gate_passed": True,
        "robot_base_physics_audit": audit,
        "robot_base_physics_audit_sha256": phase5._sha256_json(audit),
        "robot_collision_self_tests": tests,
        "measured_base_footprint_radius_m_by_robot": {"r0": 0.3},
    }
    trace = [{
        "event": "collision_monitor_ready",
        "base_physics_audit_sha256": phase5._sha256_json(audit),
        "collision_self_tests": deepcopy(tests),
    }]
    return diagnostics, trace


def _verify_base(diagnostics: dict[str, Any], trace: list[Any]) -> tuple[dict[str, set[int]], set[int]]:
    return phase5._verify_robot_base_physics_audit(
        diagnostics=diagnostics, trace=trace, robot_ids={"r0"}, planned_radius_m=0.4
    )


def test_mobile_base_attestation_resolves_exact_allowed_and_excluded_shapes() -> None:
    assert _verify_base(*_base_physics_diagnostics()) == ({"r0": {1, 2, 3, 4, 5}}, {6})


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("robot_base_contact_gate_passed",), False, "attestation is incomplete"),
        (("robot_base_physics_audit", 0), None, "entry is malformed"),
        (("robot_base_physics_audit", 0, "robot_id"), "other", "robot audit is invalid"),
        (("robot_base_physics_audit", 0, "allowed_shape_handles"), [1], "shape inventory"),
        (("robot_base_physics_audit", 0, "excluded_shape_handles"), [1], "shape inventory"),
        (("robot_base_physics_audit", 0, "shape_records", 0), None, "record is malformed"),
        (("robot_base_physics_audit", 0, "shape_records", 0, "alias"), None, "fields are invalid"),
        (("robot_base_physics_audit", 0, "shape_records", 0, "handle"), 99, "unknown handle"),
        (("robot_base_physics_audit", 0, "shape_records", 0, "after", "static"), 1, "readback is inconsistent"),
        (("robot_base_physics_audit", 0, "shape_records", 0, "alias"), "wrong-base", "footprint evidence"),
        (("robot_base_physics_audit", 0, "measured_footprint_radius_m"), 0.5, "footprint evidence"),
        (("measured_base_footprint_radius_m_by_robot", "r0"), 0.2, "footprint evidence"),
        (("robot_collision_self_tests", 0), None, "self-test record is malformed"),
        (("robot_collision_self_tests", 0, "verified"), False, "self-test is incomplete"),
        (("robot_collision_self_tests", 0, "probe_handle"), 98, "not base-bound"),
        (("robot_collision_self_tests", 0, "colliding_object_handles"), [6, 99], "not base-bound"),
    ],
)
def test_mobile_base_verification_rejects_rehashed_but_invalid_geometry_audit(
    path: tuple[Any, ...], value: Any, error: str
) -> None:
    diagnostics, trace = _base_physics_diagnostics()
    diagnostics = _changed(diagnostics, path, value)
    # Rehashing deliberately gets past integrity checks to exercise the physical
    # policy. A signed, internally invalid audit must still be rejected.
    diagnostics["robot_base_physics_audit_sha256"] = phase5._sha256_json(
        diagnostics["robot_base_physics_audit"]
    )
    with pytest.raises(ValueError, match=error):
        _verify_base(diagnostics, trace)


def test_mobile_base_audit_requires_the_matching_collision_ready_trace() -> None:
    diagnostics, _ = _base_physics_diagnostics()
    with pytest.raises(ValueError, match="trace is not bound"):
        _verify_base(diagnostics, [])


def _collision_fixture() -> tuple[ScenarioManifest, dict[str, Any], list[Any]]:
    scenario = _scenario()
    inventory = phase5._expected_collision_inventory(scenario.plan)
    groups = phase5._expected_collision_query_groups(scenario.plan)
    counts = dict(sorted(Counter(str(item["category"]) for item in inventory).items()))
    base = {"r0": [1, 2, 3, 4, 5], "r1": [11, 12, 13, 14, 15]}
    trace: list[Any] = [{
        "event": "collision_monitor_ready", "expected_queries_per_step": len(groups),
        "pair_category_counts": counts, "pair_inventory_sha256": phase5._sha256_json(inventory),
        "query_group_inventory": groups, "query_group_inventory_sha256": phase5._sha256_json(groups),
        "robot_ids": ["r0", "r1"], "module_ids": ["m0"], "obstacle_ids": ["4,4"],
        "aggregate_robot_base_membership": base,
        "aggregate_world_membership": [
            {"entity": "module:m0", "object_handle": 100},
            {"entity": "obstacle:4,4", "object_handle": 101},
        ],
        "aggregate_collection_membership_sha256": phase5._sha256_json({
            "robot_base_shape_handles": sorted([*base["r0"], *base["r1"]]),
            "world_object_handles": [100, 101],
        }),
    }, {
        "event": "collision_query_round", "physics_step": 1, "query_count": len(groups),
        "pair_category_counts": counts, "detected_collisions": [{
            **inventory[0], "permitted_logical_payload_overlap": False,
            "colliding_object_handles": [1, 11],
        }], "colliding_robot_ids": ["r0", "r1"],
    }]
    diagnostics = {
        "collision_pair_category_counts": counts,
        "physical_collision_event_count": 1, "physical_collision_stops": 2,
        "permitted_logical_payload_overlaps": 0,
    }
    return scenario, diagnostics, trace


def _verify_collisions(
    scenario: ScenarioManifest, diagnostics: dict[str, Any], trace: list[Any], *, queries: int = 2
) -> None:
    phase5._verify_collision_trace(
        scenario=scenario, trace=trace, diagnostics=diagnostics,
        physics_steps=1, expected_queries=queries,
        base_handles_by_robot={"r0": {1, 2, 3, 4, 5}, "r1": {11, 12, 13, 14, 15}},
        excluded_robot_shape_handles={6, 16},
    )


def test_collision_evidence_counts_raw_base_bound_events_and_stopped_robots() -> None:
    _verify_collisions(*_collision_fixture())


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ((0, "robot_ids"), ["r0"], "inventory trace"),
        ((0, "aggregate_robot_base_membership", "r0"), [6], "membership is incomplete"),
        ((0, "aggregate_world_membership", 0, "entity"), "module:unknown", "membership is invalid"),
        ((0, "aggregate_world_membership", 0, "object_handle"), 1, "membership is invalid"),
        ((0, "aggregate_world_membership", 1, "object_handle"), 100, "membership is invalid"),
        ((1, "query_count"), 1, "incomplete pair coverage"),
        ((1, "detected_collisions"), None, "malformed results"),
        ((1, "detected_collisions", 0), None, "malformed event"),
        ((1, "detected_collisions", 0, "entity_2"), "robot:unknown", "unknown or exempted pair"),
        ((1, "detected_collisions", 0, "permitted_logical_payload_overlap"), True, "unknown or exempted pair"),
        ((1, "detected_collisions", 0, "colliding_object_handles"), [1], "exact object pair"),
        ((1, "detected_collisions", 0, "colliding_object_handles"), [1, 16], "attested mobile base"),
        ((1, "colliding_robot_ids"), ["r0"], "inconsistent robot stops"),
        ((1, "physics_step"), 2, "one round per physics step"),
    ],
)
def test_collision_evidence_rejects_inventories_exemptions_and_counterfeit_pairs(
    path: tuple[Any, ...], value: Any, error: str
) -> None:
    scenario, diagnostics, trace = _collision_fixture()
    with pytest.raises(ValueError, match=error):
        _verify_collisions(scenario, diagnostics, _changed(trace, path, value))


@pytest.mark.parametrize(
    "metric", ["physical_collision_event_count", "physical_collision_stops", "permitted_logical_payload_overlaps"]
)
def test_collision_metrics_cannot_disagree_with_per_step_records(metric: str) -> None:
    scenario, diagnostics, trace = _collision_fixture()
    diagnostics[metric] += 1
    with pytest.raises(ValueError, match="diagnostics do not match"):
        _verify_collisions(scenario, diagnostics, trace)


def test_collision_rounds_cannot_be_duplicated_or_omitted() -> None:
    scenario, diagnostics, trace = _collision_fixture()
    with pytest.raises(ValueError, match="repeats a physics step"):
        _verify_collisions(scenario, diagnostics, [*trace, deepcopy(trace[-1])])
    with pytest.raises(ValueError, match="one round per physics step"):
        _verify_collisions(scenario, diagnostics, trace[:1])
    with pytest.raises(ValueError, match="query-group inventory"):
        _verify_collisions(scenario, diagnostics, trace, queries=1)
    diagnostics["collision_pair_category_counts"] = {}
    with pytest.raises(ValueError, match="categories do not match"):
        _verify_collisions(scenario, diagnostics, trace)


def _handshake_fixture() -> tuple[list[Any], dict[str, Any]]:
    config = DynamicCoppeliaConfig(control_hz=10)
    trace = [{
        "event": "remote_step_handshake_reconciled", "physics_step": 1,
        "error": "No such function: _*executed*_", "timestamp_s": 0.1,
        "observed_simulation_time_s": 0.1, "expected_simulation_time_s": 0.1,
    }, {
        "event": "remote_step_handshake_retried", "physics_step": 2,
        "error": "No such function: _*executed*_", "retry_index": 1,
        "timestamp_s": 0.1, "observed_simulation_time_s": 0.1,
        "previous_simulation_time_s": 0.1, "expected_simulation_time_s": 0.2,
    }]
    return trace, {
        "executor_config": config.model_dump(mode="json"),
        "remote_step_handshake_reconciliations": 1,
        "remote_step_handshake_retries": 1,
    }


def test_handshake_reconciliation_distinguishes_advanced_and_unadvanced_steps() -> None:
    trace, diagnostics = _handshake_fixture()
    phase5._verify_remote_step_handshake_reconciliations(
        trace=trace, diagnostics=diagnostics, physics_steps=2
    )


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "physics_step", 0), (0, "physics_step", 3),
        (0, "error", "other remote failure"), (0, "timestamp_s", 0.2),
        (0, "observed_simulation_time_s", 0.0), (0, "expected_simulation_time_s", 0.2),
        (1, "physics_step", 0), (1, "physics_step", 3), (1, "retry_index", 2),
        (1, "error", "other remote failure"), (1, "previous_simulation_time_s", 0),
        (1, "observed_simulation_time_s", 0.2), (1, "expected_simulation_time_s", 0.3),
        (1, "timestamp_s", 0.2),
    ],
)
def test_handshake_records_cannot_claim_an_unobserved_step_or_retry(
    index: int, field: str, value: Any
) -> None:
    trace, diagnostics = _handshake_fixture()
    trace[index][field] = value
    with pytest.raises(ValueError, match="handshake .* trace is invalid"):
        phase5._verify_remote_step_handshake_reconciliations(
            trace=trace, diagnostics=diagnostics, physics_steps=2
        )


@pytest.mark.parametrize("field", ["remote_step_handshake_reconciliations", "remote_step_handshake_retries"])
def test_handshake_summary_counts_require_raw_records(field: str) -> None:
    trace, diagnostics = _handshake_fixture()
    diagnostics[field] = 0
    with pytest.raises(ValueError, match="count is invalid"):
        phase5._verify_remote_step_handshake_reconciliations(
            trace=trace, diagnostics=diagnostics, physics_steps=2
        )


def _recovery_fixture() -> tuple[
    phase5.Phase5RunResult, list[RobotCommand], list[RobotTelemetry],
    list[Any], DynamicCoppeliaConfig,
]:
    config = DynamicCoppeliaConfig(disabled_settle_consecutive_samples=2)
    samples = {
        robot_id: [_telemetry(robot_id, 1, offset), _telemetry(robot_id, 2, 1 + offset)]
        for robot_id, offset in (("r1", -0.5), ("r2", 0.5))
    }
    routes = {
        robot_id: [Vec2(x=item.measured_pose.position.x, y=0) for item in rows]
        for robot_id, rows in samples.items()
    }
    replay = phase5.Phase5JobReplay(
        job_id="reassigned-job", module_id="m1", planned_order=1,
        planned_robot_ids=["r0", "r2"], executed_robot_ids=["r1", "r2"],
        planned_start_s=0, planned_pickup_s=1, planned_end_s=3,
        approach_routes=routes, carry_route=[Vec2(x=0, y=0), Vec2(x=1, y=0)],
        assigned_carry_routes=routes, return_routes=routes,
        formation_offsets={"r1": Vec2(x=-0.5, y=0), "r2": Vec2(x=0.5, y=0)},
        logical_carrier_offset=Vec3(x=0, y=0, z=0), logical_transport_height_m=1,
        measured_carry_routes=routes, measured_carry_samples=samples,
        approach_robot_positions={"r1": Vec3(x=-0.5, y=0), "r2": Vec3(x=0.5, y=0)},
        carry_robot_positions={"r1": Vec3(x=-0.5, y=0), "r2": Vec3(x=0.5, y=0)},
        return_robot_positions={"r1": Vec3(x=0.5, y=0), "r2": Vec3(x=1.5, y=0)},
        clearance_minima=[
            phase5.Phase5ClearanceMinimum(
                phase="carry", source=source, mover_kind="logical_payload",
                obstacle_kind="disabled_robot", minimum_surface_clearance_m=0.3,
                required_clearance_m=0.2, evaluated_pair_count=2,
                limiting_mover_id="m1", limiting_obstacle_id="r0",
            )
            for source in ("planned", "measured")
        ],
        installation_snap=phase5.Phase5InstallationSnap(
            from_pose=_pose(1), target_pose=_pose(1), at_s=2,
            target_pose_digest=phase5._sha256_json(_pose(1)),
        ),
        started_at_s=0, pickup_at_s=1, installed_at_s=2, returned_at_s=3,
        measured_poses=[row for rows in samples.values() for row in rows],
        maximum_formation_error_m=0, maximum_assignment_error_m=0,
        maximum_spacing_error_m=0, maximum_install_error_m=0,
        minimum_pickup_base_clearance_m=0.3, minimum_install_base_clearance_m=0.3,
        minimum_planned_payload_clearance_m=0.3, minimum_measured_payload_clearance_m=0.3,
        maximum_synchronized_formation_error_m=0, reassigned_after_unavailability=True,
    )
    recovery = phase5.Phase5RecoveryRecord(
        robot_id="r0", disabled_at_s=0.5, completion_fraction_at_disable=0.25,
        stop_command_index=1, remaining_module_count=3,
        reassigned_job_ids=["reassigned-job"], commands_after_stop=0,
        settled_at_s=0.7, settle_sample_count=3, settle_displacement_m=0, settled=True,
    )
    result = phase5.Phase5RunResult(
        scenario="unavailable_robot_recovery", evidence_kind="offline_harness",
        status="completed", scenario_id="recovery-boundary-fixture", scenario_seed=900,
        expected_module_count=4, installed_module_ids=["m0", "m1", "m2", "m3"],
        active_robot_ids=["r1", "r2"], replay=[replay], recovery=recovery,
        metrics={}, acceptance={}, live_gate_passed=False,
    )
    commands = [RobotCommand(
        robot_id=robot_id, timestamp_s=at, linear_velocity_mps=0,
        angular_velocity_rps=0, wheel_target_velocity_rad_s=(0, 0, 0, 0), source="recovery",
    ) for robot_id, at in (("r0", 0.5), ("r1", 1.0))]
    telemetry = [_telemetry("r0", at) for at in (0.5, 0.6, 0.7)]
    trace = [
        {"event": "module_installed", "module_id": "m0", "timestamp_s": 0.4},
        {"event": "robot_unavailable", "robot_id": "r0", "timestamp_s": 0.5},
        {"event": "disabled_robot_settled", "robot_id": "r0", "timestamp_s": 0.7},
        {"event": "module_installed", "module_id": "m1", "timestamp_s": 2},
    ]
    return result, commands, telemetry, trace, config


def _verify_recovery(
    result: phase5.Phase5RunResult, commands: list[RobotCommand],
    telemetry: list[RobotTelemetry], trace: list[Any], config: DynamicCoppeliaConfig,
) -> None:
    phase5._verify_recovery_evidence(result, result.replay, commands, telemetry, trace, config)


def test_recovery_requires_raw_stop_settle_reassignment_and_post_disable_clearance() -> None:
    _verify_recovery(*_recovery_fixture())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completion_fraction_at_disable", 0.24), ("commands_after_stop", 1),
        ("settled", False), ("settle_sample_count", 2), ("reassigned_job_ids", []),
    ],
)
def test_recovery_summary_cannot_hide_early_disable_or_missing_proof(field: str, value: Any) -> None:
    result, commands, telemetry, trace, config = _recovery_fixture()
    assert result.recovery is not None
    result.recovery = result.recovery.model_copy(update={field: value})
    with pytest.raises(ValueError, match="recovery record is incomplete"):
        _verify_recovery(result, commands, telemetry, trace, config)


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("no_record", "record is incomplete"),
        ("post_stop_command", "commands after its stop cutoff"),
        ("missing_telemetry", "missing telemetry"),
        ("still_moving", "did not measurably settle"),
        ("false_displacement", "displacement does not match"),
        ("false_reassignment", "list does not match replay"),
        ("no_post_disable_job", "no post-disable payload routes"),
        ("missing_clearance_source", "planned/measured payload clearance"),
        ("unsafe_clearance", "planned/measured payload clearance"),
        ("missing_settle_trace", "lacks disable or settle events"),
        ("early_disable_trace", "disable point does not match"),
        ("false_remaining_count", "disable point does not match"),
    ],
)
def test_recovery_evidence_is_rederived_instead_of_trusting_pass_flags(fault: str, error: str) -> None:
    result, commands, telemetry, trace, config = _recovery_fixture()
    assert result.recovery is not None
    if fault == "no_record":
        result.recovery = None
    elif fault == "post_stop_command":
        commands.append(commands[0].model_copy(update={"timestamp_s": 1}))
    elif fault == "missing_telemetry":
        telemetry.pop()
    elif fault == "still_moving":
        telemetry[-1] = telemetry[-1].model_copy(update={"linear_velocity_mps": 1})
    elif fault == "false_displacement":
        telemetry[-1] = telemetry[-1].model_copy(update={"measured_pose": _pose(1)})
    elif fault == "false_reassignment":
        result.replay[0].reassigned_after_unavailability = False
    elif fault == "no_post_disable_job":
        result.replay[0].pickup_at_s = 0.4
    elif fault == "missing_clearance_source":
        result.replay[0].clearance_minima.pop()
    elif fault == "unsafe_clearance":
        result.replay[0].clearance_minima[0].minimum_surface_clearance_m = 0.1
    elif fault == "missing_settle_trace":
        trace.pop(2)
    elif fault == "early_disable_trace":
        trace[0]["timestamp_s"] = 0.6
    else:
        result.recovery.remaining_module_count = 2
    with pytest.raises(ValueError, match=error):
        _verify_recovery(result, commands, telemetry, trace, config)


def _failing_call(*args: object) -> object:
    raise RuntimeError("bounded fixture transport failure")


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_runtime_numeric_queries_reject_bool_nonfinite_and_non_numeric_values(value: Any) -> None:
    sim = SimpleNamespace(getSimulationTime=lambda: value)
    with pytest.raises(DynamicCoppeliaError, match="value is invalid"):
        phase5._runtime_number(sim, "getSimulationTime")


@pytest.mark.parametrize("value", [True, 1.5, None, "1"])
def test_runtime_integer_queries_require_real_integers(value: Any) -> None:
    sim = SimpleNamespace(getSimulationState=lambda: value)
    with pytest.raises(DynamicCoppeliaError, match="value is invalid"):
        phase5._runtime_integer(sim, "getSimulationState")


@pytest.mark.parametrize("query", [phase5._runtime_integer, phase5._runtime_number])
def test_runtime_queries_explain_missing_methods_and_transport_errors(query: Any) -> None:
    with pytest.raises(DynamicCoppeliaError, match="simulator lacks"):
        query(object(), "getSimulationState")
    with pytest.raises(DynamicCoppeliaError, match="query failed") as exc:
        query(SimpleNamespace(getSimulationState=_failing_call), "getSimulationState")
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_runtime_queries_preserve_valid_values_and_arguments() -> None:
    assert phase5._runtime_number(SimpleNamespace(now=lambda: 2), "now") == 2.0
    assert phase5._runtime_integer(SimpleNamespace(uid=lambda handle: handle + 1), "uid", 17) == 18


def _integer_identity(**overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(**{
        "intparam_program_version": 10,
        "intparam_program_revision": 11,
        "getInt32Param": lambda parameter: 41000 if parameter == 10 else 7,
        **overrides,
    })


def test_simulator_identity_uses_string_version_or_strict_integer_fallback() -> None:
    sim = _integer_identity(
        stringparam_application_version=9, getStringParam=lambda parameter: " 4.10.0 rev 7 "
    )
    assert phase5._query_simulator_identity(sim) == ("4.10.0 rev 7", "string_parameter", None, None)
    sim.getStringParam = _failing_call
    assert phase5._query_simulator_identity(sim) == ("4.10.0 rev 7", "integer_parameters", 41000, 7)
    sim.getStringParam = lambda parameter: " "
    assert phase5._query_simulator_identity(sim) == ("4.10.0 rev 7", "integer_parameters", 41000, 7)


@pytest.mark.parametrize("version,revision", [(True, 0), (0, 0), (-1, 0), (41000, True), (41000, -1)])
def test_simulator_integer_identity_rejects_invalid_version_or_revision(version: Any, revision: Any) -> None:
    sim = _integer_identity(getInt32Param=lambda parameter: version if parameter == 10 else revision)
    with pytest.raises(DynamicCoppeliaError, match="integer application identity is invalid"):
        phase5._query_simulator_identity(sim)


def test_simulator_identity_missing_api_and_failed_query_are_not_guessed() -> None:
    with pytest.raises(DynamicCoppeliaError, match="no definitive application identity"):
        phase5._query_simulator_identity(object())
    with pytest.raises(DynamicCoppeliaError, match="identity query failed") as exc:
        phase5._query_simulator_identity(_integer_identity(getInt32Param=_failing_call))
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_remote_capability_metadata_has_deterministic_binary_and_nested_projection() -> None:
    payload = {2: memoryview(b"\x00\xff"), "nested": (None, bytearray(b"A"), True, 1.5)}
    assert phase5._runtime_jsonable(payload) == {
        "2": {"binary_hex": "00ff"}, "nested": [None, {"binary_hex": "41"}, True, 1.5],
    }
    with pytest.raises(DynamicCoppeliaError, match="unsupported set data"):
        phase5._runtime_jsonable({1, 2})


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (object(), "cannot query simulator capabilities"),
        (SimpleNamespace(call=_failing_call), "inventory query failed"),
        (SimpleNamespace(call=lambda *args: []), "not a mapping"),
        (SimpleNamespace(call=lambda *args: {}), "lacks live-attestation capabilities"),
    ],
)
def test_remote_capability_inventory_fails_closed(client: Any, error: str) -> None:
    with pytest.raises(DynamicCoppeliaError, match=error):
        phase5._capture_remote_api_info(SimpleNamespace(client=client))


def test_remote_capability_inventory_binds_complete_metadata_hash() -> None:
    metadata = {name: {"func": []} for name in phase5._LIVE_REQUIRED_REMOTE_API_CAPABILITIES}
    requests: list[tuple[object, object]] = []

    def call(function: str, arguments: list[str]) -> object:
        requests.append((function, arguments))
        return metadata

    executor = SimpleNamespace(client=SimpleNamespace(call=call))
    assert phase5._capture_remote_api_info(executor) == phase5._sha256_json(metadata)
    assert requests == [("zmqRemoteApi.info", ["sim"])]


def _runtime_binding_payload() -> dict[str, Any]:
    payload = {
        "run_id": "offline-validator-fixture", "scenario_id": "fixture", "scenario_seed": 900,
        "source_commit": "a" * 40, "source_tree_digest": "b" * 64,
        "plan_digest": "c" * 64, "configuration_digest": "d" * 64,
        "configuration_origin_digest": "e" * 64, "remote_api_client_version": "2.0.4",
        "remote_api_protocol_version": 2, "client_uuid": "12345678-1234-4234-9234-123456789abc",
        "client_send_count_before": 1, "client_send_count_after": 2,
        "endpoint_host": "127.0.0.1", "endpoint_port": 23000,
        "remote_api_capabilities": list(phase5._LIVE_REQUIRED_REMOTE_API_CAPABILITIES),
        "remote_api_info_sha256": "f" * 64, "simulator_version": "4.10.0 rev 7",
        "simulator_identity_source": "string_parameter", "scene_root_handle": 17,
        "scene_root_uid": 9001, "simulation_stopped_state": 0,
        "simulation_state_before": 0, "simulation_state_after": 0,
        "simulation_time_before_s": 0, "simulation_time_after_s": 0,
        "session_nonce": "1" * 64, "challenge_tag": "offline-validator-fixture",
        "physics_steps": 1, "command_count": 1, "telemetry_count": 1,
        "command_stream_sha256": "2" * 64, "telemetry_stream_sha256": "3" * 64,
        "installed_module_ids": ["m0"],
        "artifact_sha256": {name: "4" * 64 for name in phase5._REQUIRED_ARTIFACTS},
    }
    challenge = phase5._runtime_challenge_payload(**{
        name: payload[name] for name in (
            "session_nonce", "client_uuid", "endpoint_host", "endpoint_port", "plan_digest",
            "configuration_origin_digest", "scene_root_uid", "remote_api_info_sha256", "simulator_version",
        )
    })
    payload["challenge_payload_sha256"] = phase5._sha256_json(challenge)
    payload["challenge_response_sha256"] = payload["challenge_payload_sha256"]
    return _rebind_runtime_payload(payload)


def _rebind_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Construct is used only to include schema defaults in the canonical digest.
    # Every returned payload is then tested with full Pydantic validation below.
    canonical = phase5.Phase5RuntimeAttestation.model_construct(**payload).model_dump(
        mode="json", exclude={"binding_digest"}
    )
    payload["binding_digest"] = phase5._sha256_json(canonical)
    return payload


def test_runtime_model_accepts_a_self_consistent_binding_without_claiming_live_execution() -> None:
    record = phase5.Phase5RuntimeAttestation.model_validate(_runtime_binding_payload())
    assert record.run_id == "offline-validator-fixture"
    assert record.binding_digest == phase5._sha256_json(
        record.model_dump(mode="json", exclude={"binding_digest"})
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("client_send_count_after", 1, "did not observe remote calls"),
        ("simulation_state_before", 1, "bracket a stopped simulator"),
        ("simulation_state_after", 1, "bracket a stopped simulator"),
        ("remote_api_capabilities", [], "capability inventory differs"),
        ("artifact_sha256", {}, "artifact binding is incomplete"),
        ("artifact_sha256", {name: "not-a-digest" for name in phase5._REQUIRED_ARTIFACTS}, "artifact binding is incomplete"),
        ("challenge_response_sha256", "0" * 64, "response differs from request"),
        ("plan_digest", "0" * 64, "not bound to its origin fields"),
        ("simulator_program_version", 41000, "string simulator identity cannot include integer"),
        ("simulator_identity_source", "integer_parameters", "integer simulator identity is inconsistent"),
        ("binding_digest", "0" * 64, "binding digest differs"),
    ],
)
def test_runtime_model_rejects_rehashed_but_semantically_invalid_attestation(
    field: str, value: Any, error: str
) -> None:
    payload = _runtime_binding_payload()
    payload[field] = value
    if field != "binding_digest":
        _rebind_runtime_payload(payload)
    with pytest.raises(ValueError, match=error):
        phase5.Phase5RuntimeAttestation.model_validate(payload)


def test_runtime_integer_identity_must_match_both_version_components() -> None:
    payload = _runtime_binding_payload()
    payload.update({
        "simulator_identity_source": "integer_parameters",
        "simulator_program_version": 41000, "simulator_program_revision": 7,
    })
    assert phase5.Phase5RuntimeAttestation.model_validate(
        _rebind_runtime_payload(payload)
    ).simulator_version == "4.10.0 rev 7"
    payload["simulator_program_revision"] = 8
    with pytest.raises(ValueError, match="integer simulator identity is inconsistent"):
        phase5.Phase5RuntimeAttestation.model_validate(_rebind_runtime_payload(payload))


def _provenance_payload() -> dict[str, Any]:
    return {
        "run_id": "offline-fixture", "generated_at": "2026-10-10T00:00:00Z",
        "evidence_kind": "offline_harness", "live_evidence": False, "run_status": "completed",
        "live_gate_passed": False, "approval_gate_confirmed": False, "scenario": "nominal",
        "scenario_id": "fixture", "scenario_seed": 900, "source_commit": "a" * 40,
        "source_dirty": False, "source_tree_digest": "b" * 64, "plan_digest": "c" * 64,
        "configuration_digest": "d" * 64, "limitations": ["offline validator fixture"], "artifacts": [],
    }


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"live_evidence": True}, "flag conflicts"),
        ({"live_gate_passed": True}, "lacks a complete simulator attestation"),
        ({"runtime_attestation_digest": "0" * 64}, "requires exactly one runtime attestation"),
        ({"evidence_kind": "live_coppelia", "live_evidence": True}, "requires exactly one runtime attestation"),
    ],
)
def test_provenance_cannot_promote_offline_or_unattested_evidence(updates: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        phase5.Phase5ProvenanceManifest.model_validate({**_provenance_payload(), **updates})


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"planned_pickup_s": 4}, "planned job timestamps are not monotonic"),
        ({"pickup_at_s": 4}, "measured job timestamps are not monotonic"),
        ({"approach_routes": {}}, "route identities"),
        ({"carry_route": []}, "carrier-route horizon"),
        ({"assigned_carry_routes": {"r1": [], "r2": []}}, "carrier-route horizon"),
    ],
)
def test_replay_schema_rejects_inconsistent_temporal_and_team_contracts(
    updates: dict[str, Any], error: str
) -> None:
    result, *_ = _recovery_fixture()
    payload = result.replay[0].model_dump(mode="json")
    with pytest.raises(ValueError, match=error):
        phase5.Phase5JobReplay.model_validate({**payload, **updates})


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("not_rigid", "not rigidly derived"),
        ("missing_sample", "one point per sample"),
        ("wrong_robot", "sample identity mismatch"),
        ("out_of_interval", "outside the job interval"),
        ("not_measured", "route does not match telemetry"),
        ("not_synchronized", "share one horizon"),
    ],
)
def test_replay_schema_binds_measured_routes_to_identified_synchronized_samples(fault: str, error: str) -> None:
    result, *_ = _recovery_fixture()
    payload = result.replay[0].model_dump(mode="json")
    if fault == "not_rigid":
        payload["assigned_carry_routes"]["r1"][0]["x"] = 42
    elif fault == "missing_sample":
        payload["measured_carry_samples"]["r1"] = []
    elif fault == "wrong_robot":
        payload["measured_carry_samples"]["r1"][0]["robot_id"] = "r0"
    elif fault == "out_of_interval":
        payload["measured_carry_samples"]["r1"][0]["timestamp_s"] = 0
    elif fault == "not_measured":
        payload["measured_carry_routes"]["r1"][0]["x"] = 42
    else:
        payload["measured_carry_samples"]["r1"].pop()
        payload["measured_carry_routes"]["r1"].pop()
    with pytest.raises(ValueError, match=error):
        phase5.Phase5JobReplay.model_validate(payload)


@pytest.mark.parametrize(
    "updates", [
        {"evaluated_pair_count": 1},
        {"minimum_surface_clearance_m": 0.3},
        {"limiting_mover_id": "payload"},
        {"minimum_surface_clearance_m": 0.3, "evaluated_pair_count": 1},
    ],
)
def test_clearance_minimum_requires_a_complete_limiting_pair_or_empty_inventory(updates: dict[str, Any]) -> None:
    payload = {
        "phase": "carry", "source": "measured", "mover_kind": "logical_payload",
        "obstacle_kind": "disabled_robot", "required_clearance_m": 0.2, "evaluated_pair_count": 0,
    }
    assert phase5.Phase5ClearanceMinimum.model_validate(payload).minimum_surface_clearance_m is None
    with pytest.raises(ValueError, match="clearance minimum|clearance inventory"):
        phase5.Phase5ClearanceMinimum.model_validate({**payload, **updates})


def test_small_jsonl_loader_reports_exact_invalid_line_and_never_reuses_invalid_source(tmp_path: Path) -> None:
    path = tmp_path / "commands.jsonl"
    _, commands, *_ = _recovery_fixture()
    path.write_text(commands[0].model_dump_json() + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"commands.jsonl:2 is blank"):
        phase5._load_jsonl_models(path, RobotCommand, source_rows=commands)
    path.write_text(commands[0].model_dump_json() + "\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"commands.jsonl:2 is not valid JSON"):
        phase5._load_jsonl_models(path, RobotCommand, source_rows=commands)


def test_exact_source_reuse_compares_canonical_hash_and_row_count() -> None:
    path = Path("commands.jsonl")
    _, commands, *_ = _recovery_fixture()
    values = [item.model_dump(mode="json") for item in commands]
    with pytest.raises(ValueError, match="row count differs"):
        phase5._load_json_models(path, values[:1], RobotCommand, source_rows=commands)
    with pytest.raises(ValueError, match=r"commands.jsonl:2 differs"):
        phase5._load_json_models(path, values, RobotCommand, source_rows=commands[:1])
    values[0]["timestamp_s"] = 0.6
    with pytest.raises(ValueError, match=r"commands.jsonl:1 differs"):
        phase5._load_json_models(path, values, RobotCommand, source_rows=commands)


def _clearance_plan() -> BuildPlan:
    plan = _scenario().plan.model_copy(deep=True)
    plan.site_grid = SiteGrid(
        width=21, height=21, resolution_m=1, origin=Vec2(x=-10, y=-10),
        obstacle_cells=[(18, 18)],
    )
    plan.robots[0].start_pose = _pose(-4, -3)
    plan.robots[1].start_pose = _pose(-4, -1)
    return plan


def test_initial_clearance_allows_safe_layout_and_equivalent_full_turn_rotations() -> None:
    plan = _clearance_plan()
    phase5._validate_initial_physical_clearance(plan, phase5.Phase5PhysicalYardConfig())
    plan.modules[0].staging_pose.rotation_rpy_degrees = Vec3(x=360, y=-360, z=720)
    phase5._validate_initial_physical_clearance(plan, phase5.Phase5PhysicalYardConfig())


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("rotation", "requires in-transit payload rotation"),
        ("staging_overlap", "staging modules overlap"),
        ("robot_overlap", "dispatch robots overlap"),
        ("robot_in_payload", "overlaps staged module"),
        ("obstacle_in_cottage", "obstacle overlaps the cottage"),
        ("obstacle_in_payload", "obstacle overlaps a staged module"),
        ("obstacle_in_robot", "obstacle overlaps dispatch robot"),
    ],
)
def test_initial_clearance_rejects_every_physical_overlap_class(fault: str, error: str) -> None:
    plan = _clearance_plan()
    if fault == "rotation":
        plan.modules[0].staging_pose.rotation_rpy_degrees.z = 45
    elif fault == "staging_overlap":
        plan.modules.append(plan.modules[0].model_copy(update={"module_id": "m1"}, deep=True))
    elif fault == "robot_overlap":
        plan.robots[1].start_pose = plan.robots[0].start_pose.model_copy(deep=True)
    elif fault == "robot_in_payload":
        plan.robots[0].start_pose = plan.modules[0].staging_pose.model_copy(deep=True)
    elif fault == "obstacle_in_cottage":
        plan.site_grid.obstacle_cells = [(10, 10)]
    elif fault == "obstacle_in_payload":
        plan.site_grid.obstacle_cells = [(12, 10)]
    else:
        plan.site_grid.obstacle_cells = [(6, 7)]
    with pytest.raises(ValueError, match=error):
        phase5._validate_initial_physical_clearance(plan, phase5.Phase5PhysicalYardConfig())


@pytest.mark.parametrize("yaw", [0, 30, 45, 90, 180, -90, 450])
def test_yaw_projection_matches_rpy_projection_and_analytic_rotated_rectangle(yaw: float) -> None:
    module = _scenario().plan.modules[0]
    module.dimensions = Dimensions3D(width=2, depth=1, height=0.2)
    radians = math.radians(yaw)
    expected = (
        2 * abs(math.cos(radians)) + abs(math.sin(radians)),
        2 * abs(math.sin(radians)) + abs(math.cos(radians)),
    )
    assert phase5._oriented_size_xy(module, yaw) == pytest.approx(expected)
    assert phase5._oriented_size_xy(module, Vec3(x=0, y=0, z=yaw)) == pytest.approx(expected)


def _rigid_route_arguments() -> dict[str, Any]:
    plan = _clearance_plan()
    carrier = [Vec2(x=0, y=0), Vec2(x=1, y=0)]
    return {
        "plan": plan, "module": plan.modules[0], "installed_module_ids": set(),
        "robot_positions": {"r0": Vec2(x=0, y=-1), "r1": Vec2(x=0, y=1)},
        "active_robot_ids": {"r0", "r1"}, "carrier_route": carrier,
        "base_routes": {
            "r0": [Vec2(x=point.x, y=-1) for point in carrier],
            "r1": [Vec2(x=point.x, y=1) for point in carrier],
        },
        "transport_height_m": 1, "disabled_robot_ids": set(), "source": "planned",
        "config": phase5.Phase5PhysicalYardConfig(),
    }


def test_rigid_route_validation_returns_measured_clearance_inventories_for_safe_corridor() -> None:
    records = phase5._validate_rigid_carrier_route(**_rigid_route_arguments())
    payload = [record for record in records if record.obstacle_kind == "active_payload"]
    assert len(payload) == 1
    assert payload[0].evaluated_pair_count == 2
    assert payload[0].minimum_surface_clearance_m == pytest.approx(0.38)
    assert all(record.phase == "carry" and record.source == "planned" for record in records)


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("horizon", "do not share the carrier horizon"),
        ("payload_bounds", "logical payload leaves the site grid"),
        ("base_bounds", "r0 leaves the site grid"),
        ("base_payload_overlap", "clearance"),
    ],
)
def test_rigid_route_validation_rejects_unbounded_or_intersecting_transport(fault: str, error: str) -> None:
    arguments = _rigid_route_arguments()
    if fault == "horizon":
        arguments["base_routes"]["r0"].pop()
    elif fault == "payload_bounds":
        arguments["carrier_route"][1] = Vec2(x=10, y=0)
    elif fault == "base_bounds":
        arguments["base_routes"]["r0"][1] = Vec2(x=10, y=-1)
    else:
        arguments["base_routes"]["r0"] = arguments["carrier_route"]
    with pytest.raises(RoutingError, match=error):
        phase5._validate_rigid_carrier_route(**arguments)


def _native_bundle_manifest(tmp_path: Path) -> dict[str, Any]:
    payload = _provenance_payload()
    artifacts = []
    for filename in sorted(phase5._REQUIRED_ARTIFACTS):
        (tmp_path / filename).write_bytes(b"x")
        artifacts.append({"path": filename, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()})
    payload["artifacts"] = artifacts
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_native_bundle_requires_manifest_before_reading_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing manifest.json"):
        phase5.verify_phase5_artifact_bundle(tmp_path)


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("naive_timestamp", "timestamp must include a timezone"),
        ("duplicate_record", "duplicate artifact paths"),
        ("extra_file", "artifact inventory does not match"),
        ("missing_file", "artifact inventory does not match"),
        ("wrong_size", "artifact size mismatch"),
        ("wrong_digest", "artifact digest mismatch"),
        ("missing_required", "inventory is not canonical"),
    ],
)
def test_native_bundle_integrity_guards_precede_parsing_untrusted_evidence(
    tmp_path: Path, fault: str, error: str
) -> None:
    payload = _native_bundle_manifest(tmp_path)
    first = payload["artifacts"][0]
    if fault == "naive_timestamp":
        payload["generated_at"] = "2026-10-10T00:00:00"
    elif fault == "duplicate_record":
        payload["artifacts"].append(deepcopy(first))
    elif fault == "extra_file":
        (tmp_path / "unexpected.bin").write_bytes(b"extra")
    elif fault == "missing_file":
        (tmp_path / first["path"]).unlink()
    elif fault == "wrong_size":
        first["bytes"] = 2
    elif fault == "wrong_digest":
        first["sha256"] = "0" * 64
    else:
        payload["artifacts"].pop(0)
        (tmp_path / first["path"]).unlink()
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        phase5.verify_phase5_artifact_bundle(tmp_path)
