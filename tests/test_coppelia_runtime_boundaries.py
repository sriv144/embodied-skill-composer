"""Bounded simulator safety tests; no network sockets or live API execution.

The official client below is allocated without calling its constructor. Its
transport and simulator methods are local fixtures, never evidence of a live run.
"""

from __future__ import annotations

import math
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import embodied_skill_composer.construction.coppelia_dynamic as dynamic
import embodied_skill_composer.construction.coppelia_phase5 as phase5
from embodied_skill_composer.construction.intelligence_models import RobotCommand, RobotTelemetry, ScenarioManifest
from embodied_skill_composer.construction.models import BuildPlan


def _scenario() -> ScenarioManifest:
    plan = BuildPlan.model_validate({
        "plan_id": "runtime-fixture",
        "design": {
            "design_id": "fixture", "title": "Offline unit fixture",
            "footprint_width_m": 2, "footprint_depth_m": 2,
            "floor_plan": {"walls": [], "approved": True},
        },
        "modules": [{
            "module_id": "m0", "module_type": "foundation", "mesh_node": "m0",
            "target_pose": {"position": {"x": 0, "y": 0, "z": 0}},
            "staging_pose": {"position": {"x": 2, "y": 0, "z": 0}},
            "dimensions": {"width": 1, "depth": 1, "height": 0.2},
            "mass_kg": 1, "grip_points": [{"x": 0, "y": 0, "z": 0}],
            "required_team_size": 2, "install_duration_s": 1, "material": "fixture",
        }],
        "robots": [{
            "robot_id": robot_id, "payload_capacity_kg": 10, "speed_mps": 1,
            "battery_capacity_wh": 100,
            "start_pose": {"position": {"x": index * 2, "y": -3, "z": 0}},
        } for index, robot_id in enumerate(("r0", "r1"))],
        "site_grid": {"width": 10, "height": 10, "obstacle_cells": [[8, 8]]},
    })
    return ScenarioManifest(scenario_id="runtime-fixture", seed=900, split="test", plan=plan)


def _executor(sim: Any = None) -> dynamic.DynamicCoppeliaExecutor:
    executor = dynamic.DynamicCoppeliaExecutor(_scenario().plan)
    executor.sim = sim
    return executor


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("fixture transport failed")


class _PhysicsSim:
    object_shape_type = 1
    shapeintparam_respondable = 2
    shapeintparam_static = 3

    def __init__(self) -> None:
        self.aliases = {
            1: "construction_intelligence_r0", 2: "wheel_respondable_fl",
            3: "wheel_respondable_rl", 4: "wheel_respondable_rr",
            5: "wheel_respondable_fr", 6: "gripper_visual",
        }
        self.integers = {(handle, parameter): 1 for handle in self.aliases for parameter in (2, 3)}
        self.booleans = {(handle, "collidable"): True for handle in self.aliases}
        self.dimensions = [0.1, 0.1, 0.1]
        self.bounding_pose = [0, 0, 0, 0, 0, 0, 1]
        self.matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    def getObjectsInTree(self, root: int, shape_type: int, options: int) -> list[int]:
        assert (root, shape_type, options) == (1, 1, 0)
        return list(self.aliases)

    def getObjectAlias(self, handle: int, options: int) -> str:
        assert options == 1
        return self.aliases[handle]

    def getObjectInt32Param(self, handle: int, parameter: int) -> int:
        return self.integers[(handle, parameter)]

    def setObjectInt32Param(self, handle: int, parameter: int, value: int) -> None:
        self.integers[(handle, parameter)] = value

    def getBoolProperty(self, handle: int, name: str) -> bool:
        return self.booleans[(handle, name)]

    def setBoolProperty(self, handle: int, name: str, value: bool) -> None:
        self.booleans[(handle, name)] = value

    def getShapeBB(self, handle: int) -> list[list[float]]:
        return [self.dimensions, self.bounding_pose]

    def getObjectMatrix(self, handle: int, relative_to: int) -> list[float]:
        assert relative_to == 1
        return self.matrix


def test_base_only_physics_keeps_five_dynamic_contact_shapes_and_excludes_gripper() -> None:
    sim = _PhysicsSim()
    executor = _executor(sim)
    executor._configure_youbot_base_physics("r0", 1)
    assert executor.robot_base_shape_handles == {"r0": {1, 2, 3, 4, 5}}
    assert executor.robot_excluded_shape_handles == {"r0": {6}}
    assert sim.integers[(6, sim.shapeintparam_static)] == 1
    assert sim.booleans[(6, "collidable")] is False
    assert all(sim.integers[(handle, sim.shapeintparam_static)] == 0 for handle in range(1, 6))
    audit = executor.robot_base_physics_audit[0]
    assert audit["measured_footprint_radius_m"] == pytest.approx(math.sqrt(0.05**2 * 3))
    assert audit["base_only_policy_verified"] is True


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("missing_api", "attestation is unavailable"), ("root_absent", "root is not an auditable shape"),
        ("wheel_missing", "exactly one wheel_respondable_fr"),
        ("wheel_duplicate", "exactly one wheel_respondable_fr"),
        ("root_is_wheel", "five unique shapes"),
        ("no_excluded", "no excluded arm"), ("readback", "read back"),
        ("non_bool", "read back"), ("oversized", "exceeds planned radius"),
    ],
)
def test_base_physics_fails_closed_when_inventory_or_measured_readback_is_unsafe(
    monkeypatch: pytest.MonkeyPatch, fault: str, error: str
) -> None:
    sim = _PhysicsSim()
    if fault == "missing_api":
        monkeypatch.setattr(sim, "getShapeBB", None)
    elif fault == "root_absent":
        sim.aliases.pop(1)
    elif fault == "wheel_missing":
        sim.aliases.pop(5)
    elif fault == "wheel_duplicate":
        sim.aliases[7] = sim.aliases[5]
    elif fault == "root_is_wheel":
        sim.aliases[1] = sim.aliases.pop(5)
    elif fault == "no_excluded":
        sim.aliases.pop(6)
    elif fault == "readback":
        monkeypatch.setattr(sim, "setObjectInt32Param", lambda *args: None)
    elif fault == "non_bool":
        sim.booleans[(1, "collidable")] = 1
    else:
        sim.dimensions = [1, 1, 1]
    executor = _executor(sim)
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        executor._configure_youbot_base_physics("r0", 1)
    assert not executor.robot_base_physics_audit


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("bad_shape", "malformed bounding-box"), ("short_matrix", "invalid footprint geometry"),
        ("nan_dimension", "invalid footprint geometry"), ("zero_dimension", "invalid footprint geometry"),
        ("zero_quaternion", "zero bounding-box quaternion"),
    ],
)
def test_measured_base_footprint_requires_finite_native_bbox_and_pose_evidence(
    monkeypatch: pytest.MonkeyPatch, fault: str, error: str
) -> None:
    sim = _PhysicsSim()
    if fault == "bad_shape":
        monkeypatch.setattr(sim, "getShapeBB", lambda handle: [[1, 2], [0] * 7])
    elif fault == "short_matrix":
        sim.matrix.pop()
    elif fault == "nan_dimension":
        sim.dimensions[0] = float("nan")
    elif fault == "zero_dimension":
        sim.dimensions[0] = 0
    else:
        sim.bounding_pose[3:] = [0, 0, 0, 0]
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        dynamic._shape_xy_footprint_radius_m(sim, shape_handle=1, robot_handle=1)


class _LegacyScriptSim:
    scriptintparam_enabled = 10

    def __init__(self, initial: Any = 1, *, ignore_write: bool = False) -> None:
        self.enabled = initial
        self.ignore_write = ignore_write

    def getObjectInt32Param(self, handle: int, parameter: int) -> Any:
        assert (handle, parameter) == (17, self.scriptintparam_enabled)
        return self.enabled

    def setObjectInt32Param(self, handle: int, parameter: int, value: int) -> None:
        assert (handle, parameter) == (17, self.scriptintparam_enabled)
        if not self.ignore_write:
            self.enabled = value


@pytest.mark.parametrize("disabled", [True, False])
def test_legacy_script_control_proves_requested_ownership_with_state_readback(disabled: bool) -> None:
    sim = _LegacyScriptSim(initial=int(disabled))
    before, after, api = _executor(sim)._set_script_disabled(17, disabled=disabled)
    assert before is not disabled
    assert after is disabled
    assert api == "scriptintparam_enabled_legacy"
    assert sim.enabled == int(not disabled)


@pytest.mark.parametrize("initial", [True, None, "1"])
def test_legacy_script_control_rejects_indeterminate_enabled_state(initial: Any) -> None:
    with pytest.raises(dynamic.DynamicCoppeliaError, match="indeterminate enabled state"):
        _executor(_LegacyScriptSim(initial))._set_script_disabled(17, disabled=True)


def test_script_control_does_not_assume_write_success_or_api_availability() -> None:
    with pytest.raises(dynamic.DynamicCoppeliaError, match="state readback did not match"):
        _executor(_LegacyScriptSim(ignore_write=True))._set_script_disabled(17, disabled=True)
    with pytest.raises(dynamic.DynamicCoppeliaError, match="state cannot be verified"):
        _executor(object())._set_script_disabled(17, disabled=True)


@pytest.mark.parametrize("source", [None, "", "  ", b"source"])
def test_script_source_must_be_nonempty_auditable_text(source: Any) -> None:
    sim = SimpleNamespace(getStringProperty=lambda handle, name: source)
    with pytest.raises(dynamic.DynamicCoppeliaError, match="indeterminate source"):
        _executor(sim)._read_script_source(17)


def test_script_source_uses_legacy_audited_parameter_or_fails_closed() -> None:
    sim = SimpleNamespace(scriptstringparam_text=1, getScriptStringParam=lambda handle, parameter: "source")
    assert _executor(sim)._read_script_source(17) == "source"
    with pytest.raises(dynamic.DynamicCoppeliaError, match="source cannot be audited"):
        _executor(object())._read_script_source(17)


def test_robot_pose_writes_are_initialization_only_and_rejected_before_remote_write() -> None:
    writes = []
    executor = _executor(SimpleNamespace(setObjectPosition=lambda handle, pose: writes.append((handle, pose))))
    executor._set_initial_robot_position(17, [1, 2, 0.12])
    assert writes == [(17, [1, 2, 0.12])]
    assert executor.initial_robot_pose_writes == 1
    assert executor.runtime_events[-1]["simulation_started"] is False
    executor.started = True
    with pytest.raises(dynamic.DynamicCoppeliaError, match="forbidden after simulation start"):
        executor._set_initial_robot_position(17, [10, 20, 0.12])
    assert writes == [(17, [1, 2, 0.12])]
    assert executor.post_start_robot_pose_writes == 1


@pytest.mark.parametrize("observed,expected", [(0.0, "not_started"), (0.05, "completed")])
def test_step_handshake_classifies_only_zero_or_exactly_one_measured_step(observed: float, expected: str) -> None:
    executor = _executor(SimpleNamespace(getSimulationTime=lambda: observed))
    assert executor._classify_remote_step_handshake(RuntimeError("fixture")) == (expected, observed)


@pytest.mark.parametrize("observed", [-0.05, 0.1, float("nan"), float("inf"), "invalid"])
def test_step_handshake_does_not_accept_skipped_negative_or_unmeasurable_time(observed: Any) -> None:
    executor = _executor(SimpleNamespace(getSimulationTime=lambda: observed))
    with pytest.raises(dynamic.DynamicCoppeliaError, match="without"):
        executor._classify_remote_step_handshake(RuntimeError("fixture"))


def test_step_handshake_failure_preserves_transport_cause_without_guessing_progress() -> None:
    for sim in (object(), SimpleNamespace(getSimulationTime=_fail)):
        with pytest.raises(dynamic.DynamicCoppeliaError, match="without measurable simulator time") as exc:
            _executor(sim)._classify_remote_step_handshake(RuntimeError("fixture"))
        assert isinstance(exc.value.__cause__, RuntimeError)


def test_scene_save_requires_stopped_simulator_and_native_binary_buffer(tmp_path: Path) -> None:
    executor = _executor(SimpleNamespace(saveScene=lambda: b"VREP" + b"\0" * 124))
    path = executor.save_scene(tmp_path / "scene.ttt")
    assert path.read_bytes() == b"VREP" + b"\0" * 124
    executor.started = True
    with pytest.raises(dynamic.DynamicCoppeliaError, match="after simulation stops"):
        executor.save_scene(tmp_path / "active.ttt")
    assert not (tmp_path / "active.ttt").exists()


@pytest.mark.parametrize(
    ("sim", "error"),
    [(object(), "serialization is unavailable"), (SimpleNamespace(saveScene=_fail), "failed to serialize"),
     (SimpleNamespace(saveScene=lambda: "VREP"), "did not return a binary"),
     (SimpleNamespace(saveScene=lambda: b"bad"), "scene")],
)
def test_scene_save_refuses_unavailable_failed_or_non_native_serialization(tmp_path: Path, sim: Any, error: str) -> None:
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        _executor(sim).save_scene(tmp_path / "scene.ttt")
    assert not (tmp_path / "scene.ttt").exists()


class _AttestationSim:
    simulation_stopped = 0
    stringparam_application_version = 10

    def __init__(self, client: Any) -> None:
        self.client = client
        self.state = 0
        self.time = 0.0
        self.uid = 9001
        self.version = "4.10.0 rev 7"
        self.data: dict[tuple[int, str], bytes] = {}

    def getStringParam(self, parameter: int) -> str:
        self.client.sendCnt += 1
        return self.version

    def getObjectUid(self, handle: int) -> int:
        self.client.sendCnt += 1
        return self.uid

    def getSimulationState(self) -> int:
        self.client.sendCnt += 1
        return self.state

    def getSimulationTime(self) -> float:
        self.client.sendCnt += 1
        return self.time

    def writeCustomDataBlock(self, handle: int, tag: str, payload: bytes) -> None:
        self.client.sendCnt += 1
        self.data[(handle, tag)] = payload

    def readCustomDataBlock(self, handle: int, tag: str) -> bytes:
        self.client.sendCnt += 1
        return self.data[(handle, tag)]


@pytest.fixture
def origin_fixture(monkeypatch: pytest.MonkeyPatch):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient.__new__(RemoteAPIClient)  # No constructor, ZMQ socket or remote call.
    client.socket = SimpleNamespace(close=lambda: None)
    client.context = SimpleNamespace(term=lambda: None)
    client.uuid = "12345678-1234-4234-9234-123456789abc"
    client.VERSION = 2
    client.sendCnt = 0
    metadata = {name: {"func": []} for name in phase5._LIVE_REQUIRED_REMOTE_API_CAPABILITIES}

    def call(function: str, arguments: list[str]) -> dict[str, object]:
        assert (function, arguments) == ("zmqRemoteApi.info", ["sim"])
        client.sendCnt += 1
        return metadata

    client.call = call
    sim = _AttestationSim(client)
    scenario = _scenario()
    executor = dynamic.DynamicCoppeliaExecutor(scenario.plan)
    executor.client = client
    executor.sim = sim
    executor.root_handle = 17
    executor.is_ready = True
    monkeypatch.setattr(phase5, "version", lambda name: "2.0.4")
    yard = phase5.Phase5PhysicalYardManifest(
        source_plan_digest="a" * 64, transformed_plan_digest="b" * 64,
        allocator_configuration_digest="c" * 64, configuration=phase5.Phase5PhysicalYardConfig(),
        shelf_row_count=1, staging_bounds_xy=(1, 3, -1, 1), grid_origin=scenario.plan.site_grid.origin,
        grid_width=10, grid_height=10, module_count=1, robot_count=2, obstacle_count=1,
        initial_clearance_preflight_passed=False, sequential_route_preflight_passed=False,
        all_robots_active_preflight_passed=False, continuous_world_clearance_preflight_passed=False,
        recovery_route_preflight_passed=False, maximum_preflight_formation_offset_m=0,
        maximum_preflight_transport_height_m=0,
    )
    yield executor, scenario, yard, metadata
    phase5._LIVE_RUNTIME_SESSIONS.pop(executor, None)


def test_official_client_origin_requires_exact_class_and_available_package_identity(origin_fixture) -> None:
    executor, *_ = origin_fixture
    identity = phase5._official_remote_client_identity(executor)
    assert identity.client_uuid == executor.client.uuid
    assert identity.package_version == "2.0.4"
    assert identity.send_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("uuid", "invalid"), ("VERSION", True), ("VERSION", 0), ("sendCnt", True), ("sendCnt", -1)],
)
def test_client_origin_rejects_incomplete_counters_protocol_and_uuid(origin_fixture, field: str, value: Any) -> None:
    executor, *_ = origin_fixture
    setattr(executor.client, field, value)
    with pytest.raises(dynamic.DynamicCoppeliaError, match="identity is incomplete"):
        phase5._official_remote_client_identity(executor)


def test_client_origin_rejects_injected_clients_disconnected_scene_and_substitutes(origin_fixture) -> None:
    executor, *_ = origin_fixture
    executor.client_factory = lambda config: executor.client
    with pytest.raises(dynamic.DynamicCoppeliaError, match="injected simulator clients"):
        phase5._official_remote_client_identity(executor)
    executor.client_factory = None
    executor.is_ready = False
    with pytest.raises(dynamic.DynamicCoppeliaError, match="connected generated scene"):
        phase5._official_remote_client_identity(executor)
    executor.is_ready = True
    executor.client = SimpleNamespace(uuid="12345678-1234-4234-9234-123456789abc", VERSION=2, sendCnt=0)
    with pytest.raises(dynamic.DynamicCoppeliaError, match="official RemoteAPIClient"):
        phase5._official_remote_client_identity(executor)


@pytest.mark.parametrize("fault", ["missing", "empty"])
def test_client_origin_does_not_invent_an_unavailable_distribution_version(origin_fixture, monkeypatch, fault: str) -> None:
    executor, *_ = origin_fixture
    if fault == "missing":
        def version(name: str) -> str:
            raise PackageNotFoundError(name)
        monkeypatch.setattr(phase5, "version", version)
    else:
        monkeypatch.setattr(phase5, "version", lambda name: " ")
    with pytest.raises(dynamic.DynamicCoppeliaError, match="package version is"):
        phase5._official_remote_client_identity(executor)


def test_attestation_origin_roundtrip_and_finalization_bind_the_same_local_fixture_session(origin_fixture) -> None:
    executor, scenario, yard, _ = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    assert not session.finalized
    assert session.challenge_payload == executor.sim.data[(17, session.challenge_tag)]
    assert phase5._finalize_live_runtime_attestation(executor) is session
    assert session.finalized
    assert session.client_send_count_after > session.client_send_count_before
    event_count = len(executor.runtime_events)
    assert phase5._finalize_live_runtime_attestation(executor) is session
    assert len(executor.runtime_events) == event_count  # Finalization is idempotent.
    with pytest.raises(dynamic.DynamicCoppeliaError, match="already owns"):
        phase5._begin_live_runtime_attestation(executor, scenario, yard)


@pytest.mark.parametrize(
    ("fault", "error"),
    [("started", "before simulator execution"), ("steps", "before simulator execution"),
     ("bad_uid", "root UID is invalid"), ("bad_stopped_constant", "stopped-state constant is invalid"),
     ("running_simulator", "begin while simulation is stopped"), ("negative_time", "time cannot be negative"),
     ("no_write", "cannot round-trip"), ("read_failure", "round-trip failed"),
     ("bad_buffer", "not a byte buffer"), ("wrong_response", "challenge response differs")],
)
def test_attestation_begin_rejects_execution_without_an_unchanged_challenge_origin(
    origin_fixture, monkeypatch, fault: str, error: str
) -> None:
    executor, scenario, yard, _ = origin_fixture
    if fault == "started":
        executor.started = True
    elif fault == "steps":
        executor.physics_steps = 1
    elif fault == "bad_uid":
        executor.sim.uid = 0
    elif fault == "bad_stopped_constant":
        executor.sim.simulation_stopped = True
    elif fault == "running_simulator":
        executor.sim.state = 1
    elif fault == "negative_time":
        executor.sim.time = -1
    elif fault == "no_write":
        monkeypatch.setattr(executor.sim, "writeCustomDataBlock", None)
    elif fault == "read_failure":
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", _fail)
    elif fault == "bad_buffer":
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", lambda *args: 123)
    else:
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", lambda *args: b"counterfeit")
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        phase5._begin_live_runtime_attestation(executor, scenario, yard)
    assert executor not in phase5._LIVE_RUNTIME_SESSIONS


@pytest.mark.parametrize(
    ("fault", "error"),
    [("client_identity", "client identity changed"), ("capabilities", "capability inventory changed"),
     ("version", "application identity changed"), ("root_uid", "scene root changed"),
     ("no_read", "cannot re-read"), ("read_failure", "challenge re-read failed"),
     ("bad_buffer", "not a byte buffer"), ("lost_challenge", "challenge did not persist"),
     ("running_simulator", "end while simulation is stopped"), ("negative_time", "time cannot be negative")],
)
def test_attestation_finalize_rejects_runtime_origin_changes_instead_of_signing_them(
    origin_fixture, monkeypatch, fault: str, error: str
) -> None:
    executor, scenario, yard, metadata = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    if fault == "client_identity":
        executor.client.uuid = "12345678-1234-4234-9234-123456789abd"
    elif fault == "capabilities":
        metadata["getSimulationTime"] = {"changed": True}
    elif fault == "version":
        executor.sim.version = "4.10.1 rev 0"
    elif fault == "root_uid":
        executor.sim.uid += 1
    elif fault == "no_read":
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", None)
    elif fault == "read_failure":
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", _fail)
    elif fault == "bad_buffer":
        monkeypatch.setattr(executor.sim, "readCustomDataBlock", lambda *args: 123)
    elif fault == "lost_challenge":
        executor.sim.data[(17, session.challenge_tag)] = b"changed"
    elif fault == "running_simulator":
        executor.sim.state = 1
    else:
        executor.sim.time = -1
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        phase5._finalize_live_runtime_attestation(executor)
    assert not session.finalized
    assert not any(item["event"] == "live_runtime_attestation_finalized" for item in executor.runtime_events)


def test_finalization_cannot_create_an_origin_record_after_the_run() -> None:
    with pytest.raises(dynamic.DynamicCoppeliaError, match="not established before execution"):
        phase5._finalize_live_runtime_attestation(_executor())


class _ScriptInventorySim:
    object_script_type = 8

    def __init__(self) -> None:
        self.sources = {
            10: "sim.setJointTargetVelocity(wheelJoints[1], 1)",
            11: "sim.setObjectOrientation(slippingJoint_fl, {})",
            12: "sim.setJointTargetPosition(armJoint, 1)",
        }
        self.disabled = {handle: False for handle in self.sources}

    def getObjectsInTree(self, root: int, *args: Any) -> list[int]:
        assert root == 1
        return list(self.sources)

    def getObjectAlias(self, handle: int, options: int) -> str:
        return f"/robot/{handle}/script"

    def getStringProperty(self, handle: int, name: str) -> str:
        assert name == "code"
        return self.sources[handle]

    def getBoolProperty(self, handle: int, name: str) -> bool:
        assert name == "scriptDisabled"
        return self.disabled[handle]

    def setBoolProperty(self, handle: int, name: str, value: bool) -> None:
        assert name == "scriptDisabled"
        self.disabled[handle] = value


def test_exclusive_wheel_ownership_disables_writers_but_retains_passive_maintenance() -> None:
    sim = _ScriptInventorySim()
    executor = _executor(sim)
    executor._disable_bundled_motion_script("r0", 1)
    assert sim.disabled == {10: True, 11: False, 12: True}
    assert executor.disabled_wheel_command_scripts == 1
    assert executor.disabled_arm_gripper_scripts == 1
    assert executor.retained_bundled_maintenance_scripts == 1
    assert executor.script_control_gate_passed is True
    assert executor.script_inventory_classification_complete is True
    assert len(executor.script_control_audit[0]["scripts"]) == 3


@pytest.mark.parametrize(
    ("fault", "error"),
    [("enumeration", "exhaustively enumerate"), ("empty", "no bundled YouBot script"),
     ("duplicate_writer", "exactly one bundled wheel-command writer"),
     ("no_maintenance", "no allowlisted passive"), ("unknown_role", "unrecognized control role"),
     ("unknown_state", "indeterminate disabled state"), ("ignored_disable", "state readback did not match")],
)
def test_script_inventory_fails_without_exhaustive_classification_and_verified_ownership(
    monkeypatch: pytest.MonkeyPatch, fault: str, error: str
) -> None:
    sim = _ScriptInventorySim()
    if fault == "enumeration":
        original = sim.getObjectsInTree
        monkeypatch.setattr(sim, "getObjectsInTree", lambda root, *args: _fail() if args else original(root))
    elif fault == "empty":
        sim.sources.clear()
    elif fault == "duplicate_writer":
        sim.sources[13] = sim.sources[10]
    elif fault == "no_maintenance":
        sim.sources.pop(11)
    elif fault == "unknown_role":
        sim.sources[11] = "print('not an allowlisted robot control role')"
    elif fault == "unknown_state":
        sim.disabled[10] = None
    else:
        monkeypatch.setattr(sim, "setBoolProperty", lambda *args: None)
    executor = _executor(sim)
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        executor._disable_bundled_motion_script("r0", 1)
    assert executor.script_control_gate_passed is False


def test_stop_transport_failure_preserves_started_state_and_never_commands_disabled_robot(monkeypatch) -> None:
    executor = _executor(SimpleNamespace(stopSimulation=_fail))
    executor.is_ready = executor.started = True
    executor.robot_handles = {"r0": 1, "r1": 2}
    executor.disabled_robots.add("r1")
    commands = []
    monkeypatch.setattr(executor, "command_body_velocity", lambda *args, **kwargs: commands.append((args, kwargs)))
    with pytest.raises(RuntimeError, match="fixture transport failed"):
        executor.stop()
    assert executor.started is True
    assert [args[0] for args, kwargs in commands] == ["r0"]
    assert commands[0][0][1:] == (0.0, 0.0, 0.0)


def _collision_executor() -> dynamic.DynamicCoppeliaExecutor:
    executor = _executor()
    executor.robot_base_shape_handles = {"r0": {1, 2}, "r1": {11, 12}}
    executor.robot_excluded_shape_handles = {"r0": {3}, "r1": {13}}
    executor.module_handles = {"m0": 100}
    executor.obstacle_handles = {"8,8": 101}
    return executor


@pytest.mark.parametrize(
    ("handles", "expected"),
    [([1, 100], ("robot:r0", "module:m0", "robot_module", ("r0",))),
     ([101, 11], ("robot:r1", "obstacle:8,8", "robot_obstacle", ("r1",)))],
)
def test_aggregate_world_collision_resolves_attested_base_and_world_entity(handles, expected) -> None:
    assert _collision_executor()._resolve_robot_world_collision(handles) == expected


@pytest.mark.parametrize(
    ("handles", "error"),
    [([1], "one exact pair"), ([3, 100], "excluded YouBot geometry"),
     ([1, 11], "exactly one mobile base"), ([1, 999], "unattested object")],
)
def test_aggregate_world_collision_cannot_include_hidden_geometry_or_unknown_objects(handles, error: str) -> None:
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        _collision_executor()._resolve_robot_world_collision(handles)


@pytest.mark.parametrize(
    ("handles", "error"),
    [([1], "exact colliding object pair"), ([3, 11], "excluded YouBot geometry"),
     ([1, 100], "every implicated mobile base")],
)
def test_robot_pair_collision_requires_every_implicated_attested_base(handles, error: str) -> None:
    executor = _collision_executor()
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        executor._validate_collision_object_handles(
            first_label="robot:r0", second_label="robot:r1", robot_ids=("r0", "r1"), object_handles=handles
        )
    executor._validate_collision_object_handles(
        first_label="robot:r0", second_label="robot:r1", robot_ids=("r0", "r1"), object_handles=[1, 11]
    )


def test_online_policy_exception_still_requests_safe_stop(monkeypatch) -> None:
    executor = _executor()
    operations = []
    monkeypatch.setattr(executor, "start", lambda: operations.append("start"))
    monkeypatch.setattr(executor, "stop", lambda: operations.append("stop"))
    env = SimpleNamespace(agents=["r0"], decision_count=0, scenario=None, reset=lambda seed: ({}, {}))
    with pytest.raises(RuntimeError, match="fixture transport failed"):
        executor.execute_online(env, _fail)
    assert operations == ["start", "stop"]


@pytest.mark.parametrize("with_diagnostics", [True, False])
def test_online_policy_progress_preserves_annotations_and_stops_on_completion(monkeypatch, with_diagnostics: bool) -> None:
    executor = _executor()
    operations = []
    annotations = []
    env = SimpleNamespace(
        agents=["r0"], decision_count=0, scenario=SimpleNamespace(seed=900),
        reset=lambda seed: ({"r0": {}}, {}), metrics=lambda: {"completion": 1},
        annotate_latest_decisions=lambda *args: annotations.append(args),
    )

    def step(actions):
        assert actions == {"r0": 0}
        env.agents = []
        env.decision_count += 1
        return {}, {}, {}, {}, {"r0": {"assignments": []}}

    env.step = step
    monkeypatch.setattr(executor, "start", lambda: operations.append("start"))
    monkeypatch.setattr(executor, "stop", lambda: operations.append("stop"))
    monkeypatch.setattr(executor, "diagnostics", lambda *, logical_metrics: logical_metrics)
    def provider(*args):
        return ({"r0": 0}, {"confidence": 1}) if with_diagnostics else {"r0": 0}
    assert executor.execute_online(env, provider) == {"completion": 1}
    assert operations == ["start", "stop"]
    assert annotations == ([("online_policy", {"confidence": 1})] if with_diagnostics else [])


class _CleanupSim:
    handle_scene = -1
    handle_all = -2

    def __init__(self) -> None:
        self.aliases = {100: dynamic.GENERATED_SCENE_ROOT_ALIAS, 101: "owned_child", 999: "user_scene"}
        self.removed = []

    def getObjectsInTree(self, root: int, kind: int, options: int) -> list[int]:
        assert (kind, options) == (-2, 0)
        return list(self.aliases) if root == -1 else [100, 101]

    def getObjectAlias(self, handle: int, options: int) -> str:
        return self.aliases[handle]

    def removeObjects(self, handles: list[int], delayed: bool) -> None:
        assert delayed is False
        self.removed.append(handles)
        for handle in handles:
            self.aliases.pop(handle)


def test_scene_initialization_removes_only_project_owned_roots_and_is_idempotent() -> None:
    sim = _CleanupSim()
    executor = _executor(sim)
    executor._remove_prior_generated_scenes()
    assert sim.removed == [[100, 101]]
    assert sim.aliases == {999: "user_scene"}
    assert executor.prior_generated_scene_object_count == 2
    executor._remove_prior_generated_scenes()
    assert sim.removed == [[100, 101]]
    assert executor.generated_scene_cleanup_verified is True
    assert executor.prior_generated_scene_root_count == 0


@pytest.mark.parametrize("fault,error", [
    ("missing_api", "cleanup is unavailable"), ("enumeration", "cleanup failed"),
    ("not_removed", "could not remove every prior generated root"),
])
def test_scene_cleanup_refuses_unavailable_or_unverifiable_initialization(monkeypatch, fault: str, error: str) -> None:
    sim = _CleanupSim()
    if fault == "missing_api":
        monkeypatch.setattr(sim, "removeObjects", None)
    elif fault == "enumeration":
        monkeypatch.setattr(sim, "getObjectsInTree", _fail)
    else:
        monkeypatch.setattr(sim, "removeObjects", lambda *args: None)
    executor = _executor(sim)
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        executor._remove_prior_generated_scenes()
    assert executor.generated_scene_cleanup_verified is False


class _CollisionSim:
    handle_single = 1
    shapeintparam_respondable = 2

    def __init__(self) -> None:
        self.collections = {}
        self.removed = []

    def createCollection(self, options: int) -> int:
        handle = 20 + len(self.collections)
        self.collections[handle] = []
        return handle

    def addItemToCollection(self, collection: int, selector: int, handle: int, options: int) -> None:
        assert selector == self.handle_single
        self.collections[collection].append(handle)

    def getCollectionObjects(self, collection: int) -> list[int]:
        return list(self.collections[collection])

    def checkCollision(self, collection: int, probe: int) -> tuple[int, list[int]]:
        return 1, [self.collections[collection][0], probe]

    def getObjectPosition(self, handle: int) -> list[float]:
        return [0, 0, 0]

    def setObjectInt32Param(self, *args) -> None:
        pass

    def setBoolProperty(self, *args) -> None:
        pass

    def removeObjects(self, handles: list[int], delayed: bool) -> None:
        self.removed.extend(handles)


def _monitor_executor(monkeypatch) -> dynamic.DynamicCoppeliaExecutor:
    executor = _collision_executor()
    executor.sim = _CollisionSim()
    executor.robot_handles = {"r0": 1, "r1": 11}
    monkeypatch.setattr(executor, "_create_box", lambda *args: 700)
    return executor


def test_collision_monitor_requires_positive_base_bound_probe_and_exact_collection_membership(monkeypatch) -> None:
    executor = _monitor_executor(monkeypatch)
    executor._prepare_collision_monitoring()
    assert executor.robot_base_contact_gate_passed is True
    assert len(executor.robot_collision_self_tests) == 2
    assert all(record["verified"] for record in executor.robot_collision_self_tests)
    assert executor.sim.removed == [700, 700]
    assert executor.collision_pair_category_counts == {"robot_robot": 1, "robot_module": 2, "robot_obstacle": 2}
    assert executor.runtime_events[-1]["expected_queries_per_step"] == 2


@pytest.mark.parametrize("fault,error", [
    ("missing_api", "monitoring is unavailable"), ("no_base", "no attested mobile-base"),
    ("membership", "membership readback failed"), ("empty_scene", "no query groups or entity pairs"),
    ("no_cleanup", "self-test cleanup is unavailable"), ("negative_probe", "positive self-test"),
    ("wrong_base", "positive self-test"), ("excluded_geometry", "positive self-test"),
])
def test_collision_monitor_fails_closed_without_attested_contact_and_complete_inventory(
    monkeypatch, fault: str, error: str
) -> None:
    executor = _monitor_executor(monkeypatch)
    if fault == "missing_api":
        monkeypatch.setattr(executor.sim, "checkCollision", None)
    elif fault == "no_base":
        executor.robot_base_shape_handles.pop("r1")
    elif fault == "membership":
        monkeypatch.setattr(executor.sim, "getCollectionObjects", lambda handle: [])
    elif fault == "empty_scene":
        executor.robot_handles = {}
        executor.robot_base_shape_handles = {}
        executor.module_handles = {}
        executor.obstacle_handles = {}
    elif fault == "no_cleanup":
        monkeypatch.setattr(executor.sim, "removeObjects", None)
    elif fault == "negative_probe":
        monkeypatch.setattr(executor.sim, "checkCollision", lambda *args: (0, []))
    elif fault == "wrong_base":
        monkeypatch.setattr(executor.sim, "checkCollision", lambda *args: (1, [999, 700]))
    else:
        monkeypatch.setattr(executor.sim, "checkCollision", lambda *args: (1, [3, 700]))
    with pytest.raises(dynamic.DynamicCoppeliaError, match=error):
        executor._prepare_collision_monitoring()
    assert executor.robot_base_contact_gate_passed is False


def test_collision_probe_is_removed_even_if_native_query_raises(monkeypatch) -> None:
    executor = _monitor_executor(monkeypatch)
    executor.robot_collision_entities = {"r0": 20, "r1": 21}
    monkeypatch.setattr(executor.sim, "checkCollision", _fail)
    with pytest.raises(RuntimeError, match="fixture transport failed"):
        executor._self_test_collision_monitoring()
    assert executor.sim.removed == [700]
    assert not executor.robot_collision_self_tests


def _measured_sample(index: int, *, linear: float = 0, angular: float = 0) -> RobotTelemetry:
    return RobotTelemetry(
        timestamp_s=index / 20, robot_id="r0", measured_pose={"position": {"x": index * 0.001, "y": 0, "z": 0}},
        linear_velocity_mps=linear, angular_velocity_rps=angular, battery_remaining_wh=100,
    )


def _disabled_executor(monkeypatch, samples: list[RobotTelemetry]) -> dynamic.DynamicCoppeliaExecutor:
    executor = _executor()
    executor.config.disabled_settle_max_steps = len(samples) - 1
    executor.disabled_robots.add("r0")
    executor.disabled_command_cutoffs["r0"] = 0
    executor.disabled_settled["r0"] = False
    iterator = iter(samples)
    monkeypatch.setattr(executor, "sample_telemetry", lambda robot_id: next(iterator))
    monkeypatch.setattr(executor, "_step_physics", lambda: setattr(executor, "physics_steps", executor.physics_steps + 1))
    return executor


def test_unavailable_robot_settlement_requires_consecutive_measured_stillness_and_records_displacement(monkeypatch) -> None:
    samples = [_measured_sample(index) for index in range(7)]
    samples[3] = _measured_sample(3, linear=0.1)  # Two still samples must not survive a moving sample.
    executor = _disabled_executor(monkeypatch, samples)
    assert executor.settle_disabled_robot("r0") == samples
    assert executor.physics_steps == 6
    assert executor.disabled_settled == {"r0": True}
    assert executor.disabled_settle_displacement_m == {"r0": pytest.approx(0.006)}
    assert executor.runtime_events[-1] == {
        "timestamp_s": 0.3, "event": "disabled_robot_settled", "robot_id": "r0",
        "sample_count": 7, "displacement_m": pytest.approx(0.006), "commands_after_disable": 0,
    }


@pytest.mark.parametrize("moving_axis", ["linear", "angular"])
def test_unavailable_robot_settlement_times_out_on_measured_motion(monkeypatch, moving_axis: str) -> None:
    samples = [_measured_sample(index, **{moving_axis: 0.1}) for index in range(5)]
    executor = _disabled_executor(monkeypatch, samples)
    with pytest.raises(dynamic.DynamicCoppeliaError, match="did not settle after disable"):
        executor.settle_disabled_robot("r0")
    assert executor.disabled_settled == {"r0": False}
    assert not executor.disabled_settle_samples
    assert not executor.runtime_events


def test_settlement_refuses_available_robot_and_detects_any_post_disable_wheel_command(monkeypatch) -> None:
    executor = _disabled_executor(monkeypatch, [_measured_sample(index) for index in range(4)])
    with pytest.raises(dynamic.DynamicCoppeliaError, match="not unavailable"):
        executor.settle_disabled_robot("r1")
    executor.commands.append(RobotCommand(
        timestamp_s=0, robot_id="r0", linear_velocity_mps=0, angular_velocity_rps=0,
        wheel_target_velocity_rad_s=(0, 0, 0, 0), source="formation_hold",
    ))
    with pytest.raises(dynamic.DynamicCoppeliaError, match="received a wheel command while settling"):
        executor.settle_disabled_robot("r0")
    assert executor.disabled_settled == {"r0": False}
    assert not executor.runtime_events


def _attested_result(executor, scenario, yard) -> phase5.Phase5RunResult:
    return phase5.Phase5RunResult(
        scenario="nominal", evidence_kind="live_coppelia", status="failed",
        scenario_id=scenario.scenario_id, scenario_seed=scenario.seed, expected_module_count=1,
        installed_module_ids=[], active_robot_ids=["r0", "r1"], replay=[],
        trace=deepcopy(executor.runtime_events), metrics={"physical_yard": yard.model_dump(mode="json")},
        acceptance={}, live_gate_passed=False,
    )


def _require_session(executor, scenario, result):
    return phase5._require_live_runtime_session(
        result=result, scenario=scenario, executor=executor, simulator_version="4.10.0 rev 7",
    )


def test_finalized_session_can_bind_only_its_unchanged_executor_and_cannot_be_consumed_twice(origin_fixture) -> None:
    executor, scenario, yard, _ = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    phase5._finalize_live_runtime_attestation(executor)
    result = _attested_result(executor, scenario, yard)
    assert _require_session(executor, scenario, result) is session
    assert session.consumed is False  # Checking is not itself evidence publication.
    session.consumed = True
    with pytest.raises(ValueError, match="already bound one bundle"):
        _require_session(executor, scenario, result)


@pytest.mark.parametrize("fault", ["absent", "not_finalized", "version", "configuration", "scenario_seed",
                                      "plan", "physics_steps", "commands", "telemetry", "installed_modules"])
def test_bundle_origin_refuses_unfinalized_or_changed_runtime_evidence(origin_fixture, fault: str) -> None:
    executor, scenario, yard, _ = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    phase5._finalize_live_runtime_attestation(executor)
    result = _attested_result(executor, scenario, yard)
    version = session.simulator_version
    error = "no longer matches executor evidence"
    if fault == "absent":
        phase5._LIVE_RUNTIME_SESSIONS.pop(executor)
        error = "lacks a finalized"
    elif fault == "not_finalized":
        session.finalized = False
        error = "lacks a finalized"
    elif fault == "version":
        version = "4.10.1 rev 0"
        error = "differs from the runtime-origin"
    elif fault == "configuration":
        executor.config.control_hz = 25
    elif fault == "scenario_seed":
        scenario.seed += 1
    elif fault == "plan":
        scenario.plan.design.title = "changed after attestation"
    elif fault == "physics_steps":
        executor.physics_steps += 1
    elif fault == "commands":
        executor.commands.append(RobotCommand(
            timestamp_s=0, robot_id="r0", linear_velocity_mps=0, angular_velocity_rps=0,
            wheel_target_velocity_rad_s=(0, 0, 0, 0), source="formation_hold",
        ))
    elif fault == "telemetry":
        executor.telemetry.append(_measured_sample(0))
    else:
        executor.installed_modules.add("m0")
    with pytest.raises(ValueError, match=error):
        phase5._require_live_runtime_session(result=result, scenario=scenario, executor=executor, simulator_version=version)


@pytest.mark.parametrize("fault", ["missing_start", "missing_end", "duplicate_start", "duplicate_end",
                                      "start_client", "start_capabilities", "start_version", "start_root", "start_challenge",
                                      "start_state", "end_client", "end_root", "end_challenge", "end_state",
                                      "end_steps", "end_commands", "end_telemetry", "end_send_count"])
def test_runtime_origin_requires_exactly_one_consistent_start_and_end_event(origin_fixture, fault: str) -> None:
    executor, scenario, yard, _ = origin_fixture
    phase5._begin_live_runtime_attestation(executor, scenario, yard)
    phase5._finalize_live_runtime_attestation(executor)
    result = _attested_result(executor, scenario, yard)
    if fault.startswith("missing_"):
        result.trace.pop(0 if fault.endswith("start") else 1)
    elif fault.startswith("duplicate_"):
        result.trace.append(deepcopy(result.trace[0 if fault.endswith("start") else 1]))
    else:
        side, field = fault.split("_", 1)
        fields = {
            "client": "client_uuid", "capabilities": "remote_api_info_sha256", "version": "simulator_version",
            "root": "scene_root_uid", "challenge": "challenge_payload_sha256" if side == "start" else "challenge_response_sha256",
            "state": "simulation_state", "steps": "physics_steps", "commands": "command_count",
            "telemetry": "telemetry_count", "send_count": "client_send_count_after",
        }
        result.trace[0 if side == "start" else 1][fields[field]] = "changed"
    with pytest.raises(ValueError, match="runtime attestation"):
        _require_session(executor, scenario, result)


def _build_attestation(run_dir, executor, scenario, result, session):
    return phase5._build_phase5_runtime_attestation(
        run_dir=run_dir, run_id="offline-unit-fixture", result=result, scenario=scenario, session=session,
        source_commit="a" * 40, source_tree_digest="b" * 64, configuration_digest="c" * 64,
    )


def test_runtime_attestation_hashes_every_required_artifact_and_binds_result_scenario(origin_fixture, tmp_path: Path) -> None:
    executor, scenario, yard, _ = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    phase5._finalize_live_runtime_attestation(executor)
    result = _attested_result(executor, scenario, yard)
    for name in phase5._REQUIRED_ARTIFACTS:
        (tmp_path / name).write_bytes(f"offline test artifact: {name}".encode())
    attestation = _build_attestation(tmp_path, executor, scenario, result, session)
    assert attestation.artifact_sha256 == {
        name: phase5._sha256_file(tmp_path / name) for name in phase5._REQUIRED_ARTIFACTS
    }
    assert attestation.binding_digest == phase5._sha256_json(attestation.model_dump(mode="json", exclude={"binding_digest"}))
    assert attestation.physics_steps == 0  # This unit fixture is not a passing execution result.
    result.scenario_id = "another-scenario"
    with pytest.raises(ValueError, match="scenario differs from the result"):
        _build_attestation(tmp_path, executor, scenario, result, session)


@pytest.mark.parametrize("field", ["finalized", "simulation_state_after", "simulation_time_after_s",
                                      "client_send_count_after", "physics_steps", "command_count", "telemetry_count",
                                      "command_stream_sha256", "telemetry_stream_sha256", "installed_module_ids"])
def test_runtime_attestation_cannot_sign_incomplete_final_measurements(origin_fixture, tmp_path: Path, field: str) -> None:
    executor, scenario, yard, _ = origin_fixture
    session = phase5._begin_live_runtime_attestation(executor, scenario, yard)
    phase5._finalize_live_runtime_attestation(executor)
    result = _attested_result(executor, scenario, yard)
    setattr(session, field, False if field == "finalized" else None)
    with pytest.raises(ValueError, match="finalization is incomplete"):
        _build_attestation(tmp_path, executor, scenario, result, session)
    assert not list(tmp_path.iterdir())
