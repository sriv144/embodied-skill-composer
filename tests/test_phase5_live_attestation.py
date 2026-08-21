from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import embodied_skill_composer.construction.coppelia_phase5 as phase5
from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaExecutor,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5RunResult,
    prepare_phase5_physical_yard,
    verify_phase5_artifact_bundle,
    write_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
)


WORKSPACE = Path(__file__).resolve().parents[1]
SIMULATOR_VERSION = "4.10.0 rev 0"


class _AttestedFixtureClient:
    def __init__(self) -> None:
        self.uuid = "12345678-1234-4234-9234-123456789abc"
        self.VERSION = 2
        self.sendCnt = 0
        self.sim = _AttestedFixtureSim(self)

    def call(self, function: str, arguments: list[str]) -> dict[str, object]:
        self.sendCnt += 1
        assert function == "zmqRemoteApi.info"
        assert arguments == ["sim"]
        return {
            capability: {"func": []}
            for capability in phase5._LIVE_REQUIRED_REMOTE_API_CAPABILITIES
        }


class _AttestedFixtureSim:
    simulation_stopped = 0
    stringparam_application_version = 10

    def __init__(self, client: _AttestedFixtureClient) -> None:
        self.client = client
        self.custom_data: dict[tuple[int, str], bytes] = {}

    def _bump(self) -> None:
        self.client.sendCnt += 1

    def getStringParam(self, parameter: int) -> str:
        self._bump()
        assert parameter == self.stringparam_application_version
        return SIMULATOR_VERSION

    def getObjectUid(self, handle: int) -> int:
        self._bump()
        assert handle == 17
        return 9_001

    def getSimulationState(self) -> int:
        self._bump()
        return self.simulation_stopped

    def getSimulationTime(self) -> float:
        self._bump()
        return 0.0

    def writeCustomDataBlock(
        self,
        handle: int,
        tag: str,
        payload: bytes,
    ) -> None:
        self._bump()
        self.custom_data[(handle, tag)] = bytes(payload)

    def readCustomDataBlock(self, handle: int, tag: str) -> bytes:
        self._bump()
        return self.custom_data[(handle, tag)]

    def saveScene(self) -> bytes:
        self._bump()
        return b"VREP" + b"\0" * 124


@pytest.fixture(scope="module")
def phase5_scenario():
    design = load_house_design(
        WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
    )
    generated = generate_cottage_scenario(
        900,
        design,
        config=CottageScenarioConfig(
            widths_m=(6.0,),
            depths_m=(6.0,),
            interior_panel_range=(0, 0),
            obstacle_count_range=(0, 0),
        ),
    )
    return prepare_phase5_physical_yard(generated)


def _fixture_executor(scenario) -> DynamicCoppeliaExecutor:
    executor = DynamicCoppeliaExecutor(
        scenario.plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
    )
    client = _AttestedFixtureClient()
    executor.client = client
    executor.sim = client.sim
    executor.root_handle = 17
    executor.is_ready = True
    return executor


def _failed_result(executor, scenario, physical_yard) -> Phase5RunResult:
    metrics = executor.diagnostics()
    metrics.update(
        {
            "scenario": "nominal",
            "scenario_id": scenario.scenario_id,
            "physical_yard": physical_yard.model_dump(mode="json"),
            "physical_yard_digest": phase5._sha256_json(
                physical_yard.model_dump(mode="json")
            ),
            "maximum_formation_offset_m": 0.0,
            "maximum_preflight_formation_offset_m": (
                physical_yard.maximum_preflight_formation_offset_m
            ),
            "payload_transport": "logical_carrier",
            "live_evidence": True,
            "live_gate_passed": False,
        }
    )
    return Phase5RunResult(
        scenario="nominal",
        evidence_kind="live_coppelia",
        status="failed",
        scenario_id=scenario.scenario_id,
        scenario_seed=scenario.seed,
        expected_module_count=len(scenario.plan.modules),
        installed_module_ids=[],
        active_robot_ids=[],
        replay=[],
        trace=list(executor.runtime_events),
        metrics=metrics,
        acceptance={"runtime_completed": False},
        live_gate_passed=False,
        error_type="FixtureStop",
        error="bounded attestation fixture",
    )


def _patch_official_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identity(
        executor: DynamicCoppeliaExecutor,
    ) -> phase5._OfficialRemoteClientIdentity:
        return phase5._OfficialRemoteClientIdentity(
            client_uuid=executor.client.uuid,
            protocol_version=executor.client.VERSION,
            package_version="2.0.4",
            send_count=executor.client.sendCnt,
        )

    monkeypatch.setattr(phase5, "_official_remote_client_identity", identity)


def _write_attested_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase5_scenario,
) -> Path:
    scenario, physical_yard = phase5_scenario
    executor = _fixture_executor(scenario)
    _patch_official_identity(monkeypatch)
    phase5._begin_live_runtime_attestation(
        executor,
        scenario,
        physical_yard,
    )
    executor.physics_steps = 3
    robot = scenario.plan.robots[0]
    executor.commands.append(
        RobotCommand(
            timestamp_s=0.0,
            robot_id=robot.robot_id,
            linear_velocity_mps=0.0,
            angular_velocity_rps=0.0,
            wheel_target_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
            source="settling",
        )
    )
    executor.telemetry.append(
        RobotTelemetry(
            timestamp_s=0.0,
            robot_id=robot.robot_id,
            measured_pose=robot.start_pose,
            linear_velocity_mps=0.0,
            angular_velocity_rps=0.0,
            battery_remaining_wh=robot.battery_capacity_wh,
        )
    )
    phase5._finalize_live_runtime_attestation(executor)
    result = _failed_result(executor, scenario, physical_yard)
    run_dir = tmp_path / "attested-live-fixture"
    manifest = write_phase5_artifact_bundle(
        run_dir,
        run_id="attested-live-fixture",
        result=result,
        scenario=scenario,
        executor=executor,
        source_commit="1" * 40,
        source_tree_digest="2" * 64,
        source_dirty=False,
        approval_gate_confirmed=True,
        simulator_version=SIMULATOR_VERSION,
    )
    assert manifest.live_evidence
    assert not manifest.live_gate_passed
    assert manifest.runtime_attestation_digest
    assert verify_phase5_artifact_bundle(run_dir) == manifest
    return run_dir


def test_promoted_offline_result_cannot_claim_live_runtime(
    tmp_path: Path,
    phase5_scenario,
) -> None:
    scenario, physical_yard = phase5_scenario
    executor = _fixture_executor(scenario)
    result = _failed_result(executor, scenario, physical_yard)

    with pytest.raises(ValueError, match="runtime-origin attestation"):
        write_phase5_artifact_bundle(
            tmp_path / "relabeled-offline",
            run_id="relabeled-offline",
            result=result,
            scenario=scenario,
            executor=executor,
            source_commit="1" * 40,
            source_tree_digest="2" * 64,
            source_dirty=False,
            approval_gate_confirmed=True,
            simulator_version=SIMULATOR_VERSION,
        )

    assert not (tmp_path / "relabeled-offline").exists()


def test_runtime_origin_round_trip_writes_verifiable_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase5_scenario,
) -> None:
    run_dir = _write_attested_fixture(
        tmp_path,
        monkeypatch,
        phase5_scenario,
    )
    payload = json.loads(
        (run_dir / "runtime_attestation.json").read_text(encoding="utf-8")
    )
    assert payload["transport"] == "coppeliasim_zmq_remote_api"
    assert payload["challenge_payload_sha256"] == (
        payload["challenge_response_sha256"]
    )
    assert set(payload["artifact_sha256"]) == phase5._REQUIRED_ARTIFACTS


def test_verifier_rejects_self_rehashed_runtime_origin_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase5_scenario,
) -> None:
    run_dir = _write_attested_fixture(
        tmp_path,
        monkeypatch,
        phase5_scenario,
    )
    attestation_path = run_dir / "runtime_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["scene_root_uid"] += 1
    attestation["binding_digest"] = phase5._sha256_json(
        {
            key: value
            for key, value in attestation.items()
            if key != "binding_digest"
        }
    )
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    manifest["runtime_attestation_digest"] = digest
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["path"] == attestation_path.name
    )
    artifact["bytes"] = attestation_path.stat().st_size
    artifact["sha256"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="runtime challenge is not bound to its origin fields",
    ):
        verify_phase5_artifact_bundle(run_dir)
