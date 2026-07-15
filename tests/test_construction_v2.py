from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from embodied_skill_composer.construction.api import create_app
from embodied_skill_composer.construction.blender import generate_blender_assets
from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia import (
    CoppeliaConstructionAdapter,
    CoppeliaConstructionConfig,
)
from embodied_skill_composer.construction.floorplan import infer_orthogonal_floor_plan
from embodied_skill_composer.construction.models import BuildPlan, HouseDesign
from embodied_skill_composer.construction.reporting import render_research_report
from embodied_skill_composer.construction.recovery import Disruption, inject_disruption
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scheduler import compare_controllers, schedule_build
from embodied_skill_composer.construction.trace import build_execution_trace


WORKSPACE = Path(__file__).resolve().parents[1]
DESIGN_PATH = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"


@pytest.fixture(scope="module")
def design() -> HouseDesign:
    return load_house_design(DESIGN_PATH)


@pytest.fixture(scope="module")
def plan(design: HouseDesign) -> BuildPlan:
    return compile_house_design(design)


@pytest.fixture(scope="module")
def schedules(plan: BuildPlan):
    return compare_controllers(plan)


def test_cottage_compiles_to_mixed_team_four_robot_plan(plan: BuildPlan) -> None:
    assert len(plan.modules) == 24
    assert len(plan.robots) == 4
    assert {module.required_team_size for module in plan.modules} == {1, 2}
    assert {module.module_type.value for module in plan.modules} >= {
        "foundation",
        "wall_panel",
        "door_panel",
        "window_panel",
        "roof_panel",
    }
    assert all(module.mesh_node == f"module__{module.module_id}" for module in plan.modules)


def test_unapproved_floor_plan_cannot_compile(design: HouseDesign) -> None:
    pending = design.model_copy(deep=True)
    pending.floor_plan.approved = False
    with pytest.raises(ValueError, match="reviewed and approved"):
        compile_house_design(pending)


def test_plan_rejects_unknown_dependency_and_cycles(plan: BuildPlan) -> None:
    payload = plan.model_dump(mode="json")
    payload["modules"][0]["dependencies"] = ["missing_module"]
    with pytest.raises(ValidationError, match="unknown dependencies"):
        BuildPlan.model_validate(payload)

    payload = plan.model_dump(mode="json")
    payload["modules"][0]["dependencies"] = [payload["modules"][1]["module_id"]]
    payload["modules"][1]["dependencies"] = [payload["modules"][0]["module_id"]]
    with pytest.raises(ValidationError, match="cycle"):
        BuildPlan.model_validate(payload)


def test_controller_schedules_obey_dependencies_and_robot_exclusivity(
    plan: BuildPlan,
    schedules,
) -> None:
    modules = {item.module_id: item for item in plan.modules}
    for schedule in schedules.values():
        jobs = {item.module_id: item for item in schedule.jobs}
        for module in plan.modules:
            for dependency in module.dependencies:
                assert jobs[module.module_id].start_s >= jobs[dependency].end_s
            assert len(jobs[module.module_id].robot_ids) == module.required_team_size
        for robot in plan.robots:
            robot_jobs = sorted(
                (item for item in schedule.jobs if robot.robot_id in item.robot_ids),
                key=lambda item: item.start_s,
            )
            assert all(
                left.end_s <= right.start_s
                for left, right in zip(robot_jobs, robot_jobs[1:], strict=False)
            )
        assert set(jobs) == set(modules)


def test_cp_sat_reduces_fixture_makespan_by_more_than_twenty_percent(schedules) -> None:
    sequential = schedules["sequential"]
    optimized = schedules["optimized"]
    assert optimized.solver_status.startswith("cp_sat_")
    assert optimized.makespan_s <= sequential.makespan_s * 0.8
    assert optimized.makespan_s <= schedules["greedy"].makespan_s


def test_cp_sat_schedule_is_reproducible(plan: BuildPlan, schedules) -> None:
    repeated = schedule_build(plan, "optimized")
    assert repeated.model_dump() == schedules["optimized"].model_dump()


def test_trace_reaches_complete_house_and_exposes_brain(plan: BuildPlan, schedules) -> None:
    trace = build_execution_trace(plan, schedules["optimized"])
    assert trace.metrics.structure_completion_rate == 1.0
    assert trace.frames[-1].timestamp_s == trace.metrics.makespan_s
    assert set(trace.frames[-1].completed_module_ids) == {
        module.module_id for module in plan.modules
    }
    assert any(event.event_type == "assignment" for event in trace.brain_events)
    assert any(event.event_type == "rejection" for event in trace.brain_events)
    assert trace.brain_events[-1].event_type == "completion"


def test_marl_scripted_coordinator_completes_plan_in_dependency_order(
    plan: BuildPlan,
) -> None:
    pytest.importorskip("pettingzoo")
    from embodied_skill_composer.construction.marl_env import (
        ConstructionCoordinationEnv,
        scripted_coordination_actions,
    )

    env = ConstructionCoordinationEnv(plan)
    observations, infos = env.reset(seed=7)
    assert set(observations) == {robot.robot_id for robot in plan.robots}
    assert all(info["action_mask"][0] == 1 for info in infos.values())

    modules = {module.module_id: module for module in plan.modules}
    completed_before: set[str] = set()
    while env.agents:
        actions = scripted_coordination_actions(env)
        _, _, _, _, step_infos = env.step(actions)
        assignments = next(iter(step_infos.values()))["assignments"]
        for assignment in assignments:
            module = modules[assignment["module_id"]]
            assert set(module.dependencies) <= completed_before
            assert len(assignment["robot_ids"]) == module.required_team_size
        completed_before = set(env.completed)

    assert len(env.completed) == len(plan.modules) == 24
    assert env.decision_count == 9
    assert {item["module_id"] for item in env.assignment_history} == {
        module.module_id for module in plan.modules
    }


def test_marl_environment_passes_pettingzoo_parallel_api(plan: BuildPlan) -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from embodied_skill_composer.construction.marl_env import ConstructionCoordinationEnv

    parallel_api_test(ConstructionCoordinationEnv(plan, max_decisions=16), num_cycles=20)


@pytest.mark.parametrize(
    ("failure_type", "expected_delay"),
    [("obstacle", 14), ("robot_unavailable", 24), ("dropped_resource", 12)],
)
def test_disruption_regenerates_trace_and_records_recovery(
    plan: BuildPlan,
    schedules,
    failure_type: str,
    expected_delay: int,
) -> None:
    baseline = schedules["optimized"]
    recovered = inject_disruption(
        plan,
        baseline,
        Disruption(failure_type=failure_type, timestamp_s=60),
    )
    assert recovered.metrics.makespan_s == baseline.makespan_s + expected_delay
    assert recovered.metrics.recovery_cost_s == expected_delay
    assert any(event.event_type == "recovery" for event in recovered.brain_events)
    assert recovered.frames[-1].timestamp_s == recovered.metrics.makespan_s
    assert len(recovered.frames[-1].completed_module_ids) == len(plan.modules)


def test_research_report_compares_all_controllers(plan: BuildPlan, schedules) -> None:
    traces = {name: build_execution_trace(plan, item) for name, item in schedules.items()}
    report = render_research_report(plan, traces)
    assert "CP-SAT controller" in report
    assert "sequential" in report
    assert "greedy" in report
    assert "optimized" in report
    assert "64" in report


def test_floor_plan_parser_returns_review_required_metric_geometry() -> None:
    image = np.full((500, 700), 255, dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (620, 430), 0, thickness=18)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    inferred = infer_orthogonal_floor_plan(encoded.tobytes(), known_width_m=9.0)
    assert inferred.approved is False
    assert len(inferred.walls) == 4
    assert inferred.confidence < 1.0
    assert inferred.warnings


def test_api_exposes_project_trace_report_and_health() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["module_count"] == 24
        project = client.get("/api/project")
        assert project.status_code == 200
        assert project.json()["optimized_improvement_percent"] >= 20
        trace = client.get("/api/traces/optimized")
        assert trace.status_code == 200
        assert trace.json()["metrics"]["structure_completion_rate"] == 1.0
        assert client.get("/api/traces/not-a-controller").status_code == 404
        report = client.get("/api/report")
        assert "Construction Research Report" in report.json()["markdown"]


def test_blender_adapter_builds_safe_typed_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plan: BuildPlan,
) -> None:
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output = Path(command[-1])
        for name in (
            "house.blend",
            "house.glb",
            "assembled_preview.png",
            "exploded_modules.png",
            "geometry_manifest.json",
        ):
            (output / name).write_bytes(b"fixture")
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_run)
    artifacts = generate_blender_assets(plan, tmp_path / "output", blender_path=blender)
    assert all(path.is_file() for path in artifacts.values())
    assert observed["command"][0] == str(blender)
    assert "--background" in observed["command"]
    assert json.loads((tmp_path / "output" / "build_plan.json").read_text())["plan_id"] == plan.plan_id


class FakeCoppeliaSim:
    primitiveshape_cuboid = 0
    colorcomponent_ambient_diffuse = 0
    shapeintparam_static = 1
    simulation_stopped = 0

    def __init__(self) -> None:
        self.next_handle = 10
        self.state = self.simulation_stopped
        self.positions: dict[int, list[float]] = {}
        self.aliases: dict[int, str] = {}
        self.saved_scene: str | None = None

    def _handle(self) -> int:
        self.next_handle += 1
        return self.next_handle

    def createDummy(self, _size: float) -> int:
        return self._handle()

    def createPrimitiveShape(self, _kind: int, _dimensions, _options: int) -> int:
        return self._handle()

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[handle] = alias

    def setObjectParent(self, _handle: int, _parent: int, _keep: bool) -> None:
        pass

    def setObjectPosition(self, handle: int, position) -> None:
        self.positions[handle] = list(position)

    def setObjectOrientation(self, _handle: int, _orientation) -> None:
        pass

    def setShapeColor(self, _handle: int, _name, _component: int, _color) -> None:
        pass

    def setObjectInt32Param(self, _handle: int, _parameter: int, _value: int) -> None:
        pass

    def getSimulationState(self) -> int:
        return self.state

    def startSimulation(self) -> None:
        self.state = 1

    def stopSimulation(self) -> None:
        self.state = self.simulation_stopped

    def saveScene(self, path: str) -> None:
        self.saved_scene = path
        Path(path).write_bytes(b"fake-coppelia-scene")


class FakeCoppeliaClient:
    def __init__(self) -> None:
        self.sim = FakeCoppeliaSim()
        self.stepping = False
        self.step_count = 0

    def require(self, name: str):
        assert name == "sim"
        return self.sim

    def setStepping(self, enabled: bool) -> None:
        self.stepping = enabled

    def step(self) -> None:
        assert self.stepping
        self.step_count += 1


def test_coppelia_v2_adapter_consumes_canonical_trace(
    tmp_path: Path,
    plan: BuildPlan,
    schedules,
) -> None:
    trace = build_execution_trace(plan, schedules["optimized"])
    fake = FakeCoppeliaClient()
    adapter = CoppeliaConstructionAdapter(
        plan,
        trace,
        config=CoppeliaConstructionConfig(use_robot_models=False),
        client_factory=lambda _config: fake,
    )
    adapter.connect()
    diagnostics = adapter.play(max_frames=5)
    scene_path = adapter.save_scene(tmp_path / "construction_v2.ttt")
    assert diagnostics["module_count"] == 24
    assert diagnostics["robot_count"] == 4
    assert diagnostics["frame_count"] == 5
    assert diagnostics["physics_steps"] == 5
    assert diagnostics["logical_completion"] == 1.0
    assert scene_path.is_file()
    assert fake.sim.saved_scene == str(scene_path)
