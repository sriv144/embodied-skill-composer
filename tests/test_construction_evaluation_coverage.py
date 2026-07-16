from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from embodied_skill_composer.construction import evaluation
from embodied_skill_composer.construction.evaluation import (
    ControllerEvaluation,
    ControllerName,
    EpisodeEvaluation,
    EvaluationSuite,
    MetricSummary,
)
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.runtime import load_house_design


@pytest.fixture(scope="module")
def cottage_design() -> HouseDesign:
    workspace = Path(__file__).resolve().parents[1]
    return load_house_design(workspace / "configs" / "construction" / "cottage_v1.yaml")


class _FakeTemporalEnv:
    created: list[_FakeTemporalEnv] = []

    def __init__(self, scenario: object) -> None:
        self.scenario = scenario
        self.possible_agents = ["robot-1", "robot-2"]
        self.agents = list(self.possible_agents)
        self.decision_count = 0
        self.actions: list[dict[str, int]] = []
        self.annotations: list[tuple[str, dict[str, object]]] = []
        type(self).created.append(self)

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        assert seed >= 0
        return np.zeros((2, 3), dtype=np.float32), {}

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[np.ndarray, dict[str, float], dict[str, bool], dict[str, bool], dict[str, object]]:
        self.actions.append(actions)
        self.decision_count += 1
        self.agents = []
        return np.ones((2, 3), dtype=np.float32), {}, {}, {}, {}

    def annotate_latest_decisions(
        self,
        controller: str,
        diagnostics: dict[str, object],
    ) -> None:
        self.annotations.append((controller, diagnostics))

    @staticmethod
    def metrics() -> dict[str, object]:
        return {
            "structure_completion_rate": 0.875,
            "makespan_s": 42.5,
            "total_travel_m": 12.25,
            "total_energy_wh": 4.5,
            "idle_robot_seconds": 3.0,
            "robot_utilization": {"robot-1": 0.5, "robot-2": 0.75},
            "collision_count": 1,
            "wasted_work_s": 2.5,
            "invalid_bid_count": 2,
            "drop_count": 1,
            "routing_backend": "astar",
        }


def _install_episode_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    context: dict[str, object] = {"failures": [], "routes": [], "priorities": []}
    _FakeTemporalEnv.created = []

    def fake_scenario(seed: int, design: HouseDesign, *, config: object) -> object:
        del design
        cast(list[tuple[bool, float]], context["failures"]).append(
            (bool(getattr(config, "include_failures")), float(getattr(config, "failure_probability")))
        )
        return SimpleNamespace(
            scenario_id=f"scenario-{seed}",
            split=SimpleNamespace(value="validation"),
            plan=SimpleNamespace(),
        )

    def route(name: str, actions: dict[str, int]) -> dict[str, int]:
        cast(list[str], context["routes"]).append(name)
        return actions

    def fake_schedule(plan: object, controller: str) -> object:
        del plan
        assert controller == "optimized"
        return SimpleNamespace(
            jobs=[SimpleNamespace(module_id="module-1", start_s=1, end_s=4, robot_ids=["robot-1"])]
        )

    def fake_cp_sat(env: object, priority: object) -> dict[str, int]:
        del env
        cast(list[object], context["priorities"]).append(priority)
        return route("cp_sat", {"robot-1": 4, "robot-2": 0})

    monkeypatch.setattr(evaluation, "generate_cottage_scenario", fake_scenario)
    monkeypatch.setattr(evaluation, "TemporalConstructionCoordinationEnv", _FakeTemporalEnv)
    monkeypatch.setattr(
        evaluation,
        "sequential_temporal_actions",
        lambda env: route("sequential", {agent: 1 for agent in env.agents}),
    )
    monkeypatch.setattr(
        evaluation,
        "scripted_temporal_actions",
        lambda env: route("greedy", {agent: 2 for agent in env.agents}),
    )
    monkeypatch.setattr(
        evaluation,
        "auction_temporal_actions",
        lambda env: route("auction", {agent: 3 for agent in env.agents}),
    )
    monkeypatch.setattr(evaluation, "schedule_build", fake_schedule)
    monkeypatch.setattr(evaluation, "cp_sat_expert_actions", fake_cp_sat)
    return context


@pytest.mark.parametrize(
    ("controller", "expected_action"),
    [("sequential", 1), ("greedy", 2), ("auction", 3), ("cp_sat", 4)],
)
def test_evaluate_controller_episode_routes_deterministic_controllers(
    cottage_design: HouseDesign,
    monkeypatch: pytest.MonkeyPatch,
    controller: ControllerName,
    expected_action: int,
) -> None:
    context = _install_episode_runtime(monkeypatch)

    result = evaluation.evaluate_controller_episode(
        cottage_design,
        seed=800,
        controller=controller,
        failure_enabled=controller == "cp_sat",
    )

    assert result.scenario_id == "scenario-800"
    assert result.split == "validation"
    assert result.controller == controller
    assert result.structure_completion_rate == 0.875
    assert result.mean_robot_utilization == pytest.approx(0.625)
    assert result.decision_count == 1
    assert result.routing_backend == "astar"
    assert _FakeTemporalEnv.created[-1].actions[0]["robot-1"] == expected_action
    assert context["routes"] == [controller]
    expected_failure = controller == "cp_sat"
    assert context["failures"] == [(expected_failure, 1.0 if expected_failure else 0.0)]
    if controller == "cp_sat":
        assert context["priorities"] == [
            {"module-1": (1, 4, ("robot-1",))}
        ]


def test_evaluate_learned_controller_uses_deterministic_policy_and_annotations(
    cottage_design: HouseDesign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_episode_runtime(monkeypatch)
    actor_model = object()
    policy_calls: list[dict[str, object]] = []

    def fake_policy_actions(
        actor: object,
        observations: np.ndarray,
        agents: list[str],
        *,
        device: str,
        deterministic: bool,
    ) -> tuple[dict[str, int], dict[str, object]]:
        policy_calls.append(
            {
                "actor": actor,
                "observations": observations.copy(),
                "agents": agents,
                "device": device,
                "deterministic": deterministic,
            }
        )
        return {agent: 5 for agent in agents}, {"entropy": 0.1}

    monkeypatch.setattr(evaluation, "policy_actions", fake_policy_actions)
    bundle = SimpleNamespace(actor_model=actor_model)
    result = evaluation.evaluate_controller_episode(
        cottage_design,
        seed=804,
        controller="mappo",
        bundle=cast(Any, bundle),
        device="cuda",
    )

    assert result.controller == "mappo"
    assert policy_calls[0]["actor"] is actor_model
    assert policy_calls[0]["device"] == "cuda"
    assert policy_calls[0]["deterministic"] is True
    assert _FakeTemporalEnv.created[-1].annotations == [("mappo", {"entropy": 0.1})]


def test_evaluate_learned_controller_requires_policy_bundle(
    cottage_design: HouseDesign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_episode_runtime(monkeypatch)

    with pytest.raises(ValueError, match="ippo evaluation requires a policy bundle"):
        evaluation.evaluate_controller_episode(
            cottage_design,
            seed=801,
            controller="ippo",
        )


def _episode(
    controller: ControllerName,
    *,
    seed: int = 7,
    failure_enabled: bool = False,
    completion: float = 1.0,
    makespan: float = 100.0,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        scenario_id=f"fixture-{seed}",
        seed=seed,
        split="validation",
        controller=controller,
        failure_enabled=failure_enabled,
        structure_completion_rate=completion,
        makespan_s=makespan,
        total_travel_m=20.0 + seed,
        total_energy_wh=5.0 + seed,
        idle_robot_seconds=float(seed),
        mean_robot_utilization=0.8,
        collision_count=seed % 2,
        wasted_work_s=float(seed % 3),
        invalid_bid_count=seed % 2,
        drop_count=0,
        decision_count=5,
        routing_backend="astar",
    )


def test_run_suite_expands_modes_policies_and_aggregates_deterministically(
    cottage_design: HouseDesign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str, object, bool, str]] = []
    mappo_bundle = object()

    def fake_evaluate(
        design: HouseDesign,
        *,
        seed: int,
        controller: ControllerName,
        bundle: object,
        failure_enabled: bool,
        device: str,
    ) -> EpisodeEvaluation:
        assert design is cottage_design
        calls.append((seed, controller, bundle, failure_enabled, device))
        return _episode(
            controller,
            seed=seed,
            failure_enabled=failure_enabled,
            completion=0.9 + (seed - 7) * 0.05,
            makespan=100.0 + seed,
        )

    monkeypatch.setattr(evaluation, "evaluate_controller_episode", fake_evaluate)
    suite = evaluation.run_evaluation_suite(
        cottage_design,
        seeds=[7, 8],
        controllers=["greedy", "mappo"],
        policies={"mappo": cast(Any, mappo_bundle)},
        include_failure_suite=True,
        device="cuda",
    )

    assert len(suite.episodes) == 8
    assert len(suite.summaries) == 4
    assert suite.evaluation_id.endswith("-heldout-2seed")
    assert [(item.controller, item.failure_enabled) for item in suite.summaries] == [
        ("greedy", False),
        ("mappo", False),
        ("greedy", True),
        ("mappo", True),
    ]
    assert calls[0] == (7, "greedy", None, False, "cuda")
    assert calls[2] == (7, "mappo", mappo_bundle, False, "cuda")
    assert calls[-1] == (8, "mappo", mappo_bundle, True, "cuda")
    first_completion = suite.summaries[0].metrics["structure_completion_rate"]
    assert first_completion.mean == pytest.approx(0.925)
    assert first_completion.median == pytest.approx(0.925)
    assert first_completion.std > 0

    nominal_only = evaluation.run_evaluation_suite(
        cottage_design,
        seeds=[9],
        controllers=["greedy"],
        include_failure_suite=False,
    )
    assert len(nominal_only.episodes) == 1
    assert nominal_only.summaries[0].metrics["makespan_s"].std == 0.0


def test_summarize_skips_missing_controller_mode_combinations() -> None:
    episodes = [
        _episode("greedy", seed=7, failure_enabled=False),
        _episode("mappo", seed=7, failure_enabled=True, completion=0.7),
    ]

    summaries = evaluation.summarize_evaluations(episodes)

    assert [(item.controller, item.failure_enabled) for item in summaries] == [
        ("greedy", False),
        ("mappo", True),
    ]
    assert all(len(item.metrics) == 9 for item in summaries)


def test_metric_summary_singleton_and_bootstrap_are_reproducible() -> None:
    singleton = evaluation._metric_summary(np.array([4.5]), seed=11)
    first = evaluation._metric_summary(np.array([1.0, 3.0, 5.0]), seed=2027)
    second = evaluation._metric_summary(np.array([1.0, 3.0, 5.0]), seed=2027)

    assert singleton == MetricSummary(
        mean=4.5,
        std=0.0,
        bootstrap_ci95_low=4.5,
        bootstrap_ci95_high=4.5,
        median=4.5,
    )
    assert first == second
    assert first.mean == 3.0
    assert first.median == 3.0
    assert first.std == 2.0
    assert first.bootstrap_ci95_low <= first.mean <= first.bootstrap_ci95_high


def _metric(mean: float, *, median_value: float | None = None) -> MetricSummary:
    return MetricSummary(
        mean=mean,
        std=0.0,
        bootstrap_ci95_low=mean,
        bootstrap_ci95_high=mean,
        median=mean if median_value is None else median_value,
    )


def _controller_summary(
    controller: ControllerName,
    *,
    failure_enabled: bool,
    completion: float,
    makespan: float,
) -> ControllerEvaluation:
    return ControllerEvaluation(
        controller=controller,
        failure_enabled=failure_enabled,
        episode_count=1,
        metrics={
            "structure_completion_rate": _metric(completion),
            "makespan_s": _metric(makespan),
            "total_travel_m": _metric(20.0),
            "mean_robot_utilization": _metric(0.8),
        },
    )


def _report_suite(evaluation_id: str = "report-fixture") -> EvaluationSuite:
    summaries = [
        _controller_summary(
            "mappo", failure_enabled=False, completion=0.96, makespan=115.0
        ),
        _controller_summary(
            "ippo", failure_enabled=False, completion=0.94, makespan=120.0
        ),
        _controller_summary(
            "cp_sat", failure_enabled=False, completion=1.0, makespan=100.0
        ),
        _controller_summary(
            "mappo", failure_enabled=True, completion=0.86, makespan=130.0
        ),
    ]
    return EvaluationSuite(
        evaluation_id=evaluation_id,
        seeds=[900],
        controllers=["mappo", "ippo", "cp_sat"],
        episodes=[_episode("mappo", seed=900)],
        summaries=summaries,
    )


def test_write_artifacts_round_trips_json_csv_and_acceptance_report(tmp_path: Path) -> None:
    suite = _report_suite("artifact-fixture")

    artifacts = evaluation.write_evaluation_artifacts(suite, tmp_path)

    assert artifacts.run_dir == tmp_path.resolve() / "artifact-fixture"
    assert EvaluationSuite.model_validate_json(
        artifacts.evaluation_json.read_text(encoding="utf-8")
    ) == suite
    with artifacts.episodes_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["controller"] == "mappo"
    assert rows[0]["seed"] == "900"
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "| mappo | no | 0.960 [0.960, 0.960] | 115.0 |" in report
    assert "`mappo` no-failure completion >= 0.95: PASS" in report
    assert "`ippo` no-failure completion >= 0.95: NOT YET" in report
    assert "MAPPO median makespan within 15% of CP-SAT: PASS (1.150x)" in report
    assert "MAPPO failure completion >= 0.85: PASS" in report
    assert "construction_coordination_v1" in report
    with pytest.raises(FileExistsError):
        evaluation.write_evaluation_artifacts(suite, tmp_path)


def test_report_covers_failed_thresholds_and_missing_learned_results() -> None:
    failing = EvaluationSuite(
        evaluation_id="failing",
        seeds=[900],
        controllers=["mappo", "ippo", "cp_sat"],
        episodes=[],
        summaries=[
            _controller_summary(
                "mappo", failure_enabled=False, completion=0.8, makespan=10.0
            ),
            _controller_summary(
                "ippo", failure_enabled=False, completion=0.95, makespan=10.0
            ),
            _controller_summary(
                "cp_sat", failure_enabled=False, completion=1.0, makespan=0.0
            ),
            _controller_summary(
                "mappo", failure_enabled=True, completion=0.7, makespan=10.0
            ),
        ],
    )
    report = evaluation.render_evaluation_report(failing)
    assert "`mappo` no-failure completion >= 0.95: NOT YET" in report
    assert "`ippo` no-failure completion >= 0.95: PASS" in report
    assert "MAPPO median makespan within 15% of CP-SAT: NOT YET" in report
    assert "MAPPO failure completion >= 0.85: NOT YET" in report

    no_learned = EvaluationSuite(
        evaluation_id="baseline-only",
        seeds=[900],
        controllers=["greedy"],
        episodes=[_episode("greedy", seed=900)],
        summaries=[
            _controller_summary(
                "greedy", failure_enabled=False, completion=1.0, makespan=100.0
            )
        ],
    )
    assert "Learned-policy acceptance cannot be audited" in evaluation.render_evaluation_report(
        no_learned
    )


def test_sequential_actions_select_first_ready_capable_team() -> None:
    modules = [SimpleNamespace(module_id="module-b"), SimpleNamespace(module_id="module-a")]
    selections: list[tuple[str, list[str]]] = []

    class FakeSequentialEnv:
        agents = ["robot-1", "robot-2", "robot-3"]
        robot_runtime = {
            "robot-1": SimpleNamespace(status="idle"),
            "robot-2": SimpleNamespace(status="idle"),
            "robot-3": SimpleNamespace(status="moving"),
        }
        module_index = {"module-a": 2, "module-b": 5}

        @staticmethod
        def ready_modules() -> list[object]:
            return modules

        @staticmethod
        def _select_capable_team(module: object, available: list[str]) -> list[str] | None:
            module_id = str(getattr(module, "module_id"))
            selections.append((module_id, list(available)))
            return None if module_id == "module-a" else ["robot-1", "robot-2"]

    env = cast(Any, FakeSequentialEnv())
    assert evaluation.sequential_temporal_actions(env) == {
        "robot-1": 6,
        "robot-2": 6,
        "robot-3": 0,
    }
    assert selections == [
        ("module-a", ["robot-1", "robot-2"]),
        ("module-b", ["robot-1", "robot-2"]),
    ]

    monkey_env = FakeSequentialEnv()
    monkey_env._select_capable_team = lambda module, available: None
    assert evaluation.sequential_temporal_actions(cast(Any, monkey_env)) == {
        "robot-1": 0,
        "robot-2": 0,
        "robot-3": 0,
    }
