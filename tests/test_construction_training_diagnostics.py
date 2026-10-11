"""Construction success must not be inferred from rollout/time-limit endings."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from embodied_skill_composer.construction import training  # noqa: E402
from embodied_skill_composer.construction.marl_env_v1 import FLEET_SIZE  # noqa: E402
from embodied_skill_composer.construction.policy import TorchRLPolicyBundle  # noqa: E402
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402


class _IdleActor:
    def __call__(self, *_inputs):
        logits = torch.full_like(_inputs[-1], -1000, dtype=torch.float32)
        logits[..., 0] = 0
        return logits


def _idle_bundle() -> TorchRLPolicyBundle:
    return cast(TorchRLPolicyBundle, SimpleNamespace(actor_model=_IdleActor()))


def _critic(_bundle, _observations, _state):
    return torch.full((FLEET_SIZE, 1), 2.0)


@pytest.mark.parametrize(
    ("terminated", "truncated", "installed", "successful"),
    [
        (True, False, 4, True),
        (False, True, 2, False),
        (False, False, 1, False),
        (True, False, 2, False),
        (False, True, 4, True),
    ],
)
def test_collector_records_actual_modules_separately_from_environment_endings(
    monkeypatch,
    terminated: bool,
    truncated: bool,
    installed: int,
    successful: bool,
) -> None:
    class Environment:
        possible_agents = [f"robot-{index}" for index in range(FLEET_SIZE)]

        def __init__(self, _scenario):
            self.modules = list(range(4))
            self.completed = set()
            self.agents = self.possible_agents.copy()

        def reset(self, *, seed):
            assert 0 <= seed < 800
            return {}, {}

        def state(self):
            return np.zeros(1, dtype=np.float32)

        def step(self, actions):
            assert actions == dict.fromkeys(self.possible_agents, 0)
            self.completed = set(range(installed))
            if terminated or truncated:
                self.agents = []
            return (
                {},
                dict.fromkeys(self.possible_agents, 0),
                dict.fromkeys(self.possible_agents, terminated),
                dict.fromkeys(self.possible_agents, truncated),
                {},
            )

    def tensors(_observations, _agents, *, device):
        assert device == "cpu"
        return {
            "self": torch.zeros(FLEET_SIZE, 1),
            "robots": torch.zeros(FLEET_SIZE, FLEET_SIZE, 1),
            "modules": torch.zeros(FLEET_SIZE, 4, 1),
            "dependencies": torch.zeros(FLEET_SIZE, 4, 4),
            "action_mask": torch.ones(FLEET_SIZE, 5),
        }

    monkeypatch.setattr(training, "TemporalConstructionCoordinationEnv", Environment)
    monkeypatch.setattr(
        training,
        "generate_cottage_scenario",
        lambda *_args, **_kwargs: SimpleNamespace(failures=[]),
    )
    monkeypatch.setattr(training, "observations_to_tensors", tensors)
    monkeypatch.setattr(training, "_critic_values", _critic)
    episodes, cursor = training.collect_policy_rollouts(
        _idle_bundle(),
        cast(training.HouseDesign, None),
        training.TrainingConfig.for_profile("unit"),
        decision_target=1,
        episode_cursor=0,
        device="cpu",
    )

    assert cursor == 1
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.completed is successful
    assert episode.structure_completion_rate == installed / 4
    assert episode.terminated is terminated
    assert episode.truncated is truncated
    assert episode.steps[-1].done is (terminated or truncated)
    torch.testing.assert_close(
        episode.bootstrap_values,
        torch.full((FLEET_SIZE, 1), 0.0 if terminated or truncated else 2.0),
    )


def test_real_environment_time_limit_is_not_success(monkeypatch) -> None:
    environment_type = training.TemporalConstructionCoordinationEnv
    monkeypatch.setattr(
        training,
        "TemporalConstructionCoordinationEnv",
        lambda scenario: environment_type(scenario, max_decisions=1),
    )
    monkeypatch.setattr(training, "_critic_values", _critic)
    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    episodes, _ = training.collect_policy_rollouts(
        _idle_bundle(),
        design,
        training.TrainingConfig.for_profile("unit"),
        decision_target=1,
        episode_cursor=0,
        device="cpu",
    )
    episode = episodes[0]
    assert episode.steps[-1].done is True
    assert episode.truncated is True
    assert episode.terminated is False
    assert episode.completed is False
    assert episode.structure_completion_rate == 0
    assert training._rollout_diagnostics(episodes) == {
        "rollout_episode_count": 1,
        "rollout_terminal_fraction": 1.0,
        "rollout_termination_fraction": 0.0,
        "rollout_truncation_fraction": 1.0,
        "rollout_collector_cutoff_fraction": 0.0,
        "rollout_success_fraction": 0.0,
        "rollout_structure_completion_rate": 0.0,
        "rollout_failure_probability_mean": 0.0,
        "rollout_failure_probability_min": 0.0,
        "rollout_failure_probability_max": 0.0,
        "rollout_failure_scenario_fraction": 0.0,
    }


def test_rollout_diagnostics_separate_success_truncation_and_collector_cutoff() -> None:
    episodes = [
        training.RolloutEpisode([], torch.zeros(1), True, 1.0, True, False, 0.0, False),
        training.RolloutEpisode([], torch.zeros(1), False, 0.5, False, True, 0.175, True),
        training.RolloutEpisode([], torch.zeros(1), False, 0.25, False, False, 0.35, False),
    ]
    metrics = training._rollout_diagnostics(episodes)
    assert metrics["rollout_episode_count"] == 3
    assert metrics["rollout_terminal_fraction"] == pytest.approx(2 / 3)
    assert metrics["rollout_termination_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout_truncation_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout_collector_cutoff_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout_success_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout_structure_completion_rate"] == pytest.approx(1.75 / 3)
    assert metrics["rollout_failure_probability_mean"] == pytest.approx(0.175)
    assert metrics["rollout_failure_probability_min"] == 0
    assert metrics["rollout_failure_probability_max"] == 0.35
    assert metrics["rollout_failure_scenario_fraction"] == pytest.approx(1 / 3)


def test_rollout_diagnostics_reject_empty_batches() -> None:
    with pytest.raises(ValueError, match="at least one trajectory"):
        training._rollout_diagnostics([])


def test_curve_preserves_legacy_rows_and_leaves_unknown_diagnostics_blank(tmp_path: Path) -> None:
    legacy = {"update": 1, "transitions": 8, "rollout_terminal_fraction": 1.0}
    current = {
        "update": 2,
        "transitions": 16,
        **training._rollout_diagnostics(
            [
                training.RolloutEpisode([], torch.zeros(1), False, 0.5, False, True),
            ]
        ),
    }
    rows = [legacy.copy(), current.copy()]
    output = tmp_path / "learning_curve.csv"
    training._write_curve(output, rows)
    with output.open(newline="", encoding="utf-8") as handle:
        restored = list(csv.DictReader(handle))
    assert rows == [legacy, current]
    assert restored[0]["rollout_terminal_fraction"] == "1.0"
    for key in current.keys() - legacy.keys():
        assert restored[0][key] == ""
    assert restored[1]["rollout_truncation_fraction"] == "1.0"
    assert restored[1]["rollout_success_fraction"] == "0.0"
    assert restored[1]["rollout_structure_completion_rate"] == "0.5"


def test_empty_curve_retains_legacy_header(tmp_path: Path) -> None:
    output = tmp_path / "empty.csv"
    training._write_curve(output, [])
    assert output.read_text(encoding="utf-8") == "update,transitions\n"


@pytest.mark.parametrize(
    "transitions,expected", [(0, 0), (25, 0.175), (50, 0.35), (100, 0.35), (101, 0.35)]
)
def test_linear_failure_curriculum_uses_agent_transitions_and_clamps_at_final_probability(
    transitions: int,
    expected: float,
) -> None:
    config = training.TrainingConfig(
        transitions=100,
        failure_curriculum_mode="linear",
        failure_curriculum_start_probability=0,
        failure_curriculum_final_probability=0.35,
        failure_curriculum_ramp_fraction=0.5,
    )
    assert training._training_failure_probability(config, transitions) == pytest.approx(expected)


def test_linear_failure_curriculum_records_nonzero_start_and_custom_ramp() -> None:
    config = training.TrainingConfig(
        transitions=100,
        failure_curriculum_mode="linear",
        failure_curriculum_start_probability=0.1,
        failure_curriculum_final_probability=0.5,
        failure_curriculum_ramp_fraction=0.25,
    )
    assert training._training_failure_probability(config, 0) == 0.1
    assert training._training_failure_probability(config, 25) == 0.5
    assert training._training_failure_probability(config, 100) == 0.5
    dumped = config.model_dump(mode="json")
    assert {
        key: dumped[key]
        for key in (
            "failure_curriculum_mode",
            "failure_curriculum_start_probability",
            "failure_curriculum_final_probability",
            "failure_curriculum_ramp_fraction",
        )
    } == {
        "failure_curriculum_mode": "linear",
        "failure_curriculum_start_probability": 0.1,
        "failure_curriculum_final_probability": 0.5,
        "failure_curriculum_ramp_fraction": 0.25,
    }


@pytest.mark.parametrize(
    "transitions,expected", [(75_000, 0.035), (150_000, 0.07), (750_000, 0.35), (1_500_000, 0.35)]
)
def test_short_pilot_uses_the_recorded_full_research_curriculum_prefix(
    transitions: int, expected: float
) -> None:
    config = training.TrainingConfig(
        transitions=150_000,
        failure_curriculum_mode="linear",
        failure_curriculum_reference_transitions=1_500_000,
    )
    assert training._training_failure_probability(config, transitions) == pytest.approx(expected)
    assert config.model_dump(mode="json")["failure_curriculum_reference_transitions"] == 1_500_000


@pytest.mark.parametrize("transitions", [0, 25, 50, 100, 1000])
def test_legacy_single_run_configuration_retains_constant_35_percent_mixture(
    transitions: int,
) -> None:
    legacy = training.TrainingConfig.model_validate({"algorithm": "mappo", "transitions": 100})
    assert legacy.failure_curriculum_mode == "constant"
    assert legacy.failure_curriculum_reference_transitions is None
    assert training._training_failure_probability(legacy, transitions) == 0.35


@pytest.mark.parametrize("mode", ["constant", "linear"])
@pytest.mark.parametrize("transitions", [0, 25, 50, 100])
def test_no_failure_ablation_overrides_every_curriculum_probability(mode, transitions) -> None:
    config = training.TrainingConfig(
        transitions=100,
        include_training_failures=False,
        failure_curriculum_mode=mode,
        failure_curriculum_start_probability=0.1,
        failure_curriculum_final_probability=0.9,
    )
    assert training._training_failure_probability(config, transitions) == 0


@pytest.mark.parametrize(
    "invalid",
    [
        {"failure_curriculum_mode": "unregistered"},
        {"failure_curriculum_start_probability": -0.1},
        {"failure_curriculum_final_probability": 1.1},
        {"failure_curriculum_ramp_fraction": 0},
        {"failure_curriculum_ramp_fraction": 1.1},
        {"failure_curriculum_reference_transitions": 0},
        {
            "failure_curriculum_mode": "linear",
            "failure_curriculum_start_probability": 0.8,
            "failure_curriculum_final_probability": 0.2,
        },
    ],
)
def test_failure_curriculum_rejects_invalid_or_decreasing_schedules(invalid) -> None:
    with pytest.raises(ValueError):
        training.TrainingConfig(**invalid)


def test_failure_curriculum_rejects_negative_durable_transition_counters() -> None:
    with pytest.raises(ValueError, match="transition counter"):
        training._training_failure_probability(training.TrainingConfig(), -1)
    with pytest.raises(ValueError, match="transition counter"):
        training.collect_policy_rollouts(
            _idle_bundle(),
            cast(training.HouseDesign, None),
            training.TrainingConfig(),
            decision_target=1,
            episode_cursor=0,
            device="cpu",
            start_transitions=-1,
        )


def test_new_scenario_probabilities_and_seeds_match_across_a_resumed_collection(
    monkeypatch,
) -> None:
    environment_type = training.TemporalConstructionCoordinationEnv
    monkeypatch.setattr(
        training,
        "TemporalConstructionCoordinationEnv",
        lambda scenario: environment_type(scenario, max_decisions=1),
    )
    monkeypatch.setattr(training, "_critic_values", _critic)
    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    config = training.TrainingConfig(
        seed=7,
        transitions=100,
        failure_curriculum_mode="linear",
    )
    observed = []
    original = training.generate_cottage_scenario

    def scenario(seed, base_design, *, config):
        result = original(seed, base_design, config=config)
        observed.append((seed, config.failure_probability, result.model_dump(mode="json")))
        return result

    monkeypatch.setattr(training, "generate_cottage_scenario", scenario)
    training.collect_policy_rollouts(
        _idle_bundle(),
        design,
        config,
        decision_target=3,
        episode_cursor=4,
        device="cpu",
        start_transitions=20,
    )
    uninterrupted = observed.copy()
    observed.clear()
    _, cursor = training.collect_policy_rollouts(
        _idle_bundle(),
        design,
        config,
        decision_target=2,
        episode_cursor=4,
        device="cpu",
        start_transitions=20,
    )
    episodes, _ = training.collect_policy_rollouts(
        _idle_bundle(),
        design,
        config,
        decision_target=1,
        episode_cursor=cursor,
        device="cpu",
        start_transitions=20 + 2 * FLEET_SIZE,
    )
    assert observed == uninterrupted
    assert [item[0] for item in observed] == [11, 12, 13]
    assert [item[1] for item in observed] == pytest.approx([0.14, 0.168, 0.196])
    assert episodes[0].failure_probability == pytest.approx(0.196)


def test_curriculum_and_diagnostics_resume_at_persisted_transition_counter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    config = training.TrainingConfig.for_profile("unit")
    config.include_training_failures = True
    config.failure_curriculum_mode = "linear"
    config.transitions = 16
    config.rollout_decisions = 2
    config.expert_episodes = 0
    config.behavior_clone_epochs = 0
    config.checkpoint_fractions = [0.5, 1.0]
    config.output_root = tmp_path / "training"
    monkeypatch.setattr(
        training,
        "source_fingerprint",
        lambda: {
            "commit": "diagnostics-source",
            "dirty": False,
            "tree_digest": "diagnostics-tree",
        },
    )

    def export(_model, path, *, device):
        assert device == "cpu"
        path.write_bytes(b"fixture-onnx")
        return path

    monkeypatch.setattr(training, "export_actor_onnx", export)
    counters = []
    original_collect = training.collect_policy_rollouts

    def collect(*args, **kwargs):
        counters.append(kwargs["start_transitions"])
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(training, "collect_policy_rollouts", collect)
    with pytest.raises(RuntimeError, match="training cancelled"):
        training.train_swarm_policy(
            design,
            config,
            cancel_check=lambda: any(config.output_root.glob("*/checkpoints/checkpoint_050pct.pt")),
        )
    latest = next(config.output_root.glob("*/checkpoints/latest.pt"))
    snapshot = torch.load(latest, map_location="cpu", weights_only=False)
    assert snapshot["transitions"] == 8
    assert snapshot["curve_rows"][0]["rollout_failure_probability_mean"] == 0
    assert snapshot["config"]["failure_curriculum_mode"] == "linear"

    # A mixed-history checkpoint can retain genuinely unknown legacy diagnostics.
    old_row = snapshot["curve_rows"][0]
    for key in list(old_row):
        if key.startswith("rollout_") and key != "rollout_terminal_fraction":
            del old_row[key]
    torch.save(snapshot, latest)
    config.resume_checkpoint = latest
    events = []
    artifacts = training.train_swarm_policy(design, config, progress_callback=events.append)
    assert counters == [0, 8]
    update = next(event for event in events if event["event"] == "ppo_update")
    assert update["rollout_failure_probability_mean"] == 0.35
    assert update["rollout_failure_probability_min"] == 0.35
    assert update["rollout_failure_probability_max"] == 0.35
    assert "rollout_success_fraction" in update
    assert "rollout_structure_completion_rate" in update
    # The durable worker validates this exact event before persisting/broadcasting.
    from embodied_skill_composer.construction.lab_events import PpoUpdateEvent

    validated = PpoUpdateEvent.model_validate(update)
    assert validated.model_dump()["rollout_failure_probability_mean"] == 0.35
    from embodied_skill_composer.construction.lab_registry import LabRegistry

    registry = LabRegistry(tmp_path / "durable-lab.sqlite")
    run_id = registry.create_run("training", config.model_dump(mode="json"))
    registry.append_event(run_id, update)
    persisted = registry.list_event_envelopes(run_id)[-1].payload
    assert isinstance(persisted, PpoUpdateEvent)
    assert persisted.rollout_failure_probability_mean == 0.35
    assert persisted.rollout_success_fraction == update["rollout_success_fraction"]
    with artifacts.learning_curve_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["rollout_success_fraction"] == ""
    assert rows[1]["rollout_failure_probability_mean"] == "0.35"


def _legacy_update_payload():
    return {
        "event": "ppo_update",
        "transitions": 8,
        "update": 1,
        "loss_objective": 0.0,
        "loss_critic": 0.0,
        "loss_entropy": 0.0,
        "mean_episode_return": 0.0,
        "rollout_terminal_fraction": 1.0,
    }


def test_legacy_typed_update_diagnostics_remain_unknown_and_wire_payload_unchanged() -> None:
    from embodied_skill_composer.construction.lab_events import (
        PpoUpdateEvent,
        dump_run_event_payload,
        validate_run_event_payload,
    )

    payload = _legacy_update_payload()
    event = validate_run_event_payload(payload)
    assert isinstance(event, PpoUpdateEvent)
    assert event.rollout_success_fraction is None
    assert event.rollout_structure_completion_rate is None
    assert event.rollout_truncation_fraction is None
    assert event.rollout_failure_probability_mean is None
    assert dump_run_event_payload(event) == payload


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("rollout_episode_count", 0),
        ("rollout_episode_count", 1.5),
        *[
            (name, value)
            for name in (
                "rollout_termination_fraction",
                "rollout_truncation_fraction",
                "rollout_collector_cutoff_fraction",
                "rollout_success_fraction",
                "rollout_structure_completion_rate",
                "rollout_failure_probability_mean",
                "rollout_failure_probability_min",
                "rollout_failure_probability_max",
                "rollout_failure_scenario_fraction",
            )
            for value in (-0.01, 1.01)
        ],
    ],
)
def test_typed_update_rejects_invalid_diagnostic_values(field: str, invalid: object) -> None:
    from embodied_skill_composer.construction.lab_events import validate_run_event_payload

    with pytest.raises(ValueError):
        validate_run_event_payload({**_legacy_update_payload(), field: invalid})
