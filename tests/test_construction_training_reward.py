"""Reward interventions must not change actions, dynamics, or historical resumes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError


torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from embodied_skill_composer.construction import training  # noqa: E402
from embodied_skill_composer.construction.marl_env_v1 import (  # noqa: E402
    DEFAULT_REWARD_TIME_COEFFICIENT,
    FLEET_SIZE,
    TemporalConstructionCoordinationEnv,
    scripted_temporal_actions,
)
from embodied_skill_composer.construction.policy import (  # noqa: E402
    TorchRLPolicyBundle,
    build_torchrl_policy,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402
from embodied_skill_composer.construction.scenarios import (  # noqa: E402
    CottageScenarioConfig,
    generate_cottage_scenario,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def _scenario():
    design = load_house_design(WORKSPACE / "configs/construction/cottage_v1.yaml")
    return generate_cottage_scenario(
        7,
        design,
        config=CottageScenarioConfig(include_failures=False, obstacle_count_range=(0, 0)),
    )


@pytest.mark.parametrize(
    "value", [0, -0.15, float("inf"), -float("inf"), float("nan"), True, None, "1.5"]
)
def test_training_config_requires_finite_positive_time_coefficient(value: object) -> None:
    with pytest.raises(ValidationError):
        training.TrainingConfig(reward_time_coefficient=cast(float, value))


@pytest.mark.parametrize(
    "value", [0, -0.15, float("inf"), -float("inf"), float("nan"), True, None, "1.5"]
)
def test_environment_requires_finite_positive_time_coefficient(value: object) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TemporalConstructionCoordinationEnv(_scenario(), reward_time_coefficient=cast(float, value))


def test_legacy_default_config_and_digest_are_unchanged() -> None:
    config = training.TrainingConfig.for_profile("unit", algorithm="mappo", seed=19)
    assert config.reward_time_coefficient == DEFAULT_REWARD_TIME_COEFFICIENT == 0.15
    # Captured from the committed pre-intervention implementation, not recomputed
    # with the new digest function as an expected-value oracle.
    legacy_digest = "b45de81e1b6239302367c668d26b1a99ad0910b1b837b7879c05b7c9a98533d1"
    assert training.configuration_digest(config) == legacy_digest
    legacy_payload = config.model_dump(mode="python", exclude={"reward_time_coefficient"})
    restored = training.TrainingConfig.model_validate(legacy_payload)
    assert restored.reward_time_coefficient == 0.15
    assert training.configuration_digest(restored) == legacy_digest
    assert (
        training.configuration_digest(config.model_copy(update={"reward_time_coefficient": 1.5}))
        != legacy_digest
    )


def test_reward_time_coefficient_is_explicit_and_round_trips() -> None:
    config = training.TrainingConfig(reward_time_coefficient=1.5)
    assert config.model_dump(mode="json")["reward_time_coefficient"] == 1.5
    restored = training.TrainingConfig.model_validate_json(config.model_dump_json())
    assert restored.reward_time_coefficient == 1.5
    assert training.configuration_digest(restored) == training.configuration_digest(config)
    assert training.configuration_digest(config) != training.configuration_digest(
        training.TrainingConfig()
    )
    assert training.configuration_digest(
        config.model_copy(update={"reward_time_coefficient": 1.6})
    ) != training.configuration_digest(config)


def test_time_reward_only_changes_reward_not_dynamics_or_action_masks() -> None:
    scenario = _scenario()
    control = TemporalConstructionCoordinationEnv(scenario)
    candidate = TemporalConstructionCoordinationEnv(scenario, reward_time_coefficient=1.5)
    control_observations, _ = control.reset(seed=7)
    candidate_observations, _ = candidate.reset(seed=7)
    while control.agents:
        assert candidate.agents == control.agents
        for agent in control.possible_agents:
            for key in control_observations[agent]:
                np.testing.assert_array_equal(
                    candidate_observations[agent][key], control_observations[agent][key]
                )
        actions = scripted_temporal_actions(control)
        assert actions == scripted_temporal_actions(candidate)
        start_s = control.sim_time_s
        control_observations, control_rewards, control_done, control_truncated, _ = control.step(
            actions
        )
        candidate_observations, candidate_rewards, candidate_done, candidate_truncated, _ = (
            candidate.step(actions)
        )
        elapsed = control.sim_time_s - start_s
        for agent in control.possible_agents:
            assert candidate_rewards[agent] - control_rewards[agent] == pytest.approx(
                -(1.5 - 0.15) * elapsed / control.max_sim_time_s
            )
        assert candidate_done == control_done
        assert candidate_truncated == control_truncated
        assert candidate.metrics() == control.metrics()
        assert candidate.assignment_history == control.assignment_history
    assert control.metrics()["structure_completion_rate"] == 1.0


class _WaitActor:
    def __call__(self, *_inputs):
        logits = torch.full_like(_inputs[-1], -1000, dtype=torch.float32)
        logits[..., 0] = 0
        return logits


@pytest.mark.parametrize("coefficient", [0.15, 1.5])
def test_rollout_uses_configured_reward_coefficient(
    monkeypatch: pytest.MonkeyPatch,
    coefficient: float,
) -> None:
    observed_coefficients = []

    def environment(scenario, *, reward_time_coefficient=DEFAULT_REWARD_TIME_COEFFICIENT):
        observed_coefficients.append(reward_time_coefficient)
        return TemporalConstructionCoordinationEnv(
            scenario, max_decisions=1, reward_time_coefficient=reward_time_coefficient
        )

    monkeypatch.setattr(training, "TemporalConstructionCoordinationEnv", environment)
    monkeypatch.setattr(training, "_critic_values", lambda *_args: torch.zeros(FLEET_SIZE, 1))
    design = load_house_design(WORKSPACE / "configs/construction/cottage_v1.yaml")
    config = training.TrainingConfig.for_profile("unit").model_copy(
        update={"reward_time_coefficient": coefficient}
    )
    episodes, _ = training.collect_policy_rollouts(
        cast(TorchRLPolicyBundle, SimpleNamespace(actor_model=_WaitActor())),
        design,
        config,
        decision_target=1,
        episode_cursor=0,
        device="cpu",
    )
    assert observed_coefficients == [coefficient]
    assert episodes[0].steps[0].reward == pytest.approx(-coefficient / 1800)
    assert episodes[0].truncated is True
    assert episodes[0].structure_completion_rate == 0


@pytest.mark.parametrize("coefficient", [0.15, 1.5])
def test_resume_binds_reward_coefficient_and_accepts_legacy_default_checkpoint(
    tmp_path: Path,
    coefficient: float,
) -> None:
    config = training.TrainingConfig.for_profile("unit")
    config.reward_time_coefficient = coefficient
    config.source_tree_digest = "same-tree"
    bundle = build_torchrl_policy("mappo", hidden_dim=config.hidden_dim)
    ppo_optimizer = torch.optim.Adam(
        [*bundle.actor_model.parameters(), *bundle.critic_model.parameters()],
        lr=config.learning_rate,
    )
    bc_optimizer = torch.optim.Adam(bundle.actor_model.parameters(), lr=config.learning_rate)
    checkpoint = training.save_training_checkpoint(
        tmp_path / "legacy.pt",
        bundle=bundle,
        ppo_optimizer=ppo_optimizer,
        bc_optimizer=bc_optimizer,
        config=config,
        configuration_digest_value=training.configuration_digest(config),
        design_digest="same-design",
        source_commit_value="same-source",
        transitions=0,
        updates=0,
        episode_cursor=0,
        bc_epoch=0,
        curve_rows=[],
        checkpoint_lineage=[],
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if coefficient == 0.15:
        payload["config"].pop("reward_time_coefficient")
    torch.save(payload, checkpoint)
    restored = training.TrainingConfig.model_validate(payload["config"])
    for current in (config, restored):
        training.load_training_checkpoint(
            checkpoint,
            bundle=bundle,
            ppo_optimizer=ppo_optimizer,
            bc_optimizer=bc_optimizer,
            expected_configuration_digest=training.configuration_digest(current),
            expected_design_digest="same-design",
            expected_source_commit="same-source",
            expected_source_dirty=False,
            expected_source_tree_digest="same-tree",
            device="cpu",
        )
    with pytest.raises(ValueError, match="configuration_digest"):
        training.load_training_checkpoint(
            checkpoint,
            bundle=bundle,
            ppo_optimizer=ppo_optimizer,
            bc_optimizer=bc_optimizer,
            expected_configuration_digest=training.configuration_digest(
                config.model_copy(
                    update={"reward_time_coefficient": 1.5 if coefficient == 0.15 else 0.15}
                )
            ),
            expected_design_digest="same-design",
            expected_source_commit="same-source",
            expected_source_dirty=False,
            expected_source_tree_digest="same-tree",
            device="cpu",
        )


def test_reward_pilot_specs_expand_into_real_training_configs() -> None:
    from scripts.run_construction_validation_pilot import build_pilot_protocol

    _matrix_id, _digest, protocol = build_pilot_protocol(
        {"commit": "a" * 40, "dirty": False, "tree_digest": "b" * 64},
        design_digest="c" * 64,
        parent_protocol_digest="d" * 64,
        device="cpu",
        comparison="reward-time",
    )
    configs = [
        training.TrainingConfig.model_validate(row["config"]) for row in protocol["run_specs"]
    ]
    assert {config.reward_time_coefficient for config in configs} == {0.15, 1.5}
    assert {config.ppo_epochs for config in configs} == {6}
    assert {config.transitions for config in configs} == {150_000}
    assert {config.failure_curriculum_reference_transitions for config in configs} == {1_500_000}
    assert len({training.configuration_digest(config) for config in configs}) == 4
