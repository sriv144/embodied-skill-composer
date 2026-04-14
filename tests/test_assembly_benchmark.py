import importlib.util
from pathlib import Path

import pytest

from embodied_skill_composer.assembly.backends import IsaacBackendNotReadyError, build_assembly_backend
from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BeamTask,
    TrainingConfig,
)


torch_available = importlib.util.find_spec("torch") is not None


def build_default_assembly_config() -> AssemblyScenarioConfig:
    return AssemblyScenarioConfig(
        grid_size=12,
        max_steps=120,
        agent_starts=[(0, 2), (0, 3)],
        beams=[
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(8, 7),
                assembly_right=(8, 8),
            ),
            BeamTask(
                name="beam_beta",
                pickup_left=(2, 6),
                pickup_right=(2, 7),
                assembly_left=(9, 7),
                assembly_right=(9, 8),
            ),
        ],
        curriculum_stage_beams=[
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                )
            ],
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                ),
                BeamTask(
                    name="beam_beta_easy",
                    pickup_left=(2, 5),
                    pickup_right=(2, 6),
                    assembly_left=(8, 9),
                    assembly_right=(8, 10),
                ),
            ],
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                ),
                BeamTask(
                    name="beam_beta",
                    pickup_left=(2, 6),
                    pickup_right=(2, 7),
                    assembly_left=(9, 7),
                    assembly_right=(9, 8),
                ),
            ],
        ],
    )


def build_fast_training_config() -> TrainingConfig:
    return TrainingConfig(
        total_iterations=2,
        episodes_per_iteration=3,
        update_epochs=2,
        option_behavior_cloning_epochs=25,
        option_update_epochs=2,
        evaluation_episodes=2,
        seed=7,
    )


def test_local_backend_factory_preserves_scripted_behavior() -> None:
    config = build_default_assembly_config()
    profile = AssemblyRuntimeProfile()
    direct = CollaborativeAssemblyEnv(config=config, seed=7)
    from_factory = build_assembly_backend(config=config, runtime_profile=profile, seed=7)

    direct.reset(seed=7)
    from_factory.reset(seed=7)

    done = False
    while not done:
        direct_result = direct.execute_team_option(direct.scripted_team_option())
        factory_result = from_factory.execute_team_option(from_factory.scripted_team_option())
        done = direct_result.done
        assert direct_result.success == factory_result.success

    direct_artifact = direct.build_artifact(policy_mode="scripted")
    factory_artifact = from_factory.build_artifact(policy_mode="scripted")
    assert direct_artifact.metrics.total_reward == factory_artifact.metrics.total_reward
    assert direct_artifact.metrics.beams_installed == factory_artifact.metrics.beams_installed


def test_isaac_backend_stub_preserves_contract_shape_but_raises_on_execution() -> None:
    config = build_default_assembly_config()
    profile = AssemblyRuntimeProfile(
        name="isaac_gpu",
        backend="isaac_lab",
        device="cuda",
        requires_linux=True,
        requires_nvidia_gpu=True,
    )
    backend = build_assembly_backend(config=config, runtime_profile=profile, seed=7)

    observation, state = backend.reset(seed=9)
    diagnostics = backend.get_option_episode_diagnostics()

    assert observation.shape == (backend.team_option_obs_dim,)
    assert state.shape == (backend.state_dim,)
    assert diagnostics["status"] == "stub"
    assert diagnostics["last_seed"] == 9

    with pytest.raises(IsaacBackendNotReadyError):
        backend.execute_team_option(0)


@pytest.mark.skipif(not torch_available, reason="torch is required for benchmark smoke tests")
def test_policy_benchmark_smoke(tmp_path: Path) -> None:
    from embodied_skill_composer.assembly.benchmark import run_assembly_policy_benchmark
    from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer
    from embodied_skill_composer.assembly.trainer import MAPPOTrainer

    config = build_default_assembly_config()
    training = build_fast_training_config()
    profile = AssemblyRuntimeProfile()

    option_env = build_assembly_backend(config=config, runtime_profile=profile, seed=training.seed)
    option_trainer = HierarchicalOptionTrainer(option_env, training, device="cpu")
    option_ckpt = tmp_path / "assembly_options.pt"
    option_metrics = tmp_path / "assembly_options_metrics.json"
    option_trainer.train(option_ckpt, option_metrics)

    low_level_env = build_assembly_backend(config=config, runtime_profile=profile, seed=training.seed)
    low_level_trainer = MAPPOTrainer(low_level_env, training, device="cpu")
    low_level_ckpt = tmp_path / "assembly_marl.pt"
    low_level_metrics = tmp_path / "assembly_marl_metrics.json"
    low_level_trainer.train(low_level_ckpt, low_level_metrics)

    summary = run_assembly_policy_benchmark(
        env_config=config,
        training_config=training,
        runtime_profile=profile,
        options_checkpoint=option_ckpt,
        low_level_checkpoint=low_level_ckpt,
        episodes=2,
    )

    assert summary.scripted_options.success_rate == 1.0
    assert summary.learned_options.success_rate >= 1.0
    assert summary.learned_options.mean_beams_installed == 2.0
    assert summary.low_level_learned.success_rate < summary.learned_options.success_rate
