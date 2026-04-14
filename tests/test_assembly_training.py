import importlib.util
from pathlib import Path

import pytest

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import AssemblyScenarioConfig, BeamTask, TrainingConfig


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="torch is required for MARL training smoke tests")
def test_marl_training_smoke(tmp_path: Path) -> None:
    from embodied_skill_composer.assembly.trainer import MAPPOTrainer

    env = CollaborativeAssemblyEnv(
        config=AssemblyScenarioConfig(
            grid_size=8,
            max_steps=40,
            agent_starts=[(0, 1), (0, 2)],
            beams=[
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 1),
                    pickup_right=(2, 2),
                    assembly_left=(5, 4),
                    assembly_right=(5, 5),
                )
            ],
        ),
        seed=7,
    )
    trainer = MAPPOTrainer(
        env=env,
        config=TrainingConfig(total_iterations=2, episodes_per_iteration=3, update_epochs=2, seed=7),
        device="cpu",
    )

    summary = trainer.train(
        checkpoint_path=tmp_path / "assembly_marl.pt",
        metrics_path=tmp_path / "assembly_training_metrics.json",
    )

    assert summary.iterations == 2
    assert (tmp_path / "assembly_marl.pt").exists()
    assert (tmp_path / "assembly_training_metrics.json").exists()
    assert summary.last_mean_return >= summary.warmstart_success_rate - 1.0


@pytest.mark.skipif(not torch_available, reason="torch is required for option-policy training smoke tests")
def test_hierarchical_option_training_smoke(tmp_path: Path) -> None:
    from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer

    env = CollaborativeAssemblyEnv(
        config=AssemblyScenarioConfig(
            grid_size=10,
            max_steps=80,
            agent_starts=[(0, 2), (0, 3)],
            beams=[
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(7, 6),
                    assembly_right=(7, 7),
                ),
                BeamTask(
                    name="beam_beta",
                    pickup_left=(2, 5),
                    pickup_right=(2, 6),
                    assembly_left=(8, 6),
                    assembly_right=(8, 7),
                ),
            ],
            curriculum_stage_beams=[
                [
                    BeamTask(
                        name="beam_alpha",
                        pickup_left=(2, 2),
                        pickup_right=(2, 3),
                        assembly_left=(7, 6),
                        assembly_right=(7, 7),
                    )
                ],
                [
                    BeamTask(
                        name="beam_alpha",
                        pickup_left=(2, 2),
                        pickup_right=(2, 3),
                        assembly_left=(7, 6),
                        assembly_right=(7, 7),
                    ),
                    BeamTask(
                        name="beam_beta_easy",
                        pickup_left=(2, 4),
                        pickup_right=(2, 5),
                        assembly_left=(7, 8),
                        assembly_right=(7, 9),
                    ),
                ],
                [
                    BeamTask(
                        name="beam_alpha",
                        pickup_left=(2, 2),
                        pickup_right=(2, 3),
                        assembly_left=(7, 6),
                        assembly_right=(7, 7),
                    ),
                    BeamTask(
                        name="beam_beta",
                        pickup_left=(2, 5),
                        pickup_right=(2, 6),
                        assembly_left=(8, 6),
                        assembly_right=(8, 7),
                    ),
                ],
            ],
        ),
        seed=7,
    )
    trainer = HierarchicalOptionTrainer(
        env=env,
        config=TrainingConfig(
            total_iterations=2,
            episodes_per_iteration=3,
            option_behavior_cloning_epochs=25,
            option_update_epochs=2,
            evaluation_episodes=2,
            seed=7,
        ),
        device="cpu",
    )

    summary = trainer.train(
        checkpoint_path=tmp_path / "assembly_options.pt",
        metrics_path=tmp_path / "assembly_option_training_metrics.json",
    )

    assert summary.iterations == 2
    assert (tmp_path / "assembly_options.pt").exists()
    assert (tmp_path / "assembly_option_training_metrics.json").exists()
    assert summary.last_mean_beams_installed >= 1.0
