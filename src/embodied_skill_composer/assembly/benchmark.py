from __future__ import annotations

from pathlib import Path

import torch

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.baseline import scripted_joint_policy
from embodied_skill_composer.assembly.models import (
    AssemblyBenchmarkSummary,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    PolicyBenchmarkResult,
    TrainingConfig,
)
from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer
from embodied_skill_composer.assembly.trainer import MAPPOTrainer


def run_assembly_policy_benchmark(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    options_checkpoint: Path,
    low_level_checkpoint: Path,
    episodes: int = 5,
) -> AssemblyBenchmarkSummary:
    scripted_options = _evaluate_scripted_options(env_config, training_config, runtime_profile, episodes)
    learned_options = _evaluate_learned_options(
        env_config, training_config, runtime_profile, options_checkpoint, episodes
    )
    low_level_learned = _evaluate_low_level_learned(
        env_config, training_config, runtime_profile, low_level_checkpoint, episodes
    )
    return AssemblyBenchmarkSummary(
        backend=runtime_profile.backend,
        runtime_profile=runtime_profile.name,
        scripted_options=scripted_options,
        learned_options=learned_options,
        low_level_learned=low_level_learned,
    )


def _evaluate_scripted_options(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    episodes: int,
) -> PolicyBenchmarkResult:
    env = build_assembly_backend(env_config, runtime_profile, seed=training_config.seed)
    returns: list[float] = []
    beams: list[int] = []
    successes = 0
    for episode in range(episodes):
        env.reset(seed=training_config.seed + episode)
        done = False
        while not done:
            result = env.execute_team_option(env.scripted_team_option())
            done = result.done
        artifact = env.build_artifact(policy_mode="scripted")
        returns.append(artifact.metrics.total_reward)
        beams.append(artifact.metrics.beams_installed)
        successes += int(artifact.metrics.success)
    return PolicyBenchmarkResult(
        policy_name="scripted_options",
        success_rate=successes / max(1, episodes),
        mean_return=float(sum(returns) / max(1, len(returns))),
        mean_beams_installed=float(sum(beams) / max(1, len(beams))),
        notes="Scripted option oracle using deterministic execution.",
    )


def _evaluate_learned_options(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    checkpoint_path: Path,
    episodes: int,
) -> PolicyBenchmarkResult:
    env = build_assembly_backend(env_config, runtime_profile, seed=training_config.seed)
    trainer = HierarchicalOptionTrainer(env=env, config=training_config, device=runtime_profile.device)
    trainer.load_checkpoint(checkpoint_path)
    metrics = trainer.evaluate_policy(episodes=episodes)
    return PolicyBenchmarkResult(
        policy_name="learned_options",
        success_rate=metrics.success_rate,
        mean_return=metrics.mean_return,
        mean_beams_installed=metrics.mean_beams_installed,
        notes="Hierarchical team-options policy with imitation warm-start and PPO fine-tuning.",
    )


def _evaluate_low_level_learned(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    checkpoint_path: Path,
    episodes: int,
) -> PolicyBenchmarkResult:
    env = build_assembly_backend(env_config, runtime_profile, seed=training_config.seed)
    trainer = MAPPOTrainer(env=env, config=training_config, device=runtime_profile.device)
    trainer.load_checkpoint(checkpoint_path)
    returns: list[float] = []
    beams: list[int] = []
    successes = 0
    env.set_curriculum_stage(None)

    for episode in range(episodes):
        env.reset(seed=training_config.seed + episode)
        done = False
        while not done:
            observations = env.get_agent_observations()
            with torch.no_grad():
                logits = trainer._masked_logits(
                    trainer.actor(
                        torch.as_tensor(observations, dtype=torch.float32, device=trainer.device),
                        trainer._current_phase_indices(),
                    ),
                    torch.as_tensor(env.get_action_masks(), dtype=torch.float32, device=trainer.device),
                )
                actions = torch.argmax(logits, dim=-1).cpu().tolist()
            _, _, _, done, _ = env.step(actions)
        artifact = env.build_artifact(policy_mode="learned")
        returns.append(artifact.metrics.total_reward)
        beams.append(artifact.metrics.beams_installed)
        successes += int(artifact.metrics.success)
    return PolicyBenchmarkResult(
        policy_name="low_level_learned",
        success_rate=successes / max(1, episodes),
        mean_return=float(sum(returns) / max(1, len(returns))),
        mean_beams_installed=float(sum(beams) / max(1, len(beams))),
        notes="Retained low-level MARL baseline for comparison only.",
    )
