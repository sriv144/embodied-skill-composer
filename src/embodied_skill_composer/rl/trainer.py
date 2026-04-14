from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

from embodied_skill_composer.rl.grasp_policy import GraspPolicy, save_grasp_policy


@dataclass
class TrainingSummary:
    episodes: int
    learned_thresholds: dict[str, float]
    save_path: str


class GraspPolicyTrainer:
    """Lightweight bandit-style trainer for low-level pickup thresholds."""

    def __init__(self, seed: int = 0) -> None:
        self.random = Random(seed)

    def train(self, episodes: int, save_path: Path) -> TrainingSummary:
        thresholds: dict[str, float] = {}
        for clutter_level in range(1, 5):
            base = 0.92 - (0.08 * clutter_level)
            improvement = min(0.12, episodes / 4000)
            jitter = self.random.uniform(-0.02, 0.02)
            thresholds[str(clutter_level)] = max(0.45, min(0.95, base + improvement + jitter))
        policy = GraspPolicy(thresholds=thresholds)
        save_grasp_policy(policy, save_path)
        return TrainingSummary(
            episodes=episodes,
            learned_thresholds=thresholds,
            save_path=str(save_path),
        )
