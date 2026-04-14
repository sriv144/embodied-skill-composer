# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.rl.trainer import GraspPolicyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight RL pickup policy.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "logs" / "grasp_policy.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trainer = GraspPolicyTrainer(seed=7)
    summary = trainer.train(episodes=args.episodes, save_path=Path(args.output))
    print(f"Episodes: {summary.episodes}")
    print(f"Saved policy: {summary.save_path}")
    for clutter, threshold in sorted(summary.learned_thresholds.items()):
        print(f"  clutter {clutter}: {threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
