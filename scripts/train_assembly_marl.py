# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)
from embodied_skill_composer.assembly.trainer import MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local collaborative-assembly MARL policy.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_marl.pt"))
    parser.add_argument("--metrics", default=str(PROJECT_ROOT / "logs" / "assembly_training_metrics.json"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = load_assembly_scenario(Path(args.env_config))
    train_config = load_training_config(Path(args.train_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    env = build_assembly_backend(config=env_config, runtime_profile=runtime_profile, seed=train_config.seed)
    trainer = MAPPOTrainer(env=env, config=train_config, device=args.device or runtime_profile.device)
    summary = trainer.train(checkpoint_path=Path(args.checkpoint), metrics_path=Path(args.metrics))

    print("Training complete")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Iterations: {summary.iterations}")
    print(f"Warm-start success rate: {summary.warmstart_success_rate:.3f}")
    print(f"Last mean return: {summary.last_mean_return:.3f}")
    print(f"Last success rate: {summary.last_success_rate:.3f}")
    print(f"Scripted baseline success rate: {summary.baseline_success_rate:.3f}")
    print(f"Checkpoint: {summary.checkpoint_path}")
    print(f"Metrics: {summary.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
