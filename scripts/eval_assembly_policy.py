# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.baseline import scripted_joint_policy
from embodied_skill_composer.assembly.models import EpisodeArtifact
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)
from embodied_skill_composer.assembly.trainer import MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted or learned collaborative-assembly policies.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--policy", choices=["scripted", "learned"], default="scripted")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_marl.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    env_config = load_assembly_scenario(Path(args.env_config))
    train_config = load_training_config(Path(args.train_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    env = build_assembly_backend(config=env_config, runtime_profile=runtime_profile, seed=train_config.seed)
    env.set_curriculum_stage(None)
    trainer = MAPPOTrainer(env=env, config=train_config, device=runtime_profile.device)

    if args.policy == "learned":
        trainer.load_checkpoint(checkpoint_path)

    artifacts: list[EpisodeArtifact] = []
    for episode in range(args.episodes):
        env.reset(seed=train_config.seed + episode)
        done = False
        while not done:
            if args.policy == "scripted":
                actions = scripted_joint_policy(env)
            else:
                observations = env.get_agent_observations()
                import torch

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
        artifacts.append(env.build_artifact(policy_mode=args.policy))

    success_rate = sum(int(artifact.metrics.success) for artifact in artifacts) / max(1, len(artifacts))
    mean_return = sum(artifact.metrics.total_reward for artifact in artifacts) / max(1, len(artifacts))
    print(f"Policy: {args.policy}")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Mean return: {mean_return:.3f}")
    print(json.dumps([artifact.model_dump(mode='json') for artifact in artifacts], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
