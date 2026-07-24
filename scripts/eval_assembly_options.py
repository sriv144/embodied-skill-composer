# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.models import EpisodeArtifact
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted or learned hierarchical assembly options.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--policy", choices=["scripted", "learned"], default="scripted")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_options.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    env_config = load_assembly_scenario(Path(args.env_config))
    train_config = load_training_config(Path(args.train_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    env = build_assembly_backend(config=env_config, runtime_profile=runtime_profile, seed=train_config.seed)
    trainer = None
    if args.policy == "learned":
        if not torch_available():
            print("Learned hierarchical evaluation requires torch. Install it with `pip install -r requirements-rl.txt`.")
            return 1
        import torch

        from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer

        trainer = HierarchicalOptionTrainer(env=env, config=train_config, device=runtime_profile.device)
        trainer.load_checkpoint(checkpoint_path)

    env.set_curriculum_stage(None)
    episodes: list[dict[str, object]] = []
    artifacts: list[EpisodeArtifact] = []
    for episode in range(args.episodes):
        env.reset(seed=train_config.seed + episode)
        done = False
        while not done:
            option: int
            if args.policy == "scripted":
                option = int(env.scripted_team_option())
            else:
                import torch

                assert trainer is not None
                observation = torch.as_tensor(
                    env.get_team_option_observation(), dtype=torch.float32, device=trainer.device
                ).unsqueeze(0)
                mask = torch.as_tensor(trainer._masked_option_array(), dtype=torch.float32, device=trainer.device).unsqueeze(0)
                with torch.no_grad():
                    logits = trainer._masked_logits(trainer.actor(observation), mask)
                    option = int(torch.argmax(logits, dim=-1).item())
            result = env.execute_team_option(option, max_primitive_steps=env.config.option_max_primitive_steps)
            done = result.done
        artifact = env.build_artifact(policy_mode=args.policy)
        artifacts.append(artifact)
        episodes.append(
            {
                "artifact": artifact.model_dump(mode="json"),
                "diagnostics": env.get_option_episode_diagnostics(),
            }
        )

    success_rate = sum(int(artifact.metrics.success) for artifact in artifacts) / max(
        1, len(artifacts)
    )
    mean_return = sum(artifact.metrics.total_reward for artifact in artifacts) / max(
        1, len(artifacts)
    )
    mean_beams_installed = sum(
        artifact.metrics.beams_installed for artifact in artifacts
    ) / max(1, len(artifacts))
    print(f"Policy: {args.policy}")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Mean return: {mean_return:.3f}")
    print(f"Mean beams installed: {mean_beams_installed:.3f}")
    print(json.dumps(episodes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
