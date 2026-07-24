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
from embodied_skill_composer.assembly.mujoco_backend import MuJoCoAssemblyBackend
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the collaborative assembly task in MuJoCo.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "mujoco_local.yaml"),
    )
    parser.add_argument("--policy", choices=["scripted", "learned"], default="scripted")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_options.pt"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gui", action="store_true", help="Open MuJoCo viewer playback after the episode.")
    parser.add_argument("--record", help="Optional MP4/GIF path to record rendered MuJoCo playback.")
    parser.add_argument(
        "--diagnostics-output",
        default=str(PROJECT_ROOT / "logs" / "mujoco_assembly_episode.json"),
        help="Where to write episode artifact and diagnostics.",
    )
    return parser.parse_args()


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def main() -> int:
    args = parse_args()
    env_config = load_assembly_scenario(Path(args.env_config))
    train_config = load_training_config(Path(args.train_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    env = build_assembly_backend(config=env_config, runtime_profile=runtime_profile, seed=train_config.seed)
    if not isinstance(env, MuJoCoAssemblyBackend):
        print(f"Runtime profile '{runtime_profile.name}' must use backend 'mujoco_local'.")
        return 1
    if not env.is_ready:
        print("MuJoCo backend is not ready. Install dependencies with `pip install -r requirements-sim-mujoco.txt`.")
        return 1

    trainer = None
    if args.policy == "learned":
        if not torch_available():
            print("Learned MuJoCo playback requires torch. Install `requirements-rl.txt` first.")
            return 1
        import torch

        from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer

        trainer = HierarchicalOptionTrainer(env=env, config=train_config, device=runtime_profile.device)
        trainer.load_checkpoint(Path(args.checkpoint))

    env.set_curriculum_stage(None)
    env.reset(seed=args.seed)
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
    diagnostics = env.get_option_episode_diagnostics()
    record_path = None
    if args.record:
        record_path = env.record_episode(Path(args.record), diagnostics=diagnostics)
        diagnostics = env.get_option_episode_diagnostics()
    if args.gui:
        env.launch_viewer_playback(diagnostics=diagnostics)

    payload = {
        "artifact": artifact.model_dump(mode="json"),
        "diagnostics": diagnostics,
    }
    diagnostics_path = Path(args.diagnostics_output)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Policy: {args.policy}")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Success: {artifact.metrics.success}")
    print(f"Beams installed: {artifact.metrics.beams_installed}/{artifact.metrics.total_beams}")
    print(f"Steps: {artifact.metrics.step_count}")
    print(f"Return: {artifact.metrics.total_reward:.3f}")
    print(f"Diagnostics: {diagnostics_path}")
    if record_path:
        print(f"Recording: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
