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
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)
from embodied_skill_composer.assembly.visualizer import render_playback_frames, render_summary_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a local assembly episode as frame-by-frame 2D playback.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--policy", choices=["scripted", "learned"], default="scripted")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_options.pt"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diagnostics-json", help="Optional path to a saved diagnostics JSON file to replay.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "assembly_playback"),
        help="Directory where playback frames and summary image are written.",
    )
    return parser.parse_args()


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    env_config = load_assembly_scenario(Path(args.env_config))
    diagnostics: dict[str, object]

    if args.diagnostics_json:
        diagnostics = json.loads(Path(args.diagnostics_json).read_text(encoding="utf-8"))
    else:
        train_config = load_training_config(Path(args.train_config))
        runtime_profile = load_runtime_profile(Path(args.runtime_profile))
        env = build_assembly_backend(config=env_config, runtime_profile=runtime_profile, seed=train_config.seed)
        env.set_curriculum_stage(None)
        trainer = None
        if args.policy == "learned":
            if not torch_available():
                print("Learned playback requires torch. Install it with `pip install -r requirements-rl.txt`.")
                return 1
            import torch

            from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer

            trainer = HierarchicalOptionTrainer(env=env, config=train_config, device=runtime_profile.device)
            trainer.load_checkpoint(Path(args.checkpoint))

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
        diagnostics = env.get_option_episode_diagnostics()
        diagnostics_path = output_dir / f"diagnostics_{args.policy}.json"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    frame_paths = render_playback_frames(env_config, diagnostics, output_dir / "frames", title_prefix="assembly")
    summary_path = render_summary_figure(env_config, diagnostics, output_dir / "summary.png")

    print(f"Playback frames: {len(frame_paths)}")
    if frame_paths:
        print(f"First frame: {frame_paths[0]}")
    print(f"Summary image: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
