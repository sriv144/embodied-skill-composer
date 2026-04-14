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

from embodied_skill_composer.assembly.benchmark import run_assembly_policy_benchmark
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the assembly scripted, hierarchical, and low-level policies.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--options-checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_options.pt"))
    parser.add_argument("--low-level-checkpoint", default=str(PROJECT_ROOT / "logs" / "assembly_marl.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "logs" / "assembly_policy_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_assembly_policy_benchmark(
        env_config=load_assembly_scenario(Path(args.env_config)),
        training_config=load_training_config(Path(args.train_config)),
        runtime_profile=load_runtime_profile(Path(args.runtime_profile)),
        options_checkpoint=Path(args.options_checkpoint),
        low_level_checkpoint=Path(args.low_level_checkpoint),
        episodes=args.episodes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")

    rows = [
        summary.scripted_options,
        summary.learned_options,
        summary.low_level_learned,
    ]
    name_width = max(len("Policy"), *(len(row.policy_name) for row in rows))
    print(f"Runtime profile: {summary.runtime_profile} ({summary.backend})")
    print(f"{'Policy':<{name_width}}  Success  Return   Beams")
    for row in rows:
        print(
            f"{row.policy_name:<{name_width}}  "
            f"{row.success_rate:>7.3f}  "
            f"{row.mean_return:>6.2f}  "
            f"{row.mean_beams_installed:>5.2f}"
        )
    print(f"Summary JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
