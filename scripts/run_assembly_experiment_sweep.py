# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.experiments import (
    default_experiment_output_dir,
    run_assembly_experiment_sweep,
)
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
    load_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generated assembly scenario training/evaluation sweep.")
    parser.add_argument("--env-config", default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "assembly_training.yaml"))
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--seeds", default="7,8,9")
    parser.add_argument("--beam-count", type=int, default=2)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_experiment_output_dir(PROJECT_ROOT / "logs" / "assembly_experiments")
    summary = run_assembly_experiment_sweep(
        base_env_config=load_assembly_scenario(Path(args.env_config)),
        training_config=load_training_config(Path(args.train_config)),
        runtime_profile=load_runtime_profile(Path(args.runtime_profile)),
        scenario_count=args.scenarios,
        seeds=parse_seeds(args.seeds),
        output_dir=output_dir,
        beam_count=args.beam_count,
        evaluation_episodes=args.evaluation_episodes,
    )
    failed = sum(1 for result in summary.results if result.status == "failed")
    print(f"Assembly experiment sweep complete: {summary.scenario_count} scenarios x {len(summary.seeds)} seeds")
    print(f"Runtime profile: {summary.runtime_profile} ({summary.backend})")
    print(f"Result rows: {len(summary.results)}")
    print(f"Failed rows: {failed}")
    print(f"Summary JSON: {Path(summary.output_dir) / 'summary.json'}")
    print(f"Summary CSV: {Path(summary.output_dir) / 'summary.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
