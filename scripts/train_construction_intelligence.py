# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import TrainingConfig, train_swarm_policy


DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Behavior-clone and MAPPO/IPPO-train the Construction Intelligence swarm.",
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), default="mappo")
    parser.add_argument("--profile", choices=("unit", "smoke", "research"), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "training",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to launch training without an interactive confirmation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TrainingConfig.for_profile(
        args.profile,
        algorithm=args.algorithm,
        seed=args.seed,
    )
    config.device = args.device
    config.output_root = args.output_root
    if args.transitions is not None:
        config.transitions = args.transitions
    if not args.yes and not _confirm(config):
        print("Training cancelled. No run directory was created.")
        return 2

    design = load_house_design(args.design)
    artifacts = train_swarm_policy(
        design,
        config,
        progress_callback=lambda item: print(json.dumps(item, sort_keys=True), flush=True),
    )
    print(artifacts.model_dump_json(indent=2))
    return 0


def _confirm(config: TrainingConfig) -> bool:
    if not sys.stdin.isatty():
        print("Refusing a non-interactive training launch without --yes.", file=sys.stderr)
        return False
    print(
        f"Launch {config.algorithm.upper()} {config.profile} training for "
        f"{config.transitions:,} transitions?"
    )
    return input("Type TRAIN to continue: ").strip() == "TRAIN"


if __name__ == "__main__":
    raise SystemExit(main())
