# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.evaluation import (
    run_evaluation_suite,
    write_evaluation_artifacts,
)
from embodied_skill_composer.construction.policy import load_policy_checkpoint
from embodied_skill_composer.construction.runtime import load_house_design


DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate construction controllers on held-out procedural cottages.",
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--seeds", default="900,901,902,903,904")
    parser.add_argument(
        "--controllers",
        default="sequential,greedy,auction,cp_sat",
        help="Comma-separated sequential,greedy,auction,ippo,mappo,cp_sat controllers.",
    )
    parser.add_argument("--mappo-checkpoint", type=Path)
    parser.add_argument("--ippo-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "evaluations",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    controllers = [item.strip() for item in args.controllers.split(",") if item.strip()]
    allowed = {"sequential", "greedy", "auction", "ippo", "mappo", "cp_sat"}
    unknown = sorted(set(controllers) - allowed)
    if unknown:
        raise SystemExit(f"Unknown controllers: {', '.join(unknown)}")
    policies = {}
    if args.mappo_checkpoint:
        policies["mappo"] = load_policy_checkpoint(args.mappo_checkpoint, device=args.device)
    if args.ippo_checkpoint:
        policies["ippo"] = load_policy_checkpoint(args.ippo_checkpoint, device=args.device)
    missing = [item for item in ("mappo", "ippo") if item in controllers and item not in policies]
    if missing:
        raise SystemExit(f"Missing checkpoint arguments for: {', '.join(missing)}")

    suite = run_evaluation_suite(
        load_house_design(args.design),
        seeds=seeds,
        controllers=controllers,
        policies=policies,
        include_failure_suite=not args.no_failures,
        device=args.device,
    )
    artifacts = write_evaluation_artifacts(suite, args.output_root)
    print(artifacts.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
