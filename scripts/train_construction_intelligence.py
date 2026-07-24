# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.lab_registry import QUIESCENT_RUN_STATUSES, LabRegistry
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.training import TrainingConfig


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
    parser.add_argument(
        "--registry",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "lab.sqlite",
        help="Persistent lab queue database.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Return after the durable worker starts instead of streaming until completion.",
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
    registry = LabRegistry(args.registry)
    service = LabService(registry)
    run_id = service.launch_training(design, config)
    print(json.dumps({"run_id": run_id, "status": "queued"}), flush=True)
    try:
        sequence = 0
        while True:
            for event in registry.list_events(run_id, after=sequence):
                sequence_value = event["sequence"]
                if not isinstance(sequence_value, int):
                    raise RuntimeError("persisted event sequence is not an integer")
                sequence = sequence_value
                print(json.dumps(event, sort_keys=True), flush=True)
            run = registry.get_run(run_id)
            if run is None:
                raise RuntimeError(f"durable run disappeared: {run_id}")
            if args.detach and run["status"] != "queued":
                print(json.dumps(run, sort_keys=True), flush=True)
                return 0
            if run["status"] in QUIESCENT_RUN_STATUSES:
                print(json.dumps(run, sort_keys=True), flush=True)
                return 0 if run["status"] == "completed" else 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        service.cancel(run_id)
        print(f"Cancellation requested for {run_id}.", file=sys.stderr)
        return 130
    finally:
        service.shutdown()


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
