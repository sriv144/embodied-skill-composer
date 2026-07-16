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

from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.lab_service import LabService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and control durable construction jobs.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "lab.sqlite",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("status", "events", "cancel", "resume"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
    commands.choices["events"].add_argument("--after", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = LabRegistry(args.registry)
    if args.command == "list":
        result: object = registry.list_runs()
    elif args.command == "status":
        result = registry.get_run(args.run_id)
        if result is None:
            print(f"unknown run: {args.run_id}", file=sys.stderr)
            return 2
    elif args.command == "events":
        if registry.get_run(args.run_id) is None:
            print(f"unknown run: {args.run_id}", file=sys.stderr)
            return 2
        result = registry.list_events(args.run_id, after=args.after)
    else:
        accepted = (
            registry.request_cancel(args.run_id)
            if args.command == "cancel"
            else registry.request_resume(args.run_id)
        )
        if not accepted:
            print(f"{args.command} rejected for {args.run_id}", file=sys.stderr)
            return 2
        if args.command == "resume":
            service = LabService(registry)
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    resumed = registry.get_run(args.run_id)
                    if resumed and resumed["status"] != "resuming":
                        break
                    time.sleep(0.05)
            finally:
                service.shutdown()
        result = registry.get_run(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
