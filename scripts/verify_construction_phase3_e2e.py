# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.phase3_verification import (
    DEFAULT_DESIGN_PATH,
    DEFAULT_OUTPUT_ROOT,
    run_phase3_end_to_end_verification,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean-tree, non-synthetic Construction Intelligence Phase-3 "
            "end-to-end verification."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        ),
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    def progress(payload: dict[str, object]) -> None:
        print(json.dumps(payload, sort_keys=True), flush=True)

    result = run_phase3_end_to_end_verification(
        args.output_root,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        design_path=args.design,
        progress_callback=progress,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
