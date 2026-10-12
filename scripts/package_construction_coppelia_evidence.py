from __future__ import annotations

import argparse
from pathlib import Path

from embodied_skill_composer.construction.coppelia_replay_video import (
    package_coppelia_release_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify nominal and recovery Phase 5 bundles, render measured "
            "replays, and create the canonical public simulator descriptor."
        )
    )
    parser.add_argument("--nominal-bundle", type=Path, required=True)
    parser.add_argument("--recovery-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--duration-seconds", type=float, default=24.0)
    args = parser.parse_args()
    descriptor = package_coppelia_release_evidence(
        args.output,
        nominal_bundle=args.nominal_bundle,
        recovery_bundle=args.recovery_bundle,
        fps=args.fps,
        duration_s=args.duration_seconds,
    )
    print(descriptor)


if __name__ == "__main__":
    main()
