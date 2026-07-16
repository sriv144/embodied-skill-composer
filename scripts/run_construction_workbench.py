from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.runtime import (  # noqa: E402
    run_construction_workbench,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a reviewed house and compare multi-robot construction controllers."
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=WORKSPACE / "configs" / "construction" / "cottage_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "logs" / "construction_v2" / "runs",
    )
    args = parser.parse_args()
    run_dir, plan, traces = run_construction_workbench(args.design, args.output_root)
    print(f"Construction v2 run: {run_dir}")
    print(f"Modules: {len(plan.modules)} | Robots: {len(plan.robots)}")
    for name, trace in traces.items():
        print(
            f"{name:10s} makespan={trace.metrics.makespan_s:4d}s "
            f"travel={trace.metrics.total_travel_m:6.1f}m "
            f"idle={trace.metrics.idle_robot_seconds:4d}s"
        )
    sequential = traces["sequential"].metrics.makespan_s
    optimized = traces["optimized"].metrics.makespan_s
    print(f"Optimized improvement: {100 * (1 - optimized / sequential):.1f}%")


if __name__ == "__main__":
    main()
