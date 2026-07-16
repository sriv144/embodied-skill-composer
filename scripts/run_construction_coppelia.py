from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.compiler import compile_house_design  # noqa: E402
from embodied_skill_composer.construction.coppelia import (  # noqa: E402
    CoppeliaConstructionAdapter,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402
from embodied_skill_composer.construction.scheduler import schedule_build  # noqa: E402
from embodied_skill_composer.construction.trace import build_execution_trace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Construction v2 in CoppeliaSim.")
    parser.add_argument(
        "--design",
        type=Path,
        default=WORKSPACE / "configs" / "construction" / "cottage_v1.yaml",
    )
    parser.add_argument("--controller", choices=["sequential", "greedy", "optimized"], default="optimized")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--save-scene",
        type=Path,
        default=WORKSPACE / "artifacts" / "construction_v2" / "coppelia_cottage_v2.ttt",
    )
    args = parser.parse_args()
    plan = compile_house_design(load_house_design(args.design))
    trace = build_execution_trace(plan, schedule_build(plan, args.controller))
    adapter = CoppeliaConstructionAdapter(plan, trace)
    adapter.connect()
    diagnostics = adapter.play(max_frames=args.max_frames)
    adapter.save_scene(args.save_scene)
    print(json.dumps(diagnostics, indent=2))
    print(f"scene: {args.save_scene.resolve()}")


if __name__ == "__main__":
    main()
