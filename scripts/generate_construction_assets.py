from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.blender import (  # noqa: E402
    DEFAULT_BLENDER_PATH,
    generate_blender_assets,
)
from embodied_skill_composer.construction.compiler import compile_house_design  # noqa: E402
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modular house assets with Blender.")
    parser.add_argument(
        "--design",
        type=Path,
        default=WORKSPACE / "configs" / "construction" / "cottage_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER_PATH)
    args = parser.parse_args()
    plan = compile_house_design(load_house_design(args.design))
    artifacts = generate_blender_assets(plan, args.output, blender_path=args.blender)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
