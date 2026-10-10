from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.release_assets import (  # noqa: E402
    ReleaseAssetError,
    stage_release_public_demo,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the complete indexed v1.0.0 release asset set and atomically replace "
            "the Pages build's public demo with their canonical evidence."
        )
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--release-assets", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = stage_release_public_demo(args.release_assets, workspace=args.workspace)
    except (OSError, ReleaseAssetError, ValueError) as exc:
        parser.exit(1, f"Pages release staging failed: {exc}\n")
    print(
        json.dumps(
            {
                "channel": "release",
                "release_tag": manifest.release_tag,
                "source_commit": manifest.source.commit,
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
