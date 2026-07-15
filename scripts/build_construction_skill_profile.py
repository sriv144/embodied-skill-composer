# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.skill_profiles import (
    skill_profile_from_mujoco_campaign,
    write_skill_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Construction v1 SkillProfile from a MuJoCo campaign.",
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "skill_profile.json",
    )
    args = parser.parse_args()
    profile = skill_profile_from_mujoco_campaign(args.campaign)
    output = write_skill_profile(profile, args.output)
    print(f"Wrote {profile.profile_id} to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
