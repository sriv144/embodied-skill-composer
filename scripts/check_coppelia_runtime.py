# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.coppelia_backend import inspect_coppelia_runtime
from embodied_skill_composer.assembly.runtime import load_runtime_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the local CoppeliaSim installation and ZeroMQ connection."
    )
    parser.add_argument(
        "--runtime-profile",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "assembly_profiles"
            / "coppelia_local.yaml"
        ),
    )
    parser.add_argument("--host", help="Override the profile ZeroMQ host.")
    parser.add_argument("--port", type=int, help="Override the profile ZeroMQ port.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_runtime_profile(Path(args.runtime_profile))
    if args.host or args.port:
        profile = profile.model_copy(
            update={
                "coppelia": profile.coppelia.model_copy(
                    update={
                        "host": args.host or profile.coppelia.host,
                        "port": args.port or profile.coppelia.port,
                    }
                )
            }
        )
    status = inspect_coppelia_runtime(profile)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Runtime profile: {status['runtime_profile']} ({status['backend']})")
        print(f"Executable: {status['executable_path']}")
        print(f"Executable exists: {status['executable_exists']}")
        print(f"Python client installed: {status['client_installed']}")
        print(f"ZeroMQ endpoint: {status['host']}:{status['port']}")
        print(f"Connected: {status['connected']}")
        if status["connected"]:
            print(f"Scene objects: {status['object_count']}")
            print(f"Physics engine id: {status['physics_engine']}")
        if status["error"]:
            print(f"Connection note: {status['error']}")
    return 0 if status["connected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
