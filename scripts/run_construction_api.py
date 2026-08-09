from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn  # noqa: E402
from embodied_skill_composer.construction.api import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Construction v2 API.")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--registry-path",
        type=Path,
        help="Optional isolated durable-run database (useful for acceptance tests).",
    )
    args = parser.parse_args()
    registry_path = args.registry_path.resolve() if args.registry_path else None
    if registry_path:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reload and registry_path:
        parser.error("--reload cannot be combined with --registry-path")
    application = (
        create_app(registry_path=registry_path)
        if registry_path
        else "embodied_skill_composer.construction.api:app"
    )
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
