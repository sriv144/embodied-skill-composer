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
    parser.add_argument(
        "--coppelia-evidence-root",
        type=Path,
        help="Existing directory containing native Phase 5 run bundles.",
    )
    args = parser.parse_args()
    registry_path = args.registry_path.resolve() if args.registry_path else None
    evidence_root = (
        args.coppelia_evidence_root.resolve() if args.coppelia_evidence_root else None
    )
    if evidence_root is not None and not evidence_root.is_dir():
        parser.error("--coppelia-evidence-root must be an existing directory")
    if args.reload and (registry_path or evidence_root):
        parser.error("--reload cannot be combined with custom registry or evidence paths")
    if registry_path:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
    application = (
        create_app(registry_path=registry_path, coppelia_evidence_root=evidence_root)
        if registry_path or evidence_root
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
