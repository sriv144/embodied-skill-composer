from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.research_release_evidence import (
    package_research_release_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the completed Phase 4 matrix and atomically create its "
            "canonical public research descriptor."
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--heldout-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_registry = args.registry.resolve()
    if not source_registry.is_file():
        parser.error(f"registry does not exist: {source_registry}")
    with tempfile.TemporaryDirectory(prefix="construction-research-registry-") as raw:
        snapshot = Path(raw) / "lab.sqlite"
        _snapshot_registry(source_registry, snapshot)
        descriptor = package_research_release_evidence(
            args.output,
            registry=LabRegistry(snapshot),
            matrix_id=args.matrix_id,
            selection_evidence_path=args.selection_evidence,
            heldout_run_dir=args.heldout_run_dir,
        )
    print(descriptor)


def _snapshot_registry(source_path: Path, destination_path: Path) -> None:
    """Use SQLite's online backup API while keeping the source strictly read-only."""

    source_uri = f"{source_path.as_uri()}?mode=ro"
    with (
        sqlite3.connect(source_uri, uri=True, timeout=30) as source,
        sqlite3.connect(destination_path) as destination,
    ):
        source.backup(destination)


if __name__ == "__main__":
    main()
