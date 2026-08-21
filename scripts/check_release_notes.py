from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_RELEASE_TITLE = "Construction Intelligence v1.0.0"


class ReleaseNotesError(ValueError):
    """Raised when GitHub draft metadata differs from the committed release notes."""


def canonical_release_notes(value: str) -> str:
    """Normalize transport-only newline differences while preserving Markdown."""
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def verify_release_notes(
    expected_path: Path,
    actual_payload: Any,
    *,
    expected_title: str = EXPECTED_RELEASE_TITLE,
) -> str:
    if not isinstance(actual_payload, dict):
        raise ReleaseNotesError("GitHub release metadata must be a JSON object")
    title = actual_payload.get("name")
    body = actual_payload.get("body")
    if title != expected_title:
        raise ReleaseNotesError("GitHub release title differs from the committed title")
    if not isinstance(body, str):
        raise ReleaseNotesError("GitHub release body must be a string")
    expected = canonical_release_notes(expected_path.read_text(encoding="utf-8"))
    actual = canonical_release_notes(body)
    expected_sha256 = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    actual_sha256 = hashlib.sha256(actual.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReleaseNotesError(
            "GitHub release notes differ from the committed v1 release notes"
        )
    return actual_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind a GitHub draft title/body to the committed v1 release notes.",
    )
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual-json", type=Path, required=True)
    parser.add_argument("--expected-title", default=EXPECTED_RELEASE_TITLE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = json.loads(args.actual_json.read_text(encoding="utf-8"))
        digest = verify_release_notes(
            args.expected,
            payload,
            expected_title=args.expected_title,
        )
    except (OSError, json.JSONDecodeError, ReleaseNotesError) as exc:
        build_parser().exit(1, f"release notes check failed: {exc}\n")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
