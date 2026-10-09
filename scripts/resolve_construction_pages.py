from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal


def resolve_pages_channel(
    release: object,
    *,
    requested: Literal["auto", "release"] = "auto",
    allow_draft: bool = False,
) -> Literal["preview", "release"]:
    """Only an authoritative missing release or an unpublished draft permits preview."""

    if release is None:
        channel: Literal["preview", "release"] = "preview"
    else:
        if not isinstance(release, dict) or release.get("tag_name") != "v1.0.0":
            raise ValueError("Pages requires metadata for the fixed v1.0.0 release")
        draft, prerelease = release.get("draft"), release.get("prerelease")
        if not isinstance(draft, bool) or not isinstance(prerelease, bool):
            raise ValueError("Pages release state is missing or malformed")
        if prerelease:
            raise ValueError("Pages v1.0.0 cannot consume prerelease evidence")
        channel = "release" if not draft or allow_draft else "preview"
    if requested == "release" and channel != "release":
        raise ValueError("canonical Pages requires an available published v1.0.0 release")
    return channel


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the fail-closed v1 Pages data channel.")
    parser.add_argument("--requested", choices=("auto", "release"), default="auto")
    metadata = parser.add_mutually_exclusive_group(required=True)
    metadata.add_argument("--release-json", type=Path)
    metadata.add_argument("--not-found", action="store_true")
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    try:
        release = (
            json.loads(args.release_json.read_text(encoding="utf-8"))
            if args.release_json is not None
            else None
        )
        channel = resolve_pages_channel(
            release, requested=args.requested, allow_draft=args.allow_draft
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Pages channel resolution failed: {exc}\n")
    print(channel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
