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
    stage_release_assets,
    verify_release_assets,
)
from embodied_skill_composer.construction.release_identity import (  # noqa: E402
    RepositoryVersionError,
    repository_release_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage or fail-closed verify the five Construction Intelligence v1 "
            "GitHub release assets."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "output" / "construction-intelligence-v1-release",
    )
    parser.add_argument("--deterministic-bundle", type=Path)
    parser.add_argument("--research-bundle", type=Path)
    parser.add_argument("--simulator-bundle", type=Path)
    parser.add_argument("--release-version")
    parser.add_argument("--release-tag")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing exact release-asset directory without writing it.",
    )
    parser.add_argument(
        "--expected-tag",
        help="With --verify-only, require this tag in the identity manifest.",
    )
    parser.add_argument(
        "--require-tag-at-head",
        action="store_true",
        help="With --verify-only, require the recorded Git tag to resolve to HEAD.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    try:
        repository_identity = repository_release_identity(workspace)
        if args.verify_only:
            if any(
                value is not None
                for value in (
                    args.deterministic_bundle,
                    args.research_bundle,
                    args.simulator_bundle,
                )
            ):
                parser.error("input evidence descriptors cannot be used with --verify-only")
            manifest = verify_release_assets(
                args.output,
                workspace=workspace,
                expected_release_version=args.release_version,
                expected_release_tag=args.expected_tag or args.release_tag,
                require_tag_at_head=args.require_tag_at_head,
            )
        else:
            if args.expected_tag is not None or args.require_tag_at_head:
                parser.error(
                    "--expected-tag and --require-tag-at-head require --verify-only"
                )
            descriptors = (
                args.deterministic_bundle,
                args.research_bundle,
                args.simulator_bundle,
            )
            if any(path is None for path in descriptors):
                parser.error(
                    "staging requires --deterministic-bundle, --research-bundle, "
                    "and --simulator-bundle"
                )
            release_version = args.release_version or repository_identity.version
            release_tag = args.release_tag or repository_identity.tag
            manifest = stage_release_assets(
                args.output,
                workspace=workspace,
                deterministic_bundle=args.deterministic_bundle,
                research_bundle=args.research_bundle,
                simulator_bundle=args.simulator_bundle,
                release_version=release_version,
                release_tag=release_tag,
            )
    except (OSError, ReleaseAssetError, RepositoryVersionError, ValueError) as exc:
        parser.exit(1, f"release asset check failed: {exc}\n")
    print(
        json.dumps(
            {
                "asset_count": len(manifest.assets),
                "output": str(args.output.resolve()),
                "release_tag": manifest.release_tag,
                "release_version": manifest.release_version,
                "source_commit": manifest.source.commit,
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
