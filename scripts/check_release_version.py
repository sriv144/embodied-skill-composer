from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.release_identity import (  # noqa: E402
    RepositoryVersionError,
    repository_release_identity,
    validate_release_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Require Python and workbench package versions to agree and print "
            "the canonical release tag."
        )
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-tag")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        identity = repository_release_identity(args.workspace)
        if (args.expected_version is None) != (args.expected_tag is None):
            raise RepositoryVersionError(
                "--expected-version and --expected-tag must be supplied together"
            )
        if args.expected_version is not None and args.expected_tag is not None:
            validate_release_identity(args.expected_version, args.expected_tag)
            if (
                identity.version != args.expected_version
                or identity.tag != args.expected_tag
            ):
                raise RepositoryVersionError(
                    "repository release identity does not match the expected "
                    f"{args.expected_version}/{args.expected_tag}"
                )
    except (RepositoryVersionError, ValueError) as exc:
        parser.exit(1, f"release version check failed: {exc}\n")

    payload = {
        "version": identity.version,
        "tag": identity.tag,
        "declarations": dict(identity.declarations),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"{identity.version} {identity.tag}")


if __name__ == "__main__":
    main()
