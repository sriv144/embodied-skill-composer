from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_skill_composer.construction.release_identity import (
    RepositoryVersionError,
    repository_release_identity,
    validate_release_identity,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def test_repository_release_versions_are_consistent() -> None:
    identity = repository_release_identity(WORKSPACE)

    assert identity.tag == f"v{identity.version}"
    assert len(identity.declarations) == 5
    assert {version for _, version in identity.declarations} == {
        identity.version
    }


def test_repository_release_identity_rejects_a_frontend_mismatch(
    tmp_path: Path,
) -> None:
    _write_repository_versions(
        tmp_path,
        python_version="0.1.0",
        workbench_version="0.1.1",
    )

    with pytest.raises(RepositoryVersionError, match="declarations disagree"):
        repository_release_identity(tmp_path)


@pytest.mark.parametrize(
    ("version", "tag"),
    [
        ("01.0.0", "v01.0.0"),
        ("1.0", "v1.0"),
        ("1.0.0", "1.0.0"),
        ("1.0.0", "v1.0.1"),
    ],
)
def test_release_identity_requires_semver_and_exact_v_tag(
    version: str,
    tag: str,
) -> None:
    with pytest.raises(ValueError):
        validate_release_identity(version, tag)


def _write_repository_versions(
    root: Path,
    *,
    python_version: str,
    workbench_version: str,
) -> None:
    package = root / "src" / "embodied_skill_composer"
    package.mkdir(parents=True)
    (root / "workbench").mkdir()
    (root / "pyproject.toml").write_text(
        (
            "[project]\n"
            'name = "embodied-skill-composer"\n'
            f'version = "{python_version}"\n'
        ),
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{python_version}"\n',
        encoding="utf-8",
    )
    (root / "workbench" / "package.json").write_text(
        json.dumps(
            {
                "name": "workbench",
                "version": workbench_version,
            }
        ),
        encoding="utf-8",
    )
    (root / "workbench" / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "workbench",
                "version": workbench_version,
                "lockfileVersion": 3,
                "packages": {
                    "": {
                        "name": "workbench",
                        "version": workbench_version,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
