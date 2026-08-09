from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


_SEMVER_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class RepositoryVersionError(ValueError):
    """Raised when repository release-version declarations disagree."""


@dataclass(frozen=True)
class RepositoryReleaseIdentity:
    version: str
    tag: str
    declarations: tuple[tuple[str, str], ...]


def validate_release_identity(version: str, tag: str) -> None:
    """Validate the package version and its canonical Git tag."""

    if _SEMVER_PATTERN.fullmatch(version) is None:
        raise ValueError(f"release version must be SemVer: {version!r}")
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(
            f"release tag must exactly match {expected_tag!r}; got {tag!r}"
        )


def repository_release_identity(workspace: Path) -> RepositoryReleaseIdentity:
    """Read every authoritative repository version and require exact agreement."""

    root = workspace.resolve()
    declarations = {
        "pyproject.toml:[project].version": _pyproject_version(
            root / "pyproject.toml"
        ),
        "src/embodied_skill_composer/__init__.py:__version__": _python_version(
            root / "src" / "embodied_skill_composer" / "__init__.py"
        ),
        "workbench/package.json:version": _json_version(
            root / "workbench" / "package.json"
        ),
        "workbench/package-lock.json:version": _json_version(
            root / "workbench" / "package-lock.json"
        ),
        "workbench/package-lock.json:packages[\"\"]:version": (
            _package_lock_root_version(
                root / "workbench" / "package-lock.json"
            )
        ),
    }
    versions = set(declarations.values())
    if len(versions) != 1:
        detail = ", ".join(
            f"{label}={version!r}"
            for label, version in sorted(declarations.items())
        )
        raise RepositoryVersionError(
            f"repository version declarations disagree: {detail}"
        )
    version = next(iter(versions))
    tag = f"v{version}"
    try:
        validate_release_identity(version, tag)
    except ValueError as exc:
        raise RepositoryVersionError(str(exc)) from exc
    return RepositoryReleaseIdentity(
        version=version,
        tag=tag,
        declarations=tuple(sorted(declarations.items())),
    )


def _pyproject_version(path: Path) -> str:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload["project"]
        value = project["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError, TypeError) as exc:
        raise RepositoryVersionError(
            f"cannot read project version from {path}: {exc}"
        ) from exc
    return _require_string(value, f"{path} [project].version")


def _python_version(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RepositoryVersionError(
            f"cannot read package version from {path}: {exc}"
        ) from exc
    values: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise RepositoryVersionError(
                    f"{path} __version__ must be a string literal"
                ) from exc
            values.append(_require_string(value, f"{path} __version__"))
    if len(values) != 1:
        raise RepositoryVersionError(
            f"{path} must assign __version__ exactly once"
        )
    return values[0]


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepositoryVersionError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RepositoryVersionError(f"{path} must contain a JSON object")
    return payload


def _json_version(path: Path) -> str:
    return _require_string(
        _read_json_object(path).get("version"),
        f"{path} version",
    )


def _package_lock_root_version(path: Path) -> str:
    payload = _read_json_object(path)
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        raise RepositoryVersionError(f"{path} packages must be an object")
    root = packages.get("")
    if not isinstance(root, dict):
        raise RepositoryVersionError(f"{path} packages[\"\"] must be an object")
    return _require_string(root.get("version"), f"{path} packages[\"\"] version")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepositoryVersionError(f"{label} must be a non-empty string")
    return value
