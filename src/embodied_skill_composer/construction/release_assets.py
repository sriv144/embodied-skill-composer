from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from embodied_skill_composer.construction.experiment_protocol import (
    SelectedCheckpoint,
)
from embodied_skill_composer.construction.policy import (
    export_actor_onnx,
    file_sha256,
    load_policy_checkpoint,
    load_policy_checkpoint_metadata,
)
from embodied_skill_composer.construction.public_demo_provenance import (
    PublicDemoExportError,
    SourceIdentity,
    export_public_demo_bundle,
    verify_public_demo_export,
)
from embodied_skill_composer.construction.release_identity import (
    RepositoryReleaseIdentity,
    repository_release_identity,
    validate_release_identity,
)


ReleaseAssetRole = Literal[
    "public_demo",
    "research_evidence",
    "coppelia_evidence",
    "selected_policies",
]
RELEASE_ASSET_SCHEMA_VERSION: Literal[
    "construction-intelligence-release-assets-v1"
] = "construction-intelligence-release-assets-v1"
POLICY_ASSET_SCHEMA_VERSION: Literal[
    "construction-intelligence-selected-policies-v1"
] = "construction-intelligence-selected-policies-v1"
RELEASE_IDENTITY_FILE = "construction-intelligence-v1-release-assets.json"
POLICY_INDEX_FILE = "policy-assets.json"
_GENERATED_AT: Literal["1970-01-01T00:00:00Z"] = "1970-01-01T00:00:00Z"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_RUN_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o100644 << 16
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_MEMBER_UNCOMPRESSED_BYTES = 256 * 1024**2
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1536 * 1024**2
_MAX_ARCHIVE_COMPRESSED_BYTES = 1024 * 1024**2
_MAX_COMPRESSION_RATIO = 200.0
_MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024**2
_MAX_ARCHIVE_PATH_BYTES = 1024
_MAX_RELEASE_VERIFICATION_DISK_BYTES = 5 * 1024**3
_ASSET_FILES: dict[ReleaseAssetRole, str] = {
    "public_demo": "construction-intelligence-v1-public-demo.zip",
    "research_evidence": "construction-intelligence-v1-research-evidence.zip",
    "coppelia_evidence": "construction-intelligence-v1-coppelia-evidence.zip",
    "selected_policies": "construction-intelligence-v1-selected-policies.zip",
}


@dataclass(frozen=True)
class _ArchiveBudget:
    max_members: int
    max_member_uncompressed_bytes: int
    max_uncompressed_bytes: int
    max_compressed_bytes: int


_DEFAULT_ARCHIVE_BUDGET = _ArchiveBudget(
    max_members=1_000,
    max_member_uncompressed_bytes=64 * 1024**2,
    max_uncompressed_bytes=128 * 1024**2,
    max_compressed_bytes=96 * 1024**2,
)
_ASSET_ARCHIVE_BUDGETS: dict[str, _ArchiveBudget] = {
    _ASSET_FILES["public_demo"]: _ArchiveBudget(
        max_members=10_000,
        max_member_uncompressed_bytes=256 * 1024**2,
        max_uncompressed_bytes=768 * 1024**2,
        max_compressed_bytes=512 * 1024**2,
    ),
    _ASSET_FILES["research_evidence"]: _ArchiveBudget(
        max_members=5_000,
        max_member_uncompressed_bytes=64 * 1024**2,
        max_uncompressed_bytes=128 * 1024**2,
        max_compressed_bytes=96 * 1024**2,
    ),
    _ASSET_FILES["coppelia_evidence"]: _ArchiveBudget(
        max_members=5_000,
        max_member_uncompressed_bytes=256 * 1024**2,
        max_uncompressed_bytes=512 * 1024**2,
        max_compressed_bytes=384 * 1024**2,
    ),
    _ASSET_FILES["selected_policies"]: _ArchiveBudget(
        max_members=1_000,
        max_member_uncompressed_bytes=256 * 1024**2,
        max_uncompressed_bytes=1536 * 1024**2,
        max_compressed_bytes=1024 * 1024**2,
    ),
}
_REQUIRED_ROLES: tuple[ReleaseAssetRole, ...] = tuple(sorted(_ASSET_FILES))
_EXPECTED_FILES = {RELEASE_IDENTITY_FILE, *_ASSET_FILES.values()}
_EXPECTED_POLICY_IDENTITIES = {
    (variant, seed)
    for variant in (
        "mappo_full",
        "ippo_full",
        "mappo_no_bc",
        "mappo_no_failure_curriculum",
    )
    for seed in range(7, 12)
}
_EXPECTED_ALGORITHMS = {
    "mappo_full": "mappo",
    "ippo_full": "ippo",
    "mappo_no_bc": "mappo",
    "mappo_no_failure_curriculum": "mappo",
}


class ReleaseAssetError(ValueError):
    """Raised when release assets cannot be staged or verified safely."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchiveMember(_FrozenModel):
    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_path(self) -> ArchiveMember:
        _safe_archive_name(self.path)
        return self


class PackagedPolicy(_FrozenModel):
    run_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    selected_checkpoint: SelectedCheckpoint
    checkpoint: ArchiveMember
    onnx_artifacts: list[ArchiveMember] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> PackagedPolicy:
        if self.checkpoint.sha256 != self.selected_checkpoint.checkpoint_sha256:
            raise ValueError("packaged checkpoint hash does not match the selection")
        paths = [self.checkpoint.path, *(item.path for item in self.onnx_artifacts)]
        if len(paths) != len(set(paths)):
            raise ValueError("packaged policy paths must be unique")
        expected_prefix = f"policies/{self.run_key}/"
        if any(not path.startswith(expected_prefix) for path in paths):
            raise ValueError("packaged policy files must stay below their run directory")
        if f"{expected_prefix}actor.onnx" not in paths:
            raise ValueError("every packaged policy requires actor.onnx")
        return self


class PolicyAssetIndex(_FrozenModel):
    schema_version: Literal["construction-intelligence-selected-policies-v1"]
    generated_at: Literal["1970-01-01T00:00:00Z"]
    matrix_id: str = Field(min_length=1)
    protocol_digest: str = Field(pattern=_SHA256_PATTERN)
    policy_count: Literal[20]
    policies: list[PackagedPolicy] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_index(self) -> PolicyAssetIndex:
        run_keys = [item.run_key for item in self.policies]
        identities = {
            (
                item.selected_checkpoint.experiment_variant,
                item.selected_checkpoint.training_seed,
            )
            for item in self.policies
        }
        if len(run_keys) != len(set(run_keys)):
            raise ValueError("policy index run keys must be unique")
        if identities != _EXPECTED_POLICY_IDENTITIES:
            raise ValueError("policy index does not cover the frozen 20-run matrix")
        return self


class ReleaseAssetRecord(_FrozenModel):
    role: ReleaseAssetRole
    file: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    entry_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_file(self) -> ReleaseAssetRecord:
        if PurePosixPath(self.file).name != self.file:
            raise ValueError("release asset file must be a basename")
        if self.file != _ASSET_FILES[self.role]:
            raise ValueError("release asset filename does not match its role")
        return self


class ReleaseAssetManifest(_FrozenModel):
    schema_version: Literal["construction-intelligence-release-assets-v1"]
    generated_at: Literal["1970-01-01T00:00:00Z"]
    release_version: str = Field(min_length=1)
    release_tag: str = Field(min_length=1)
    source: SourceIdentity
    public_demo_provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_roles: list[str]
    assets: list[ReleaseAssetRecord] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_manifest(self) -> ReleaseAssetManifest:
        validate_release_identity(self.release_version, self.release_tag)
        if self.source.dirty:
            raise ValueError("release source must be clean")
        if re.fullmatch(_COMMIT_PATTERN, self.source.commit) is None:
            raise ValueError("release source commit must contain exactly 40 hex characters")
        if re.fullmatch(_SHA256_PATTERN, self.source.tree_digest) is None:
            raise ValueError("release source tree digest must be SHA-256")
        if self.required_roles != list(_REQUIRED_ROLES):
            raise ValueError("release manifest required roles are incomplete")
        roles = [item.role for item in self.assets]
        files = [item.file for item in self.assets]
        if sorted(roles) != list(_REQUIRED_ROLES) or len(roles) != len(set(roles)):
            raise ValueError("release manifest must contain every asset role exactly once")
        if set(files) != set(_ASSET_FILES.values()):
            raise ValueError("release manifest asset filenames are incomplete")
        return self


def stage_release_assets(
    output_dir: Path,
    *,
    workspace: Path,
    deterministic_bundle: Path,
    research_bundle: Path,
    simulator_bundle: Path,
    release_version: str,
    release_tag: str,
) -> ReleaseAssetManifest:
    """Stage and independently verify all deterministic v1 release assets."""

    root = workspace.resolve()
    repository_identity = repository_release_identity(root)
    _require_repository_release_identity(
        repository_identity,
        release_version=release_version,
        release_tag=release_tag,
    )
    source = repository_source_identity(root)
    destination = output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-stage-",
            dir=destination.parent,
        )
    )
    work_root = temporary_root / "work"
    assets_root = temporary_root / "assets"
    public_root = work_root / "public-demo"
    policy_root = work_root / "selected-policies"
    assets_root.mkdir(parents=True)
    try:
        export_public_demo_bundle(
            public_root,
            deterministic_bundle=deterministic_bundle,
            research_bundle=research_bundle,
            simulator_bundle=simulator_bundle,
            source=source,
            channel="release",
            release_version=release_version,
            release_tag=release_tag,
        )
        verify_public_demo_export(
            public_root,
            expected_channel="release",
            expected_source=source,
            expected_release_version=release_version,
            expected_release_tag=release_tag,
        )
        policy_index = _stage_selected_policies(
            public_root / "evidence" / "research" / "selections.json",
            policy_root,
        )
        _write_json(
            policy_root / POLICY_INDEX_FILE,
            policy_index.model_dump(mode="json"),
        )

        archive_roots = {
            "public_demo": (public_root, None),
            "research_evidence": (
                public_root / "evidence" / "research",
                PurePosixPath("evidence/research"),
            ),
            "coppelia_evidence": (
                public_root / "evidence" / "coppelia",
                PurePosixPath("evidence/coppelia"),
            ),
            "selected_policies": (policy_root, None),
        }
        records: list[ReleaseAssetRecord] = []
        for role in _REQUIRED_ROLES:
            source_root, prefix = archive_roots[role]
            archive_path = assets_root / _ASSET_FILES[role]
            entry_count = write_deterministic_zip(
                archive_path,
                _directory_archive_entries(source_root, prefix=prefix),
            )
            records.append(
                ReleaseAssetRecord(
                    role=role,
                    file=archive_path.name,
                    bytes=archive_path.stat().st_size,
                    sha256=file_sha256(archive_path),
                    entry_count=entry_count,
                )
            )
        provenance_sha = file_sha256(public_root / "provenance.json")
        manifest = ReleaseAssetManifest(
            schema_version=RELEASE_ASSET_SCHEMA_VERSION,
            generated_at=_GENERATED_AT,
            release_version=release_version,
            release_tag=release_tag,
            source=source,
            public_demo_provenance_sha256=provenance_sha,
            required_roles=list(_REQUIRED_ROLES),
            assets=sorted(records, key=lambda item: item.role),
        )
        _write_json(
            assets_root / RELEASE_IDENTITY_FILE,
            manifest.model_dump(mode="json"),
        )
        verified = verify_release_assets(
            assets_root,
            workspace=root,
            expected_release_version=release_version,
            expected_release_tag=release_tag,
            expected_source=source,
        )
        if verified != manifest:
            raise ReleaseAssetError("staged release manifest changed during verification")
        _replace_directory(assets_root, destination)
        return manifest
    except (OSError, ValueError, PublicDemoExportError) as exc:
        if isinstance(exc, ReleaseAssetError):
            raise
        raise ReleaseAssetError(f"release asset staging failed: {exc}") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def verify_release_assets(
    asset_dir: Path,
    *,
    workspace: Path,
    expected_release_version: str | None = None,
    expected_release_tag: str | None = None,
    expected_source: SourceIdentity | None = None,
    require_tag_at_head: bool = False,
) -> ReleaseAssetManifest:
    """Fail closed unless a release directory is exact, safe, and self-consistent."""

    root = asset_dir.resolve()
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed_files != _EXPECTED_FILES:
        raise ReleaseAssetError(
            "release asset inventory mismatch; "
            f"missing={sorted(_EXPECTED_FILES - observed_files)}, "
            f"extra={sorted(observed_files - _EXPECTED_FILES)}"
        )
    try:
        manifest = ReleaseAssetManifest.model_validate_json(
            (root / RELEASE_IDENTITY_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ReleaseAssetError(f"release identity manifest is invalid: {exc}") from exc
    repository_identity = repository_release_identity(workspace.resolve())
    _require_repository_release_identity(
        repository_identity,
        release_version=manifest.release_version,
        release_tag=manifest.release_tag,
    )
    if (
        expected_release_version is not None
        and manifest.release_version != expected_release_version
    ):
        raise ReleaseAssetError("release asset version does not match expectation")
    if expected_release_tag is not None and manifest.release_tag != expected_release_tag:
        raise ReleaseAssetError("release asset tag does not match expectation")
    actual_source = repository_source_identity(workspace.resolve())
    if manifest.source != actual_source:
        raise ReleaseAssetError("release asset source does not match the clean checkout")
    if expected_source is not None and manifest.source != expected_source:
        raise ReleaseAssetError("release asset source does not match expectation")
    if require_tag_at_head:
        require_tag_points_at_head(workspace.resolve(), manifest.release_tag)

    records = {item.role: item for item in manifest.assets}
    for role in _REQUIRED_ROLES:
        record = records[role]
        path = root / record.file
        if path.stat().st_size != record.bytes or file_sha256(path) != record.sha256:
            raise ReleaseAssetError(f"release asset integrity check failed: {record.file}")
        entry_count = verify_deterministic_zip(path)
        if entry_count != record.entry_count:
            raise ReleaseAssetError(
                f"release asset entry count changed: {record.file}"
            )

    public_zip = root / _ASSET_FILES["public_demo"]
    research_zip = root / _ASSET_FILES["research_evidence"]
    coppelia_zip = root / _ASSET_FILES["coppelia_evidence"]
    _require_archive_subset(
        public_zip,
        research_zip,
        prefix="evidence/research/",
    )
    _require_archive_subset(
        public_zip,
        coppelia_zip,
        prefix="evidence/coppelia/",
    )
    temporary_root = Path(tempfile.mkdtemp(prefix="construction-release-verify-"))
    try:
        public_root = temporary_root / "public"
        extract_verified_zip(public_zip, public_root)
        try:
            verify_public_demo_export(
                public_root,
                expected_channel="release",
                expected_source=manifest.source,
                expected_release_version=manifest.release_version,
                expected_release_tag=manifest.release_tag,
            )
        except (OSError, ValueError, PublicDemoExportError) as exc:
            raise ReleaseAssetError(
                f"public demo release verification failed: {exc}"
            ) from exc
        if (
            file_sha256(public_root / "provenance.json")
            != manifest.public_demo_provenance_sha256
        ):
            raise ReleaseAssetError("public demo provenance identity does not match")

        research_root = temporary_root / "research"
        policy_root = temporary_root / "policies"
        extract_verified_zip(research_zip, research_root)
        extract_verified_zip(
            root / _ASSET_FILES["selected_policies"],
            policy_root,
        )
        _verify_packaged_policies(
            research_root / "evidence" / "research" / "selections.json",
            policy_root,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return manifest


def repository_source_identity(workspace: Path) -> SourceIdentity:
    """Return a clean 40-character Git commit and SHA-256 tree identity."""

    root = workspace.resolve()
    commit = _git_bytes(root, "rev-parse", "--verify", "HEAD").decode().strip()
    if re.fullmatch(_COMMIT_PATTERN, commit) is None:
        raise ReleaseAssetError(
            "release packaging requires a full 40-character Git commit"
        )
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status.strip():
        raise ReleaseAssetError(
            "release packaging requires a completely clean Git worktree"
        )
    tree_listing = _git_bytes(root, "ls-tree", "-r", "--full-tree", "HEAD")
    if not tree_listing:
        raise ReleaseAssetError("release Git tree listing is empty")
    return SourceIdentity(
        commit=commit,
        dirty=False,
        tree_digest=hashlib.sha256(tree_listing).hexdigest(),
    )


def require_tag_points_at_head(workspace: Path, tag: str) -> None:
    """Require the exact annotated or lightweight tag target to equal HEAD."""

    root = workspace.resolve()
    head = _git_bytes(root, "rev-parse", "--verify", "HEAD").decode().strip()
    try:
        tag_commit = (
            _git_bytes(root, "rev-list", "-n", "1", f"refs/tags/{tag}")
            .decode()
            .strip()
        )
    except ReleaseAssetError as exc:
        raise ReleaseAssetError(f"release tag does not exist: {tag}") from exc
    if not tag_commit or tag_commit != head:
        raise ReleaseAssetError(
            f"release tag {tag!r} does not point at checked-out HEAD"
        )


def write_deterministic_zip(
    destination: Path,
    entries: Mapping[str, Path],
) -> int:
    """Write sorted files with fixed ZIP metadata and deterministic compression."""

    if not entries:
        raise ReleaseAssetError(f"cannot create an empty release archive: {destination}")
    normalized: list[tuple[str, Path]] = []
    for name, source in entries.items():
        safe_name = _safe_archive_name(name)
        resolved = source.resolve()
        if source.is_symlink() or not resolved.is_file():
            raise ReleaseAssetError(f"release archive source is not a regular file: {source}")
        normalized.append((safe_name, resolved))
    names = [name for name, _ in normalized]
    if len(names) != len(set(names)):
        raise ReleaseAssetError("release archive contains duplicate member names")
    budget = _archive_budget(destination.name)
    _validate_source_resource_limits(normalized, destination.name, budget)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, source in sorted(normalized):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = _ZIP_EXTERNAL_ATTR
            info.extra = b""
            info.comment = b""
            info.file_size = source.stat().st_size
            with source.open("rb") as source_handle, archive.open(
                info,
                mode="w",
                force_zip64=False,
            ) as archive_handle:
                shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)
    verify_deterministic_zip(destination)
    return len(normalized)


def verify_deterministic_zip(path: Path) -> int:
    """Reject unsafe names, duplicate members, changed metadata, or corrupt bytes."""

    try:
        budget = _archive_budget(path.name)
        compressed_bytes = path.stat().st_size
        if compressed_bytes > budget.max_compressed_bytes:
            raise ReleaseAssetError(
                f"release archive exceeds the compressed-size limit: {path.name}"
            )
        _preflight_zip_central_directory(path, budget)
        with zipfile.ZipFile(path, mode="r") as archive:
            if archive.comment:
                raise ReleaseAssetError(f"release archive has a mutable comment: {path.name}")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or names != sorted(names) or len(names) != len(set(names)):
                raise ReleaseAssetError(
                    f"release archive entries must be non-empty, unique, and sorted: {path.name}"
                )
            _validate_zip_resource_limits(infos, path.name, budget)
            for info in infos:
                _safe_archive_name(info.filename)
                if info.is_dir():
                    raise ReleaseAssetError(
                        f"release archives may contain files only: {info.filename}"
                    )
                if (
                    info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.create_system != 3
                    or info.external_attr != _ZIP_EXTERNAL_ATTR
                    or info.extra
                    or info.comment
                    or info.flag_bits & 0x1
                ):
                    raise ReleaseAssetError(
                        f"release archive metadata is not canonical: {info.filename}"
                    )
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ReleaseAssetError(
                    f"release archive contains corrupt data: {corrupt}"
                )
            return len(infos)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseAssetError(f"release archive is unreadable: {path}: {exc}") from exc


def _validate_zip_resource_limits(
    infos: list[zipfile.ZipInfo],
    archive_name: str,
    budget: _ArchiveBudget,
) -> None:
    if len(infos) > budget.max_members:
        raise ReleaseAssetError(
            f"release archive exceeds the member-count limit: {archive_name}"
        )
    total_uncompressed = 0
    for info in infos:
        if info.file_size > budget.max_member_uncompressed_bytes:
            raise ReleaseAssetError(
                "release archive member exceeds the uncompressed-size limit: "
                f"{info.filename}"
            )
        total_uncompressed += info.file_size
        if total_uncompressed > budget.max_uncompressed_bytes:
            raise ReleaseAssetError(
                f"release archive exceeds the total uncompressed-size limit: {archive_name}"
            )
        if info.file_size:
            if info.compress_size <= 0:
                raise ReleaseAssetError(
                    f"release archive member has an invalid compressed size: {info.filename}"
                )
            ratio = info.file_size / info.compress_size
            if ratio > _MAX_COMPRESSION_RATIO:
                raise ReleaseAssetError(
                    f"release archive member exceeds the compression-ratio limit: {info.filename}"
                )


def _archive_budget(archive_name: str) -> _ArchiveBudget:
    configured = _ASSET_ARCHIVE_BUDGETS.get(
        archive_name,
        _DEFAULT_ARCHIVE_BUDGET,
    )
    return _ArchiveBudget(
        max_members=min(configured.max_members, _MAX_ARCHIVE_MEMBERS),
        max_member_uncompressed_bytes=min(
            configured.max_member_uncompressed_bytes,
            _MAX_MEMBER_UNCOMPRESSED_BYTES,
        ),
        max_uncompressed_bytes=min(
            configured.max_uncompressed_bytes,
            _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
        ),
        max_compressed_bytes=min(
            configured.max_compressed_bytes,
            _MAX_ARCHIVE_COMPRESSED_BYTES,
        ),
    )


def _validate_source_resource_limits(
    entries: list[tuple[str, Path]],
    archive_name: str,
    budget: _ArchiveBudget,
) -> None:
    if len(entries) > budget.max_members:
        raise ReleaseAssetError(
            f"release archive exceeds the member-count limit: {archive_name}"
        )
    total_bytes = 0
    for name, source in entries:
        path_bytes = len(name.encode("utf-8"))
        if path_bytes > _MAX_ARCHIVE_PATH_BYTES:
            raise ReleaseAssetError(f"release archive path is too long: {name}")
        size = source.stat().st_size
        if size > budget.max_member_uncompressed_bytes:
            raise ReleaseAssetError(
                f"release archive member exceeds the uncompressed-size limit: {name}"
            )
        total_bytes += size
        if total_bytes > budget.max_uncompressed_bytes:
            raise ReleaseAssetError(
                f"release archive exceeds the total uncompressed-size limit: {archive_name}"
            )


def _preflight_zip_central_directory(
    path: Path,
    budget: _ArchiveBudget,
) -> None:
    """Bound central-directory parsing before ZipFile can allocate per-entry objects."""

    file_size = path.stat().st_size
    eocd_size = 22
    if file_size < eocd_size:
        raise ReleaseAssetError(f"release archive has no complete EOCD: {path.name}")
    tail_size = min(file_size, eocd_size + 65_535)
    with path.open("rb") as handle:
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
        relative_offset = tail.rfind(b"PK\x05\x06")
        if relative_offset < 0 or len(tail) - relative_offset < eocd_size:
            raise ReleaseAssetError(f"release archive has no complete EOCD: {path.name}")
        (
            _signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack_from("<4s4H2IH", tail, relative_offset)
        eocd_offset = file_size - tail_size + relative_offset
        if comment_length != 0 or eocd_offset + eocd_size != file_size:
            raise ReleaseAssetError(
                f"release archive EOCD is not canonical: {path.name}"
            )
        if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
            raise ReleaseAssetError(
                f"release archive cannot span disks: {path.name}"
            )
        if (
            total_entries == 0xFFFF
            or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF
        ):
            raise ReleaseAssetError(
                f"release archive cannot use ZIP64 metadata: {path.name}"
            )
        if total_entries == 0 or total_entries > budget.max_members:
            raise ReleaseAssetError(
                f"release archive exceeds the member-count limit: {path.name}"
            )
        if central_size > _MAX_CENTRAL_DIRECTORY_BYTES:
            raise ReleaseAssetError(
                f"release archive central directory is too large: {path.name}"
            )
        central_end = central_offset + central_size
        if central_offset > eocd_offset or central_end != eocd_offset:
            raise ReleaseAssetError(
                f"release archive central directory is invalid: {path.name}"
            )
        handle.seek(central_offset)
        position = central_offset
        for _ in range(total_entries):
            fixed_header = handle.read(46)
            if len(fixed_header) != 46 or fixed_header[:4] != b"PK\x01\x02":
                raise ReleaseAssetError(
                    f"release archive central directory is invalid: {path.name}"
                )
            name_length, extra_length, member_comment_length = struct.unpack_from(
                "<3H",
                fixed_header,
                28,
            )
            if (
                name_length == 0
                or name_length > _MAX_ARCHIVE_PATH_BYTES
                or extra_length != 0
                or member_comment_length != 0
            ):
                raise ReleaseAssetError(
                    f"release archive central metadata is not canonical: {path.name}"
                )
            variable_length = name_length + extra_length + member_comment_length
            position += 46 + variable_length
            if position > central_end:
                raise ReleaseAssetError(
                    f"release archive central directory is invalid: {path.name}"
                )
            handle.seek(variable_length, os.SEEK_CUR)
        if position != central_end:
            raise ReleaseAssetError(
                f"release archive central directory entry count changed: {path.name}"
            )


def extract_verified_zip(path: Path, destination: Path) -> None:
    """Extract only after the complete archive has passed path and metadata checks."""

    verify_deterministic_zip(path)
    destination.mkdir(parents=True, exist_ok=False)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(path, mode="r") as archive:
        for info in archive.infolist():
            relative = Path(*PurePosixPath(info.filename).parts)
            target = (resolved_destination / relative).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ReleaseAssetError(
                    f"release archive path escapes extraction root: {info.filename}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _stage_selected_policies(
    selection_path: Path,
    destination: Path,
) -> PolicyAssetIndex:
    matrix_id, protocol_digest, selections = _load_selected_checkpoints(selection_path)
    destination.mkdir(parents=True, exist_ok=False)
    packaged: list[PackagedPolicy] = []
    for run_key, selected in selections:
        checkpoint_path = Path(selected.checkpoint_path).resolve()
        _validate_selected_checkpoint_file(checkpoint_path, selected)
        bundle = load_policy_checkpoint(checkpoint_path, device="cpu")
        expected_algorithm = _EXPECTED_ALGORITHMS[selected.experiment_variant]
        if bundle.algorithm != expected_algorithm:
            raise ReleaseAssetError(
                f"selected policy algorithm mismatch for {run_key}: "
                f"{bundle.algorithm} != {expected_algorithm}"
            )
        run_root = destination / "policies" / run_key
        run_root.mkdir(parents=True, exist_ok=False)
        packaged_checkpoint = run_root / "checkpoint.pt"
        shutil.copyfile(checkpoint_path, packaged_checkpoint)
        exported_onnx = export_actor_onnx(
            bundle.actor_model,
            run_root / "actor.onnx",
            device="cpu",
        )
        if exported_onnx.resolve() != (run_root / "actor.onnx").resolve():
            raise ReleaseAssetError(
                f"ONNX exporter returned an unexpected path for {run_key}"
            )
        onnx_paths = sorted(
            path
            for path in run_root.rglob("*")
            if path.is_file() and path != packaged_checkpoint
        )
        if run_root / "actor.onnx" not in onnx_paths:
            raise ReleaseAssetError(f"ONNX export is missing for {run_key}")
        _validate_onnx_model(run_root / "actor.onnx")
        packaged.append(
            PackagedPolicy(
                run_key=run_key,
                selected_checkpoint=selected,
                checkpoint=_archive_member(packaged_checkpoint, destination),
                onnx_artifacts=[
                    _archive_member(path, destination) for path in onnx_paths
                ],
            )
        )
    return PolicyAssetIndex(
        schema_version=POLICY_ASSET_SCHEMA_VERSION,
        generated_at=_GENERATED_AT,
        matrix_id=matrix_id,
        protocol_digest=protocol_digest,
        policy_count=20,
        policies=sorted(packaged, key=lambda item: item.run_key),
    )


def _verify_packaged_policies(
    selection_path: Path,
    policy_root: Path,
) -> None:
    matrix_id, protocol_digest, selections = _load_selected_checkpoints(selection_path)
    try:
        index = PolicyAssetIndex.model_validate_json(
            (policy_root / POLICY_INDEX_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ReleaseAssetError(f"selected-policy index is invalid: {exc}") from exc
    if index.matrix_id != matrix_id or index.protocol_digest != protocol_digest:
        raise ReleaseAssetError("selected-policy index research identity mismatch")
    expected = dict(selections)
    observed_paths = {
        path.relative_to(policy_root).as_posix()
        for path in policy_root.rglob("*")
        if path.is_file()
    }
    declared_paths = {POLICY_INDEX_FILE}
    for policy in index.policies:
        selected = expected.get(policy.run_key)
        if selected is None or policy.selected_checkpoint != selected:
            raise ReleaseAssetError(
                f"packaged policy does not match frozen selection: {policy.run_key}"
            )
        members = [policy.checkpoint, *policy.onnx_artifacts]
        for member in members:
            path = policy_root / Path(*PurePosixPath(member.path).parts)
            if (
                not path.is_file()
                or path.stat().st_size != member.bytes
                or file_sha256(path) != member.sha256
            ):
                raise ReleaseAssetError(
                    f"packaged policy member integrity failed: {member.path}"
                )
            declared_paths.add(member.path)
        _validate_selected_checkpoint_file(
            policy_root / Path(*PurePosixPath(policy.checkpoint.path).parts),
            selected,
        )
        bundle = load_policy_checkpoint(
            policy_root / Path(*PurePosixPath(policy.checkpoint.path).parts),
            device="cpu",
        )
        if bundle.algorithm != _EXPECTED_ALGORITHMS[selected.experiment_variant]:
            raise ReleaseAssetError(
                f"packaged policy algorithm mismatch: {policy.run_key}"
            )
        _validate_onnx_model(policy_root / "policies" / policy.run_key / "actor.onnx")
    if set(expected) != {item.run_key for item in index.policies}:
        raise ReleaseAssetError("selected-policy archive does not cover all 20 selections")
    if observed_paths != declared_paths:
        raise ReleaseAssetError(
            "selected-policy archive inventory mismatch; "
            f"missing={sorted(declared_paths - observed_paths)}, "
            f"extra={sorted(observed_paths - declared_paths)}"
        )


def _load_selected_checkpoints(
    path: Path,
) -> tuple[str, str, list[tuple[str, SelectedCheckpoint]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseAssetError(f"selection evidence is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseAssetError("selection evidence must be a JSON object")
    matrix_id = payload.get("matrix_id")
    protocol_digest = payload.get("protocol_digest")
    raw_selections = payload.get("selections")
    if (
        not isinstance(matrix_id, str)
        or not matrix_id
        or not isinstance(protocol_digest, str)
        or re.fullmatch(_SHA256_PATTERN, protocol_digest) is None
        or not isinstance(raw_selections, list)
        or not all(isinstance(item, dict) for item in raw_selections)
    ):
        raise ReleaseAssetError("selection evidence identity or records are malformed")
    selections: list[tuple[str, SelectedCheckpoint]] = []
    for item in cast(list[dict[str, object]], raw_selections):
        run_key = item.get("run_key")
        selected_payload = item.get("selected", item.get("selection"))
        if (
            not isinstance(run_key, str)
            or _RUN_KEY_PATTERN.fullmatch(run_key) is None
            or not isinstance(selected_payload, dict)
        ):
            raise ReleaseAssetError("selection record is missing a safe run key or policy")
        try:
            selected = SelectedCheckpoint.model_validate(selected_payload)
        except ValueError as exc:
            raise ReleaseAssetError(
                f"selected checkpoint is malformed for {run_key}: {exc}"
            ) from exc
        selections.append((run_key, selected))
    identities = {
        (selected.experiment_variant, selected.training_seed)
        for _, selected in selections
    }
    if (
        len(selections) != 20
        or len({run_key for run_key, _ in selections}) != 20
        or identities != _EXPECTED_POLICY_IDENTITIES
    ):
        raise ReleaseAssetError(
            "selection evidence must contain exactly the frozen 20 policy identities"
        )
    return matrix_id, protocol_digest, sorted(selections)


def _validate_selected_checkpoint_file(
    path: Path,
    selected: SelectedCheckpoint,
) -> None:
    if not path.is_file():
        raise ReleaseAssetError(f"selected checkpoint is missing: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != selected.checkpoint_sha256:
        raise ReleaseAssetError(
            f"selected checkpoint SHA-256 mismatch: {path}"
        )
    try:
        metadata = load_policy_checkpoint_metadata(path, device="cpu")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ReleaseAssetError(
            f"selected checkpoint metadata is invalid for {path}: {exc}"
        ) from exc
    expected: dict[str, object] = {
        "experiment_id": selected.experiment_id,
        "experiment_variant": selected.experiment_variant,
        "training_seed": selected.training_seed,
        "transition_count": selected.transition_count,
        "configuration_digest": selected.configuration_digest,
        "source_commit": selected.source_commit,
        "checkpoint_lineage": selected.checkpoint_lineage,
        "resume_provenance": selected.resume_provenance,
    }
    mismatches = [
        name for name, value in expected.items() if metadata.get(name) != value
    ]
    fraction = metadata.get("checkpoint_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isclose(
            float(fraction),
            selected.checkpoint_fraction,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        mismatches.append("checkpoint_fraction")
    if mismatches:
        raise ReleaseAssetError(
            "selected checkpoint metadata does not match its frozen selection "
            f"for {path}: {sorted(set(mismatches))}"
        )


def _validate_onnx_model(path: Path) -> None:
    try:
        import onnx
        from onnx import external_data_helper

        model = onnx.load_model(path, load_external_data=False)
        for tensor in external_data_helper._get_all_tensors(model):
            if not external_data_helper.uses_external_data(tensor):
                continue
            locations = [
                item.value for item in tensor.external_data if item.key == "location"
            ]
            if len(locations) != 1:
                raise ReleaseAssetError(
                    f"ONNX tensor has invalid external-data locations: {path}"
                )
            relative = _safe_archive_name(locations[0])
            external_path = (
                path.parent / Path(*PurePosixPath(relative).parts)
            ).resolve()
            if (
                not external_path.is_relative_to(path.parent.resolve())
                or not external_path.is_file()
            ):
                raise ReleaseAssetError(
                    f"ONNX external data is missing or unsafe: {relative}"
                )
        external_data_helper.load_external_data_for_model(
            model,
            str(path.parent.resolve()),
        )
        onnx.checker.check_model(model)
    except ReleaseAssetError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize ONNX/protobuf failures.
        raise ReleaseAssetError(
            f"ONNX model or its external data is invalid: {path}: {exc}"
        ) from exc


def _directory_archive_entries(
    root: Path,
    *,
    prefix: PurePosixPath | None,
) -> dict[str, Path]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ReleaseAssetError(f"release archive source directory is missing: {root}")
    entries: dict[str, Path] = {}
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ReleaseAssetError(f"release archives cannot contain symlinks: {path}")
        relative = PurePosixPath(path.relative_to(resolved_root).as_posix())
        archive_name = (prefix / relative).as_posix() if prefix else relative.as_posix()
        entries[archive_name] = path
    return entries


def _archive_member(path: Path, root: Path) -> ArchiveMember:
    return ArchiveMember(
        path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def _require_archive_subset(
    parent_path: Path,
    subset_path: Path,
    *,
    prefix: str,
) -> None:
    parent = _zip_member_identities(parent_path)
    subset = _zip_member_identities(subset_path)
    expected = {
        name: identity
        for name, identity in parent.items()
        if name.startswith(prefix)
    }
    if subset != expected:
        raise ReleaseAssetError(
            f"release evidence archive does not match public demo subset: {prefix}"
        )


def _zip_member_identities(path: Path) -> dict[str, tuple[int, str]]:
    verify_deterministic_zip(path)
    identities: dict[str, tuple[int, str]] = {}
    with zipfile.ZipFile(path, mode="r") as archive:
        for info in archive.infolist():
            digest = hashlib.sha256()
            with archive.open(info, "r") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            identities[info.filename] = (info.file_size, digest.hexdigest())
    return identities


def _safe_archive_name(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ReleaseAssetError(f"unsafe release archive path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
        or path.as_posix() != value
    ):
        raise ReleaseAssetError(f"unsafe release archive path: {value!r}")
    return value


def _require_repository_release_identity(
    identity: RepositoryReleaseIdentity,
    *,
    release_version: str,
    release_tag: str,
) -> None:
    validate_release_identity(release_version, release_tag)
    if identity.version != release_version or identity.tag != release_tag:
        raise ReleaseAssetError(
            "release asset identity does not match repository package versions"
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(staging, destination)
    except Exception:
        if moved_existing and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _git_bytes(workspace: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        stderr = completed.stderr.decode(errors="replace").strip()
        raise ReleaseAssetError(
            f"Git command failed ({' '.join(args)}): {stderr or completed.returncode}"
        )
    return completed.stdout
