"""Bounded, byte-preserving storage for complete live simulator evidence."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

COPPELIA_ARCHIVE_INDEX_FILE = "construction-intelligence-v1-coppelia-evidence.json"
COPPELIA_PART_PREFIX = "construction-intelligence-v1-coppelia-evidence-part-"
MAX_CHUNK_BYTES = 128 * 1024**2
MAX_CANONICAL_FILE_BYTES = 8 * 1024**3
MAX_CANONICAL_TOTAL_BYTES = 16 * 1024**3
MAX_SCENARIO_BYTES = 8 * 1024**3
MAX_INDEX_BYTES = 4 * 1024**2
MAX_PARTS = 256
_HASH = r"^[0-9a-f]{64}$"
_NATIVE_FILES = (
    "manifest.json",
    "scenario.json",
    "planned_jobs.json",
    "planned_vs_measured_replay.json",
    "wheel_commands.jsonl",
    "measured_telemetry.jsonl",
    "trace.json",
    "metrics.json",
    "report.md",
    "construction_intelligence.ttt",
    "runtime_attestation.json",
)
CANONICAL_PATHS = frozenset(
    [
        f"evidence/coppelia/{scenario}/{name}"
        for scenario in ("nominal", "recovery")
        for name in _NATIVE_FILES
    ]
    + [f"evidence/coppelia/visualizations/{scenario}.mp4" for scenario in ("nominal", "recovery")]
)


class CoppeliaArchiveError(ValueError):
    """Evidence storage is incomplete, corrupt, unsafe, or exceeds its budget."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArchiveSource(_Model):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False] = False
    tree_digest: str = Field(pattern=_HASH)


class ArchiveChunk(_Model):
    part: str
    member: str
    bytes: int = Field(gt=0, le=MAX_CHUNK_BYTES)
    sha256: str = Field(pattern=_HASH)


class ArchivedFile(_Model):
    path: str
    bytes: int = Field(gt=0, le=MAX_CANONICAL_FILE_BYTES)
    sha256: str = Field(pattern=_HASH)
    chunks: list[ArchiveChunk] = Field(min_length=1, max_length=512)


class ArchivePart(_Model):
    file: str
    bytes: int = Field(gt=0, le=384 * 1024**2)
    sha256: str = Field(pattern=_HASH)
    entry_count: int = Field(ge=1, le=2)


def part_filename(index: int) -> str:
    return f"{COPPELIA_PART_PREFIX}{index:06d}.zip"


class CoppeliaArchiveIndex(_Model):
    schema_version: Literal["construction-intelligence-coppelia-archive-v1"] = (
        "construction-intelligence-coppelia-archive-v1"
    )
    generated_at: Literal["1970-01-01T00:00:00Z"] = "1970-01-01T00:00:00Z"
    source: ArchiveSource
    descriptor_sha256: str = Field(pattern=_HASH)
    chunk_bytes: int = Field(gt=0, le=MAX_CHUNK_BYTES)
    files: list[ArchivedFile] = Field(min_length=24, max_length=24)
    parts: list[ArchivePart] = Field(min_length=1, max_length=MAX_PARTS)

    @model_validator(mode="after")
    def validate_inventory(self) -> CoppeliaArchiveIndex:
        paths = [item.path for item in self.files]
        if paths != sorted(CANONICAL_PATHS):
            raise ValueError("canonical Coppelia file inventory must be exact and sorted")
        if sum(item.bytes for item in self.files) > MAX_CANONICAL_TOTAL_BYTES:
            raise ValueError("canonical Coppelia evidence exceeds its total byte budget")
        for scenario in ("nominal", "recovery"):
            if self.scenario_bytes(scenario) > MAX_SCENARIO_BYTES:
                raise ValueError("canonical Coppelia scenario exceeds its byte budget")
        part_names = [item.file for item in self.parts]
        if part_names != [part_filename(i + 1) for i in range(len(self.parts))]:
            raise ValueError("Coppelia archive part inventory must be contiguous and sorted")
        counts: Counter[str] = Counter()
        ordinal = 0
        for item in self.files:
            if len(item.chunks) != math.ceil(item.bytes / self.chunk_bytes):
                raise ValueError("canonical file chunk count differs from its size")
            for offset, chunk in enumerate(item.chunks):
                expected_bytes = min(self.chunk_bytes, item.bytes - offset * self.chunk_bytes)
                if (
                    chunk.bytes != expected_bytes
                    or chunk.member != f"chunks/{item.path}/{offset + 1:06d}.bin"
                    or chunk.part != part_filename(ordinal // 2 + 1)
                ):
                    raise ValueError("Coppelia chunk sequence, path, or size is not canonical")
                counts[chunk.part] += 1
                ordinal += 1
        if counts != Counter({part.file: part.entry_count for part in self.parts}):
            raise ValueError("Coppelia part members differ from the logical index")
        return self

    def scenario_bytes(self, scenario: Literal["nominal", "recovery"]) -> int:
        prefix = f"evidence/coppelia/{scenario}/"
        video = f"evidence/coppelia/visualizations/{scenario}.mp4"
        return sum(
            item.bytes for item in self.files if item.path.startswith(prefix) or item.path == video
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coppelia_archive_index(path: Path) -> CoppeliaArchiveIndex:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INDEX_BYTES:
        raise CoppeliaArchiveError("Coppelia archive index is missing, unsafe, or oversized")
    try:
        return CoppeliaArchiveIndex.model_validate_json(path.read_bytes())
    except ValueError as exc:
        raise CoppeliaArchiveError(f"invalid Coppelia archive index: {exc}") from exc


def stage_coppelia_archive(
    destination: Path,
    files: Mapping[str, Path],
    *,
    source: ArchiveSource,
    descriptor_sha256: str,
    expected_sha256: Mapping[str, str],
    verify_zip: Callable[[Path], int],
    chunk_bytes: int = MAX_CHUNK_BYTES,
) -> CoppeliaArchiveIndex:
    """Stream fixed-size slices directly into guarded deterministic ZIP shards."""
    if set(files) != CANONICAL_PATHS or set(expected_sha256) != CANONICAL_PATHS:
        raise CoppeliaArchiveError("canonical Coppelia file inventory is incomplete")
    if not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise CoppeliaArchiveError("Coppelia chunk size is outside its supported bound")
    sizes = {name: path.stat().st_size for name, path in files.items()}
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise CoppeliaArchiveError("canonical Coppelia sources must be regular files")
    if any(not 0 < size <= MAX_CANONICAL_FILE_BYTES for size in sizes.values()):
        raise CoppeliaArchiveError("canonical Coppelia file exceeds its byte budget")
    if sum(sizes.values()) > MAX_CANONICAL_TOTAL_BYTES:
        raise CoppeliaArchiveError("canonical Coppelia evidence exceeds its total byte budget")
    for scenario in ("nominal", "recovery"):
        scenario_size = sum(
            size
            for name, size in sizes.items()
            if name.startswith(f"evidence/coppelia/{scenario}/")
            or name == f"evidence/coppelia/visualizations/{scenario}.mp4"
        )
        if scenario_size > MAX_SCENARIO_BYTES:
            raise CoppeliaArchiveError("canonical Coppelia scenario exceeds its byte budget")
    slice_count = sum(math.ceil(size / chunk_bytes) for size in sizes.values())
    if slice_count > MAX_PARTS * 2:
        raise CoppeliaArchiveError("Coppelia evidence exceeds its part-count budget")
    slices = [
        (name, offset, min(chunk_bytes, sizes[name] - offset))
        for name in sorted(files)
        for offset in range(0, sizes[name], chunk_bytes)
    ]
    destination.mkdir(parents=True, exist_ok=True)
    digests = {name: hashlib.sha256() for name in files}
    chunks: dict[str, list[ArchiveChunk]] = {name: [] for name in files}
    parts: list[ArchivePart] = []
    created: list[Path] = []
    try:
        for start in range(0, len(slices), 2):
            # Only the next bounded shard is written at once. Reserve twice its
            # raw bytes for conservative DEFLATE overhead plus the final index.
            # Existing shards are already reflected in the current free space.
            required_free = sum(size for _name, _offset, size in slices[start : start + 2]) * 2
            if shutil.disk_usage(destination).free < required_free + MAX_INDEX_BYTES:
                raise CoppeliaArchiveError("insufficient free disk for bounded Coppelia staging")
            filename = part_filename(start // 2 + 1)
            path = destination / filename
            if path.exists() or path.is_symlink():
                raise CoppeliaArchiveError(f"Coppelia archive destination exists: {filename}")
            part_output = path.open("xb")
            created.append(path)
            with (
                part_output,
                zipfile.ZipFile(
                    part_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
                ) as archive,
            ):
                for name, offset, size in slices[start : start + 2]:
                    member = f"chunks/{name}/{offset // chunk_bytes + 1:06d}.bin"
                    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.file_size = size
                    digest = hashlib.sha256()
                    with (
                        files[name].open("rb") as input_handle,
                        archive.open(info, "w", force_zip64=False) as output,
                    ):
                        input_handle.seek(offset)
                        remaining = size
                        while remaining:
                            data = input_handle.read(min(remaining, 1024 * 1024))
                            if not data:
                                raise CoppeliaArchiveError("canonical source shrank during staging")
                            remaining -= len(data)
                            digest.update(data)
                            digests[name].update(data)
                            output.write(data)
                    chunks[name].append(
                        ArchiveChunk(
                            part=filename, member=member, bytes=size, sha256=digest.hexdigest()
                        )
                    )
            count = verify_zip(path)
            parts.append(
                ArchivePart(
                    file=filename,
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                    entry_count=count,
                )
            )
        originals = []
        for name in sorted(files):
            file_digest = digests[name].hexdigest()
            if files[name].stat().st_size != sizes[name] or file_digest != expected_sha256[name]:
                raise CoppeliaArchiveError(f"canonical source changed during staging: {name}")
            originals.append(
                ArchivedFile(path=name, bytes=sizes[name], sha256=file_digest, chunks=chunks[name])
            )
        index = CoppeliaArchiveIndex(
            source=source,
            descriptor_sha256=descriptor_sha256,
            chunk_bytes=chunk_bytes,
            files=originals,
            parts=parts,
        )
        index_path = destination / COPPELIA_ARCHIVE_INDEX_FILE
        if index_path.exists() or index_path.is_symlink():
            raise CoppeliaArchiveError("Coppelia archive index destination exists")
        with index_path.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(index_path)
            json.dump(index.model_dump(mode="json"), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return index
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def verify_coppelia_archive_parts(
    root: Path, index: CoppeliaArchiveIndex, *, verify_zip: Callable[[Path], int]
) -> None:
    expected = {chunk.member: chunk for item in index.files for chunk in item.chunks}
    for part in index.parts:
        path = root / part.file
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != part.bytes
            or _sha256(path) != part.sha256
        ):
            raise CoppeliaArchiveError(f"Coppelia part integrity differs: {part.file}")
        if verify_zip(path) != part.entry_count:
            raise CoppeliaArchiveError("Coppelia part entry count differs")
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != {
                name for name, chunk in expected.items() if chunk.part == part.file
            }:
                raise CoppeliaArchiveError("Coppelia part member inventory differs")
            for info in archive.infolist():
                chunk = expected[info.filename]
                digest = hashlib.sha256()
                count = 0
                with archive.open(info) as handle:
                    for data in iter(lambda: handle.read(1024 * 1024), b""):
                        count += len(data)
                        if count > chunk.bytes:
                            raise CoppeliaArchiveError("Coppelia chunk exceeds declared bytes")
                        digest.update(data)
                if count != chunk.bytes or digest.hexdigest() != chunk.sha256:
                    raise CoppeliaArchiveError("Coppelia chunk identity differs")


def reconstruct_coppelia_scenario(
    root: Path,
    index: CoppeliaArchiveIndex,
    destination: Path,
    *,
    scenario: Literal["nominal", "recovery"],
    disk_budget_bytes: int,
) -> Path:
    """Reassemble exact native bytes under an enforced per-operation disk budget.

    Call verify_coppelia_archive_parts first. Reconstruction independently checks
    every consumed chunk and original hash, and never extracts arbitrary names.
    """
    required = index.scenario_bytes(scenario)
    if required > min(disk_budget_bytes, MAX_SCENARIO_BYTES):
        raise CoppeliaArchiveError("Coppelia reconstruction exceeds its disk budget")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + MAX_INDEX_BYTES:
        raise CoppeliaArchiveError("insufficient free disk for Coppelia reconstruction")
    destination.mkdir(exist_ok=False)
    total = 0
    try:
        for item in index.files:
            if not (
                item.path.startswith(f"evidence/coppelia/{scenario}/")
                or item.path == f"evidence/coppelia/visualizations/{scenario}.mp4"
            ):
                continue
            target = destination / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with target.open("xb") as output:
                for chunk in item.chunks:
                    chunk_digest = hashlib.sha256()
                    chunk_count = 0
                    with (
                        zipfile.ZipFile(root / chunk.part) as archive,
                        archive.open(chunk.member) as input_handle,
                    ):
                        for data in iter(lambda: input_handle.read(1024 * 1024), b""):
                            count += len(data)
                            chunk_count += len(data)
                            total += len(data)
                            if (
                                count > item.bytes
                                or chunk_count > chunk.bytes
                                or total > required
                                or total > disk_budget_bytes
                            ):
                                raise CoppeliaArchiveError(
                                    "Coppelia reconstruction exceeded declared bytes"
                                )
                            output.write(data)
                            digest.update(data)
                            chunk_digest.update(data)
                    if chunk_count != chunk.bytes or chunk_digest.hexdigest() != chunk.sha256:
                        raise CoppeliaArchiveError("reconstructed Coppelia chunk differs")
            if count != item.bytes or digest.hexdigest() != item.sha256:
                raise CoppeliaArchiveError("reconstructed canonical Coppelia file differs")
        if total != required:
            raise CoppeliaArchiveError("Coppelia reconstruction byte count differs")
        return destination / "evidence" / "coppelia" / scenario
    except Exception:
        shutil.rmtree(destination)
        raise
