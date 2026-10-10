from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_skill_composer.construction import coppelia_archive as storage
from embodied_skill_composer.construction.coppelia_archive import (
    CANONICAL_PATHS,
    COPPELIA_ARCHIVE_INDEX_FILE,
    ArchiveSource,
    CoppeliaArchiveError,
    CoppeliaArchiveIndex,
    load_coppelia_archive_index,
    reconstruct_coppelia_scenario,
    stage_coppelia_archive,
    verify_coppelia_archive_parts,
)
from embodied_skill_composer.construction.release_assets import verify_deterministic_zip


def _sources(root: Path) -> dict[str, Path]:
    sources = {}
    for number, name in enumerate(sorted(CANONICAL_PATHS)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(range(64)) + f"{number}:{name}".encode())
        sources[name] = path
    return sources


def _stage(root: Path, sources: dict[str, Path]) -> CoppeliaArchiveIndex:
    return stage_coppelia_archive(
        root,
        sources,
        source=ArchiveSource(commit="a" * 40, tree_digest="b" * 64),
        descriptor_sha256="c" * 64,
        expected_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        },
        verify_zip=verify_deterministic_zip,
        chunk_bytes=32,
    )


def test_shards_are_deterministic_and_reconstruct_every_native_byte(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "source")
    first, second = tmp_path / "first", tmp_path / "second"
    index = _stage(first, sources)
    assert _stage(second, sources) == index
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert load_coppelia_archive_index(first / COPPELIA_ARCHIVE_INDEX_FILE) == index
    assert b"\r\n" not in (first / COPPELIA_ARCHIVE_INDEX_FILE).read_bytes()
    verify_coppelia_archive_parts(first, index, verify_zip=verify_deterministic_zip)
    for scenario in ("nominal", "recovery"):
        destination = tmp_path / f"restored-{scenario}"
        native = reconstruct_coppelia_scenario(
            first,
            index,
            destination,
            scenario=scenario,
            disk_budget_bytes=index.scenario_bytes(scenario),
        )
        assert native == destination / "evidence" / "coppelia" / scenario
        for path in destination.rglob("*"):
            if path.is_file():
                relative = path.relative_to(destination).as_posix()
                assert path.read_bytes() == sources[relative].read_bytes()


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "reorder", "traversal", "size", "part"]
)
def test_index_rejects_noncanonical_logical_inventory(tmp_path: Path, mutation: str) -> None:
    index = _stage(tmp_path / "assets", _sources(tmp_path / "sources"))
    payload = index.model_dump(mode="json")
    if mutation == "missing":
        payload["files"].pop()
    elif mutation == "extra":
        payload["files"].append(payload["files"][0])
    elif mutation == "duplicate":
        payload["files"][1] = payload["files"][0]
    elif mutation == "reorder":
        payload["files"][0]["chunks"].reverse()
    elif mutation == "traversal":
        payload["files"][0]["chunks"][0]["member"] = "../escape"
    elif mutation == "size":
        payload["files"][0]["chunks"][0]["bytes"] += 1
    else:
        payload["parts"][0]["file"] = "../escape.zip"
    with pytest.raises(ValueError):
        CoppeliaArchiveIndex.model_validate(payload)


def test_reconstruction_enforces_budget_before_creating_destination(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    index = _stage(assets, _sources(tmp_path / "sources"))
    target = tmp_path / "restored"
    with pytest.raises(CoppeliaArchiveError, match="disk budget"):
        reconstruct_coppelia_scenario(
            assets,
            index,
            target,
            scenario="nominal",
            disk_budget_bytes=index.scenario_bytes("nominal") - 1,
        )
    assert not target.exists()


def test_reconstruction_cleans_up_after_corrupt_chunk(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    index = _stage(assets, _sources(tmp_path / "sources"))
    payload = index.model_dump(mode="json")
    payload["files"][0]["chunks"][0]["sha256"] = "0" * 64
    changed = CoppeliaArchiveIndex.model_validate(payload)
    with pytest.raises(CoppeliaArchiveError, match="chunk differs"):
        reconstruct_coppelia_scenario(
            assets,
            changed,
            tmp_path / "restored",
            scenario="nominal",
            disk_budget_bytes=index.scenario_bytes("nominal"),
        )
    assert not (tmp_path / "restored").exists()


def test_part_hash_is_verified_before_zip_is_opened(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    index = _stage(assets, _sources(tmp_path / "sources"))
    path = assets / index.parts[0].file
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(CoppeliaArchiveError, match="integrity differs"):
        verify_coppelia_archive_parts(
            assets,
            index,
            verify_zip=lambda _path: pytest.fail("must not open corrupt ZIP"),
        )


def test_staging_checks_source_hashes_and_cleans_partial_assets(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "sources")
    target = tmp_path / "assets"
    with pytest.raises(CoppeliaArchiveError, match="source changed"):
        stage_coppelia_archive(
            target,
            sources,
            source=ArchiveSource(commit="a" * 40, tree_digest="b" * 64),
            descriptor_sha256="c" * 64,
            expected_sha256={name: "0" * 64 for name in sources},
            verify_zip=verify_deterministic_zip,
            chunk_bytes=32,
        )
    assert not list(target.iterdir())


def test_free_disk_is_checked_before_archive_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path / "sources")
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    with pytest.raises(CoppeliaArchiveError, match="insufficient free disk"):
        _stage(tmp_path / "assets", sources)
    assert not list((tmp_path / "assets").iterdir())


def test_disk_exhaustion_between_shards_removes_already_written_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path / "sources")
    checks = 0

    def available(_path):
        nonlocal checks
        checks += 1
        return SimpleNamespace(free=storage.MAX_INDEX_BYTES + 128 if checks == 1 else 0)

    monkeypatch.setattr(storage.shutil, "disk_usage", available)
    with pytest.raises(CoppeliaArchiveError, match="insufficient free disk"):
        _stage(tmp_path / "assets", sources)
    assert checks == 2
    assert not list((tmp_path / "assets").iterdir())


def test_oserror_cleans_created_shards_and_preserves_unrelated_files(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "sources")
    destination = tmp_path / "assets"
    destination.mkdir()
    keep = destination / "keep.txt"
    keep.write_text("user file", encoding="utf-8")
    calls = 0

    def verify(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return verify_deterministic_zip(path)

    with pytest.raises(OSError, match="disk full"):
        stage_coppelia_archive(
            destination,
            sources,
            source=ArchiveSource(commit="a" * 40, tree_digest="b" * 64),
            descriptor_sha256="c" * 64,
            expected_sha256={
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in sources.items()
            },
            verify_zip=verify,
            chunk_bytes=32,
        )
    assert list(destination.iterdir()) == [keep]
    assert keep.read_text(encoding="utf-8") == "user file"


@pytest.mark.parametrize("filename", [storage.part_filename(1), COPPELIA_ARCHIVE_INDEX_FILE])
def test_staging_rejects_dangling_destination_links_without_following_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    sources = _sources(tmp_path / "sources")
    destination = tmp_path / "assets"
    target = destination / filename
    original_is_symlink = Path.is_symlink
    # Exercise the dangling-link branch without Windows symlink privileges.
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == target or original_is_symlink(path),
    )
    with pytest.raises(CoppeliaArchiveError, match="destination exists"):
        _stage(destination, sources)
    assert not list(destination.iterdir())


@pytest.mark.parametrize(
    ("filename", "exclusive_mode"),
    [(storage.part_filename(1), "xb"), (COPPELIA_ARCHIVE_INDEX_FILE, "x")],
)
def test_exclusive_creation_preserves_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    exclusive_mode: str,
) -> None:
    sources = _sources(tmp_path / "sources")
    destination = tmp_path / "assets"
    target = destination / filename
    original_open = Path.open

    def raced_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == target and mode == exclusive_mode:
            with original_open(path, "wb") as output:
                output.write(b"another owner's file")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(FileExistsError):
        _stage(destination, sources)
    assert target.read_bytes() == b"another owner's file"
    assert list(destination.iterdir()) == [target]


def test_index_loader_bounds_bytes_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"too_large": True}), encoding="utf-8")
    monkeypatch.setattr(storage, "MAX_INDEX_BYTES", 1)
    with pytest.raises(CoppeliaArchiveError, match="oversized"):
        load_coppelia_archive_index(path)


def test_slice_count_is_bounded_before_building_or_writing_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path / "sources")
    monkeypatch.setattr(storage, "MAX_PARTS", 1)
    with pytest.raises(CoppeliaArchiveError, match="part-count budget"):
        _stage(tmp_path / "assets", sources)
    assert not (tmp_path / "assets").exists()


def test_index_rejects_boolean_byte_counts(tmp_path: Path) -> None:
    index = _stage(tmp_path / "assets", _sources(tmp_path / "sources"))
    payload = index.model_dump(mode="json")
    payload["files"][0]["chunks"][-1]["bytes"] = True
    with pytest.raises(ValueError):
        CoppeliaArchiveIndex.model_validate(payload)


@pytest.mark.parametrize("scope", ["scenario", "total"])
def test_index_rejects_native_byte_budgets_before_expanding_chunks(
    tmp_path: Path, scope: str
) -> None:
    index = _stage(tmp_path / "assets", _sources(tmp_path / "sources"))
    payload = index.model_dump(mode="json")
    payload["files"][0]["bytes"] = storage.MAX_SCENARIO_BYTES
    if scope == "total":
        payload["files"][-1]["bytes"] = storage.MAX_SCENARIO_BYTES
    with pytest.raises(ValueError, match=f"{scope}.*byte budget"):
        CoppeliaArchiveIndex.model_validate(payload)


@pytest.mark.parametrize("scope", ["file", "scenario", "total"])
def test_staging_checks_native_byte_budgets_before_creating_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    sources = _sources(tmp_path / "sources")
    if scope == "file":
        monkeypatch.setattr(storage, "MAX_CANONICAL_FILE_BYTES", 1)
    elif scope == "scenario":
        monkeypatch.setattr(storage, "MAX_SCENARIO_BYTES", 1)
    else:
        monkeypatch.setattr(storage, "MAX_CANONICAL_TOTAL_BYTES", 1)
    destination = tmp_path / "assets"
    with pytest.raises(CoppeliaArchiveError, match=f"{scope}.*byte budget"):
        _stage(destination, sources)
    assert not destination.exists()
