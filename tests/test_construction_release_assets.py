from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from embodied_skill_composer.construction import release_assets
from embodied_skill_composer.construction.experiment_protocol import (
    SelectedCheckpoint,
)
from embodied_skill_composer.construction.public_demo_provenance import (
    PublicDemoExportError,
    SourceIdentity,
)
from embodied_skill_composer.construction.release_assets import (
    RELEASE_IDENTITY_FILE,
    ReleaseAssetError,
    stage_release_assets,
    verify_release_assets,
)
from embodied_skill_composer.construction.release_identity import (
    RepositoryReleaseIdentity,
)


RELEASE_VERSION = "0.1.0"
RELEASE_TAG = "v0.1.0"
SOURCE = SourceIdentity(
    commit="a" * 40,
    dirty=False,
    tree_digest=hashlib.sha256(b"release-tree").hexdigest(),
)
PROTOCOL_DIGEST = hashlib.sha256(b"protocol").hexdigest()
MATRIX_ID = "construction-intelligence-v1-research"
VARIANTS = (
    ("mappo_full", "mappo"),
    ("ippo_full", "ippo"),
    ("mappo_no_bc", "mappo"),
    ("mappo_no_failure_curriculum", "mappo"),
)


@pytest.fixture
def packaged_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkpoints = tmp_path / "checkpoints"
    selections = _selection_fixture(checkpoints)
    _install_release_fakes(monkeypatch, selections)
    descriptors = {
        "deterministic_bundle": tmp_path / "deterministic.json",
        "research_bundle": tmp_path / "research.json",
        "simulator_bundle": tmp_path / "simulator.json",
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    for destination in (first, second):
        stage_release_assets(
            destination,
            workspace=workspace,
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
            **descriptors,
        )
    return {
        "first": first,
        "second": second,
        "workspace": workspace,
    }


def test_release_assets_are_byte_deterministic_and_include_external_onnx_data(
    packaged_assets: dict[str, Path],
) -> None:
    first = packaged_assets["first"]
    second = packaged_assets["second"]

    assert _directory_bytes(first) == _directory_bytes(second)
    manifest = verify_release_assets(
        first,
        workspace=packaged_assets["workspace"],
        expected_release_version=RELEASE_VERSION,
        expected_release_tag=RELEASE_TAG,
        expected_source=SOURCE,
    )
    assert manifest.required_roles == [
        "coppelia_evidence",
        "public_demo",
        "research_evidence",
        "selected_policies",
    ]
    policy_zip = first / "construction-intelligence-v1-selected-policies.zip"
    with zipfile.ZipFile(policy_zip) as archive:
        names = archive.namelist()
    assert sum(name.endswith("/actor.onnx") for name in names) == 20
    assert sum(name.endswith("/actor.onnx.data") for name in names) == 20


def test_release_verifier_rejects_missing_and_extra_assets(
    packaged_assets: dict[str, Path],
) -> None:
    root = packaged_assets["first"]
    missing = root / "construction-intelligence-v1-research-evidence.zip"
    missing.unlink()
    with pytest.raises(ReleaseAssetError, match="inventory mismatch.*missing"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])

    missing.write_bytes(b"replacement-does-not-matter")
    (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ReleaseAssetError, match="inventory mismatch.*extra"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def test_release_verifier_rejects_tampered_asset_bytes(
    packaged_assets: dict[str, Path],
) -> None:
    root = packaged_assets["first"]
    research = root / "construction-intelligence-v1-research-evidence.zip"
    research.write_bytes(research.read_bytes() + b"tampered")

    with pytest.raises(ReleaseAssetError, match="integrity check failed"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def test_release_verifier_rejects_unsafe_archive_names_after_rehash(
    packaged_assets: dict[str, Path],
) -> None:
    root = packaged_assets["first"]
    public_zip = root / "construction-intelligence-v1-public-demo.zip"
    members = _zip_bytes(public_zip)
    members["../escape.txt"] = b"escape"
    _write_test_zip(public_zip, members)
    _rehash_asset(root, public_zip.name, entry_count=len(members))

    with pytest.raises(ReleaseAssetError, match="unsafe release archive path"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def test_zip_verifier_bounds_member_and_total_uncompressed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "bounded.zip"
    _write_test_zip(archive, {"a.bin": b"123456", "b.bin": b"abcdef"})
    monkeypatch.setattr(
        release_assets,
        "_MAX_MEMBER_UNCOMPRESSED_BYTES",
        5,
    )
    with pytest.raises(ReleaseAssetError, match="member exceeds.*size limit"):
        release_assets.verify_deterministic_zip(archive)

    monkeypatch.setattr(
        release_assets,
        "_MAX_MEMBER_UNCOMPRESSED_BYTES",
        10,
    )
    monkeypatch.setattr(
        release_assets,
        "_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
        10,
    )
    with pytest.raises(ReleaseAssetError, match="total uncompressed-size limit"):
        release_assets.verify_deterministic_zip(archive)


def test_zip_verifier_bounds_member_count_and_compression_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "bounded.zip"
    _write_test_zip(archive, {"a.txt": b"a" * 4096, "b.txt": b"b"})
    monkeypatch.setattr(release_assets, "_MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ReleaseAssetError, match="member-count limit"):
        release_assets.verify_deterministic_zip(archive)

    monkeypatch.setattr(release_assets, "_MAX_ARCHIVE_MEMBERS", 10)
    monkeypatch.setattr(release_assets, "_MAX_COMPRESSION_RATIO", 2.0)
    with pytest.raises(ReleaseAssetError, match="compression-ratio limit"):
        release_assets.verify_deterministic_zip(archive)


def test_release_archive_budgets_fit_hosted_runner_guardrail() -> None:
    budgets = {
        role: release_assets._archive_budget(filename)
        for role, filename in release_assets._ASSET_FILES.items()
    }
    verification_disk_bytes = sum(
        budget.max_compressed_bytes for budget in budgets.values()
    ) + sum(
        budgets[role].max_uncompressed_bytes
        for role in ("public_demo", "research_evidence", "selected_policies")
    )

    assert (
        verification_disk_bytes
        <= release_assets._MAX_RELEASE_VERIFICATION_DISK_BYTES
        <= 5 * 1024**3
    )
    assert max(
        budget.max_member_uncompressed_bytes for budget in budgets.values()
    ) <= 256 * 1024**2
    assert release_assets._MAX_CENTRAL_DIRECTORY_BYTES <= 16 * 1024**2
    assert release_assets._MAX_COMPRESSION_RATIO <= 200.0


def test_zip_preflight_rejects_excessive_declared_count_before_zipfile_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "bounded.zip"
    _write_test_zip(archive, {"a.txt": b"a"})
    payload = bytearray(archive.read_bytes())
    eocd_offset = payload.rfind(b"PK\x05\x06")
    excessive_count = release_assets._DEFAULT_ARCHIVE_BUDGET.max_members + 1
    struct.pack_into("<HH", payload, eocd_offset + 8, excessive_count, excessive_count)
    archive.write_bytes(payload)

    def fail_if_opened(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ZipFile must not parse an excessive central directory")

    monkeypatch.setattr(release_assets.zipfile, "ZipFile", fail_if_opened)
    with pytest.raises(ReleaseAssetError, match="member-count limit"):
        release_assets.verify_deterministic_zip(archive)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("release_version", "9.9.9", "release tag must exactly match"),
        ("release_tag", "v9.9.9", "release tag must exactly match"),
        (
            "source",
            {
                "commit": "b" * 40,
                "dirty": False,
                "tree_digest": hashlib.sha256(b"other-tree").hexdigest(),
            },
            "source does not match",
        ),
    ],
)
def test_release_verifier_rejects_wrong_release_identity(
    packaged_assets: dict[str, Path],
    field: str,
    value: object,
    match: str,
) -> None:
    root = packaged_assets["first"]
    manifest_path = root / RELEASE_IDENTITY_FILE
    manifest = _read_json(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(ReleaseAssetError, match=match):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def test_release_verifier_rejects_wrong_public_demo_channel_after_rehash(
    packaged_assets: dict[str, Path],
) -> None:
    root = packaged_assets["first"]
    public_zip = root / "construction-intelligence-v1-public-demo.zip"
    members = _zip_bytes(public_zip)
    provenance = json.loads(members["provenance.json"])
    provenance["channel"] = "preview"
    members["provenance.json"] = _json_bytes(provenance)
    _write_test_zip(public_zip, members)
    _rehash_asset(root, public_zip.name, entry_count=len(members))

    with pytest.raises(ReleaseAssetError, match="public demo release verification"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def test_release_verifier_rejects_policy_hash_mismatch_after_full_rehash(
    packaged_assets: dict[str, Path],
) -> None:
    root = packaged_assets["first"]
    policy_zip = root / "construction-intelligence-v1-selected-policies.zip"
    members = _zip_bytes(policy_zip)
    index = json.loads(members["policy-assets.json"])
    policy = index["policies"][0]
    checkpoint_path = policy["checkpoint"]["path"]
    members[checkpoint_path] = b"forged checkpoint"
    forged_sha = hashlib.sha256(members[checkpoint_path]).hexdigest()
    policy["checkpoint"]["bytes"] = len(members[checkpoint_path])
    policy["checkpoint"]["sha256"] = forged_sha
    policy["selected_checkpoint"]["checkpoint_sha256"] = forged_sha
    members["policy-assets.json"] = _json_bytes(index)
    _write_test_zip(policy_zip, members)
    _rehash_asset(root, policy_zip.name, entry_count=len(members))

    with pytest.raises(ReleaseAssetError, match="does not match frozen selection"):
        verify_release_assets(root, workspace=packaged_assets["workspace"])


def _selection_fixture(checkpoint_root: Path) -> dict[str, object]:
    selections: list[dict[str, object]] = []
    for variant, algorithm in VARIANTS:
        for seed in range(7, 12):
            run_key = f"{variant}-seed-{seed}"
            configuration_digest = hashlib.sha256(run_key.encode()).hexdigest()
            lineage = [f"{run_key}-origin"]
            metadata = {
                "experiment_id": "construction_intelligence_v1",
                "experiment_variant": variant,
                "training_seed": seed,
                "transition_count": 1_500_000,
                "checkpoint_fraction": 1.0,
                "configuration_digest": configuration_digest,
                "source_commit": "b" * 40,
                "checkpoint_lineage": lineage,
                "resume_provenance": {},
            }
            checkpoint = checkpoint_root / f"{run_key}.pt"
            _write_json(
                checkpoint,
                {
                    "algorithm": algorithm,
                    "metadata": metadata,
                    "run_key": run_key,
                },
            )
            checkpoint_id = f"{run_key}-checkpoint-100pct"
            selected = SelectedCheckpoint(
                checkpoint_id=checkpoint_id,
                experiment_id="construction_intelligence_v1",
                experiment_variant=variant,
                training_seed=seed,
                checkpoint_fraction=1.0,
                transition_count=1_500_000,
                split="validation",
                scenario_seeds=[800, 801, 802, 803, 804],
                mean_completion_rate=1.0,
                mean_makespan_s=100.0,
                checkpoint_path=str(checkpoint.resolve()),
                checkpoint_sha256=_sha256(checkpoint),
                checkpoint_lineage=lineage,
                configuration_digest=configuration_digest,
                source_commit="b" * 40,
                resume_provenance={},
                required_checkpoint_fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
                candidate_ranking=[
                    checkpoint_id,
                    f"{run_key}-checkpoint-075pct",
                    f"{run_key}-checkpoint-050pct",
                    f"{run_key}-checkpoint-025pct",
                    f"{run_key}-checkpoint-010pct",
                ],
            )
            selections.append(
                {
                    "run_key": run_key,
                    "selected": selected.model_dump(mode="json"),
                }
            )
    return {
        "matrix_id": MATRIX_ID,
        "protocol_digest": PROTOCOL_DIGEST,
        "selections": selections,
    }


def _install_release_fakes(
    monkeypatch: pytest.MonkeyPatch,
    selections: dict[str, object],
) -> None:
    identity = RepositoryReleaseIdentity(
        version=RELEASE_VERSION,
        tag=RELEASE_TAG,
        declarations=(("fixture", RELEASE_VERSION),),
    )
    monkeypatch.setattr(
        release_assets,
        "repository_release_identity",
        lambda _workspace: identity,
    )
    monkeypatch.setattr(
        release_assets,
        "repository_source_identity",
        lambda _workspace: SOURCE,
    )

    def fake_export(output_dir: Path, **kwargs: Any) -> dict[str, object]:
        assert kwargs["channel"] == "release"
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "evidence" / "research" / "selections.json", selections)
        _write_json(
            output_dir / "evidence" / "research" / "evaluation.json",
            {"validated": True},
        )
        _write_json(
            output_dir / "evidence" / "coppelia" / "nominal.json",
            {"validated": True},
        )
        (output_dir / "index.html").write_text("release", encoding="utf-8")
        provenance = {
            "channel": "release",
            "release_version": kwargs["release_version"],
            "release_tag": kwargs["release_tag"],
            "source": kwargs["source"].model_dump(mode="json"),
        }
        _write_json(output_dir / "provenance.json", provenance)
        return provenance

    def fake_verify(
        output_dir: Path,
        *,
        expected_channel: str | None = None,
        expected_source: SourceIdentity | None = None,
        expected_release_version: str | None = None,
        expected_release_tag: str | None = None,
    ) -> dict[str, object]:
        provenance = _read_json(output_dir / "provenance.json")
        if provenance.get("channel") != expected_channel:
            raise PublicDemoExportError("wrong release channel")
        if provenance.get("release_version") != expected_release_version:
            raise PublicDemoExportError("wrong release version")
        if provenance.get("release_tag") != expected_release_tag:
            raise PublicDemoExportError("wrong release tag")
        if provenance.get("source") != expected_source.model_dump(mode="json"):
            raise PublicDemoExportError("wrong release source")
        return provenance

    def fake_metadata(path: Path, *, device: str = "cpu") -> dict[str, object]:
        assert device == "cpu"
        payload = _read_json(path)
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        return metadata

    def fake_load(path: Path, *, device: str = "cpu") -> SimpleNamespace:
        assert device == "cpu"
        payload = _read_json(path)
        return SimpleNamespace(
            algorithm=payload["algorithm"],
            actor_model=payload["run_key"],
        )

    def fake_onnx(
        model: object,
        path: Path,
        *,
        device: str = "cpu",
    ) -> Path:
        assert device == "cpu"
        path.write_bytes(f"onnx:{model}".encode())
        path.with_suffix(".onnx.data").write_bytes(f"data:{model}".encode())
        return path

    monkeypatch.setattr(release_assets, "export_public_demo_bundle", fake_export)
    monkeypatch.setattr(release_assets, "verify_public_demo_export", fake_verify)
    monkeypatch.setattr(release_assets, "load_policy_checkpoint_metadata", fake_metadata)
    monkeypatch.setattr(release_assets, "load_policy_checkpoint", fake_load)
    monkeypatch.setattr(release_assets, "export_actor_onnx", fake_onnx)
    monkeypatch.setattr(release_assets, "_validate_onnx_model", lambda _path: None)


def _rehash_asset(
    root: Path,
    filename: str,
    *,
    entry_count: int,
) -> None:
    manifest_path = root / RELEASE_IDENTITY_FILE
    manifest = _read_json(manifest_path)
    assets = manifest["assets"]
    assert isinstance(assets, list)
    record = next(
        item
        for item in assets
        if isinstance(item, dict) and item["file"] == filename
    )
    path = root / filename
    record["bytes"] = path.stat().st_size
    record["sha256"] = _sha256(path)
    record["entry_count"] = entry_count
    _write_json(manifest_path, manifest)


def _write_test_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, payload in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compresslevel=9)


def _zip_bytes(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
