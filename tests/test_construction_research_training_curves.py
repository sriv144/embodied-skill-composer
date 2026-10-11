from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_skill_composer.construction.research_training_curves import (
    CURVE_PREFIX,
    INDEX_ROLE,
    SEMANTICS,
    curve_contract,
    stage_training_curves,
    validate_curve,
    verify_training_curves,
)
from tests.construction_training_curve_fixtures import attach_training_curve_sources, training_csv


@pytest.fixture
def curves(tmp_path):
    configs = [{
        "algorithm": "ippo" if variant == "ippo_full" else "mappo",
        "experiment_variant": variant, "training_seed": seed, "transitions": 1_500_000,
        "source_commit": "a" * 40, "source_tree_digest": "b" * 64, "source_dirty": False,
        "configuration_digest": hashlib.sha256(f"{variant}-{seed}".encode()).hexdigest(),
    } for variant in ("mappo_full", "ippo_full", "mappo_no_bc", "mappo_no_failure_curriculum")
        for seed in range(7, 12)]
    runs = [{"run_key": f"{config['experiment_variant']}-seed-{config['training_seed']}",
             "status": "completed", "config": config} for config in configs]
    matrix = {"id": "research-matrix", "runs": attach_training_curve_sources(tmp_path / "private", runs)}
    output = tmp_path / "public"
    output.mkdir()
    artifacts = stage_training_curves(output, matrix)
    files = {role: output / source for role, (source, _target, _mime) in artifacts.items()}
    return matrix, output, artifacts, files


def test_twenty_real_training_csvs_are_byte_preserved_hash_bound_and_not_validation_points(curves):
    matrix, output, artifacts, files = curves
    configs = {run["run_key"]: run["config"] for run in matrix["runs"]}
    verify_training_curves(files, configs, matrix_id=matrix["id"])
    assert len(artifacts) == 21
    index = json.loads(files[INDEX_ROLE].read_text())
    assert len(index["runs"]) == 20
    assert index["semantics"] == SEMANTICS
    assert index["semantics"]["curve_kind"] != index["semantics"]["validation_curve_kind"]
    for record in index["runs"]:
        run = next(run for run in matrix["runs"] if run["run_key"] == record["run_key"])
        original = (Path(run["artifact_dir"]) / "learning_curve.csv").read_bytes()
        published = files[record["role"]].read_bytes()
        assert original == published == training_csv()
        assert record["sha256"] == hashlib.sha256(original).hexdigest()
        assert record["bytes"] == len(original)
        assert record["row_count"] == 8  # Five validation checkpoints, eight source PPO updates.
        assert "rollout_success_fraction" not in record["unavailable_diagnostics"]
        assert "kl_approx" in record["unavailable_diagnostics"]
    assert str(output.parent) not in files[INDEX_ROLE].read_text()
    assert len(list(output.glob("training/*/learning_curve.csv"))) == 20


def _csv(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


@pytest.mark.parametrize("mutation", [
    "empty", "missing-column", "unknown-column", "nan", "infinity", "formula", "path",
    "missing-first", "missing-middle", "missing-final", "fraction", "counter", "stride",
])
def test_incomplete_unsafe_or_nonfinite_csv_is_rejected(mutation):
    rows = list(csv.DictReader(io.StringIO(training_csv().decode())))
    if mutation == "empty":
        data = b""
    else:
        if mutation == "missing-column":
            for row in rows:
                row.pop("loss_critic")
        elif mutation == "unknown-column":
            for row in rows:
                row["local_path"] = r"C:\Users\private\run"
        elif mutation in {"nan", "infinity", "formula", "path"}:
            rows[0]["loss_critic"] = {"nan": "NaN", "infinity": "inf", "formula": "=1+1",
                                      "path": r"C:\Users\private\run"}[mutation]
        elif mutation.startswith("missing-"):
            rows.pop({"missing-first": 0, "missing-middle": 2, "missing-final": -1}[mutation])
        elif mutation == "fraction":
            rows[0]["rollout_success_fraction"] = "1.1"
        elif mutation == "counter":
            rows[1]["transitions"] = rows[0]["transitions"]
        else:
            rows[1]["transitions"] = "1000000"
        data = _csv(rows)
    with pytest.raises(ValueError):
        validate_curve(data, transitions=1_500_000, rollout_decisions=125_000)


def test_legacy_terminal_diagnostic_is_not_invented_as_success():
    core = {"update", "transitions", "loss_objective", "loss_critic", "loss_entropy",
            "mean_episode_return", "rollout_terminal_fraction"}
    rows = [{key: value for key, value in row.items() if key in core}
            for row in csv.DictReader(io.StringIO(training_csv().decode()))]
    summary = validate_curve(_csv(rows), transitions=1_500_000, rollout_decisions=125_000)
    assert "rollout_success_fraction" in summary["unavailable_diagnostics"]
    assert "not task success" in SEMANTICS["rollout_terminal_fraction"]


def test_source_update_cannot_be_dropped_and_hidden_by_renumbering():
    original = list(csv.DictReader(io.StringIO(training_csv().decode())))
    rows = [original[index] for index in (0, 1, 3, 5, 7)]
    rows.pop(0)  # Two checkpoint-shortened updates fit within one maximum stride.
    for update, row in enumerate(rows, start=1):
        row["update"] = str(update)
    validate_curve(_csv(rows), transitions=1_500_000, rollout_decisions=125_000)
    with pytest.raises(ValueError, match="every configured source update"):
        validate_curve(_csv(rows), transitions=1_500_000, rollout_decisions=125_000,
                       checkpoint_fractions=[0.1, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("fractions", [[], "not-a-list", [None], [True], [float("nan")], [0], [1.1]])
def test_malformed_checkpoint_schedule_is_rejected(fractions):
    with pytest.raises(ValueError, match="checkpoint fractions"):
        validate_curve(training_csv(), transitions=1_500_000, rollout_decisions=62_500,
                       checkpoint_fractions=fractions)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "source", "digest", "hash", "path", "semantics"])
def test_index_membership_identity_hashes_and_semantics_are_verified(curves, mutation):
    matrix, _output, _artifacts, files = curves
    index_path = files[INDEX_ROLE]
    index = json.loads(index_path.read_text())
    if mutation == "missing":
        index["runs"].pop()
    elif mutation == "duplicate":
        index["runs"][1] = copy.deepcopy(index["runs"][0])
    elif mutation == "semantics":
        index["semantics"]["curve_kind"] = "validation_checkpoint_completion"
    else:
        field = {"source": "source_commit", "digest": "configuration_digest", "hash": "sha256",
                 "path": "path"}[mutation]
        index["runs"][0][field] = "changed"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="training curve"):
        verify_training_curves(files, {run["run_key"]: run["config"] for run in matrix["runs"]},
                               matrix_id=matrix["id"])


@pytest.mark.parametrize("mutation", ["missing", "truncated", "metadata", "unsafe-key", "extra", "duplicate"])
def test_source_artifacts_must_be_complete_and_provenance_compatible(curves, mutation, tmp_path):
    matrix, _output, _artifacts, _files = curves
    run = matrix["runs"][0]
    source = Path(run["artifact_dir"]) / "learning_curve.csv"
    if mutation == "missing":
        source.unlink()
    elif mutation == "truncated":
        source.write_bytes(source.read_bytes().rsplit(b"\r\n", 2)[0] + b"\r\n")
    elif mutation == "metadata":
        metadata = Path(run["artifact_dir"]) / "training_config.json"
        config = json.loads(metadata.read_text())
        config["source_commit"] = "e" * 40
        metadata.write_text(json.dumps(config))
    elif mutation == "unsafe-key":
        run["run_key"] = "../escape"
    elif mutation == "extra":
        matrix["runs"].append(copy.deepcopy(run))
    else:
        matrix["runs"][1] = copy.deepcopy(run)
    with pytest.raises(ValueError, match="training curve"):
        stage_training_curves(tmp_path / "rejected", matrix)


def test_curve_contract_does_not_accept_active_paths_or_other_roles():
    assert curve_contract("unrelated") is None
    with pytest.raises(ValueError, match="public-safe"):
        curve_contract(f"{CURVE_PREFIX}../../unsafe.html")


def test_legacy_nine_role_bundle_remains_readable_but_cannot_stage_new_release(tmp_path):
    from embodied_skill_composer.construction import public_demo_provenance as provenance
    from tests.test_construction_public_demo_provenance import SOURCE, _research_bundle

    descriptor = _research_bundle(tmp_path / "legacy")
    payload = json.loads(descriptor.read_text())
    payload["artifacts"] = [item for item in payload["artifacts"]
                            if item["role"] != INDEX_ROLE and not item["role"].startswith(CURVE_PREFIX)]
    descriptor.write_text(json.dumps(payload))
    assert len(provenance.verify_research_release_evidence(descriptor).artifacts) == 9
    legacy = provenance._load_bundle(descriptor, expected_kind="research")
    deterministic = SimpleNamespace(
        files=dict.fromkeys(provenance._RELEASE_DETERMINISTIC_ROLES),
        manifest=SimpleNamespace(configuration_digests=["a" * 64], evidence_status="canonical"),
    )
    with pytest.raises(provenance.PublicDemoExportError, match="twenty per-update"):
        provenance._validate_export_claims(
            "release", SOURCE, deterministic, legacy, object(), object(), object(),
        )


def test_descriptor_verifier_rejects_rehashed_csv_when_index_content_identity_changed(tmp_path):
    from embodied_skill_composer.construction import public_demo_provenance as provenance
    from tests.test_construction_public_demo_provenance import _research_bundle

    descriptor = _research_bundle(tmp_path / "research")
    provenance.verify_research_release_evidence(descriptor)
    payload = json.loads(descriptor.read_text())
    artifact = next(item for item in payload["artifacts"] if item["role"].startswith(CURVE_PREFIX))
    path = descriptor.parent / artifact["path"]
    changed = path.read_bytes().replace(b"-0.1", b"-0.2")
    path.write_bytes(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    descriptor.write_text(json.dumps(payload))
    with pytest.raises(provenance.PublicDemoExportError, match="hash/content/provenance"):
        provenance.verify_research_release_evidence(descriptor)


def test_public_export_and_research_archive_retain_real_training_curves_separately(tmp_path):
    import zipfile

    from embodied_skill_composer.construction import public_demo_provenance as provenance
    from embodied_skill_composer.construction.release_assets import (
        verify_deterministic_zip,
        write_deterministic_zip,
    )
    from tests.test_construction_public_demo_provenance import (
        SOURCE,
        _deterministic_bundle,
        _research_bundle,
    )

    descriptor = _research_bundle(tmp_path / "research")
    output = tmp_path / "public"
    provenance.export_public_demo_bundle(
        output,
        deterministic_bundle=_deterministic_bundle(tmp_path / "deterministic", status="fixture", complete=True),
        research_bundle=descriptor, source=SOURCE, channel="preview",
    )
    provenance.verify_public_demo_export(output, expected_channel="preview")
    summary = json.loads((output / "research-summary.json").read_text())
    assert len(summary["learning_curves"]) == 20
    assert all(curve["curve_kind"] == "validation_checkpoint_completion"
               and len(curve["points"]) == 5 for curve in summary["learning_curves"])
    assert all("validation_completion_rate" in point and "rollout_terminal_fraction" not in point
               for curve in summary["learning_curves"] for point in curve["points"])
    assert any(reference["path"].endswith("training-curves.json")
               for reference in summary["artifact_references"])
    research_root = output / "evidence/research"
    archive = tmp_path / "construction-intelligence-v1-research-evidence.zip"
    write_deterministic_zip(archive, {
        path.relative_to(output).as_posix(): path for path in research_root.rglob("*") if path.is_file()
    })
    verify_deterministic_zip(archive)
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.endswith("/learning_curve.csv")]
        assert len(names) == 20
        assert all(zipped.read(name) == training_csv() for name in names)
