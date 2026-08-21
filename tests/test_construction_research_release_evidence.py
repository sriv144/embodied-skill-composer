from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from embodied_skill_composer.construction.evaluation import (
    EvaluationSuite,
    render_evaluation_report,
)
from embodied_skill_composer.construction.public_demo_provenance import (
    PublicDemoExportError,
    verify_research_release_evidence,
)
from embodied_skill_composer.construction.research_release_evidence import (
    package_research_release_evidence,
)
from tests.test_construction_public_demo_provenance import (
    MATRIX_ID,
    SOURCE,
    _research_bundle,
)
from scripts.package_construction_research_evidence import _snapshot_registry


class _StaticRegistry:
    def __init__(
        self,
        matrix: dict[str, object],
        frozen: list[dict[str, object]] | None = None,
    ) -> None:
        self.matrix = matrix
        self.frozen = frozen or []

    def get_experiment_matrix(self, matrix_id: str) -> dict[str, object] | None:
        return self.matrix if matrix_id == self.matrix.get("id") else None

    def list_policy_selections(self, matrix_id: str) -> list[dict[str, object]]:
        return list(self.frozen) if matrix_id == self.matrix.get("id") else []


def test_registry_snapshot_uses_a_complete_read_only_sqlite_backup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("create table evidence (value text not null)")
        connection.execute("insert into evidence values ('canonical')")
    original = source.read_bytes()
    snapshot = tmp_path / "snapshot.sqlite"

    _snapshot_registry(source, snapshot)

    with sqlite3.connect(f"{snapshot.as_uri()}?mode=ro", uri=True) as connection:
        assert connection.execute("select value from evidence").fetchone() == ("canonical",)
    assert source.read_bytes() == original


def test_research_packager_is_atomic_deterministic_and_public_safe(
    tmp_path: Path,
) -> None:
    registry, selections, heldout, selected_sources = _canonical_inputs(tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    (first / "stale.txt").write_text("stale", encoding="utf-8")
    second = tmp_path / "second"

    first_descriptor = package_research_release_evidence(
        first,
        registry=registry,
        matrix_id=MATRIX_ID,
        selection_evidence_path=selections,
        heldout_run_dir=heldout,
    )
    package_research_release_evidence(
        second,
        registry=registry,
        matrix_id=MATRIX_ID,
        selection_evidence_path=selections,
        heldout_run_dir=heldout,
    )

    manifest = verify_research_release_evidence(first_descriptor)
    assert manifest.kind == "research"
    assert manifest.evidence_status == "canonical"
    assert manifest.matrix_id == MATRIX_ID
    assert len(manifest.configuration_digests) == 20
    assert set(_directory_bytes(first)) == set(_directory_bytes(second))
    assert _directory_bytes(first) == _directory_bytes(second)
    assert not (first / "stale.txt").exists()
    assert len(list((first / "policies").glob("*/checkpoint.pt"))) == 20

    selections_payload = _read_object(first / "selections.json")
    for item in selections_payload["selections"]:
        selected = item["selected"]
        assert selected["checkpoint_path"].startswith("policies/")
        assert "\\" not in selected["checkpoint_path"]
        source = selected_sources[item["run_key"]]
        packaged = first / selected["checkpoint_path"]
        assert packaged.read_bytes() == source.read_bytes()
        assert hashlib.sha256(packaged.read_bytes()).hexdigest() == selected["checkpoint_sha256"]

    declared = [first / item.path for item in manifest.artifacts]
    published_text = "\n".join(
        path.read_text(encoding="utf-8", errors="strict") for path in declared
    )
    assert str(tmp_path) not in published_text
    assert "C:\\Users\\" not in published_text

    unsafe = tmp_path / "unsafe"
    shutil.copytree(first, unsafe)
    unsafe_matrix = _read_object(unsafe / "matrix.json")
    unsafe_runs = unsafe_matrix["runs"]
    assert isinstance(unsafe_runs, list)
    unsafe_runs[0]["config"]["output_root"] = r"C:\Users\leaked\training"
    _write_json(unsafe / "matrix.json", unsafe_matrix)
    _rehash_artifact(unsafe / "research-bundle.json", "matrix")
    with pytest.raises(PublicDemoExportError, match="machine-local paths"):
        verify_research_release_evidence(unsafe / "research-bundle.json")


def test_research_packager_rejects_incomplete_matrix_without_replacing_output(
    tmp_path: Path,
) -> None:
    registry, selections, heldout, _ = _canonical_inputs(tmp_path)
    incomplete = copy.deepcopy(registry.matrix)
    runs = incomplete["runs"]
    assert isinstance(runs, list)
    runs[0]["status"] = "running"
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(PublicDemoExportError, match="20 completed runs"):
        package_research_release_evidence(
            destination,
            registry=_StaticRegistry(incomplete),
            matrix_id=MATRIX_ID,
            selection_evidence_path=selections,
            heldout_run_dir=heldout,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_research_packager_rejects_a_tampered_selected_checkpoint(
    tmp_path: Path,
) -> None:
    registry, selections, heldout, selected_sources = _canonical_inputs(tmp_path)
    selected_sources[sorted(selected_sources)[0]].write_bytes(b"tampered")

    with pytest.raises(PublicDemoExportError, match="checkpoint hash mismatch"):
        package_research_release_evidence(
            tmp_path / "output",
            registry=registry,
            matrix_id=MATRIX_ID,
            selection_evidence_path=selections,
            heldout_run_dir=heldout,
        )


def _canonical_inputs(
    tmp_path: Path,
) -> tuple[_StaticRegistry, Path, Path, dict[str, Path]]:
    root = tmp_path / "inputs"
    _research_bundle(root)
    matrix = _read_object(root / "matrix.json")
    matrix["created_at"] = "2026-08-21T00:00:00+00:00"
    runs = matrix["runs"]
    assert isinstance(runs, list)
    for ordinal, run in enumerate(runs):
        run["ordinal"] = ordinal
        run["created_at"] = matrix["created_at"]
        config = run["config"]
        assert isinstance(config, dict)
        config["seed"] = config["training_seed"]
        config["source_tree_digest"] = SOURCE.tree_digest
        config["output_root"] = rf"C:\Users\fixture\training\{run['run_key']}"

    selection_path = root / "selections.json"
    selections = _read_object(selection_path)
    selected_sources: dict[str, Path] = {}
    for item in selections["selections"]:
        item["matrix_id"] = MATRIX_ID
        run_key = item["run_key"]
        source = root / "private" / run_key / "selected.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"selected policy {run_key}\n".encode())
        selected_sources[run_key] = source
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        selected = item["selected"]
        selected["checkpoint_path"] = str(source)
        selected["checkpoint_sha256"] = digest
        selected["checkpoint_lineage"] = [
            rf"C:\Users\fixture\training\{run_key}\checkpoint_100pct.pt"
        ]
        chosen_id = selected["checkpoint_id"]
        for candidate in item["candidates"]:
            result = candidate["result"]
            if result["checkpoint_id"] != chosen_id:
                continue
            result["checkpoint_path"] = str(source)
            result["checkpoint_sha256"] = digest
            result["checkpoint_lineage"] = list(selected["checkpoint_lineage"])
            for episode in candidate["episodes"]:
                episode["checkpoint_path"] = str(source)
                episode["checkpoint_sha256"] = digest
                episode["checkpoint_lineage"] = list(selected["checkpoint_lineage"])
    for item in selections["selections"]:
        run_key = item["run_key"]
        evidence_path = root / "validation" / run_key / "validation_selection.json"
        evidence_path.parent.mkdir(parents=True)
        item["evidence_path"] = str(evidence_path)
        _write_json(evidence_path, item)
    _write_json(selection_path, selections)

    evaluation_path = root / "evaluation.json"
    evaluation = _read_object(evaluation_path)
    selected_by_identity = {
        (item["selected"]["experiment_variant"], item["selected"]["training_seed"]): item[
            "selected"
        ]
        for item in selections["selections"]
    }
    for episode in evaluation["episodes"]:
        if episode["controller"] not in {"mappo", "ippo"}:
            continue
        selected = selected_by_identity[(episode["experiment_variant"], episode["training_seed"])]
        episode["checkpoint_path"] = selected["checkpoint_path"]
        episode["checkpoint_sha256"] = selected["checkpoint_sha256"]
        episode["checkpoint_lineage"] = list(selected["checkpoint_lineage"])
    _write_json(evaluation_path, evaluation)
    suite = EvaluationSuite.model_validate(evaluation)
    rows = [episode.model_dump(mode="json") for episode in suite.episodes]
    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "report.md").write_text(
        render_evaluation_report(suite),
        encoding="utf-8",
        newline="\n",
    )
    audited_paths = [
        selection_path,
        evaluation_path,
        root / "episodes.csv",
        root / "report.md",
        root / "acceptance.json",
        root / "ablations.json",
        root / "release_completeness.json",
        *selected_sources.values(),
        *(Path(item["evidence_path"]) for item in selections["selections"]),
    ]
    reproducibility = _read_object(root / "reproducibility_audit.json")
    reproducibility["checked_file_count"] = len(audited_paths)
    reproducibility["file_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in audited_paths
    }
    _write_json(root / "reproducibility_audit.json", reproducibility)
    frozen = [
        {"run_key": item["run_key"], "selection": item["selected"]}
        for item in selections["selections"]
    ]
    return _StaticRegistry(matrix, frozen), selection_path, root, selected_sources


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _rehash_artifact(descriptor_path: Path, role: str) -> None:
    descriptor = _read_object(descriptor_path)
    artifacts = descriptor["artifacts"]
    assert isinstance(artifacts, list)
    artifact = next(item for item in artifacts if item["role"] == role)
    source = descriptor_path.parent / artifact["path"]
    artifact["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_json(descriptor_path, descriptor)
