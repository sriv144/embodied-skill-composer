"""Cheap research-only archive gate tests; no native scene or live simulator fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embodied_skill_composer.construction import public_demo_provenance as provenance
from embodied_skill_composer.construction.experiment_execution import audit_primary_acceptance
from embodied_skill_composer.construction.research_training_curves import CURVE_PREFIX, INDEX_ROLE
from tests.test_construction_public_demo_provenance import (
    MATRIX_ID,
    PROTOCOL,
    SOURCE,
    _deterministic_bundle,
    _research_bundle,
    _rewrite_evaluation_artifacts,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_bytes(provenance._json_bytes(payload))


def _rebind_public_hashes(root: Path) -> None:
    """Model an internally consistent repackaging, rather than a trivial hash error."""
    path = root / "provenance.json"
    payload = _read(path)
    for item in payload["artifacts"]:
        data = (root / item["path"]).read_bytes()
        item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    payload["artifact_count"] = len(payload["artifacts"])
    payload["manifest_payload_sha256"] = None
    payload["manifest_payload_sha256"] = hashlib.sha256(provenance._json_bytes(payload)).hexdigest()
    _write(path, payload)


@pytest.fixture
def exported_research(tmp_path: Path) -> Path:
    root = tmp_path / "public"
    provenance.export_public_demo_bundle(
        root,
        deterministic_bundle=_deterministic_bundle(
            tmp_path / "deterministic", status="fixture", complete=True,
        ),
        research_bundle=_research_bundle(tmp_path / "research"),
        source=SOURCE,
        channel="preview",
    )
    # This fixture exercises only the research part of downloaded public-demo
    # verification. Native simulator semantics remain an independent release-
    # asset gate; no mock metadata here is real live evidence or published data.
    payload = _read(root / "provenance.json")
    payload.update(channel="release", release_version="1.0.0", release_tag="v1.0.0")
    for item in payload["inputs"]:
        item["evidence_status"] = "canonical"
    payload["inputs"].append({
        "kind": "simulator", "manifest_name": "simulator-bundle.json",
        "manifest_sha256": "c" * 64, "evidence_status": "canonical",
        "source": SOURCE.model_dump(mode="json"), "created_at": "2026-08-21T00:00:00Z",
        "configuration_digests": [], "protocol_digest": None, "profile": None, "matrix_id": None,
    })
    _write(root / "provenance.json", payload)
    _write(root / "release-status.json", {
        "schema_version": "construction-intelligence-public-status-v1",
        "channel": "release", "release_ready": True, "release_version": "1.0.0",
        "release_tag": "v1.0.0", "claim_status": "release_evidence", "limitations": [],
        "evidence": {"deterministic": "canonical", "research": "validated", "coppelia": "validated"},
    })
    _rebind_public_hashes(root)
    return root


def test_downloaded_release_recomputes_research_without_original_descriptors(exported_research):
    root = exported_research
    assert not (root / "research-bundle.json").exists()
    # A zero reduction is calculated from the validation candidates. Omitting
    # those candidates would instead produce None and must not pass this gate.
    assert _read(root / "evidence/research/ablations.json")[0]["transitions_to_95_reduction"] == 0.0
    assert provenance.verify_public_demo_export(root, expected_channel="release")["channel"] == "release"


@pytest.mark.parametrize("index,field,value", [
    (0, "supported", True), (0, "completion_gain", 0.5), (0, "transitions_to_95_reduction", 0.9),
    (0, "completion_boundary_met", True), (0, "transition_boundary_met", True),
    (0, "interpretation", "Behavior cloning is supported by invented evidence."),
    (1, "supported", True), (1, "failure_completion_gain", 0.2),
    (1, "no_failure_completion_delta", -0.5), (1, "failure_boundary_met", True),
    (1, "no_failure_safety_boundary_met", False),
])
def test_downloaded_release_rejects_rehashed_false_ablation_claims(exported_research, index, field, value):
    root = exported_research
    path = root / "evidence/research/ablations.json"
    payload = _read(path)
    payload[index][field] = value
    _write(path, payload)
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="declared ablation decisions"):
        provenance.verify_public_demo_export(root)


@pytest.mark.parametrize("mutation", ["duplicate-decision", "unsupported-summary"])
def test_ablation_membership_and_release_summary_cannot_disagree(exported_research, mutation):
    root = exported_research
    if mutation == "duplicate-decision":
        path = root / "evidence/research/ablations.json"
        payload = _read(path)
        payload.append(payload[0])
        expected = "declared ablation decisions"
    else:
        path = root / "evidence/research/release_completeness.json"
        payload = _read(path)
        payload["unsupported_ablation_hypotheses"] = []
        expected = "release-completeness decision"
    _write(path, payload)
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match=expected):
        provenance.verify_public_demo_export(root)


def test_downloaded_release_rejects_rehashed_false_primary_acceptance(exported_research):
    root = exported_research
    path = root / "evidence/research/acceptance.json"
    payload = _read(path)
    payload["results"][0]["observed"] = 0.5
    _write(path, payload)
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="declared acceptance"):
        provenance.verify_public_demo_export(root)


def test_downloaded_release_rejects_failing_primary_even_with_honest_rebound_audit(exported_research):
    root = exported_research
    research = root / "evidence/research"
    payload = _read(research / "evaluation.json")
    for episode in payload["episodes"]:
        if episode["experiment_variant"] == "mappo_full" and not episode["failure_enabled"]:
            episode["structure_completion_rate"] = 0.0
    suite = _rewrite_evaluation_artifacts(research, payload)
    _write(research / "acceptance.json", audit_primary_acceptance(suite, PROTOCOL).model_dump(mode="json"))
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="primary research thresholds failed"):
        provenance.verify_public_demo_export(root)


@pytest.mark.parametrize("mutation", ["legacy", "missing-curve", "truncated-curve", "index-identity"])
def test_downloaded_release_requires_all_twenty_full_training_curves(exported_research, mutation):
    root = exported_research
    path = root / "provenance.json"
    payload = _read(path)
    if mutation == "legacy":
        removed = [item for item in payload["artifacts"]
                   if item["role"] == INDEX_ROLE or item["role"].startswith(CURVE_PREFIX)]
        payload["artifacts"] = [item for item in payload["artifacts"] if item not in removed]
        for item in removed:
            (root / item["path"]).unlink()
        _write(path, payload)
        expected = "twenty per-update training curves"
    elif mutation == "missing-curve":
        item = next(item for item in payload["artifacts"] if item["role"].startswith(CURVE_PREFIX))
        payload["artifacts"].remove(item)
        (root / item["path"]).unlink()
        _write(path, payload)
        expected = "membership is incomplete"
    elif mutation == "truncated-curve":
        item = next(item for item in payload["artifacts"] if item["role"].startswith(CURVE_PREFIX))
        curve = root / item["path"]
        curve.write_bytes(curve.read_bytes().rsplit(b"\r\n", 2)[0] + b"\r\n")
        expected = "truncated before"
    else:
        index_path = root / "evidence/research/training-curves.json"
        index = _read(index_path)
        index["matrix_id"] = "other-matrix"
        _write(index_path, index)
        expected = "index identity/semantics"
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match=expected):
        provenance.verify_public_demo_export(root)


@pytest.mark.parametrize("relative", [
    "research-summary.json", "experiment-matrices.json", "evaluations.json",
    f"experiment-matrices/{MATRIX_ID}/selections.json", "policies.json",
])
def test_downloaded_release_rejects_generated_views_that_disagree_with_evidence(exported_research, relative):
    root = exported_research
    _write(root / relative, {"plausible_but_unverified": True})
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="view does not match canonical"):
        provenance.verify_public_demo_export(root)


@pytest.mark.parametrize("mutation", ["unregistered", "duplicate"])
def test_downloaded_release_rejects_extra_learned_policy_claims(exported_research, mutation):
    root = exported_research
    path = root / "policies.json"
    policies = _read(path)
    extra = dict(policies[0])
    if mutation == "unregistered":
        extra["id"] = "not-selected-by-validation"
    policies.append(extra)
    _write(path, policies)
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="view does not match canonical"):
        provenance.verify_public_demo_export(root)


@pytest.mark.parametrize("mutation", ["duplicate-role", "unknown-role", "wrong-target", "missing-created-at"])
def test_exported_research_contract_must_be_complete_and_unambiguous(exported_research, mutation):
    root = exported_research
    path = root / "provenance.json"
    payload = _read(path)
    records = [item for item in payload["artifacts"] if item["source_kind"] == "research"]
    if mutation == "duplicate-role":
        records[1]["role"] = records[0]["role"]
    elif mutation == "unknown-role":
        records[0]["role"] = "unrecognized_research_role"
    elif mutation == "wrong-target":
        records[0]["role"], records[1]["role"] = records[1]["role"], records[0]["role"]
    else:
        next(item for item in payload["inputs"] if item["kind"] == "research").pop("created_at")
    _write(path, payload)
    _rebind_public_hashes(root)
    with pytest.raises(provenance.PublicDemoExportError, match="exported research contract|passive public contract"):
        provenance.verify_public_demo_export(root)
