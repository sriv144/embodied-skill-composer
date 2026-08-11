from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaExecutor,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5FullCottageRunner,
    prepare_phase5_physical_yard,
    write_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.evaluation import (
    ControllerEvaluation,
    EpisodeEvaluation,
    EvaluationSuite,
    MetricSummary,
    render_evaluation_report,
    summarize_by_scenario_seed,
    summarize_by_training_seed,
    summarize_evaluations,
)
from embodied_skill_composer.construction.experiment_execution import (
    audit_primary_acceptance,
)
from embodied_skill_composer.construction.experiment_protocol import (
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.public_demo_provenance import (
    BUNDLE_SCHEMA_VERSION,
    PublicDemoExportError,
    SourceIdentity,
    export_public_demo_bundle,
    verify_public_demo_export,
    verify_public_demo_regeneration_identity,
    _recompute_controller_summaries,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
)
from tests.test_construction_dynamic_coppelia import FakeDynamicClient

from scripts.export_construction_public_demo import export_public_demo


SOURCE = SourceIdentity(
    commit="a" * 40,
    dirty=False,
    tree_digest=hashlib.sha256(b"").hexdigest(),
)
PROTOCOL = load_experiment_protocol()
PROTOCOL_DIGEST = protocol_digest(PROTOCOL)
RELEASE_VERSION = "0.1.0"
RELEASE_TAG = f"v{RELEASE_VERSION}"
MATRIX_ID = "construction-intelligence-v1-research"
VARIANTS = (
    ("mappo_full", "mappo"),
    ("ippo_full", "ippo"),
    ("mappo_no_bc", "mappo"),
    ("mappo_no_failure_curriculum", "mappo"),
)


def test_preview_regeneration_is_stable_and_labels_absent_evidence(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "inputs",
        status="fixture",
        complete=False,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    export_public_demo_bundle(
        first,
        deterministic_bundle=deterministic,
        source=SOURCE,
    )
    export_public_demo_bundle(
        second,
        deterministic_bundle=deterministic,
        source=SOURCE,
    )

    assert _directory_bytes(first) == _directory_bytes(second)
    status = _read_json(first / "release-status.json")
    assert status["claim_status"] == "fixture_preview"
    assert status["release_ready"] is False
    assert status["release_version"] is None
    assert status["release_tag"] is None
    assert status["evidence"] == {
        "coppelia": "absent",
        "deterministic": "fixture",
        "research": "absent",
    }
    assert _read_json(first / "research-summary.json")["claim_allowed"] is False
    assert _read_json(first / "coppelia-evidence.json")["claim_allowed"] is False


def test_real_default_preview_export_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    canonical_robot = (
        Path(__file__).resolve().parents[1]
        / "workbench"
        / "public"
        / "demo"
        / "construction_robot.glb"
    )

    export_public_demo(first, source=SOURCE)
    export_public_demo(second, source=SOURCE)

    assert _directory_bytes(first) == _directory_bytes(second)
    assert b"\r\n" not in (first / "report.md").read_bytes()
    assert (first / "construction_robot.glb").read_bytes() == (
        canonical_robot.read_bytes()
    )
    assert verify_public_demo_regeneration_identity(first, second) == (
        verify_public_demo_export(first)
    )


def test_release_recomputes_primary_acceptance_from_episodes(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    evaluation_path = tmp_path / "research" / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    assert isinstance(evaluation, dict)
    episodes = evaluation["episodes"]
    assert isinstance(episodes, list)
    for episode in episodes:
        assert isinstance(episode, dict)
        if (
            episode["experiment_variant"] == "mappo_full"
            and episode["failure_enabled"] is False
        ):
            episode["structure_completion_rate"] = 0.0
    _rewrite_evaluation_artifacts(tmp_path / "research", evaluation)
    acceptance_path = tmp_path / "research" / "acceptance.json"
    acceptance = _read_json(acceptance_path)
    assert isinstance(acceptance, dict)
    results = acceptance["results"]
    assert isinstance(results, list)
    mappo_nominal = next(
        item
        for item in results
        if isinstance(item, dict)
        and item["name"] == "mappo_no_failure_mean_completion"
    )
    mappo_nominal["observed"] = 0.0
    _write_json(acceptance_path, acceptance)
    _rehash_research_files(
        research,
        "evaluation",
        "episodes",
        "report",
        "acceptance",
    )

    with pytest.raises(
        PublicDemoExportError,
        match="declared acceptance does not match recomputed",
    ):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
        )


def test_release_recomputes_all_published_evaluation_summaries(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    evaluation_path = tmp_path / "research" / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    assert isinstance(evaluation, dict)
    summaries = evaluation["summaries"]
    assert isinstance(summaries, list)
    summary = summaries[0]
    assert isinstance(summary, dict)
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    completion = metrics["structure_completion_rate"]
    assert isinstance(completion, dict)
    completion["bootstrap_ci95_high"] = 0.123
    _write_json(evaluation_path, evaluation)
    _rehash_research_files(research, "evaluation")

    with pytest.raises(
        PublicDemoExportError,
        match="published evaluation summaries do not match",
    ):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
        )


def test_fast_release_summary_recomputation_matches_canonical_bootstrap() -> None:
    episodes = [
        EpisodeEvaluation.model_validate(
            _episode(
                seed=scenario_seed,
                failure=False,
                controller="mappo",
                variant="mappo_full",
                training_seed=training_seed,
                configuration_digest=hashlib.sha256(
                    f"seed-{training_seed}".encode()
                ).hexdigest(),
                structure_completion_rate=(
                    (training_seed - 7) * 5 + scenario_seed - 900
                )
                / 24,
                makespan_s=float(training_seed * 10 + scenario_seed - 899),
                policy_id=f"policy-{training_seed}",
                transition_count=1_500_000,
                checkpoint_path=f"release/{training_seed}.pt",
                checkpoint_sha256=hashlib.sha256(
                    f"checkpoint-{training_seed}".encode()
                ).hexdigest(),
                checkpoint_lineage=[f"origin-{training_seed}"],
            )
        )
        for training_seed in range(7, 12)
        for scenario_seed in range(900, 905)
    ]

    assert _recompute_controller_summaries(episodes) == summarize_evaluations(
        episodes
    )


def test_release_requires_episode_csv_to_match_evaluation_json(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    csv_path = tmp_path / "research" / "episodes.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["scenario_id"] = "forged-with-same-row-count"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _rehash_research_files(research, "episodes")

    with pytest.raises(
        PublicDemoExportError,
        match="episode CSV does not exactly match",
    ):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
        )


def test_release_recomputes_validation_checkpoint_ranking(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    selections_path = tmp_path / "research" / "selections.json"
    payload = _read_json(selections_path)
    assert isinstance(payload, dict)
    selections = payload["selections"]
    assert isinstance(selections, list)
    record = selections[0]
    assert isinstance(record, dict)
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    forged_result = candidates[-2]["result"]
    assert isinstance(forged_result, dict)
    selected = record["selected"]
    assert isinstance(selected, dict)
    forged_id = str(forged_result["checkpoint_id"])
    ranking = [
        forged_id,
        *[
            str(candidate["result"]["checkpoint_id"])
            for candidate in reversed(candidates)
            if candidate["result"]["checkpoint_id"] != forged_id
        ],
    ]
    record["selected"] = {
        **forged_result,
        "selection_rule": selected["selection_rule"],
        "required_checkpoint_fractions": selected[
            "required_checkpoint_fractions"
        ],
        "candidate_ranking": ranking,
    }
    _write_json(selections_path, payload)
    _rehash_research_files(research, "selections")

    with pytest.raises(
        PublicDemoExportError,
        match="does not match deterministic validation ranking",
    ):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
        )


def test_release_ties_heldout_episodes_to_selected_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    evaluation_path = tmp_path / "research" / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    assert isinstance(evaluation, dict)
    episodes = evaluation["episodes"]
    assert isinstance(episodes, list)
    learned = next(
        episode
        for episode in episodes
        if isinstance(episode, dict)
        and episode["experiment_variant"] == "mappo_full"
    )
    learned["checkpoint_lineage"] = ["forged-lineage"]
    _rewrite_evaluation_artifacts(tmp_path / "research", evaluation)
    _rehash_research_files(research, "evaluation", "episodes", "report")

    with pytest.raises(
        PublicDemoExportError,
        match="does not match its frozen selected checkpoint lineage",
    ):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
        )


def test_release_export_covers_every_emitted_artifact_and_round_trips(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    simulator = _simulator_bundle(tmp_path / "simulator")
    output = tmp_path / "public"

    manifest = export_public_demo_bundle(
        output,
        deterministic_bundle=deterministic,
        research_bundle=research,
        simulator_bundle=simulator,
        source=SOURCE,
        channel="release",
        release_version=RELEASE_VERSION,
        release_tag=RELEASE_TAG,
    )
    verified = verify_public_demo_export(
        output,
        expected_channel="release",
        expected_source=SOURCE,
        expected_release_version=RELEASE_VERSION,
        expected_release_tag=RELEASE_TAG,
    )

    assert verified == manifest
    assert manifest["channel"] == "release"
    assert manifest["release_version"] == RELEASE_VERSION
    assert manifest["release_tag"] == RELEASE_TAG
    assert manifest["artifact_count"] == len(manifest["artifacts"])
    emitted = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "provenance.json"
    }
    recorded = {item["path"] for item in manifest["artifacts"]}
    assert recorded == emitted
    assert all(item["bytes"] > 0 for item in manifest["artifacts"])
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert _read_json(output / "release-status.json")["release_ready"] is True
    research_summary = _read_json(output / "research-summary.json")
    assert research_summary["status"] == "validated"
    assert research_summary["confidence_intervals"]
    assert research_summary["per_training_seed"]
    assert research_summary["per_scenario_seed"]
    assert len(research_summary["learning_curves"]) == 20
    assert research_summary["acceptance"]["passed"] is True
    assert len(research_summary["ablations"]) == 2
    research_references = research_summary["artifact_references"]
    assert research_references
    assert all(
        item["path"].startswith("evidence/research/")
        and Path(item["path"]).suffix
        for item in research_references
    )
    coppelia_summary = _read_json(output / "coppelia-evidence.json")
    assert coppelia_summary["status"] == "validated"
    assert coppelia_summary["ready"] is True
    for scenario in ("nominal", "recovery"):
        references = coppelia_summary[scenario]["artifact_references"]
        assert references
        assert all(
            item["path"].startswith(f"evidence/coppelia/{scenario}/")
            and Path(item["path"]).suffix
            for item in references
        )
    assert len(_read_json(output / "experiment-matrices.json")) == 1
    assert len(_read_json(output / "policies.json")) == 20


def test_input_and_output_tampering_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    deterministic = _deterministic_bundle(
        root,
        status="fixture",
        complete=False,
    )
    (root / "project.json").write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(PublicDemoExportError, match="artifact hash mismatch"):
        export_public_demo_bundle(
            tmp_path / "rejected",
            deterministic_bundle=deterministic,
            source=SOURCE,
        )

    deterministic = _deterministic_bundle(
        tmp_path / "clean-inputs",
        status="fixture",
        complete=False,
    )
    output = tmp_path / "output"
    export_public_demo_bundle(
        output,
        deterministic_bundle=deterministic,
        source=SOURCE,
    )
    (output / "project.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(PublicDemoExportError, match="artifact integrity"):
        verify_public_demo_export(output)


@pytest.mark.parametrize(
    "commit",
    [
        "a" * 7,
        "A" * 40,
        "g" * 40,
        "a" * 41,
    ],
)
def test_source_identity_rejects_noncanonical_git_commits(commit: str) -> None:
    with pytest.raises(ValueError, match="commit"):
        SourceIdentity(
            commit=commit,
            dirty=False,
            tree_digest=hashlib.sha256(b"tree").hexdigest(),
        )

    assert SourceIdentity(
        commit="b" * 64,
        dirty=False,
        tree_digest=hashlib.sha256(b"tree").hexdigest(),
    ).commit == "b" * 64


def test_bundle_descriptor_requires_timezone_aware_created_at(
    tmp_path: Path,
) -> None:
    descriptor_path = _deterministic_bundle(
        tmp_path / "inputs",
        status="fixture",
        complete=False,
    )
    descriptor = _read_json(descriptor_path)
    descriptor["created_at"] = "2026-07-26T00:00:00"
    _write_json(descriptor_path, descriptor)

    with pytest.raises(PublicDemoExportError, match="include a timezone"):
        export_public_demo_bundle(
            tmp_path / "output",
            deterministic_bundle=descriptor_path,
            source=SOURCE,
        )


@pytest.mark.parametrize(
    ("kind", "role", "target", "media_type"),
    [
        (
            "research",
            "report",
            "evidence/research/report.html",
            "text/html",
        ),
        (
            "simulator",
            "nominal_scene",
            "evidence/coppelia/nominal/construction_intelligence.svg",
            "image/svg+xml",
        ),
        (
            "research",
            "matrix",
            "evidence/research/matrix.json",
            "text/html",
        ),
    ],
)
def test_public_evidence_rejects_active_targets_and_media_types(
    tmp_path: Path,
    kind: Literal["research", "simulator"],
    role: str,
    target: str,
    media_type: str,
) -> None:
    root = tmp_path / kind
    root.mkdir(parents=True)
    source_path = root / "payload.bin"
    source_path.write_bytes(b"passive evidence fixture")
    descriptor = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": kind,
        "evidence_status": "canonical",
        "source": SOURCE.model_dump(mode="json"),
        "created_at": "2026-07-26T00:00:00Z",
        "configuration_digests": [hashlib.sha256(b"config").hexdigest()],
        "protocol_digest": PROTOCOL_DIGEST if kind == "research" else None,
        "profile": "research" if kind == "research" else None,
        "matrix_id": "construction_intelligence_v1-research"
        if kind == "research"
        else None,
        "artifacts": [
            {
                "role": role,
                "path": source_path.name,
                "target": target,
                "sha256": _sha256(source_path),
                "media_type": media_type,
            }
        ],
    }
    descriptor_path = root / "bundle.json"
    _write_json(descriptor_path, descriptor)
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="fixture",
        complete=False,
    )

    with pytest.raises(PublicDemoExportError, match="passive public contract"):
        export_public_demo_bundle(
            tmp_path / "output",
            deterministic_bundle=deterministic,
            research_bundle=descriptor_path if kind == "research" else None,
            simulator_bundle=descriptor_path if kind == "simulator" else None,
            source=SOURCE,
        )


def test_channel_specific_release_identity_is_fail_closed(tmp_path: Path) -> None:
    descriptor_path = _deterministic_bundle(
        tmp_path / "inputs",
        status="fixture",
        complete=False,
    )
    with pytest.raises(ValueError, match="require both"):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=descriptor_path,
            source=SOURCE,
            channel="release",
        )
    with pytest.raises(ValueError, match="must not declare"):
        export_public_demo_bundle(
            tmp_path / "preview",
            deterministic_bundle=descriptor_path,
            source=SOURCE,
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_release_requires_a_deterministic_configuration_digest(
    tmp_path: Path,
) -> None:
    descriptor_path = _deterministic_bundle(
        tmp_path / "inputs",
        status="canonical",
        complete=True,
    )
    descriptor = _read_json(descriptor_path)
    descriptor["configuration_digests"] = []
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        PublicDemoExportError,
        match="at least one configuration digest",
    ):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=descriptor_path,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_regeneration_identity_compares_two_verified_exports(
    tmp_path: Path,
) -> None:
    descriptor = _deterministic_bundle(
        tmp_path / "same-input",
        status="fixture",
        complete=False,
    )
    first = tmp_path / "first"
    same = tmp_path / "same"
    export_public_demo_bundle(first, deterministic_bundle=descriptor, source=SOURCE)
    export_public_demo_bundle(same, deterministic_bundle=descriptor, source=SOURCE)

    assert verify_public_demo_regeneration_identity(first, same) == (
        verify_public_demo_export(first)
    )

    changed_descriptor = _deterministic_bundle(
        tmp_path / "changed-input",
        status="fixture",
        complete=False,
    )
    changed_project = tmp_path / "changed-input" / "project.json"
    _write_json(changed_project, {"role": "project", "revision": 2})
    changed_manifest = _read_json(changed_descriptor)
    project_record = next(
        item
        for item in changed_manifest["artifacts"]
        if item["role"] == "project"
    )
    project_record["sha256"] = _sha256(changed_project)
    _write_json(changed_descriptor, changed_manifest)
    changed = tmp_path / "changed"
    export_public_demo_bundle(
        changed,
        deterministic_bundle=changed_descriptor,
        source=SOURCE,
    )

    with pytest.raises(PublicDemoExportError, match="regeneration identity mismatch"):
        verify_public_demo_regeneration_identity(first, changed)


def test_release_refuses_an_incomplete_research_matrix(tmp_path: Path) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    matrix_path = tmp_path / "research" / "matrix.json"
    matrix = _read_json(matrix_path)
    matrix["runs"].pop()
    _write_json(matrix_path, matrix)
    descriptor = _read_json(research)
    matrix_artifact = next(
        item for item in descriptor["artifacts"] if item["role"] == "matrix"
    )
    matrix_artifact["sha256"] = _sha256(matrix_path)
    _write_json(research, descriptor)

    with pytest.raises(
        PublicDemoExportError,
        match="complete 20-run research matrix",
    ):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_release_refuses_internal_source_commit_mismatch(tmp_path: Path) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    descriptor = _read_json(research)
    descriptor["source"]["commit"] = "b" * 40
    _write_json(research, descriptor)

    with pytest.raises(PublicDemoExportError, match="run provenance mismatch"):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=deterministic,
            research_bundle=research,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_release_refuses_a_declared_simulator_gate_failure(tmp_path: Path) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    simulator = _simulator_bundle(tmp_path / "simulator")
    nominal_path = tmp_path / "simulator" / "nominal" / "metrics.json"
    nominal = _read_json(nominal_path)
    nominal["metrics"]["collision_stops"] = 1
    _write_json(nominal_path, nominal)
    phase5_manifest_path = tmp_path / "simulator" / "nominal" / "manifest.json"
    phase5_manifest = _read_json(phase5_manifest_path)
    metrics_record = next(
        item
        for item in phase5_manifest["artifacts"]
        if item["path"] == "metrics.json"
    )
    metrics_record["sha256"] = _sha256(nominal_path)
    metrics_record["bytes"] = nominal_path.stat().st_size
    _write_json(phase5_manifest_path, phase5_manifest)
    descriptor = _read_json(simulator)
    metrics_artifact = next(
        item
        for item in descriptor["artifacts"]
        if item["role"] == "nominal_metrics"
    )
    metrics_artifact["sha256"] = _sha256(nominal_path)
    manifest_artifact = next(
        item
        for item in descriptor["artifacts"]
        if item["role"] == "nominal_manifest"
    )
    manifest_artifact["sha256"] = _sha256(phase5_manifest_path)
    _write_json(simulator, descriptor)

    with pytest.raises(
        PublicDemoExportError,
        match="bundle verification failed.*collision stop",
    ):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=deterministic,
            research_bundle=research,
            simulator_bundle=simulator,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_release_rejects_rehashed_non_coppelia_scene_bytes(tmp_path: Path) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    simulator = _simulator_bundle(tmp_path / "simulator")
    scene_path = (
        tmp_path
        / "simulator"
        / "nominal"
        / "construction_intelligence.ttt"
    )
    scene_path.write_bytes(b"arbitrary bytes with freshly updated hashes")
    _rehash_native_phase5_file(
        simulator,
        prefix="nominal",
        file_name="construction_intelligence.ttt",
    )

    with pytest.raises(
        PublicDemoExportError,
        match="bundle verification failed.*scene",
    ):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=deterministic,
            research_bundle=research,
            simulator_bundle=simulator,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def test_release_rejects_rehashed_collision_booleans_without_raw_rounds(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_bundle(
        tmp_path / "deterministic",
        status="canonical",
        complete=True,
    )
    research = _research_bundle(tmp_path / "research")
    simulator = _simulator_bundle(tmp_path / "simulator")
    trace_path = tmp_path / "simulator" / "nominal" / "trace.json"
    trace = _read_json(trace_path)
    assert isinstance(trace, list)
    _write_json(
        trace_path,
        [
            item
            for item in trace
            if not isinstance(item, dict)
            or item.get("event") != "collision_query_round"
        ],
    )
    _rehash_native_phase5_file(
        simulator,
        prefix="nominal",
        file_name="trace.json",
    )

    with pytest.raises(
        PublicDemoExportError,
        match="bundle verification failed.*collision trace",
    ):
        export_public_demo_bundle(
            tmp_path / "release",
            deterministic_bundle=deterministic,
            research_bundle=research,
            simulator_bundle=simulator,
            source=SOURCE,
            channel="release",
            release_version=RELEASE_VERSION,
            release_tag=RELEASE_TAG,
        )


def _deterministic_bundle(
    root: Path,
    *,
    status: Literal["fixture", "canonical"],
    complete: bool,
) -> Path:
    root.mkdir(parents=True)
    role_targets = {
        "project": "project.json",
        "report": "report.md",
    }
    if complete:
        role_targets.update(
            {
                "scenarios": "scenarios.json",
                "policies": "policies.json",
                "runs": "runs.json",
                "house": "house.glb",
                "robot": "construction_robot.glb",
                "trace_sequential": "traces/sequential.json",
                "trace_greedy": "traces/greedy.json",
                "trace_optimized": "traces/optimized.json",
                "trace_recovery": "traces/recovery.json",
            }
        )
    for role, target in role_targets.items():
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            _write_json(path, [] if role in {"scenarios", "policies", "runs"} else {"role": role})
        else:
            path.write_bytes(f"canonical {role}\n".encode())
    return _write_bundle(
        root,
        kind="deterministic",
        status=status,
        role_targets=role_targets,
        configuration_digests=[hashlib.sha256(b"cottage-v1").hexdigest()],
    )


def _research_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    digests: list[str] = []
    for variant, algorithm in VARIANTS:
        for seed in range(7, 12):
            run_key = f"{variant}-seed-{seed}"
            digest = hashlib.sha256(run_key.encode()).hexdigest()
            digests.append(digest)
            runs.append(
                {
                    "id": f"run-{run_key}",
                    "run_key": run_key,
                    "status": "completed",
                    "config": {
                        "profile": "research",
                        "experiment_id": "construction_intelligence_v1",
                        "experiment_variant": variant,
                        "training_seed": seed,
                        "algorithm": algorithm,
                        "source_commit": SOURCE.commit,
                        "source_dirty": False,
                        "configuration_digest": digest,
                        "transitions": 1_500_000,
                        "checkpoint_fractions": [0.1, 0.25, 0.5, 0.75, 1.0],
                    },
                }
            )
            candidates: list[dict[str, object]] = []
            candidate_results: list[dict[str, object]] = []
            for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
                checkpoint = _checkpoint_identity(run_key, fraction)
                transition_count = int(1_500_000 * fraction)
                completion = float(fraction)
                makespan = 200.0 - 100.0 * completion
                result = {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "experiment_id": "construction_intelligence_v1",
                    "experiment_variant": variant,
                    "training_seed": seed,
                    "checkpoint_fraction": fraction,
                    "transition_count": transition_count,
                    "split": "validation",
                    "scenario_seeds": [800, 801, 802, 803, 804],
                    "mean_completion_rate": completion,
                    "mean_makespan_s": makespan,
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "checkpoint_lineage": checkpoint["checkpoint_lineage"],
                    "configuration_digest": digest,
                    "source_commit": SOURCE.commit,
                    "resume_provenance": {},
                }
                candidate_results.append(result)
                candidates.append(
                    {
                        "result": result,
                        "episodes": [
                            _episode(
                                seed=scenario_seed,
                                failure=failure,
                                controller=algorithm,
                                variant=variant,
                                training_seed=seed,
                                configuration_digest=digest,
                                split="validation",
                                structure_completion_rate=completion,
                                makespan_s=makespan,
                                policy_id=str(checkpoint["checkpoint_id"]),
                                transition_count=transition_count,
                                checkpoint_path=str(checkpoint["checkpoint_path"]),
                                checkpoint_sha256=str(
                                    checkpoint["checkpoint_sha256"]
                                ),
                                checkpoint_lineage=list(
                                    checkpoint["checkpoint_lineage"]
                                ),
                            )
                            for scenario_seed in range(800, 805)
                            for failure in (False, True)
                        ],
                    }
                )
            selected_result = candidate_results[-1]
            selections.append(
                {
                    "run_key": run_key,
                    "candidates": candidates,
                    "selected": {
                        **selected_result,
                        "selection_rule": (
                            "completion_rate_desc,makespan_s_asc,"
                            "transition_count_asc,checkpoint_id_asc"
                        ),
                        "required_checkpoint_fractions": [
                            0.1,
                            0.25,
                            0.5,
                            0.75,
                            1.0,
                        ],
                        "candidate_ranking": [
                            str(item["checkpoint_id"])
                            for item in reversed(candidate_results)
                        ],
                    },
                    "evidence_path": (
                        f"evidence/{run_key}/validation_selection.json"
                    ),
                }
            )
    matrix = {
        "id": MATRIX_ID,
        "protocol_digest": PROTOCOL_DIGEST,
        "execution_profile": "research",
        "expected_run_count": 20,
        "selection_count": 20,
        "status": "selected",
        "runs": runs,
    }
    _write_json(root / "matrix.json", matrix)
    _write_json(
        root / "selections.json",
        {
            "matrix_id": matrix["id"],
            "protocol_digest": PROTOCOL_DIGEST,
            "selections": selections,
            "evidence_path": "evidence/matrix_selections.json",
        },
    )
    suite = _evaluation_suite(runs, selections)
    _write_json(root / "evaluation.json", suite)
    typed_suite = EvaluationSuite.model_validate(suite)
    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = [
            episode.model_dump(mode="json")
            for episode in typed_suite.episodes
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "report.md").write_text(
        render_evaluation_report(typed_suite),
        encoding="utf-8",
    )
    _write_json(
        root / "acceptance.json",
        audit_primary_acceptance(typed_suite, PROTOCOL).model_dump(mode="json"),
    )
    _write_json(
        root / "ablations.json",
        [
            {
                "hypothesis": hypothesis,
                "supported": False,
                "completion_gain": 0.0,
                "transitions_to_95_reduction": 0.0,
                "failure_completion_gain": 0.0,
                "no_failure_completion_delta": 0.0,
                "completion_boundary_met": False,
                "transition_boundary_met": False,
                "failure_boundary_met": False,
                "no_failure_safety_boundary_met": True,
                "interpretation": "Complete evidence does not support the hypothesis.",
            }
            for hypothesis in ("behavior_cloning", "failure_curriculum")
        ],
    )
    _write_json(
        root / "reproducibility_audit.json",
        {
            "matrix_id": matrix["id"],
            "complete": True,
            "checked_file_count": 200,
            "file_hashes": {"fixture": hashlib.sha256(b"fixture").hexdigest()},
            "missing_paths": [],
            "invalid_artifacts": [],
        },
    )
    _write_json(
        root / "release_completeness.json",
        {
            "complete": True,
            "missing_training_run_ids": [],
            "missing_selected_policy_run_ids": [],
            "missing_heldout_evaluation_run_ids": [],
            "missing_ablation_hypotheses": [],
            "unsupported_ablation_hypotheses": [
                "behavior_cloning",
                "failure_curriculum",
            ],
            "blocking_reasons": [],
        },
    )
    role_targets = {
        "matrix": "evidence/research/matrix.json",
        "selections": "evidence/research/selections.json",
        "evaluation": "evidence/research/evaluation.json",
        "episodes": "evidence/research/episodes.csv",
        "report": "evidence/research/report.md",
        "acceptance": "evidence/research/acceptance.json",
        "ablations": "evidence/research/ablations.json",
        "reproducibility_audit": "evidence/research/reproducibility_audit.json",
        "release_completeness": "evidence/research/release_completeness.json",
    }
    source_paths = {role: target.rsplit("/", 1)[-1] for role, target in role_targets.items()}
    return _write_bundle(
        root,
        kind="research",
        status="canonical",
        role_targets=role_targets,
        source_paths=source_paths,
        configuration_digests=digests,
        protocol_digest=PROTOCOL_DIGEST,
        profile="research",
        matrix_id=str(matrix["id"]),
    )


def _evaluation_suite(
    runs: list[dict[str, object]],
    selections: list[dict[str, object]],
) -> dict[str, object]:
    selected_by_run = {
        str(item["run_key"]): item["selected"]
        for item in selections
    }
    episodes: list[dict[str, object]] = []
    for run in runs:
        config = run["config"]
        run_key = str(run["run_key"])
        selected = selected_by_run[run_key]
        assert isinstance(config, dict)
        assert isinstance(selected, dict)
        for seed in range(900, 905):
            for failure in (False, True):
                episodes.append(
                    _episode(
                        seed=seed,
                        failure=failure,
                        controller=str(config["algorithm"]),
                        variant=str(config["experiment_variant"]),
                        training_seed=int(config["training_seed"]),
                        configuration_digest=str(config["configuration_digest"]),
                        policy_id=f"{MATRIX_ID}-{run_key}-selected",
                        transition_count=int(selected["transition_count"]),
                        checkpoint_path=str(selected["checkpoint_path"]),
                        checkpoint_sha256=str(selected["checkpoint_sha256"]),
                        checkpoint_lineage=list(selected["checkpoint_lineage"]),
                    )
                )
    for controller in ("sequential", "greedy", "auction", "cp_sat"):
        for seed in range(900, 905):
            for failure in (False, True):
                episodes.append(
                    _episode(
                        seed=seed,
                        failure=failure,
                        controller=controller,
                    )
                )
    typed_episodes = [
        EpisodeEvaluation.model_validate(episode) for episode in episodes
    ]
    return EvaluationSuite(
        evaluation_id="canonical-heldout",
        seeds=[900, 901, 902, 903, 904],
        controllers=[
            "sequential",
            "greedy",
            "auction",
            "ippo",
            "mappo",
            "cp_sat",
        ],
        episodes=typed_episodes,
        summaries=_fixture_controller_summaries(typed_episodes),
        expected_split="test",
        grid_validation={
            "complete": True,
            "expected_episode_count": 240,
            "observed_episode_count": 240,
            "expected_split": "test",
        },
        per_training_seed=summarize_by_training_seed(typed_episodes),
        per_scenario_seed=summarize_by_scenario_seed(typed_episodes),
    ).model_dump(mode="json")


def _fixture_controller_summaries(
    episodes: list[EpisodeEvaluation],
) -> list[ControllerEvaluation]:
    metric_fields = (
        "structure_completion_rate",
        "makespan_s",
        "total_travel_m",
        "total_energy_wh",
        "idle_robot_seconds",
        "mean_robot_utilization",
        "collision_count",
        "wasted_work_s",
        "invalid_bid_count",
    )
    groups: dict[
        tuple[str, str | None, str | None, bool],
        list[EpisodeEvaluation],
    ] = {}
    for episode in episodes:
        groups.setdefault(
            (
                episode.controller,
                episode.experiment_id,
                episode.experiment_variant,
                episode.failure_enabled,
            ),
            [],
        ).append(episode)
    summaries: list[ControllerEvaluation] = []
    for key in sorted(
        groups,
        key=lambda item: (
            item[3],
            item[0],
            item[1] or "",
            item[2] or "",
        ),
    ):
        controller, experiment_id, experiment_variant, failure_enabled = key
        subset = groups[key]
        metrics: dict[str, MetricSummary] = {}
        for field in metric_fields:
            values = {float(getattr(episode, field)) for episode in subset}
            assert len(values) == 1
            value = values.pop()
            metrics[field] = MetricSummary(
                mean=value,
                std=0.0,
                bootstrap_ci95_low=value,
                bootstrap_ci95_high=value,
                median=value,
            )
        summaries.append(
            ControllerEvaluation(
                controller=controller,
                failure_enabled=failure_enabled,
                episode_count=len(subset),
                metrics=metrics,
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
                training_seed_count=len(
                    {
                        episode.training_seed
                        for episode in subset
                        if episode.training_seed is not None
                    }
                ),
                scenario_seed_count=len({episode.seed for episode in subset}),
            )
        )
    return summaries


def _episode(
    *,
    seed: int,
    failure: bool,
    controller: str,
    variant: str | None = None,
    training_seed: int | None = None,
    configuration_digest: str | None = None,
    split: str = "test",
    structure_completion_rate: float = 1.0,
    makespan_s: float = 100.0,
    policy_id: str | None = None,
    transition_count: int | None = None,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_lineage: list[str] | None = None,
) -> dict[str, object]:
    return {
        "scenario_id": f"heldout-{seed}",
        "seed": seed,
        "split": split,
        "controller": controller,
        "failure_enabled": failure,
        "structure_completion_rate": structure_completion_rate,
        "makespan_s": makespan_s,
        "total_travel_m": 10.0,
        "total_energy_wh": 5.0,
        "idle_robot_seconds": 0.0,
        "mean_robot_utilization": 1.0,
        "collision_count": 0,
        "wasted_work_s": 0.0,
        "invalid_bid_count": 0,
        "drop_count": 0,
        "decision_count": 1,
        "routing_backend": "fixture",
        "policy_id": policy_id,
        "experiment_id": "construction_intelligence_v1" if variant else None,
        "experiment_variant": variant,
        "training_seed": training_seed,
        "transition_count": transition_count,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_lineage": checkpoint_lineage or [],
        "configuration_digest": configuration_digest,
        "source_commit": SOURCE.commit if variant else None,
        "resume_provenance": {},
    }


def _checkpoint_identity(
    run_key: str,
    fraction: float,
) -> dict[str, object]:
    percentage = int(round(fraction * 100))
    checkpoint_id = f"{run_key}-checkpoint-{percentage:03d}pct"
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": f"release/{run_key}/policy_{percentage:03d}pct.pt",
        "checkpoint_sha256": hashlib.sha256(
            f"policy-{run_key}-{percentage:03d}".encode()
        ).hexdigest(),
        "checkpoint_lineage": [f"{run_key}-training-origin"],
    }


def _simulator_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    role_targets: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    configuration_digests: list[str] = []
    role_files = {
        "manifest": "manifest.json",
        "scenario": "scenario.json",
        "planned_jobs": "planned_jobs.json",
        "replay": "planned_vs_measured_replay.json",
        "wheel_commands": "wheel_commands.jsonl",
        "telemetry": "measured_telemetry.jsonl",
        "trace": "trace.json",
        "metrics": "metrics.json",
        "report": "report.md",
        "scene": "construction_intelligence.ttt",
    }
    workspace = Path(__file__).resolve().parents[1]
    design = load_house_design(
        workspace / "configs" / "construction" / "cottage_v1.yaml"
    )
    generated_cottage = generate_cottage_scenario(
        900,
        design,
        config=CottageScenarioConfig(
            widths_m=(6.0,),
            depths_m=(6.0,),
            interior_panel_range=(0, 0),
            obstacle_count_range=(1, 1),
        ),
    )
    cottage, physical_yard = prepare_phase5_physical_yard(
        generated_cottage
    )
    for prefix, scenario in (
        ("nominal", "nominal"),
        ("recovery", "unavailable_robot_recovery"),
    ):
        run_dir = root / prefix
        fake = FakeDynamicClient()
        executor = DynamicCoppeliaExecutor(
            cottage.plan,
            config=DynamicCoppeliaConfig(
                control_hz=5,
                settle_steps=0,
                maximum_wheel_speed=10.0,
                effective_wheel_radius_m=0.2,
                position_gain=5.0,
                waypoint_tolerance_m=0.22,
                formation_tolerance_m=0.5,
                install_tolerance_m=0.3,
                safety_distance_m=0.05,
                max_steps_per_waypoint=1_000,
            ),
            client_factory=lambda _config, fake=fake: fake,
        )
        executor.connect()
        offline = Phase5FullCottageRunner(
            executor,
            cottage,
            physical_yard=physical_yard,
        ).run(scenario)
        assert offline.status == "completed"
        live_metrics = dict(offline.metrics)
        live_metrics.update(
            {
                "payload_transport": "logical_carrier",
                "live_evidence": True,
                "live_gate_passed": True,
            }
        )
        live = offline.model_copy(
            update={
                "evidence_kind": "live_coppelia",
                "live_gate_passed": True,
                "metrics": live_metrics,
            }
        )
        phase5_manifest = write_phase5_artifact_bundle(
            run_dir,
            run_id=f"synthetic-live-{prefix}",
            result=live,
            scenario=cottage,
            executor=executor,
            source_commit=SOURCE.commit,
            source_tree_digest=SOURCE.tree_digest,
            approval_gate_confirmed=True,
            simulator_version="CoppeliaSim coherent test fixture",
        )
        configuration_digests.append(phase5_manifest.configuration_digest)
        for artifact, file_name in role_files.items():
            role = f"{prefix}_{artifact}"
            source_paths[role] = f"{prefix}/{file_name}"
            role_targets[role] = f"evidence/coppelia/{prefix}/{file_name}"
    return _write_bundle(
        root,
        kind="simulator",
        status="canonical",
        role_targets=role_targets,
        source_paths=source_paths,
        configuration_digests=configuration_digests,
    )


def _rewrite_evaluation_artifacts(
    root: Path,
    payload: dict[str, object],
) -> EvaluationSuite:
    parsed = EvaluationSuite.model_validate(payload)
    suite = parsed.model_copy(
        update={
            "summaries": _fixture_controller_summaries(parsed.episodes),
            "per_training_seed": summarize_by_training_seed(parsed.episodes),
            "per_scenario_seed": summarize_by_scenario_seed(parsed.episodes),
        }
    )
    _write_json(root / "evaluation.json", suite.model_dump(mode="json"))
    rows = [episode.model_dump(mode="json") for episode in suite.episodes]
    with (root / "episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "report.md").write_text(
        render_evaluation_report(suite),
        encoding="utf-8",
    )
    return suite


def _rehash_research_files(
    descriptor_path: Path,
    *roles: str,
) -> None:
    descriptor = _read_json(descriptor_path)
    assert isinstance(descriptor, dict)
    artifacts = descriptor["artifacts"]
    assert isinstance(artifacts, list)
    by_role = {
        str(item["role"]): item
        for item in artifacts
        if isinstance(item, dict)
    }
    for role in roles:
        artifact = by_role[role]
        source_path = descriptor_path.parent / str(artifact["path"])
        artifact["sha256"] = _sha256(source_path)
    _write_json(descriptor_path, descriptor)


def _write_bundle(
    root: Path,
    *,
    kind: Literal["deterministic", "research", "simulator"],
    status: Literal["fixture", "canonical"],
    role_targets: dict[str, str],
    configuration_digests: list[str],
    source_paths: dict[str, str] | None = None,
    protocol_digest: str | None = None,
    profile: str | None = None,
    matrix_id: str | None = None,
) -> Path:
    source_paths = source_paths or role_targets
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": kind,
        "evidence_status": status,
        "source": SOURCE.model_dump(mode="json"),
        "created_at": "2026-07-26T00:00:00Z",
        "configuration_digests": sorted(configuration_digests),
        "protocol_digest": protocol_digest,
        "profile": profile,
        "matrix_id": matrix_id,
        "artifacts": [
            {
                "role": role,
                "path": source_paths[role],
                "target": target,
                "sha256": _sha256(root / source_paths[role]),
                "media_type": _fixture_media_type(target),
            }
            for role, target in sorted(role_targets.items())
        ],
    }
    path = root / f"{kind}-bundle.json"
    _write_json(path, manifest)
    return path


def _fixture_media_type(target: str) -> str:
    return {
        ".csv": "text/csv",
        ".glb": "model/gltf-binary",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".ttt": "application/octet-stream",
    }.get(Path(target).suffix.lower(), "application/octet-stream")


def _rehash_native_phase5_file(
    descriptor_path: Path,
    *,
    prefix: Literal["nominal", "recovery"],
    file_name: str,
) -> None:
    run_dir = descriptor_path.parent / prefix
    changed_path = run_dir / file_name
    manifest_path = run_dir / "manifest.json"
    phase5_manifest = _read_json(manifest_path)
    assert isinstance(phase5_manifest, dict)
    artifact = next(
        item
        for item in phase5_manifest["artifacts"]
        if item["path"] == file_name
    )
    artifact["sha256"] = _sha256(changed_path)
    artifact["bytes"] = changed_path.stat().st_size
    _write_json(manifest_path, phase5_manifest)

    descriptor = _read_json(descriptor_path)
    assert isinstance(descriptor, dict)
    changed_artifact = next(
        item
        for item in descriptor["artifacts"]
        if item["path"] == f"{prefix}/{file_name}"
    )
    changed_artifact["sha256"] = _sha256(changed_path)
    manifest_artifact = next(
        item
        for item in descriptor["artifacts"]
        if item["path"] == f"{prefix}/manifest.json"
    )
    manifest_artifact["sha256"] = _sha256(manifest_path)
    _write_json(descriptor_path, descriptor)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
