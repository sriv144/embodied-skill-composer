from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Protocol, cast

from embodied_skill_composer.construction.evaluation import (
    EpisodeEvaluation,
    EvaluationSuite,
    render_evaluation_report,
)
from embodied_skill_composer.construction.experiment_execution import (
    MatrixSelectionEvidence,
    RunSelectionEvidence,
)
from embodied_skill_composer.construction.policy import file_sha256
from embodied_skill_composer.construction.public_demo_provenance import (
    BundleArtifact,
    EvidenceBundleManifest,
    PublicDemoExportError,
    SourceIdentity,
    _replace_directory,
    verify_research_release_evidence,
)
from embodied_skill_composer.construction.training import TrainingConfig


_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "matrix": ("matrix.json", "evidence/research/matrix.json", "application/json"),
    "selections": (
        "selections.json",
        "evidence/research/selections.json",
        "application/json",
    ),
    "evaluation": (
        "evaluation.json",
        "evidence/research/evaluation.json",
        "application/json",
    ),
    "episodes": ("episodes.csv", "evidence/research/episodes.csv", "text/csv"),
    "report": ("report.md", "evidence/research/report.md", "text/markdown"),
    "acceptance": (
        "acceptance.json",
        "evidence/research/acceptance.json",
        "application/json",
    ),
    "ablations": (
        "ablations.json",
        "evidence/research/ablations.json",
        "application/json",
    ),
    "reproducibility_audit": (
        "reproducibility_audit.json",
        "evidence/research/reproducibility_audit.json",
        "application/json",
    ),
    "release_completeness": (
        "release_completeness.json",
        "evidence/research/release_completeness.json",
        "application/json",
    ),
}
_PATH_FIELDS = {"output_root", "resume_checkpoint"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ExperimentMatrixRegistry(Protocol):
    def get_experiment_matrix(self, matrix_id: str) -> dict[str, object] | None: ...

    def list_policy_selections(self, matrix_id: str) -> list[dict[str, object]]: ...


def package_research_release_evidence(
    output_dir: Path,
    *,
    registry: ExperimentMatrixRegistry,
    matrix_id: str,
    selection_evidence_path: Path,
    heldout_run_dir: Path,
) -> Path:
    """Atomically create the canonical public Phase 4 evidence descriptor."""

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise PublicDemoExportError(f"research matrix does not exist: {matrix_id}")
    selection_path = selection_evidence_path.resolve()
    heldout = heldout_run_dir.resolve()
    try:
        selection_evidence = MatrixSelectionEvidence.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )
        suite = EvaluationSuite.model_validate_json(
            (heldout / "evaluation.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(
            f"canonical research inputs are unreadable or invalid: {exc}"
        ) from exc
    if selection_evidence.matrix_id != matrix_id:
        raise PublicDemoExportError("selection evidence matrix id does not match")

    public_matrix, source, configuration_digests = _public_matrix(matrix)
    _verify_frozen_selections(registry, matrix_id, selection_evidence)
    if public_matrix["protocol_digest"] != selection_evidence.protocol_digest:
        raise PublicDemoExportError("selection protocol digest does not match the matrix")

    destination = output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ) as raw:
        staging = Path(raw) / "bundle"
        staging.mkdir()
        public_selection, selected_by_identity = _stage_public_selections(
            selection_evidence,
            staging,
        )
        public_suite = _public_evaluation_suite(suite, selected_by_identity)

        _write_json(staging / "matrix.json", public_matrix)
        _write_json(staging / "selections.json", public_selection)
        _write_json(staging / "evaluation.json", public_suite.model_dump(mode="json"))
        (staging / "episodes.csv").write_bytes(_evaluation_csv_text(public_suite).encode("utf-8"))
        (staging / "report.md").write_bytes(render_evaluation_report(public_suite).encode("utf-8"))
        _copy_required(heldout / "acceptance.json", staging / "acceptance.json")
        _copy_required(heldout / "ablations.json", staging / "ablations.json")
        _write_json(
            staging / "reproducibility_audit.json",
            _public_reproducibility_audit(heldout / "reproducibility_audit.json"),
        )
        _copy_required(
            heldout / "release_completeness.json",
            staging / "release_completeness.json",
        )

        artifacts = [
            BundleArtifact(
                role=role,
                path=file_name,
                target=target,
                sha256=file_sha256(staging / file_name),
                media_type=media_type,
            )
            for role, (file_name, target, media_type) in _ARTIFACTS.items()
        ]
        manifest = EvidenceBundleManifest(
            schema_version="construction-intelligence-public-demo-input-v1",
            kind="research",
            evidence_status="canonical",
            source=source,
            created_at=cast(str, public_matrix["created_at"]),
            configuration_digests=configuration_digests,
            protocol_digest=public_matrix["protocol_digest"],
            profile="research",
            matrix_id=matrix_id,
            artifacts=sorted(artifacts, key=lambda item: item.role),
        )
        descriptor = staging / "research-bundle.json"
        _write_json(descriptor, manifest.model_dump(mode="json"))
        verify_research_release_evidence(descriptor)
        _replace_directory(staging, destination)

    descriptor = destination / "research-bundle.json"
    verify_research_release_evidence(descriptor)
    return descriptor


def _public_matrix(
    matrix: dict[str, object],
) -> tuple[dict[str, object], SourceIdentity, list[str]]:
    runs_value = matrix.get("runs")
    if not isinstance(runs_value, list) or not all(isinstance(item, dict) for item in runs_value):
        raise PublicDemoExportError("research matrix runs are malformed")
    runs = cast(list[dict[str, object]], runs_value)
    if (
        matrix.get("execution_profile") != "research"
        or matrix.get("expected_run_count") != 20
        or matrix.get("selection_count") != 20
        or len(runs) != 20
        or any(run.get("status") != "completed" for run in runs)
    ):
        raise PublicDemoExportError(
            "research packaging requires 20 completed runs and 20 frozen selections"
        )

    public_runs: list[dict[str, object]] = []
    commits: set[str] = set()
    tree_digests: set[str] = set()
    configuration_digests: set[str] = set()
    for run in runs:
        try:
            config = TrainingConfig.model_validate(run.get("config"))
        except ValueError as exc:
            raise PublicDemoExportError(
                f"research run config is invalid: {run.get('run_key')}: {exc}"
            ) from exc
        if (
            config.profile != "research"
            or config.source_dirty
            or config.source_commit is None
            or config.source_tree_digest is None
            or config.configuration_digest is None
            or _COMMIT_RE.fullmatch(config.source_commit) is None
            or _SHA256_RE.fullmatch(config.source_tree_digest) is None
            or _SHA256_RE.fullmatch(config.configuration_digest) is None
        ):
            raise PublicDemoExportError(
                f"research run lacks clean hashable provenance: {run.get('run_key')}"
            )
        commits.add(config.source_commit)
        tree_digests.add(config.source_tree_digest)
        configuration_digests.add(config.configuration_digest)
        public_config = {
            key: _public_value(value, run_key=str(run.get("run_key", "")))
            for key, value in config.model_dump(mode="json").items()
            if key not in _PATH_FIELDS
        }
        public_runs.append(
            {
                "id": run.get("id"),
                "run_key": run.get("run_key"),
                "ordinal": run.get("ordinal"),
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "started_at": run.get("started_at"),
                "ended_at": run.get("ended_at"),
                "attempt": run.get("attempt"),
                "resume_provenance_history": _public_value(
                    run.get("resume_provenance_history", []),
                    run_key=str(run.get("run_key", "")),
                ),
                "config": public_config,
            }
        )
    if len(commits) != 1 or len(tree_digests) != 1:
        raise PublicDemoExportError(
            "all research runs must share one clean source commit and tree digest"
        )
    created_at = matrix.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise PublicDemoExportError("research matrix is missing its creation timestamp")
    public_matrix = {
        "id": matrix.get("id"),
        "protocol_digest": matrix.get("protocol_digest"),
        "execution_profile": "research",
        "expected_run_count": 20,
        "selection_count": 20,
        "status": "selected",
        "created_at": created_at,
        "runs": public_runs,
    }
    return (
        public_matrix,
        SourceIdentity(
            commit=next(iter(commits)),
            dirty=False,
            tree_digest=next(iter(tree_digests)),
        ),
        sorted(configuration_digests),
    )


def _stage_public_selections(
    evidence: MatrixSelectionEvidence,
    staging: Path,
) -> tuple[dict[str, object], dict[tuple[str, int], dict[str, object]]]:
    selections: list[dict[str, object]] = []
    selected_by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for item in evidence.selections:
        selection_payload, selected = _public_run_selection(item, staging)
        identity = (
            cast(str, selected["experiment_variant"]),
            cast(int, selected["training_seed"]),
        )
        if identity in selected_by_identity:
            raise PublicDemoExportError(f"duplicate selected policy identity: {identity}")
        selected_by_identity[identity] = selected
        selections.append(selection_payload)
    return (
        {
            "matrix_id": evidence.matrix_id,
            "protocol_digest": evidence.protocol_digest,
            "selections": sorted(selections, key=lambda item: cast(str, item["run_key"])),
            "evidence_path": "evidence/research/selections.json",
        },
        selected_by_identity,
    )


def _public_run_selection(
    item: RunSelectionEvidence,
    staging: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    run_key = item.run_key
    try:
        persisted = RunSelectionEvidence.model_validate_json(
            item.evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(
            f"per-run selection evidence is invalid for {run_key}: {exc}"
        ) from exc
    if persisted != item:
        raise PublicDemoExportError(f"matrix and per-run selection evidence disagree: {run_key}")
    selected_candidate = next(
        (
            candidate.result
            for candidate in item.candidates
            if candidate.result.checkpoint_id == item.selected.checkpoint_id
        ),
        None,
    )
    selected_base = item.selected.model_dump(mode="json")
    for key in (
        "selection_rule",
        "required_checkpoint_fractions",
        "candidate_ranking",
    ):
        selected_base.pop(key)
    if selected_candidate is None or selected_base != selected_candidate.model_dump(mode="json"):
        raise PublicDemoExportError(
            f"selected checkpoint does not match its validation candidate: {run_key}"
        )
    selected_source = _resolve_input_path(
        Path(item.selected.checkpoint_path),
        item.evidence_path.parent,
    )
    if not selected_source.is_file():
        raise PublicDemoExportError(
            f"selected checkpoint is missing for {run_key}: {selected_source}"
        )
    if file_sha256(selected_source) != item.selected.checkpoint_sha256:
        raise PublicDemoExportError(
            f"selected checkpoint hash mismatch for {run_key}: {selected_source}"
        )
    public_selected_path = f"policies/{run_key}/checkpoint.pt"
    policy_source = staging / public_selected_path
    policy_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_source, policy_source)

    public_candidates: list[dict[str, object]] = []
    selected_result: dict[str, object] | None = None
    for candidate in item.candidates:
        result = candidate.result
        if result.checkpoint_id == item.selected.checkpoint_id:
            public_path = public_selected_path
        else:
            public_path = (
                f"validation-only/{run_key}/"
                f"checkpoint_{int(round(result.checkpoint_fraction * 100)):03d}pct.pt"
            )
        lineage = [
            f"validation-only/{run_key}/{_path_name(value)}" for value in result.checkpoint_lineage
        ]
        resume = cast(
            dict[str, object],
            _public_value(result.resume_provenance, run_key=run_key),
        )
        public_result = result.model_copy(
            update={
                "checkpoint_path": public_path,
                "checkpoint_lineage": lineage,
                "resume_provenance": resume,
            }
        )
        public_episodes = [
            episode.model_copy(
                update={
                    "checkpoint_path": public_path,
                    "checkpoint_lineage": lineage,
                    "resume_provenance": resume,
                }
            )
            for episode in candidate.episodes
        ]
        public_candidates.append(
            {
                "result": public_result.model_dump(mode="json"),
                "episodes": [episode.model_dump(mode="json") for episode in public_episodes],
            }
        )
        if result.checkpoint_id == item.selected.checkpoint_id:
            selected_result = public_result.model_dump(mode="json")
    if selected_result is None:
        raise PublicDemoExportError(
            f"selected checkpoint is absent from validation candidates: {run_key}"
        )
    public_selected = {
        **selected_result,
        "selection_rule": item.selected.selection_rule,
        "required_checkpoint_fractions": item.selected.required_checkpoint_fractions,
        "candidate_ranking": item.selected.candidate_ranking,
    }
    return (
        {
            "matrix_id": item.matrix_id,
            "run_key": run_key,
            "candidates": public_candidates,
            "selected": public_selected,
            "evidence_path": f"validation/{run_key}/validation_selection.json",
        },
        public_selected,
    )


def _public_evaluation_suite(
    suite: EvaluationSuite,
    selected_by_identity: Mapping[tuple[str, int], dict[str, object]],
) -> EvaluationSuite:
    episodes: list[EpisodeEvaluation] = []
    for episode in suite.episodes:
        if episode.controller not in {"mappo", "ippo"}:
            episodes.append(episode)
            continue
        identity = (cast(str, episode.experiment_variant), cast(int, episode.training_seed))
        selected = selected_by_identity.get(identity)
        if selected is None:
            raise PublicDemoExportError(f"held-out episode has no selected policy: {identity}")
        episodes.append(
            episode.model_copy(
                update={
                    "checkpoint_path": selected["checkpoint_path"],
                    "checkpoint_lineage": selected["checkpoint_lineage"],
                    "resume_provenance": selected["resume_provenance"],
                }
            )
        )
    return suite.model_copy(update={"episodes": episodes})


def _public_reproducibility_audit(path: Path) -> dict[str, object]:
    payload = _read_object(path, "reproducibility audit")
    hashes = payload.get("file_hashes")
    if (
        payload.get("complete") is not True
        or payload.get("missing_paths") != []
        or payload.get("invalid_artifacts") != []
        or not isinstance(hashes, dict)
        or not hashes
        or not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in hashes.values()
        )
    ):
        raise PublicDemoExportError("reproducibility audit is incomplete or invalid")
    public_hashes = {
        f"verified-source/{index:04d}-{_safe_label(_path_name(str(source)))}": digest
        for index, (source, digest) in enumerate(sorted(hashes.items()), start=1)
    }
    for source, digest in hashes.items():
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if not source_path.is_file() or file_sha256(source_path) != digest:
            raise PublicDemoExportError(f"reproducibility audit file hash mismatch: {source}")
    return {
        **payload,
        "checked_file_count": len(public_hashes),
        "file_hashes": public_hashes,
        "source_path_redaction": (
            "Machine-local source paths were replaced with stable public labels; "
            "the verified SHA-256 values are unchanged."
        ),
    }


def _verify_frozen_selections(
    registry: ExperimentMatrixRegistry,
    matrix_id: str,
    evidence: MatrixSelectionEvidence,
) -> None:
    frozen = registry.list_policy_selections(matrix_id)
    frozen_by_run = {str(item.get("run_key", "")): item.get("selection") for item in frozen}
    if len(frozen_by_run) != 20:
        raise PublicDemoExportError("registry does not contain all 20 frozen policy selections")
    for item in evidence.selections:
        if frozen_by_run.get(item.run_key) != item.selected.model_dump(mode="json"):
            raise PublicDemoExportError(
                f"selection evidence differs from the frozen registry: {item.run_key}"
            )


def _public_value(value: object, *, run_key: str) -> object:
    if isinstance(value, dict):
        return {str(key): _public_value(item, run_key=run_key) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_value(item, run_key=run_key) for item in value]
    if isinstance(value, str) and _is_absolute_path_text(value):
        return f"training/{run_key}/{_path_name(value)}"
    return value


def _is_absolute_path_text(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _path_name(value: str) -> str:
    return PureWindowsPath(value.replace("/", "\\")).name or "artifact"


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return label or "artifact"


def _resolve_input_path(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDemoExportError(f"{label} is unreadable or invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicDemoExportError(f"{label} must contain a JSON object")
    return cast(dict[str, object], payload)


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise PublicDemoExportError(f"required research artifact is missing: {source}")
    shutil.copyfile(source, destination)


def _evaluation_csv_text(suite: EvaluationSuite) -> str:
    rows = [episode.model_dump(mode="json") for episode in suite.episodes]
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
    )
