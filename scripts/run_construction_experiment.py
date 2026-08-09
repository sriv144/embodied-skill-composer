# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.experiment_execution import (
    MatrixSelectionEvidence,
    audit_ablation_hypotheses,
    evaluate_and_freeze_matrix_selections,
    evaluate_frozen_heldout_matrix,
    write_acceptance_artifact,
)
from embodied_skill_composer.construction.experiment_protocol import (
    ExperimentProfile,
    ReleaseEvidence,
    decide_release_completeness,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.research_evidence import (
    audit_reproducibility_artifacts,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import TrainingConfig


DEFAULT_REGISTRY = (
    WORKSPACE / "logs" / "construction_intelligence" / "lab.sqlite"
)
DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
DEFAULT_EVIDENCE = (
    WORKSPACE / "logs" / "construction_intelligence" / "experiments"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the frozen Construction Intelligence v1 matrix.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Evaluation device; training always uses the frozen profile setting.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser(
        "execute",
        help="Launch/reattach, train, validate, select, and evaluate held-out.",
    )
    execute.add_argument(
        "--profile",
        choices=("unit", "smoke", "research"),
        default="research",
    )
    execute.add_argument("--poll-seconds", type=float, default=2.0)
    execute.add_argument("--max-resume-attempts", type=int, default=3)
    status = commands.add_parser("status")
    status.add_argument("matrix_id", nargs="?")
    select = commands.add_parser("select")
    select.add_argument("matrix_id")
    heldout = commands.add_parser("heldout")
    heldout.add_argument("matrix_id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = LabRegistry(args.registry)
    protocol = load_experiment_protocol()
    design = load_house_design(args.design)
    if args.command == "status":
        payload = (
            registry.get_experiment_matrix(args.matrix_id)
            if args.matrix_id
            else registry.list_experiment_matrices()
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload else 2
    if args.command == "select":
        evidence = evaluate_and_freeze_matrix_selections(
            registry,
            args.matrix_id,
            design,
            protocol,
            output_root=args.evidence_root / "validation",
            device=args.device,
        )
        print(evidence.model_dump_json(indent=2))
        return 0
    if args.command == "heldout":
        selection_evidence = _load_selection_evidence(
            args.evidence_root,
            args.matrix_id,
        )
        suite, audit = evaluate_frozen_heldout_matrix(
            registry,
            args.matrix_id,
            design,
            protocol,
            output_root=args.evidence_root / "heldout",
            device=args.device,
        )
        acceptance_path = write_acceptance_artifact(
            audit,
            args.evidence_root
            / "heldout"
            / suite.evaluation_id
            / "acceptance.json",
        )
        ablations = audit_ablation_hypotheses(
            suite,
            protocol,
            selection_evidence=selection_evidence,
        )
        ablation_path = _write_json_artifact(
            [item.model_dump(mode="json") for item in ablations],
            acceptance_path.parent / "ablations.json",
        )
        print(
            json.dumps(
                {
                    "evaluation_id": suite.evaluation_id,
                    "acceptance": audit.model_dump(mode="json"),
                    "acceptance_path": str(acceptance_path),
                    "ablation_path": str(ablation_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if audit.passed else 1
    return _execute(args, registry, protocol, design)


def _execute(
    args: argparse.Namespace,
    registry: LabRegistry,
    protocol,
    design,
) -> int:
    profile: ExperimentProfile = args.profile
    digest = protocol_digest(protocol)
    matrix_id = f"{protocol.experiment_id}-{profile}-{digest[:12]}"
    matrix = registry.get_experiment_matrix(matrix_id)
    service = LabService(registry)
    try:
        if matrix is None:
            launch_runs: list[tuple[str, TrainingConfig]] = []
            for spec in expand_experiment_matrix(protocol, profile):
                config = TrainingConfig.model_validate(
                    spec.training_config_payload()
                )
                launch_runs.append((spec.run_id, config))
            service.launch_training_matrix(
                design,
                matrix_id=matrix_id,
                protocol_digest=digest,
                protocol=protocol.model_dump(mode="json"),
                execution_profile=profile,
                runs=launch_runs,
            )
        last_summary = None
        while True:
            matrix = registry.get_experiment_matrix(matrix_id)
            if matrix is None:
                raise RuntimeError(f"matrix disappeared: {matrix_id}")
            summary = {
                "matrix_id": matrix_id,
                "status": matrix["status"],
                "status_counts": matrix["status_counts"],
                "selection_count": matrix["selection_count"],
            }
            if summary != last_summary:
                print(json.dumps(summary, sort_keys=True), flush=True)
                last_summary = summary
            persisted_runs = _matrix_runs(matrix)
            if _resume_interrupted_runs(
                service,
                persisted_runs,
                max_attempts=args.max_resume_attempts,
            ):
                continue
            if all(
                str(run["status"]) == "completed"
                for run in persisted_runs
            ):
                break
            blocking = [
                run
                for run in persisted_runs
                if (
                    str(run["status"]) == "cancelled"
                    or (
                        str(run["status"]) in {"failed", "interrupted"}
                        and (
                            _integer_field(run, "attempt")
                            >= args.max_resume_attempts
                            or not _has_resume_source(run)
                        )
                    )
                )
            ]
            if blocking:
                print(
                    json.dumps(
                        {
                            "blocking_runs": [
                                {
                                    "id": run["id"],
                                    "status": run["status"],
                                    "error": run["error"],
                                }
                                for run in blocking
                            ]
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
                return 1
            time.sleep(max(args.poll_seconds, 0.05))
    except KeyboardInterrupt:
        print(
            "Detached cleanly; the active worker retains its persisted claim.",
            file=sys.stderr,
        )
        return 130
    finally:
        service.shutdown()

    selection_evidence = evaluate_and_freeze_matrix_selections(
        registry,
        matrix_id,
        design,
        protocol,
        output_root=args.evidence_root / "validation",
        device=args.device,
    )
    suite, audit = evaluate_frozen_heldout_matrix(
        registry,
        matrix_id,
        design,
        protocol,
        output_root=args.evidence_root / "heldout",
        device=args.device,
    )
    acceptance_path = write_acceptance_artifact(
        audit,
        args.evidence_root
        / "heldout"
        / suite.evaluation_id
        / "acceptance.json",
    )
    ablations = audit_ablation_hypotheses(
        suite,
        protocol,
        selection_evidence=selection_evidence,
    )
    ablation_path = _write_json_artifact(
        [item.model_dump(mode="json") for item in ablations],
        acceptance_path.parent / "ablations.json",
    )
    release_path: Path | None = None
    reproducibility_audit_path: Path | None = None
    release_complete = True
    if profile == "research":
        reproducibility_audit = audit_reproducibility_artifacts(
            registry,
            matrix_id,
            protocol,
            selection_evidence,
            suite,
            heldout_run_dir=acceptance_path.parent,
            acceptance_path=acceptance_path,
            ablation_path=ablation_path,
        )
        reproducibility_audit_path = _write_json_artifact(
            reproducibility_audit.model_dump(mode="json"),
            acceptance_path.parent / "reproducibility_audit.json",
        )
        expected_run_ids = [
            spec.run_id
            for spec in expand_experiment_matrix(protocol, "research")
        ]
        release = decide_release_completeness(
            protocol,
            ReleaseEvidence(
                completed_training_run_ids=expected_run_ids,
                selected_policy_run_ids=expected_run_ids,
                heldout_evaluated_run_ids=expected_run_ids,
                ablation_decisions=ablations,
                primary_acceptance_passed=audit.passed,
                report_generated=True,
                reproducibility_artifacts_complete=reproducibility_audit.complete,
            ),
        )
        release_complete = release.complete
        release_path = _write_json_artifact(
            release.model_dump(mode="json"),
            acceptance_path.parent / "release_completeness.json",
        )
    print(
        json.dumps(
            {
                "matrix_id": matrix_id,
                "evaluation_id": suite.evaluation_id,
                "acceptance_passed": audit.passed,
                "acceptance_path": str(acceptance_path),
                "ablation_path": str(ablation_path),
                "release_complete": release_complete,
                "release_path": str(release_path) if release_path else None,
                "reproducibility_audit_path": (
                    str(reproducibility_audit_path)
                    if reproducibility_audit_path
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit.passed and release_complete else 1


def _resume_interrupted_runs(
    service: LabService,
    runs: list[dict[str, object]],
    *,
    max_attempts: int,
) -> bool:
    resumed = False
    for run in runs:
        if (
            str(run["status"]) in {"interrupted", "failed"}
            and _integer_field(run, "attempt") < max_attempts
        ):
            resumed = service.resume(str(run["id"])) or resumed
    return resumed


def _has_resume_source(run: dict[str, object]) -> bool:
    checkpoint = run.get("latest_checkpoint")
    if checkpoint is None:
        return True
    return isinstance(checkpoint, str) and bool(checkpoint) and Path(checkpoint).is_file()


def _load_selection_evidence(
    evidence_root: Path,
    matrix_id: str,
) -> MatrixSelectionEvidence:
    path = (
        evidence_root.resolve()
        / "validation"
        / matrix_id
        / "matrix_selections.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"validation selection evidence is required before held-out audit: {path}"
        )
    evidence = MatrixSelectionEvidence.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if evidence.matrix_id != matrix_id:
        raise ValueError(
            f"selection evidence matrix mismatch: {evidence.matrix_id} != {matrix_id}"
        )
    return evidence


def _matrix_runs(matrix: dict[str, object]) -> list[dict[str, object]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list) or not all(isinstance(item, dict) for item in runs):
        raise ValueError("matrix runs are malformed")
    return runs


def _integer_field(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _write_json_artifact(payload: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
