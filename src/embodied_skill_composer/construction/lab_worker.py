from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import cast

from embodied_skill_composer.construction.lab_registry import (
    LabRegistry,
    LostRunClaimError,
    QuiescentRunStatus,
    _process_identity,
)
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.training import TrainingConfig, train_swarm_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one claimed construction lab job.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--claim-token", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = LabRegistry(args.registry)
    run = registry.get_run(args.run_id)
    if run is None:
        raise SystemExit(f"unknown run: {args.run_id}")
    if not registry.verify_claim(args.run_id, args.claim_token):
        raise SystemExit("claim token is stale or invalid")
    run_kind = str(run["kind"])
    if run_kind not in {"training", "matrix_evaluation"}:
        raise SystemExit(f"unsupported worker job kind: {run['kind']}")

    input_payload = cast(dict[str, object], run["input"])
    design = HouseDesign.model_validate(input_payload.get("design"))
    attempt = _integer_field(run, "attempt")
    training_config: TrainingConfig | None = None
    if run_kind == "training":
        training_config = TrainingConfig.model_validate(run["config"])
        latest_checkpoint = run.get("latest_checkpoint")
        if latest_checkpoint:
            training_config.resume_checkpoint = Path(str(latest_checkpoint))
            resume_provenance: dict[str, object] = {
                "run_id": args.run_id,
                "attempt": attempt,
                "checkpoint": str(latest_checkpoint),
            }
            registry.record_resume_provenance(
                args.run_id,
                resume_provenance,
                claim_token=args.claim_token,
            )
            training_config.resume_provenance = resume_provenance

    stop_heartbeat = Event()
    claim_lost = Event()
    process_identity = _process_identity(os.getpid())

    def heartbeat() -> None:
        while not stop_heartbeat.wait(timeout=5):
            try:
                registry.update_run(
                    args.run_id,
                    pid=os.getpid(),
                    process_identity=process_identity,
                    heartbeat=True,
                    claim_token=args.claim_token,
                )
            except LostRunClaimError:
                claim_lost.set()
                return

    thread = Thread(target=heartbeat, name=f"lab-heartbeat-{args.run_id}", daemon=True)
    registry.update_run(
        args.run_id,
        pid=os.getpid(),
        process_identity=process_identity,
        heartbeat=True,
        claim_token=args.claim_token,
    )
    started_event: dict[str, object] = {
        "event": (
            "training_started"
            if run_kind == "training"
            else "matrix_evaluation_started"
        ),
        "mode": "subprocess",
        "attempt": attempt,
        "pid": os.getpid(),
    }
    if run_kind == "matrix_evaluation":
        started_event["matrix_id"] = _string_field(
            cast(dict[str, object], run["config"]),
            "matrix_id",
        )
    registry.append_event(args.run_id, started_event, claim_token=args.claim_token)
    thread.start()
    try:
        def cancel_requested() -> bool:
            if claim_lost.is_set() or not registry.verify_claim(
                args.run_id, args.claim_token
            ):
                raise LostRunClaimError(args.run_id)
            return registry.cancel_requested(args.run_id)

        if training_config is not None:
            return _execute_training(
                registry,
                args.run_id,
                args.claim_token,
                design,
                training_config,
                cancel_requested,
            )
        return _execute_matrix_evaluation(
            registry,
            args.run_id,
            args.claim_token,
            design,
            cast(dict[str, object], run["config"]),
            cancel_requested,
        )
    except LostRunClaimError:
        return 3
    except Exception as exc:
        cancelled = registry.cancel_requested(args.run_id) or "cancelled" in str(exc).lower()
        status: QuiescentRunStatus = "cancelled" if cancelled else "failed"
        try:
            registry.finalize_run(
                args.run_id,
                status=status,
                event={"event": status, "error": str(exc)},
                error=str(exc),
                claim_token=args.claim_token,
            )
        except LostRunClaimError:
            return 3
        return 2
    finally:
        stop_heartbeat.set()
        thread.join(timeout=1)


def _execute_training(
    registry: LabRegistry,
    run_id: str,
    claim_token: str,
    design: HouseDesign,
    config: TrainingConfig,
    cancel_requested: Callable[[], bool],
) -> int:
    def progress(payload: dict[str, object]) -> None:
        transitions_value = payload.get("transitions", 0)
        if not isinstance(transitions_value, int):
            raise ValueError("progress transitions must be an integer")
        checkpoint = payload.get("checkpoint_path")
        registry.update_run(
            run_id,
            progress=transitions_value / max(config.transitions, 1),
            latest_checkpoint=str(checkpoint) if checkpoint else None,
            pid=os.getpid(),
            heartbeat=True,
            claim_token=claim_token,
        )
        registry.append_event(run_id, payload, claim_token=claim_token)

    artifacts = train_swarm_policy(
        design,
        config,
        progress_callback=progress,
        cancel_check=cancel_requested,
    )
    manifest_payload = json.loads(
        artifacts.policy_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(manifest_payload, dict):
        raise ValueError("policy manifest must contain an object")
    manifest = cast(dict[str, object], manifest_payload)
    registry.finalize_run(
        run_id,
        status="completed",
        event={
            "event": "training_completed",
            "artifacts": artifacts.model_dump(mode="json"),
        },
        progress=1.0,
        artifact_dir=str(artifacts.run_dir),
        claim_token=claim_token,
        policy=(
            str(manifest["policy_id"]),
            str(manifest["controller"]),
            manifest,
        ),
    )
    return 0


def _execute_matrix_evaluation(
    registry: LabRegistry,
    run_id: str,
    claim_token: str,
    design: HouseDesign,
    config: dict[str, object],
    cancel_requested: Callable[[], bool],
) -> int:
    from embodied_skill_composer.construction.experiment_execution import (
        MatrixSelectionEvidence,
        audit_ablation_hypotheses,
        evaluate_frozen_heldout_matrix,
    )
    from embodied_skill_composer.construction.experiment_protocol import (
        ExperimentProtocol,
        protocol_digest,
    )

    matrix_id = _string_field(config, "matrix_id")
    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise KeyError(matrix_id)
    protocol = ExperimentProtocol.model_validate(matrix["protocol"])
    expected_protocol_digest = _string_field(config, "protocol_digest")
    if protocol_digest(protocol) != expected_protocol_digest:
        raise ValueError("matrix evaluation protocol digest mismatch")
    output_root = Path(_string_field(config, "output_root")).resolve()
    device = _string_field(config, "device")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported matrix evaluation device: {device}")
    selection_evidence: MatrixSelectionEvidence | None = None
    selection_evidence_value = config.get("selection_evidence_path")
    if selection_evidence_value is not None:
        if not isinstance(selection_evidence_value, str) or not selection_evidence_value:
            raise ValueError("selection_evidence_path must be a non-empty string or null")
        selection_evidence_path = Path(selection_evidence_value)
        if selection_evidence_path.is_file():
            selection_evidence = MatrixSelectionEvidence.model_validate_json(
                selection_evidence_path.read_text(encoding="utf-8")
            )
            if selection_evidence.matrix_id != matrix_id:
                raise ValueError("selection evidence matrix id mismatch")
        else:
            registry.append_event(
                run_id,
                {
                    "event": "selection_evidence_unavailable",
                    "path": str(selection_evidence_path),
                    "effect": "transitions_to_95_ablation_evidence_omitted",
                },
                claim_token=claim_token,
            )
    suite, acceptance = evaluate_frozen_heldout_matrix(
        registry,
        matrix_id,
        design,
        protocol,
        output_root=output_root,
        device=device,
        cancel_check=cancel_requested,
    )
    if cancel_requested():
        raise RuntimeError("matrix evaluation cancelled")
    ablations = audit_ablation_hypotheses(
        suite,
        protocol,
        selection_evidence=selection_evidence,
    )
    registry.finalize_run(
        run_id,
        status="completed",
        event={
            "event": "matrix_evaluation_completed",
            "matrix_id": matrix_id,
            "evaluation_id": suite.evaluation_id,
            "acceptance": acceptance.model_dump(mode="json"),
            "ablations": [item.model_dump(mode="json") for item in ablations],
        },
        progress=1.0,
        artifact_dir=str(output_root / suite.evaluation_id),
        claim_token=claim_token,
    )
    return 0


def _integer_field(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"run field {key!r} must be an integer")
    return value


def _string_field(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run field {key!r} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
