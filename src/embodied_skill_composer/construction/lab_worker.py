from __future__ import annotations

import argparse
import json
import os
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
    if run["kind"] != "training":
        raise SystemExit(f"unsupported worker job kind: {run['kind']}")

    config = TrainingConfig.model_validate(run["config"])
    input_payload = cast(dict[str, object], run["input"])
    design = HouseDesign.model_validate(input_payload.get("design"))
    latest_checkpoint = run.get("latest_checkpoint")
    attempt = _integer_field(run, "attempt")
    if latest_checkpoint:
        config.resume_checkpoint = Path(str(latest_checkpoint))
        config.resume_provenance = {
            "run_id": args.run_id,
            "attempt": attempt,
            "checkpoint": str(latest_checkpoint),
        }

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
    registry.append_event(
        args.run_id,
        {
            "event": "training_started",
            "mode": "subprocess",
            "attempt": attempt,
            "pid": os.getpid(),
        },
        claim_token=args.claim_token,
    )
    thread.start()
    try:
        def progress(payload: dict[str, object]) -> None:
            transitions_value = payload.get("transitions", 0)
            if not isinstance(transitions_value, int):
                raise ValueError("progress transitions must be an integer")
            transitions = transitions_value
            checkpoint = payload.get("checkpoint_path")
            registry.update_run(
                args.run_id,
                progress=transitions / max(config.transitions, 1),
                latest_checkpoint=str(checkpoint) if checkpoint else None,
                pid=os.getpid(),
                heartbeat=True,
                claim_token=args.claim_token,
            )
            registry.append_event(args.run_id, payload, claim_token=args.claim_token)

        def cancel_requested() -> bool:
            if claim_lost.is_set() or not registry.verify_claim(
                args.run_id, args.claim_token
            ):
                raise LostRunClaimError(args.run_id)
            return registry.cancel_requested(args.run_id)

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
            args.run_id,
            status="completed",
            event={
                "event": "training_completed",
                "artifacts": artifacts.model_dump(mode="json"),
            },
            progress=1.0,
            artifact_dir=str(artifacts.run_dir),
            claim_token=args.claim_token,
            policy=(
                str(manifest["policy_id"]),
                str(manifest["controller"]),
                manifest,
            ),
        )
        return 0
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


def _integer_field(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"run field {key!r} must be an integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
