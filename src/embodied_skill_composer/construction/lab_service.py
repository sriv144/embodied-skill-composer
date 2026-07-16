from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from embodied_skill_composer.construction.lab_registry import (
    QUIESCENT_RUN_STATUSES,
    LabRegistry,
    QuiescentRunStatus,
    _process_identity,
)
from embodied_skill_composer.construction.models import HouseDesign

if TYPE_CHECKING:
    from embodied_skill_composer.construction.evaluation import ControllerName
    from embodied_skill_composer.construction.training import TrainingArtifacts, TrainingConfig


class ArtifactResult(Protocol):
    run_dir: Path

    def model_dump(self, *, mode: str = "python") -> dict[str, object]: ...


class TrainingArtifactResult(ArtifactResult, Protocol):
    policy_manifest_path: Path


class TrainingRunner(Protocol):
    def __call__(
        self,
        design: HouseDesign,
        config: TrainingConfig,
        *,
        progress_callback: object,
        cancel_check: object,
    ) -> TrainingArtifacts: ...


class EvaluationRunner(Protocol):
    def __call__(self, design: HouseDesign, config: EvaluationJobConfig) -> ArtifactResult: ...


class EvaluationJobConfig(TypedDict):
    seeds: list[int]
    controllers: list[ControllerName]
    policy_checkpoints: dict[str, str]
    include_failures: bool
    output_root: str


class LabService:
    """Durable lab facade with a single-slot subprocess queue for training."""

    def __init__(
        self,
        registry: LabRegistry,
        *,
        training_runner: TrainingRunner | None = None,
        evaluation_runner: EvaluationRunner | None = None,
        max_workers: int = 2,
    ) -> None:
        self.registry = registry
        self.training_runner = training_runner
        self.evaluation_runner = evaluation_runner
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="construction-lab",
        )
        self._lock = Lock()
        self._futures: dict[str, Future[None]] = {}
        self._stop = Event()
        self._wake = Event()
        self._dispatcher: Thread | None = None
        if self.training_runner is None:
            self.registry.reconcile_stale_runs()
            self._dispatcher = Thread(
                target=self._dispatch_training,
                name="construction-training-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

    def launch_training(self, design: HouseDesign, config: TrainingConfig) -> str:
        from embodied_skill_composer.construction.training import (
            configuration_digest,
            environment_fingerprint,
            source_fingerprint,
        )

        config.output_root = config.output_root.resolve()
        source = source_fingerprint()
        config.source_commit = config.source_commit or str(source["commit"])
        config.source_dirty = bool(source["dirty"])
        config.source_tree_digest = str(source["tree_digest"])
        if config.profile == "research" and config.source_dirty:
            raise ValueError("research training requires a clean source worktree")
        config.environment_fingerprint = (
            config.environment_fingerprint or environment_fingerprint()
        )
        config.configuration_digest = configuration_digest(config)
        run_id = self.registry.create_run(
            "training",
            config.model_dump(mode="json"),
            input_payload={"design": design.model_dump(mode="json")},
            config_digest=config.configuration_digest,
            source_commit=config.source_commit,
        )
        if self.training_runner is not None:
            future = self.executor.submit(
                self._run_training_inline,
                run_id,
                design.model_copy(deep=True),
                config,
            )
            with self._lock:
                self._futures[run_id] = future
        else:
            self._wake.set()
        return run_id

    def launch_evaluation(
        self,
        design: HouseDesign,
        *,
        seeds: list[int],
        controllers: list[ControllerName],
        policy_checkpoints: dict[str, str],
        include_failures: bool,
        output_root: Path,
    ) -> str:
        config: EvaluationJobConfig = {
            "seeds": seeds,
            "controllers": controllers,
            "policy_checkpoints": policy_checkpoints,
            "include_failures": include_failures,
            "output_root": str(output_root),
        }
        run_id = self.registry.create_run(
            "evaluation",
            cast(dict[str, object], config),
            input_payload={"design": design.model_dump(mode="json")},
        )
        future = self.executor.submit(
            self._run_evaluation,
            run_id,
            design.model_copy(deep=True),
            config,
        )
        with self._lock:
            self._futures[run_id] = future
        return run_id

    def cancel(self, run_id: str) -> bool:
        accepted = self.registry.request_cancel(run_id)
        self._wake.set()
        return accepted

    def resume(self, run_id: str) -> bool:
        accepted = self.registry.request_resume(run_id)
        if accepted:
            self._wake.set()
        return accepted

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=2)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _dispatch_training(self) -> None:
        while not self._stop.is_set():
            claimed = self.registry.claim_next_training()
            if claimed is None:
                self.registry.reconcile_stale_runs()
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            run_id = str(claimed["id"])
            claim_token = str(claimed["claim_token"])
            process_log = self.registry.path.parent / "process" / f"{run_id}.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "embodied_skill_composer.construction.lab_worker",
                "--registry",
                str(self.registry.path),
                "--run-id",
                run_id,
                "--claim-token",
                claim_token,
            ]
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
            worker_environment = os.environ.copy()
            worker_environment["PYTHONIOENCODING"] = "utf-8"
            worker_environment["PYTHONUTF8"] = "1"
            try:
                with process_log.open("a", encoding="utf-8") as handle:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        creationflags=creationflags,
                        env=worker_environment,
                    )
                    process_identity = _process_identity(process.pid)
                    self.registry.update_run(
                        run_id,
                        pid=process.pid,
                        process_identity=process_identity,
                        heartbeat=True,
                        claim_token=claim_token,
                    )
                    self.registry.append_event(
                        run_id,
                        {
                            "event": "worker_started",
                            "pid": process.pid,
                            "process_log": str(process_log),
                        },
                        claim_token=claim_token,
                    )
                    while process.poll() is None and not self._stop.wait(timeout=0.5):
                        self.registry.reconcile_stale_runs()
                    if process.poll() is None:
                        # Deliberately leave the detached worker alive; it owns its persisted claim.
                        return
                    return_code = int(process.returncode or 0)
            except OSError as exc:
                return_code = -1
                self.registry.append_event(
                    run_id,
                    {"event": "worker_launch_failed", "error": str(exc)},
                    claim_token=claim_token,
                )
            current = self.registry.get_run(run_id)
            if (
                current
                and current.get("claim_token") == claim_token
                and str(current["status"]) not in QUIESCENT_RUN_STATUSES
            ):
                interrupted = current.get("latest_checkpoint") is not None
                self.registry.finalize_run(
                    run_id,
                    status="interrupted" if interrupted else "failed",
                    event={"event": "worker_exited", "return_code": return_code},
                    error=f"training worker exited with code {return_code}",
                    claim_token=claim_token,
                )

    def _run_training_inline(
        self,
        run_id: str,
        design: HouseDesign,
        config: TrainingConfig,
    ) -> None:
        self.registry.update_run(run_id, status="running", heartbeat=True)
        self.registry.append_event(run_id, {"event": "training_started", "mode": "inline"})
        try:
            assert self.training_runner is not None

            def progress(payload: dict[str, object]) -> None:
                transitions_value = payload.get("transitions", 0)
                if not isinstance(transitions_value, int):
                    raise ValueError("progress transitions must be an integer")
                transitions = transitions_value
                fraction = transitions / max(config.transitions, 1)
                checkpoint = payload.get("checkpoint_path")
                self.registry.update_run(
                    run_id,
                    progress=fraction,
                    latest_checkpoint=str(checkpoint) if checkpoint else None,
                    heartbeat=True,
                )
                self.registry.append_event(run_id, payload)

            artifacts = self.training_runner(
                design,
                config,
                progress_callback=progress,
                cancel_check=lambda: self.registry.cancel_requested(run_id),
            )
            manifest = _read_manifest(artifacts.policy_manifest_path)
            self.registry.finalize_run(
                run_id,
                status="completed",
                event={
                    "event": "training_completed",
                    "artifacts": artifacts.model_dump(mode="json"),
                },
                progress=1.0,
                artifact_dir=str(artifacts.run_dir),
                policy=(
                    str(manifest["policy_id"]),
                    str(manifest["controller"]),
                    manifest,
                ),
            )
        except Exception as exc:
            cancelled = self.registry.cancel_requested(run_id) or "cancelled" in str(exc).lower()
            status: QuiescentRunStatus = "cancelled" if cancelled else "failed"
            self.registry.finalize_run(
                run_id,
                status=status,
                event={"event": status, "error": str(exc)},
                error=str(exc),
            )

    def _run_evaluation(
        self,
        run_id: str,
        design: HouseDesign,
        config: EvaluationJobConfig,
    ) -> None:
        self.registry.update_run(run_id, status="running")
        self.registry.append_event(run_id, {"event": "evaluation_started"})
        try:
            runner = self.evaluation_runner
            if runner is None:
                from embodied_skill_composer.construction.evaluation import (
                    run_evaluation_suite,
                    write_evaluation_artifacts,
                )
                from embodied_skill_composer.construction.policy import load_policy_checkpoint

                policies = {
                    controller: load_policy_checkpoint(Path(path))
                    for controller, path in config["policy_checkpoints"].items()
                }
                suite = run_evaluation_suite(
                    design,
                    seeds=config["seeds"],
                    controllers=config["controllers"],
                    policies=policies,
                    include_failure_suite=config["include_failures"],
                )
                evaluation_artifacts: ArtifactResult = write_evaluation_artifacts(
                    suite, Path(config["output_root"])
                )
            else:
                evaluation_artifacts = runner(design, config)
            self.registry.finalize_run(
                run_id,
                status="completed",
                event={
                    "event": "evaluation_completed",
                    "artifacts": evaluation_artifacts.model_dump(mode="json"),
                },
                progress=1.0,
                artifact_dir=str(evaluation_artifacts.run_dir),
            )
        except Exception as exc:
            self.registry.finalize_run(
                run_id,
                status="failed",
                event={"event": "failed", "error": str(exc)},
                error=str(exc),
            )


def _read_manifest(path: Path) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"policy manifest must contain an object: {path}")
    return cast(dict[str, object], parsed)
