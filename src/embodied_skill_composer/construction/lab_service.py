from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.models import HouseDesign


class LabService:
    def __init__(
        self,
        registry: LabRegistry,
        *,
        training_runner=None,
        evaluation_runner=None,
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
        self._futures = {}

    def launch_training(self, design: HouseDesign, config) -> str:
        run_id = self.registry.create_run("training", config.model_dump(mode="json"))
        future = self.executor.submit(self._run_training, run_id, design.model_copy(deep=True), config)
        with self._lock:
            self._futures[run_id] = future
        return run_id

    def launch_evaluation(
        self,
        design: HouseDesign,
        *,
        seeds: list[int],
        controllers: list[str],
        policy_checkpoints: dict[str, str],
        include_failures: bool,
        output_root: Path,
    ) -> str:
        config = {
            "seeds": seeds,
            "controllers": controllers,
            "policy_checkpoints": policy_checkpoints,
            "include_failures": include_failures,
            "output_root": str(output_root),
        }
        run_id = self.registry.create_run("evaluation", config)
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
        return self.registry.request_cancel(run_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run_training(self, run_id: str, design: HouseDesign, config) -> None:
        self.registry.update_run(run_id, status="running")
        self.registry.append_event(run_id, {"event": "training_started"})
        try:
            runner = self.training_runner
            if runner is None:
                from embodied_skill_composer.construction.training import train_swarm_policy

                runner = train_swarm_policy

            def progress(payload: dict[str, object]) -> None:
                transitions = int(payload.get("transitions", 0))
                fraction = transitions / max(config.transitions, 1)
                self.registry.update_run(run_id, progress=fraction)
                self.registry.append_event(run_id, payload)

            artifacts = runner(
                design,
                config,
                progress_callback=progress,
                cancel_check=lambda: self.registry.cancel_requested(run_id),
            )
            manifest = json.loads(artifacts.policy_manifest_path.read_text(encoding="utf-8"))
            self.registry.upsert_policy(
                manifest["policy_id"],
                manifest["controller"],
                manifest,
            )
            self.registry.update_run(
                run_id,
                status="completed",
                progress=1.0,
                artifact_dir=str(artifacts.run_dir),
            )
            self.registry.append_event(
                run_id,
                {"event": "training_completed", "artifacts": artifacts.model_dump(mode="json")},
            )
        except Exception as exc:
            cancelled = self.registry.cancel_requested(run_id) or "cancelled" in str(exc).lower()
            status = "cancelled" if cancelled else "failed"
            self.registry.update_run(run_id, status=status, error=str(exc))
            self.registry.append_event(
                run_id,
                {"event": status, "error": str(exc)},
            )

    def _run_evaluation(
        self,
        run_id: str,
        design: HouseDesign,
        config: dict[str, object],
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
                artifacts = write_evaluation_artifacts(suite, Path(config["output_root"]))
            else:
                artifacts = runner(design, config)
            self.registry.update_run(
                run_id,
                status="completed",
                progress=1.0,
                artifact_dir=str(artifacts.run_dir),
            )
            self.registry.append_event(
                run_id,
                {"event": "evaluation_completed", "artifacts": artifacts.model_dump(mode="json")},
            )
        except Exception as exc:
            self.registry.update_run(run_id, status="failed", error=str(exc))
            self.registry.append_event(run_id, {"event": "failed", "error": str(exc)})
