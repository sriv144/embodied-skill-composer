from __future__ import annotations

import asyncio
import pickle
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, cast

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.experiment_execution import (
    evaluate_and_freeze_run_selection,
    materialize_matrix_selection_evidence,
)
from embodied_skill_composer.construction.experiment_protocol import (
    ExperimentProfile,
    ExperimentProtocol,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.floorplan import infer_orthogonal_floor_plan
from embodied_skill_composer.construction.evaluation import ControllerName
from embodied_skill_composer.construction.lab_registry import (
    QUIESCENT_RUN_STATUSES,
    LabRegistry,
)
from embodied_skill_composer.construction.lab_events import RunEventEnvelope
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.reporting import render_research_report
from embodied_skill_composer.construction.recovery import Disruption, inject_disruption
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scheduler import compare_controllers
from embodied_skill_composer.construction.scenarios import generate_cottage_scenario
from embodied_skill_composer.construction.trace import build_execution_trace


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
DEFAULT_ASSETS = WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1"
DEFAULT_LAB_DATABASE = WORKSPACE / "logs" / "construction_intelligence" / "lab.sqlite"
LOCAL_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}


class WorkbenchState:
    def __init__(self, design: HouseDesign):
        self.replace_design(design)

    def replace_design(self, design: HouseDesign) -> None:
        self.design = design
        self.plan = compile_house_design(design)
        self.schedules = compare_controllers(self.plan)
        self.traces = {
            name: build_execution_trace(self.plan, schedule)
            for name, schedule in self.schedules.items()
        }


class RebuildRequest(BaseModel):
    design: HouseDesign


class ScenarioGenerationRequest(BaseModel):
    seed: int = Field(ge=0, le=999)


class TrainingLaunchRequest(BaseModel):
    algorithm: Literal["mappo", "ippo"] = "mappo"
    profile: Literal["unit", "smoke", "research"] = "smoke"
    seed: int = Field(default=7, ge=0)
    transitions: int | None = Field(default=None, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    confirmed: bool = False


class ExperimentMatrixLaunchRequest(BaseModel):
    profile: ExperimentProfile = "smoke"
    confirmed: bool = False


class ValidationSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool = False


class HeldoutMatrixLaunchRequest(BaseModel):
    confirmed: bool = False


def _default_evaluation_controllers() -> list[ControllerName]:
    return ["sequential", "greedy", "auction", "cp_sat"]


class EvaluationLaunchRequest(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [900, 901, 902, 903, 904])
    controllers: list[ControllerName] = Field(default_factory=_default_evaluation_controllers)
    policy_ids: dict[str, str] = Field(default_factory=dict)
    include_failures: bool = True


def create_app(
    *,
    registry_path: Path | None = None,
    training_runner=None,
    evaluation_runner=None,
) -> FastAPI:
    registry = LabRegistry(registry_path or DEFAULT_LAB_DATABASE)
    service = LabService(
        registry,
        training_runner=training_runner,
        evaluation_runner=evaluation_runner,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        service.shutdown()

    app = FastAPI(
        title="Embodied Skill Composer Construction Workbench",
        version="3.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(LOCAL_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"],
    )
    state = WorkbenchState(load_house_design(DEFAULT_DESIGN))
    registry.upsert_scenario(
        state.design.design_id,
        seed=None,
        split="fixture",
        payload={
            "design": state.design.model_dump(mode="json"),
            "plan": state.plan.model_dump(mode="json"),
        },
    )
    app.state.lab_registry = registry
    app.state.lab_service = service
    app.state.experiment_protocol = load_experiment_protocol()

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ready",
            "design_id": state.design.design_id,
            "module_count": len(state.plan.modules),
            "robot_count": len(state.plan.robots),
            "cp_sat": state.schedules["optimized"].solver_status,
            "lab_database": str(registry.path),
        }

    @app.get("/api/project")
    def project() -> dict[str, object]:
        sequential = state.schedules["sequential"].makespan_s
        optimized = state.schedules["optimized"].makespan_s
        return {
            "design": state.design.model_dump(mode="json"),
            "plan": state.plan.model_dump(mode="json"),
            "controllers": {
                name: trace.metrics.model_dump(mode="json")
                for name, trace in state.traces.items()
            },
            "optimized_improvement_percent": round(100 * (1 - optimized / sequential), 1),
            "geometry_asset_url": (
                "/artifacts/house.glb" if (DEFAULT_ASSETS / "house.glb").is_file() else None
            ),
            "robot_asset_url": "/demo/construction_robot.glb",
        }

    @app.get("/api/traces/{controller}")
    def trace(controller: str) -> dict[str, object]:
        if controller not in state.traces:
            raise HTTPException(status_code=404, detail=f"unknown controller: {controller}")
        return state.traces[controller].model_dump(mode="json")

    @app.get("/api/report")
    def report() -> dict[str, str]:
        return {"markdown": render_research_report(state.plan, state.traces)}

    @app.post("/api/traces/{controller}/disrupt")
    def disrupt(controller: str, payload: Disruption) -> dict[str, object]:
        if controller not in state.schedules:
            raise HTTPException(status_code=404, detail=f"unknown controller: {controller}")
        try:
            recovered = inject_disruption(
                state.plan,
                state.schedules[controller],
                payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return recovered.model_dump(mode="json")

    @app.post("/api/design/rebuild")
    def rebuild(payload: RebuildRequest) -> dict[str, object]:
        try:
            state.replace_design(payload.design)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        registry.upsert_scenario(
            state.design.design_id,
            seed=None,
            split="reviewed",
            payload={
                "design": state.design.model_dump(mode="json"),
                "plan": state.plan.model_dump(mode="json"),
            },
        )
        return project()

    @app.post("/api/intent/parse")
    async def parse_intent(
        file: UploadFile = File(...),
        known_width_m: float = Query(gt=1, le=40),
    ) -> dict[str, object]:
        try:
            inferred = infer_orthogonal_floor_plan(
                await file.read(), known_width_m=known_width_m
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return inferred.model_dump(mode="json")

    @app.get("/api/lab/scenarios")
    def scenarios() -> list[dict[str, object]]:
        return registry.list_scenarios()

    @app.post("/api/lab/scenarios", status_code=201)
    def generate_scenario(payload: ScenarioGenerationRequest) -> dict[str, object]:
        scenario = generate_cottage_scenario(payload.seed, state.design)
        serialized = scenario.model_dump(mode="json")
        registry.upsert_scenario(
            scenario.scenario_id,
            seed=scenario.seed,
            split=scenario.split.value,
            payload=serialized,
        )
        return serialized

    @app.get("/api/lab/scenarios/{scenario_id}")
    def scenario(scenario_id: str) -> dict[str, object]:
        item = registry.get_scenario(scenario_id)
        if item is None:
            raise HTTPException(status_code=404, detail="scenario not found")
        return item

    @app.get("/api/lab/policies")
    def policies() -> list[dict[str, object]]:
        return registry.list_policies()

    @app.get("/api/lab/evaluations")
    def evaluations(
        matrix_id: str | None = Query(default=None),
    ) -> list[dict[str, object]]:
        return registry.list_evaluations(matrix_id=matrix_id)

    @app.get("/api/lab/runs")
    def runs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, object]]:
        return registry.list_runs(limit=limit)

    @app.get("/api/lab/runs/{run_id}")
    def run(run_id: str) -> dict[str, object]:
        item = registry.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item

    @app.get(
        "/api/lab/runs/{run_id}/events",
        response_model_exclude_unset=True,
    )
    def run_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> list[RunEventEnvelope]:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return registry.list_event_envelopes(run_id, after=after)

    @app.post("/api/lab/experiment-matrices", status_code=202)
    def launch_experiment_matrix(
        payload: ExperimentMatrixLaunchRequest,
    ) -> dict[str, object]:
        if not payload.confirmed:
            raise HTTPException(
                status_code=409,
                detail="experiment matrix launch requires confirmed=true",
            )
        protocol = cast(ExperimentProtocol, app.state.experiment_protocol)
        digest = protocol_digest(protocol)
        specs = expand_experiment_matrix(protocol, payload.profile)
        from embodied_skill_composer.construction.training import TrainingConfig

        runs = [
            (
                spec.run_id,
                TrainingConfig.model_validate(spec.training_config_payload()),
            )
            for spec in specs
        ]
        matrix_id = f"{protocol.experiment_id}-{payload.profile}-{digest[:12]}"
        try:
            run_ids = service.launch_training_matrix(
                state.design,
                matrix_id=matrix_id,
                protocol_digest=digest,
                protocol=protocol.model_dump(mode="json"),
                execution_profile=payload.profile,
                runs=runs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "matrix_id": matrix_id,
            "protocol_digest": digest,
            "profile": payload.profile,
            "run_count": len(run_ids),
            "run_ids": run_ids,
        }

    @app.get("/api/lab/experiment-matrices")
    def experiment_matrices() -> list[dict[str, object]]:
        return registry.list_experiment_matrices()

    @app.get("/api/lab/experiment-matrices/{matrix_id}")
    def experiment_matrix(matrix_id: str) -> dict[str, object]:
        item = registry.get_experiment_matrix(matrix_id)
        if item is None:
            raise HTTPException(status_code=404, detail="experiment matrix not found")
        return item

    @app.get("/api/lab/experiment-matrices/{matrix_id}/selections")
    def experiment_matrix_selections(matrix_id: str) -> list[dict[str, object]]:
        if registry.get_experiment_matrix(matrix_id) is None:
            raise HTTPException(status_code=404, detail="experiment matrix not found")
        return registry.list_policy_selections(matrix_id)

    @app.put(
        "/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
        status_code=201,
    )
    def register_validation_selection(
        matrix_id: str,
        run_key: str,
        payload: ValidationSelectionRequest,
    ) -> dict[str, object]:
        if not payload.confirmed:
            raise HTTPException(
                status_code=409,
                detail="validation selection requires confirmed=true",
            )
        matrix = registry.get_experiment_matrix(matrix_id)
        if matrix is None:
            raise HTTPException(status_code=404, detail="experiment matrix not found")
        matrix_protocol = ExperimentProtocol.model_validate(matrix["protocol"])
        profile = cast(ExperimentProfile, matrix["execution_profile"])
        expected = {
            item.run_id: item
            for item in expand_experiment_matrix(matrix_protocol, profile)
        }
        if run_key not in expected:
            raise HTTPException(status_code=404, detail="matrix run not found")
        matrix_runs = cast(list[dict[str, object]], matrix["runs"])
        persisted_run = next(item for item in matrix_runs if item["run_key"] == run_key)
        persisted_input = cast(dict[str, object], persisted_run["input"])
        persisted_design = HouseDesign.model_validate(persisted_input["design"])
        evidence_root = registry.path.parent / "evidence" / "validation_selections"
        try:
            evidence = evaluate_and_freeze_run_selection(
                registry,
                matrix_id,
                run_key,
                persisted_design,
                matrix_protocol,
                output_root=evidence_root,
                device="cpu",
            )
            matrix_evidence = materialize_matrix_selection_evidence(
                registry,
                matrix_id,
                matrix_protocol,
                output_root=evidence_root,
            )
        except (
            EOFError,
            KeyError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "matrix_id": matrix_id,
            "run_key": run_key,
            "selection": evidence.selected.model_dump(mode="json"),
            "evidence_path": str(evidence.evidence_path),
            "matrix_evidence_path": (
                str(matrix_evidence.evidence_path)
                if matrix_evidence is not None
                else None
            ),
        }

    @app.post(
        "/api/lab/experiment-matrices/{matrix_id}/heldout",
        status_code=202,
    )
    def launch_heldout_matrix(
        matrix_id: str,
        payload: HeldoutMatrixLaunchRequest,
    ) -> dict[str, object]:
        if not payload.confirmed:
            raise HTTPException(
                status_code=409,
                detail="held-out evaluation launch requires confirmed=true",
            )
        matrix = registry.get_experiment_matrix(matrix_id)
        if matrix is None:
            raise HTTPException(status_code=404, detail="experiment matrix not found")
        matrix_protocol = ExperimentProtocol.model_validate(matrix["protocol"])
        profile = cast(ExperimentProfile, matrix["execution_profile"])
        expected = {
            item.run_id: item
            for item in expand_experiment_matrix(matrix_protocol, profile)
        }
        selections = registry.list_policy_selections(matrix_id)
        selected = {
            cast(str, item["run_key"]): cast(dict[str, object], item["selection"])
            for item in selections
        }
        missing = sorted(set(expected) - set(selected))
        if len(selected) != 20 or missing:
            raise HTTPException(
                status_code=409,
                detail=(
                    "held-out evaluation requires all 20 frozen validation "
                    f"selections; registered={len(selected)}, missing={missing}"
                ),
            )
        for run_key in expected:
            checkpoint_path = selected[run_key].get("checkpoint_path")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                raise HTTPException(
                    status_code=409,
                    detail=f"selection has no checkpoint path: {run_key}",
                )
        matrix_runs = cast(list[dict[str, object]], matrix["runs"])
        persisted_input = cast(dict[str, object], matrix_runs[0]["input"])
        persisted_design = HouseDesign.model_validate(persisted_input["design"])
        evidence_root = registry.path.parent / "evidence"
        try:
            matrix_evidence = materialize_matrix_selection_evidence(
                registry,
                matrix_id,
                matrix_protocol,
                output_root=evidence_root / "validation_selections",
            )
            if matrix_evidence is None:
                raise ValueError(
                    "held-out evaluation requires canonical matrix selection evidence"
                )
            evaluation_run_id = service.launch_matrix_evaluation(
                persisted_design,
                matrix_id=matrix_id,
                protocol=matrix_protocol,
                output_root=evidence_root / "heldout" / matrix_id,
                selection_evidence_path=matrix_evidence.evidence_path,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "matrix_id": matrix_id,
            "selection_count": len(selected),
            "evaluation_run_count": 1,
            "run_ids": [evaluation_run_id],
            "selection_evidence_path": str(matrix_evidence.evidence_path),
        }

    @app.post("/api/lab/training", status_code=202)
    def launch_training(payload: TrainingLaunchRequest) -> dict[str, str]:
        if not payload.confirmed:
            raise HTTPException(
                status_code=409,
                detail="training requires confirmed=true",
            )
        if payload.algorithm not in {"mappo", "ippo"}:
            raise HTTPException(status_code=422, detail="algorithm must be mappo or ippo")
        if payload.profile not in {"unit", "smoke", "research"}:
            raise HTTPException(status_code=422, detail="unknown training profile")
        if payload.device not in {"auto", "cpu", "cuda"}:
            raise HTTPException(status_code=422, detail="unknown training device")
        from embodied_skill_composer.construction.training import TrainingConfig

        config = TrainingConfig.for_profile(
            payload.profile,
            algorithm=payload.algorithm,
            seed=payload.seed,
        )
        config.device = payload.device
        if payload.transitions is not None:
            config.transitions = payload.transitions
        try:
            return {"run_id": service.launch_training(state.design, config)}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/lab/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str) -> dict[str, object]:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        accepted = service.cancel(run_id)
        if not accepted:
            raise HTTPException(status_code=409, detail="run is already terminal")
        return {"run_id": run_id, "cancel_requested": True}

    @app.post("/api/lab/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str) -> dict[str, object]:
        item = registry.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        accepted = service.resume(run_id)
        if not accepted:
            raise HTTPException(
                status_code=409,
                detail="run is not resumable or has no durable checkpoint",
            )
        return {"run_id": run_id, "status": "resuming"}

    @app.post("/api/lab/evaluations", status_code=202)
    def launch_evaluation(payload: EvaluationLaunchRequest) -> dict[str, str]:
        allowed = {"sequential", "greedy", "auction", "ippo", "mappo", "cp_sat"}
        unknown = sorted(set(payload.controllers) - allowed)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown controllers: {unknown}")
        policy_records = {item["id"]: item for item in registry.list_policies()}
        checkpoints: dict[str, str] = {}
        for controller in ("mappo", "ippo"):
            if controller not in payload.controllers:
                continue
            policy_id = payload.policy_ids.get(controller)
            record = policy_records.get(policy_id)
            if record is None:
                raise HTTPException(status_code=422, detail=f"missing {controller} policy")
            manifest = PolicyManifest.model_validate(record["manifest"])
            if not manifest.checkpoint_path:
                raise HTTPException(
                    status_code=422,
                    detail=f"{controller} policy has no checkpoint",
                )
            checkpoints[controller] = manifest.checkpoint_path
        run_id = service.launch_evaluation(
            state.design,
            seeds=payload.seeds,
            controllers=payload.controllers,
            policy_checkpoints=checkpoints,
            include_failures=payload.include_failures,
            output_root=WORKSPACE / "logs" / "construction_intelligence" / "evaluations",
        )
        return {"run_id": run_id}

    @app.get("/api/lab/coppelia/health")
    def coppelia_health(
        port: int = Query(default=23000, ge=1, le=65535),
    ) -> dict[str, object]:
        host = "127.0.0.1"
        try:
            with socket.create_connection((host, port), timeout=0.4):
                reachable = True
                detail = "ZeroMQ remote API port is reachable"
        except OSError as exc:
            reachable = False
            detail = str(exc)
        return {
            "reachable": reachable,
            "host": host,
            "port": port,
            "detail": detail,
            "controller": "dynamic_base_logical_payload",
        }

    @app.websocket("/api/lab/runs/{run_id}/events/ws")
    async def stream_run_events(websocket: WebSocket, run_id: str) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in LOCAL_ORIGINS:
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        await websocket.accept()
        if registry.get_run(run_id) is None:
            await websocket.close(code=1008, reason="run not found")
            return
        sequence = 0
        while True:
            events = registry.list_event_envelopes(run_id, after=sequence)
            for event in events:
                sequence = event.sequence
                await websocket.send_json(
                    event.model_dump(mode="json", exclude_unset=True)
                )
            current = registry.get_run(run_id)
            if current and current["status"] in QUIESCENT_RUN_STATUSES and not events:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(0.25)

    if DEFAULT_ASSETS.is_dir():
        app.mount("/artifacts", StaticFiles(directory=DEFAULT_ASSETS), name="artifacts")
    return app


app = create_app()
