from __future__ import annotations

import asyncio
import pickle
import socket
from contextlib import asynccontextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.design_validation import (
    DesignValidationError,
    DesignValidationResult,
    validate_house_design,
)
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
from embodied_skill_composer.construction.floorplan import (
    MAX_FLOORPLAN_ENCODED_BYTES,
    infer_orthogonal_floor_plan,
)
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
from embodied_skill_composer.construction.workbench_evidence import (
    artifact_references,
    build_local_coppelia_summary,
    build_local_research_summary,
    phase5_run_directory,
    resolve_artifact,
)


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
DEFAULT_ASSETS = WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1"
DEFAULT_LAB_DATABASE = WORKSPACE / "logs" / "construction_intelligence" / "lab.sqlite"
DEFAULT_COPPELIA_EVIDENCE = (
    WORKSPACE / "logs" / "construction_intelligence" / "coppelia_phase5"
)
LOCAL_ORIGINS = frozenset(
    {
        "http://127.0.0.1:4173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://localhost:5173",
    }
)
UNSAFE_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
TRUSTED_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})
MAX_FLOORPLAN_MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAX_FLOORPLAN_MULTIPART_BODY_BYTES = (
    MAX_FLOORPLAN_ENCODED_BYTES + MAX_FLOORPLAN_MULTIPART_OVERHEAD_BYTES
)
API_SECURITY_RESPONSE_HEADERS = (
    (
        b"content-security-policy",
        b"frame-ancestors 'none'; object-src 'none'; base-uri 'none'; "
        b"form-action 'self'",
    ),
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
)


class SecurityResponseHeadersMiddleware:
    """Apply browser hardening headers to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _value in headers}
                headers.extend(
                    (name, value)
                    for name, value in API_SECURITY_RESPONSE_HEADERS
                    if name not in existing
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class FloorplanUploadBodyLimitMiddleware:
    """Bound the multipart request before Starlette parses or spools it."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or str(scope.get("method", "")).upper() != "POST"
            or scope.get("path") != "/api/intent/parse"
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=400,
                    detail="invalid Content-Length for floor-plan upload",
                )
                return
            if declared_length < 0:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=400,
                    detail="invalid Content-Length for floor-plan upload",
                )
                return
            if declared_length > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return

        received_bytes = 0
        request_messages: list[Message] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            request_messages.append(message)
            if (
                message["type"] != "http.request"
                or not message.get("more_body", False)
            ):
                break

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index
            if message_index < len(request_messages):
                message = request_messages[message_index]
                message_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int = 413,
        detail: str = (
            "floor-plan multipart request exceeds the bounded upload limit"
        ),
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
        )
        await response(scope, receive, send)


class UnsafeMethodOriginMiddleware:
    """Reject browser cross-site mutations while retaining trusted CLI access."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: frozenset[str],
    ) -> None:
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or str(scope.get("method", "")).upper() not in UNSAFE_HTTP_METHODS
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        origin = headers.get(b"origin")
        fetch_site = headers.get(b"sec-fetch-site")
        rejected = (
            origin is not None and origin not in self.allowed_origins
        ) or (
            fetch_site is not None
            and fetch_site.lower() not in TRUSTED_FETCH_SITES
        )
        if rejected:
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "cross-site mutation rejected; use the loopback "
                        "workbench origin or a trusted no-Origin CLI client"
                    )
                },
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class WorkbenchState:
    def __init__(self, design: HouseDesign):
        self.replace_design(design)

    def replace_design(self, design: HouseDesign) -> None:
        candidate_design = design.model_copy(deep=True)
        candidate_plan = compile_house_design(candidate_design)
        candidate_schedules = compare_controllers(candidate_plan)
        candidate_traces = {
            name: build_execution_trace(candidate_plan, schedule)
            for name, schedule in candidate_schedules.items()
        }
        self.design = candidate_design
        self.plan = candidate_plan
        self.schedules = candidate_schedules
        self.traces = candidate_traces


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
    allowed_origins: Iterable[str] | None = None,
    coppelia_evidence_root: Path | None = None,
) -> FastAPI:
    local_origins = _validated_local_origins(allowed_origins or LOCAL_ORIGINS)
    phase5_root = (coppelia_evidence_root or DEFAULT_COPPELIA_EVIDENCE).resolve()
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
        FloorplanUploadBodyLimitMiddleware,
        max_body_bytes=MAX_FLOORPLAN_MULTIPART_BODY_BYTES,
    )
    app.add_middleware(
        UnsafeMethodOriginMiddleware,
        allowed_origins=local_origins,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(local_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityResponseHeadersMiddleware)
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
        except DesignValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.result.model_dump(mode="json"),
            ) from exc
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

    @app.post("/api/design/validate", response_model=DesignValidationResult)
    def validate_design(payload: RebuildRequest) -> DesignValidationResult:
        return validate_house_design(payload.design)

    @app.post("/api/intent/parse")
    async def parse_intent(
        file: UploadFile = File(...),
        known_width_m: float = Query(gt=1, le=40),
    ) -> dict[str, object]:
        if (
            file.size is not None
            and file.size > MAX_FLOORPLAN_ENCODED_BYTES
        ):
            raise HTTPException(
                status_code=413,
                detail="floor-plan upload exceeds the 8 MiB encoded-image limit",
            )
        image_bytes = await file.read(MAX_FLOORPLAN_ENCODED_BYTES + 1)
        if len(image_bytes) > MAX_FLOORPLAN_ENCODED_BYTES:
            raise HTTPException(
                status_code=413,
                detail="floor-plan upload exceeds the 8 MiB encoded-image limit",
            )
        try:
            inferred = infer_orthogonal_floor_plan(
                image_bytes,
                known_width_m=known_width_m,
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

    @app.get("/api/lab/evidence/research-summary")
    def research_summary() -> dict[str, object]:
        return build_local_research_summary(registry)

    @app.get("/api/lab/evidence/coppelia")
    def coppelia_evidence() -> dict[str, object]:
        return build_local_coppelia_summary(phase5_root)

    @app.get("/api/lab/runs")
    def runs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, object]]:
        return registry.list_runs(limit=limit)

    @app.get("/api/lab/runs/{run_id}")
    def run(run_id: str) -> dict[str, object]:
        item = registry.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item

    @app.get("/api/lab/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str) -> list[dict[str, str]]:
        item = registry.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        root = _registered_artifact_root(item)
        if root is None:
            return []
        return artifact_references(
            root,
            f"/api/lab/runs/{quote(run_id, safe='')}/artifacts",
        )

    @app.get("/api/lab/runs/{run_id}/artifacts/{artifact_path:path}")
    def run_artifact(run_id: str, artifact_path: str) -> FileResponse:
        item = registry.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        root = _registered_artifact_root(item)
        if root is None:
            raise HTTPException(status_code=404, detail="run has no artifacts")
        return _artifact_response(root, artifact_path)

    @app.get("/api/lab/evaluations/{evaluation_id}/artifacts")
    def evaluation_artifacts(evaluation_id: str) -> list[dict[str, str]]:
        item = _evaluation_record(registry, evaluation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        root = Path(str(item["artifact_dir"]))
        return artifact_references(
            root,
            f"/api/lab/evaluations/{quote(evaluation_id, safe='')}/artifacts",
        )

    @app.get(
        "/api/lab/evaluations/{evaluation_id}/artifacts/{artifact_path:path}"
    )
    def evaluation_artifact(
        evaluation_id: str,
        artifact_path: str,
    ) -> FileResponse:
        item = _evaluation_record(registry, evaluation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return _artifact_response(Path(str(item["artifact_dir"])), artifact_path)

    @app.get("/api/lab/coppelia/evidence/{run_id}/artifacts")
    def coppelia_artifacts(run_id: str) -> list[dict[str, str]]:
        try:
            root = phase5_run_directory(phase5_root, run_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail="validated Coppelia evidence run not found",
            ) from exc
        return artifact_references(
            root,
            f"/api/lab/coppelia/evidence/{quote(run_id, safe='')}/artifacts",
        )

    @app.get(
        "/api/lab/coppelia/evidence/{run_id}/artifacts/{artifact_path:path}"
    )
    def coppelia_artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            root = phase5_run_directory(phase5_root, run_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail="validated Coppelia evidence run not found",
            ) from exc
        return _artifact_response(root, artifact_path)

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
        if origin is not None and origin not in local_origins:
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        await websocket.accept()
        if registry.get_run(run_id) is None:
            await websocket.close(code=1008, reason="run not found")
            return
        await _stream_run_events(websocket, registry, run_id)

    if DEFAULT_ASSETS.is_dir():
        app.mount("/artifacts", StaticFiles(directory=DEFAULT_ASSETS), name="artifacts")
    return app


async def _stream_run_events(
    websocket: WebSocket,
    registry: LabRegistry,
    run_id: str,
) -> None:
    """Stream persisted envelopes until the run or browser connection closes."""

    try:
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
    except WebSocketDisconnect:
        return


def _validated_local_origins(origins: Iterable[str]) -> frozenset[str]:
    validated: set[str] = set()
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "workbench origins must be explicit HTTP(S) loopback origins "
                "with a port and no path, query, credentials, or fragment"
            )
        validated.add(origin.rstrip("/"))
    if not validated:
        raise ValueError("at least one explicit loopback workbench origin is required")
    return frozenset(validated)


def _registered_artifact_root(item: dict[str, object]) -> Path | None:
    value = item.get("artifact_dir")
    return Path(value) if isinstance(value, str) and value else None


def _evaluation_record(
    registry: LabRegistry,
    evaluation_id: str,
) -> dict[str, object] | None:
    return next(
        (
            item
            for item in registry.list_evaluations()
            if item.get("id") == evaluation_id
        ),
        None,
    )


def _artifact_response(root: Path, artifact_path: str) -> FileResponse:
    try:
        path = resolve_artifact(root, artifact_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="attachment",
    )


app = create_app()
