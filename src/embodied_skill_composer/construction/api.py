from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.floorplan import infer_orthogonal_floor_plan
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.reporting import render_research_report
from embodied_skill_composer.construction.recovery import Disruption, inject_disruption
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scheduler import compare_controllers
from embodied_skill_composer.construction.trace import build_execution_trace


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
DEFAULT_ASSETS = WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1"


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


def create_app() -> FastAPI:
    app = FastAPI(title="Embodied Skill Composer Construction Workbench", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = WorkbenchState(load_house_design(DEFAULT_DESIGN))

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ready",
            "design_id": state.design.design_id,
            "module_count": len(state.plan.modules),
            "robot_count": len(state.plan.robots),
            "cp_sat": state.schedules["optimized"].solver_status,
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

    if DEFAULT_ASSETS.is_dir():
        app.mount("/artifacts", StaticFiles(directory=DEFAULT_ASSETS), name="artifacts")
    return app


app = create_app()
