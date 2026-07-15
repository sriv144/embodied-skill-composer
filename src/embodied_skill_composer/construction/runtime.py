from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.models import (
    ArchitecturalIntent,
    BuildPlan,
    ExecutionTrace,
    HouseDesign,
)
from embodied_skill_composer.construction.reporting import render_research_report
from embodied_skill_composer.construction.scheduler import compare_controllers
from embodied_skill_composer.construction.trace import build_execution_trace


def load_house_design(path: Path) -> HouseDesign:
    with path.open(encoding="utf-8") as stream:
        return HouseDesign.model_validate(yaml.safe_load(stream))


def load_architectural_intent(path: Path) -> ArchitecturalIntent:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) if path.suffix.lower() in {".yaml", ".yml"} else json.load(stream)
    return ArchitecturalIntent.model_validate(payload)


def run_construction_workbench(
    design_path: Path,
    output_root: Path,
) -> tuple[Path, BuildPlan, dict[str, ExecutionTrace]]:
    design = load_house_design(design_path)
    plan = compile_house_design(design)
    schedules = compare_controllers(plan)
    traces = {name: build_execution_trace(plan, schedule) for name, schedule in schedules.items()}
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{design.design_id}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "house_design.json", design.model_dump(mode="json"))
    _write_json(run_dir / "build_plan.json", plan.model_dump(mode="json"))
    for name, trace in traces.items():
        _write_json(run_dir / f"schedule_{name}.json", trace.schedule.model_dump(mode="json"))
        _write_json(run_dir / f"execution_trace_{name}.json", trace.model_dump(mode="json"))
    _write_json(
        run_dir / "metrics.json",
        {name: trace.metrics.model_dump(mode="json") for name, trace in traces.items()},
    )
    (run_dir / "report.md").write_text(
        render_research_report(plan, traces), encoding="utf-8"
    )
    return run_dir, plan, traces


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
