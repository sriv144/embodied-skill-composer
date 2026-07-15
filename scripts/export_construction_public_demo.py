from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.browser_assets import (  # noqa: E402
    generate_browser_robot_asset,
)
from embodied_skill_composer.construction.compiler import compile_house_design  # noqa: E402
from embodied_skill_composer.construction.recovery import (  # noqa: E402
    Disruption,
    inject_disruption,
)
from embodied_skill_composer.construction.reporting import (  # noqa: E402
    render_research_report,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402
from embodied_skill_composer.construction.scheduler import compare_controllers  # noqa: E402
from embodied_skill_composer.construction.trace import build_execution_trace  # noqa: E402


def export_public_demo(output_dir: Path, *, regenerate_robot: bool = True) -> None:
    design = load_house_design(WORKSPACE / "configs" / "construction" / "cottage_v1.yaml")
    plan = compile_house_design(design)
    schedules = compare_controllers(plan)
    traces = {
        name: build_execution_trace(plan, schedule)
        for name, schedule in schedules.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    house_source = WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1" / "house.glb"
    if not house_source.is_file():
        raise FileNotFoundError(
            "Generate the cottage first with scripts/generate_construction_assets.py"
        )
    shutil.copy2(house_source, output_dir / "house.glb")
    robot_path = output_dir / "construction_robot.glb"
    if regenerate_robot or not robot_path.is_file():
        generate_browser_robot_asset(robot_path)

    sequential = schedules["sequential"].makespan_s
    optimized = schedules["optimized"].makespan_s
    project = {
        "design": design.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "controllers": {
            name: trace.metrics.model_dump(mode="json")
            for name, trace in traces.items()
        },
        "optimized_improvement_percent": round(100 * (1 - optimized / sequential), 1),
        "geometry_asset_url": "house.glb",
        "robot_asset_url": "construction_robot.glb",
    }
    _write_json(output_dir / "project.json", project)
    for name, trace in traces.items():
        _write_json(trace_dir / f"{name}.json", trace.model_dump(mode="json"))
    recovery = inject_disruption(
        plan,
        schedules["optimized"],
        Disruption(failure_type="obstacle", timestamp_s=72),
    )
    _write_json(trace_dir / "recovery.json", recovery.model_dump(mode="json"))
    _write_json(
        output_dir / "scenarios.json",
        [
            {
                "id": design.design_id,
                "seed": None,
                "split": "fixture",
                "payload": {"module_count": len(plan.modules), "title": design.title},
                "created_at": "2026-07-15T00:00:00Z",
            }
        ],
    )
    _write_json(output_dir / "policies.json", [])
    _write_json(
        output_dir / "runs.json",
        [
            {
                "id": "fixture_baseline_evaluation",
                "kind": "evaluation",
                "status": "completed",
                "config": {
                    "controllers": ["sequential", "greedy", "cp_sat"],
                    "scope": "deterministic_cottage_fixture",
                },
                "created_at": "2026-07-15T00:00:00Z",
                "started_at": "2026-07-15T00:00:00Z",
                "ended_at": "2026-07-15T00:00:01Z",
                "progress": 1.0,
                "artifact_dir": "demo",
                "error": None,
            }
        ],
    )
    report = render_research_report(plan, traces)
    report += (
        "\n## Public Demo Provenance\n\n"
        "Exported deterministically from the reviewed `cottage_v1` fixture. "
        "No trained MAPPO or IPPO result is included in this curated bundle.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the read-only public research demo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "workbench" / "public" / "demo",
    )
    parser.add_argument("--reuse-robot", action="store_true")
    args = parser.parse_args()
    export_public_demo(args.output, regenerate_robot=not args.reuse_robot)
    print(args.output)


if __name__ == "__main__":
    main()
