# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.core.executor import TaskExecutor
from embodied_skill_composer.core.planner import RuleBasedPlanner
from embodied_skill_composer.tasks.catalog import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Embodied Skill Composer demo task.")
    parser.add_argument("--task", help="Task name to execute")
    parser.add_argument("--list", action="store_true", help="List available task names")
    parser.add_argument(
        "--backend",
        choices=["mock", "pybullet"],
        default="mock",
        help="Simulation backend to use. PyBullet is optional on Windows.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a PyBullet GUI window. Only applies to the pybullet backend.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="When using --gui, keep the simulator open briefly after execution for inspection.",
    )
    parser.add_argument(
        "--save-camera",
        help="Optional output path for a top-down camera snapshot after execution.",
    )
    return parser.parse_args()


def save_ppm(rgb: list[list[list[int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = len(rgb)
    width = len(rgb[0]) if height else 0
    with output_path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in rgb:
            for pixel in row:
                handle.write(bytes(pixel[:3]))


def build_adapter(backend: str, runtime_config: dict, scene_config: dict, gui: bool = False):
    if backend == "mock":
        from embodied_skill_composer.sim.mock_adapter import MockTabletopAdapter

        return MockTabletopAdapter(runtime_config=runtime_config, scene_config=scene_config)

    try:
        from embodied_skill_composer.sim.pybullet_adapter import PyBulletTabletopAdapter
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyBullet backend requested, but pybullet is not installed. "
            "On Windows, run scripts\\setup_pybullet_backend.ps1 and then launch through "
            ".\\.tools\\micromamba\\micromamba.exe run -p .\\.mmenv python ... "
            "or use --backend mock."
        ) from exc
    return PyBulletTabletopAdapter(
        runtime_config=runtime_config,
        scene_config=scene_config,
        gui=gui,
    )


def main() -> int:
    args = parse_args()
    runtime_config = yaml.safe_load((PROJECT_ROOT / "configs" / "runtime.yaml").read_text())
    scene_config = yaml.safe_load((PROJECT_ROOT / "configs" / "scene.yaml").read_text())
    tasks = load_tasks(PROJECT_ROOT / "configs" / "tasks.yaml")

    if args.list:
        print("Available tasks:")
        for name in tasks:
            print(f"  - {name}")
        return 0

    if not args.task:
        print("Please pass --task or use --list to inspect demos.")
        return 1
    if args.task not in tasks:
        print(f"Unknown task: {args.task}")
        return 1

    task = tasks[args.task]
    adapter = build_adapter(args.backend, runtime_config, scene_config, gui=args.gui)
    try:
        planner = RuleBasedPlanner()
        plan = planner.plan(task, adapter.get_world_state())

        print(f"Task: {task.name}")
        print(task.description)
        print(f"Backend: {args.backend}")
        if args.backend == "pybullet":
            print(f"GUI: {args.gui}")
        print("Planned skill sequence:")
        for index, step in enumerate(plan, start=1):
            print(f"  {index}. {step.name} {step.params}")

        executor = TaskExecutor(adapter=adapter, log_dir=PROJECT_ROOT / runtime_config["log_dir"])
        report = executor.run(task, plan)

        print(f"\nSuccess: {report.success}")
        if report.failure_step:
            print(f"Failure step: {report.failure_step}")
        print(f"Execution log: {report.log_path}")
        for event in report.events:
            status = "ok" if event.success else "fail"
            print(f"  - [{status}] {event.step_name} (attempt {event.attempt}): {event.message}")
        if args.save_camera and hasattr(adapter, "capture_observation"):
            observation = adapter.capture_observation()
            output_path = Path(args.save_camera)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path
            save_ppm(observation.rgb, output_path)
            print(f"Saved camera snapshot: {output_path}")
        if args.gui and args.backend == "pybullet" and args.hold_seconds > 0:
            print(f"Holding GUI open for {args.hold_seconds:.1f}s for inspection...")
            time.sleep(args.hold_seconds)
        return 0 if report.success else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
