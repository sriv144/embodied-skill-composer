# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.pipelines.collection import CollectionEpisodeRunner
from embodied_skill_composer.sim.mock_warehouse_adapter import MockWarehouseAdapter
from embodied_skill_composer.tasks.catalog import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the warehouse collection flagship task.")
    parser.add_argument("--task", default="warehouse_multi_object_collection")
    parser.add_argument("--perception", choices=["oracle", "classical_cv"], default="classical_cv")
    parser.add_argument("--policy", choices=["scripted", "rl"], default="scripted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = yaml.safe_load((PROJECT_ROOT / "configs" / "warehouse_runtime.yaml").read_text())
    scene = yaml.safe_load((PROJECT_ROOT / "configs" / "warehouse_scene.yaml").read_text())
    tasks = load_tasks(PROJECT_ROOT / "configs" / "warehouse_tasks.yaml")
    task = tasks[args.task].model_copy(update={"perception_mode": args.perception, "policy_mode": args.policy})

    adapter = MockWarehouseAdapter(runtime_config=runtime, scene_config=scene)
    try:
        runner = CollectionEpisodeRunner(adapter=adapter, log_dir=PROJECT_ROOT / runtime["log_dir"])
        result = runner.run(task)
        print(f"Task: {task.name}")
        print(task.description)
        print(f"Perception: {task.perception_mode}")
        print(f"Pickup policy: {task.policy_mode}")
        print(f"Success: {result.report.success}")
        print(f"Collected targets: {result.objects_collected}/{len(task.target_objects)}")
        print(f"Completion rate: {result.target_completion_rate:.2f}")
        print(f"Grasp retry rate: {result.grasp_retry_rate:.2f}")
        print(f"Perception miss rate: {result.perception_miss_rate:.2f}")
        print(f"Execution log: {result.report.log_path}")
        return 0 if result.report.success else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
