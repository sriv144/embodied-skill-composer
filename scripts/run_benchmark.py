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

from embodied_skill_composer.benchmark import BenchmarkRunner
from embodied_skill_composer.sim.mock_warehouse_adapter import MockWarehouseAdapter
from embodied_skill_composer.tasks.catalog import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark warehouse collection episodes.")
    parser.add_argument("--task", default="warehouse_multi_object_collection")
    parser.add_argument("--perception", choices=["oracle", "classical_cv"], default="classical_cv")
    parser.add_argument("--policy", choices=["scripted", "rl"], default="scripted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = yaml.safe_load((PROJECT_ROOT / "configs" / "warehouse_runtime.yaml").read_text())
    scene = yaml.safe_load((PROJECT_ROOT / "configs" / "warehouse_scene.yaml").read_text())
    task = load_tasks(PROJECT_ROOT / "configs" / "warehouse_tasks.yaml")[args.task]
    task = task.model_copy(update={"perception_mode": args.perception, "policy_mode": args.policy})
    seeds = [int(seed) for seed in runtime["benchmark"]["seeds"]]

    runner = BenchmarkRunner(
        adapter_factory=lambda seed: MockWarehouseAdapter(
            runtime_config=runtime | {"seed": seed},
            scene_config=scene,
        ),
        log_dir=PROJECT_ROOT / runtime["log_dir"],
    )
    summary = runner.run(task, seeds=seeds)
    print(f"Episodes: {summary.episodes}")
    print(f"Success rate: {summary.success_rate:.2f}")
    print(f"Collection completion: {summary.collection_completion_rate:.2f}")
    print(f"Average objects collected: {summary.average_objects_collected:.2f}")
    print(f"Grasp retry rate: {summary.grasp_retry_rate:.2f}")
    print(f"Perception miss rate: {summary.perception_miss_rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
