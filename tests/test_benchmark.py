from pathlib import Path

from embodied_skill_composer.benchmark import BenchmarkRunner
from embodied_skill_composer.core.models import TaskSpec, TaskType
from embodied_skill_composer.sim.mock_warehouse_adapter import MockWarehouseAdapter


def build_runtime(seed: int = 7) -> dict:
    return {"seed": seed, "log_dir": "logs", "rl_policy_path": "logs/grasp_policy.json"}


def build_scene() -> dict:
    return {
        "randomize_object_positions": False,
        "zones": {"tote": {"center": [0.0, -0.8, 0.0], "size": [0.2, 0.2, 0.05]}},
        "stations": {
            "dock": {"position": [0.0, -1.2, 0.0], "kind": "dock"},
            "shelf_a": {"position": [-0.8, 0.0, 0.0], "kind": "pickup"},
            "shelf_b": {"position": [0.0, 0.0, 0.0], "kind": "pickup"},
        },
        "objects": {
            "target_red": {"color_name": "red", "size": [0.03, 0.03, 0.03], "station_name": "shelf_a"},
            "target_blue": {"color_name": "blue", "size": [0.03, 0.03, 0.03], "station_name": "shelf_b"},
        },
    }


def test_benchmark_smoke_outputs_summary(tmp_path: Path) -> None:
    task = TaskSpec(
        name="collection",
        task_type=TaskType.MULTI_OBJECT_COLLECTION,
        source_object="target_red",
        target_objects=["target_red", "target_blue"],
        drop_zone="tote",
        perception_mode="oracle",
        policy_mode="scripted",
    )
    runner = BenchmarkRunner(
        adapter_factory=lambda seed: MockWarehouseAdapter(build_runtime(seed), build_scene()),
        log_dir=tmp_path,
    )

    summary = runner.run(task, seeds=[7, 11])

    assert summary.episodes == 2
    assert 0.0 <= summary.success_rate <= 1.0
