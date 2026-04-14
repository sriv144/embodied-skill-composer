from pathlib import Path

from embodied_skill_composer.core.models import TaskSpec, TaskType
from embodied_skill_composer.pipelines.collection import CollectionEpisodeRunner
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
            "shelf_c": {"position": [0.8, 0.0, 0.0], "kind": "pickup"},
        },
        "objects": {
            "target_red": {"color_name": "red", "size": [0.03, 0.03, 0.03], "station_name": "shelf_a"},
            "target_blue": {"color_name": "blue", "size": [0.03, 0.03, 0.03], "station_name": "shelf_b"},
            "distractor_green": {"color_name": "green", "size": [0.03, 0.03, 0.03], "station_name": "shelf_c"},
        },
    }


def test_collection_episode_oracle_mode_collects_all_targets(tmp_path: Path) -> None:
    adapter = MockWarehouseAdapter(runtime_config=build_runtime(), scene_config=build_scene())
    task = TaskSpec(
        name="collection",
        task_type=TaskType.MULTI_OBJECT_COLLECTION,
        source_object="target_red",
        target_objects=["target_red", "target_blue"],
        drop_zone="tote",
        perception_mode="oracle",
        policy_mode="scripted",
    )

    result = CollectionEpisodeRunner(adapter=adapter, log_dir=tmp_path).run(task)

    assert result.report.success is True
    assert result.objects_collected == 2
    assert result.report.final_world.objects["target_red"].collected is True
