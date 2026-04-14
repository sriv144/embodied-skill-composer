from embodied_skill_composer.core.models import TaskSpec, TaskType
from embodied_skill_composer.perception.modules import build_perception
from embodied_skill_composer.sim.mock_warehouse_adapter import MockWarehouseAdapter


def build_runtime() -> dict:
    return {"seed": 7, "log_dir": "logs", "rl_policy_path": "logs/grasp_policy.json"}


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


def test_classical_perception_detects_station_objects() -> None:
    adapter = MockWarehouseAdapter(runtime_config=build_runtime(), scene_config=build_scene())
    task = TaskSpec(
        name="collection",
        task_type=TaskType.MULTI_OBJECT_COLLECTION,
        source_object="target_red",
        target_objects=["target_red", "target_blue"],
        drop_zone="tote",
    )

    world, report = build_perception("classical_cv").build_world(adapter, task)

    assert set(world.objects) == {"target_red", "target_blue"}
    assert report.missed_targets == []
    assert report.station_predictions["shelf_a"] == "target_red"
