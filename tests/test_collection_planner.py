from embodied_skill_composer.core.models import (
    ObjectState,
    RobotState,
    StationState,
    TaskSpec,
    TaskType,
    WorldState,
    ZoneState,
)
from embodied_skill_composer.core.planner import RuleBasedPlanner


def test_collection_planner_sequences_navigation_pick_and_delivery() -> None:
    world = WorldState(
        robot=RobotState(
            end_effector_position=(0.0, 0.0, 0.5),
            gripper_opening=0.08,
            base_position=(0.0, 0.0, 0.0),
            navigation_node="dock",
        ),
        objects={
            "target_red": ObjectState(
                name="target_red",
                color_name="red",
                position=(-0.8, 0.0, 0.0),
                size=(0.03, 0.03, 0.03),
                station_name="shelf_a",
            ),
            "target_blue": ObjectState(
                name="target_blue",
                color_name="blue",
                position=(0.0, 0.0, 0.0),
                size=(0.03, 0.03, 0.03),
                station_name="shelf_b",
            ),
        },
        zones={"tote": ZoneState(name="tote", center=(0.0, -0.8, 0.0), size=(0.2, 0.2, 0.05))},
        stations={
            "shelf_a": StationState(name="shelf_a", position=(-0.8, 0.0, 0.0)),
            "shelf_b": StationState(name="shelf_b", position=(0.0, 0.0, 0.0)),
        },
    )
    task = TaskSpec(
        name="collection",
        task_type=TaskType.MULTI_OBJECT_COLLECTION,
        source_object="target_red",
        target_objects=["target_red", "target_blue"],
        drop_zone="tote",
        perception_mode="classical_cv",
        policy_mode="scripted",
    )

    plan = RuleBasedPlanner().plan(task, world)

    assert [step.name for step in plan[:5]] == [
        "observe_scene",
        "navigate_to_waypoint",
        "pick_object",
        "navigate_to_waypoint",
        "deliver_object",
    ]
    assert sum(1 for step in plan if step.name == "pick_object") == 2
