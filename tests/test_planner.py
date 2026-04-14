from embodied_skill_composer.core.models import ObjectState, RobotState, TaskSpec, TaskType, WorldState, ZoneState
from embodied_skill_composer.core.planner import RuleBasedPlanner


def build_world() -> WorldState:
    return WorldState(
        robot=RobotState(end_effector_position=(0.4, 0.0, 0.3), gripper_opening=0.08),
        objects={
            "red_block": ObjectState(
                name="red_block",
                color_name="red",
                position=(0.45, -0.1, 0.03),
                size=(0.025, 0.025, 0.025),
            ),
            "blue_block": ObjectState(
                name="blue_block",
                color_name="blue",
                position=(0.45, 0.02, 0.03),
                size=(0.025, 0.025, 0.025),
            ),
            "green_block": ObjectState(
                name="green_block",
                color_name="green",
                position=(0.45, 0.16, 0.03),
                size=(0.03, 0.03, 0.03),
            ),
        },
        zones={
            "tray": ZoneState(name="tray", center=(0.62, -0.22, 0.02), size=(0.12, 0.12, 0.01)),
            "blue_zone": ZoneState(
                name="blue_zone", center=(0.62, 0.22, 0.02), size=(0.12, 0.12, 0.01)
            ),
        },
    )


def test_pick_and_place_plan_sequence() -> None:
    planner = RuleBasedPlanner()
    world = build_world()
    task = TaskSpec(
        name="demo",
        task_type=TaskType.PICK_AND_PLACE,
        source_object="red_block",
        target_zone="tray",
    )
    plan = planner.plan(task, world)

    assert [step.name for step in plan] == [
        "open_gripper",
        "move_to_pose",
        "grasp_object",
        "lift_object",
        "move_to_pose",
        "place_object",
    ]
    assert plan[-1].params["target_zone"] == "tray"


def test_sort_uses_color_routing() -> None:
    planner = RuleBasedPlanner()
    world = build_world()
    task = TaskSpec(
        name="sort_demo",
        task_type=TaskType.SORT_TO_ZONE,
        source_object="blue_block",
        color_routing={"blue": "blue_zone"},
    )
    plan = planner.plan(task, world)

    assert plan[-1].params["target_zone"] == "blue_zone"

