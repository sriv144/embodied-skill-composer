from pathlib import Path

from embodied_skill_composer.core.executor import TaskExecutor
from embodied_skill_composer.core.models import (
    ObjectState,
    RobotState,
    SkillStep,
    TaskSpec,
    TaskType,
    WorldState,
    ZoneState,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.world = WorldState(
            robot=RobotState(end_effector_position=(0.4, 0.0, 0.3), gripper_opening=0.08),
            objects={
                "red_block": ObjectState(
                    name="red_block",
                    color_name="red",
                    position=(0.45, -0.1, 0.03),
                    size=(0.025, 0.025, 0.025),
                )
            },
            zones={"tray": ZoneState(name="tray", center=(0.62, -0.22, 0.02), size=(0.12, 0.12, 0.01))},
        )
        self.grasp_attempts = 0

    def reset(self, seed: int | None = None) -> None:
        return None

    def get_world_state(self) -> WorldState:
        return self.world

    def move_to(self, target_position: tuple[float, float, float], yaw: float = 0.0) -> bool:
        self.world.robot.end_effector_position = target_position
        return True

    def open_gripper(self) -> bool:
        self.world.robot.gripper_opening = 0.08
        return True

    def close_gripper(self) -> bool:
        self.world.robot.gripper_opening = 0.0
        return True

    def attempt_grasp(self, object_name: str, approach_offset: float = 0.04) -> bool:
        self.grasp_attempts += 1
        if self.grasp_attempts < 2:
            return False
        self.world.robot.holding_object = object_name
        self.world.objects[object_name].held = True
        return True

    def lift_object(self, height: float) -> bool:
        return self.world.robot.holding_object is not None

    def place_held_object(self, target_position: tuple[float, float, float]) -> bool:
        held_name = self.world.robot.holding_object
        if held_name is None:
            return False
        self.world.objects[held_name].position = target_position
        self.world.objects[held_name].held = False
        self.world.robot.holding_object = None
        return True

    def resolve_zone_center(self, zone_name: str) -> tuple[float, float, float]:
        return self.world.zones[zone_name].center

    def resolve_object_position(self, object_name: str) -> tuple[float, float, float]:
        return self.world.objects[object_name].position

    def resolve_stack_position(self, object_name: str) -> tuple[float, float, float]:
        pos = self.world.objects[object_name].position
        return (pos[0], pos[1], pos[2] + 0.05)

    def close(self) -> None:
        return None


def test_executor_retries_failed_grasp(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    executor = TaskExecutor(adapter=adapter, log_dir=tmp_path)
    task = TaskSpec(
        name="retry_demo",
        task_type=TaskType.PICK_AND_PLACE,
        source_object="red_block",
        target_zone="tray",
    )
    plan = [
        SkillStep(name="open_gripper"),
        SkillStep(
            name="grasp_object",
            params={"object_name": "red_block", "approach_offset": 0.04},
            max_retries=2,
        ),
        SkillStep(name="lift_object", params={"height": 0.16}),
        SkillStep(name="place_object", params={"target_zone": "tray"}),
    ]

    report = executor.run(task, plan)

    assert report.success is True
    assert adapter.grasp_attempts == 2
    assert any(event.step_name == "grasp_object" and not event.success for event in report.events)
    assert report.log_path is not None

