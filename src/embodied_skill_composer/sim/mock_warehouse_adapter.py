from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from embodied_skill_composer.core.models import (
    ObjectState,
    RobotState,
    SensorObservation,
    StationState,
    WorldState,
    ZoneState,
)
from embodied_skill_composer.rl.grasp_policy import load_grasp_policy


def _as_vector3(values: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


class MockWarehouseAdapter:
    """Waypoint-based warehouse collection environment for tests, demos, and RL experiments."""

    COLOR_MAP: dict[str, tuple[int, int, int]] = {
        "red": (220, 64, 64),
        "blue": (64, 96, 220),
        "green": (64, 190, 96),
        "yellow": (220, 200, 80),
    }

    def __init__(self, runtime_config: dict[str, Any], scene_config: dict[str, Any]) -> None:
        self.runtime_config = runtime_config
        self.scene_config = scene_config
        self.random = Random(int(runtime_config.get("seed", 0)))
        self.robot = RobotState(
            end_effector_position=(0.0, 0.0, 0.5),
            gripper_opening=0.08,
            base_position=(0.0, 0.0, 0.0),
            navigation_node="dock",
        )
        self.objects: dict[str, ObjectState] = {}
        self.zones: dict[str, ZoneState] = {}
        self.stations: dict[str, StationState] = {}
        self.policy_path = Path(runtime_config.get("rl_policy_path", "logs/grasp_policy.json"))
        self.reset(runtime_config.get("seed", 0))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.random.seed(int(seed))
        self.robot = RobotState(
            end_effector_position=(0.0, 0.0, 0.5),
            gripper_opening=0.08,
            base_position=(0.0, 0.0, 0.0),
            navigation_node="dock",
        )
        self.zones = {
            name: ZoneState(
                name=name,
                center=_as_vector3(payload["center"]),
                size=_as_vector3(payload["size"]),
            )
            for name, payload in self.scene_config["zones"].items()
        }
        self.stations = {
            name: StationState(
                name=name,
                position=_as_vector3(payload["position"]),
                kind=payload.get("kind", "pickup"),
                capacity=int(payload.get("capacity", 1)),
            )
            for name, payload in self.scene_config["stations"].items()
        }

        station_names = [name for name, station in self.stations.items() if station.kind == "pickup"]
        randomized = self.scene_config.get("randomize_object_positions", True)
        self.objects = {}
        for name, payload in self.scene_config["objects"].items():
            station_name = str(payload.get("station_name", station_names[0]))
            if randomized:
                station_name = station_names[self.random.randrange(len(station_names))]
            station = self.stations[station_name]
            x_jitter = self.random.uniform(-0.03, 0.03)
            y_jitter = self.random.uniform(-0.03, 0.03)
            self.objects[name] = ObjectState(
                name=name,
                color_name=payload["color_name"],
                position=(station.position[0] + x_jitter, station.position[1] + y_jitter, station.position[2]),
                size=_as_vector3(payload["size"]),
                station_name=station_name,
            )

    def get_world_state(self) -> WorldState:
        return WorldState(
            robot=self.robot.model_copy(),
            objects={name: obj.model_copy() for name, obj in self.objects.items()},
            zones={name: zone.model_copy() for name, zone in self.zones.items()},
            stations={name: station.model_copy() for name, station in self.stations.items()},
        )

    def get_oracle_world_state(self) -> WorldState:
        return self.get_world_state()

    def capture_observation(self) -> SensorObservation:
        width = 180
        height = 90
        rgb = [[[245, 245, 245] for _ in range(width)] for _ in range(height)]
        pickup_stations = [name for name, station in self.stations.items() if station.kind == "pickup"]
        pickup_stations.sort()
        slot_map: dict[str, tuple[int, int]] = {}
        if not pickup_stations:
            return SensorObservation(camera_name="warehouse_topdown", rgb=rgb, resolution=(width, height))

        x_step = width // (len(pickup_stations) + 1)
        y_mid = height // 2
        for index, station_name in enumerate(pickup_stations, start=1):
            x_center = x_step * index
            slot_map[station_name] = (x_center, y_mid)
            for y in range(y_mid - 16, y_mid + 16):
                for x in range(x_center - 16, x_center + 16):
                    rgb[y][x] = [230, 230, 230]

        for obj in self.objects.values():
            if obj.collected or obj.held or not obj.station_name or obj.station_name not in slot_map:
                continue
            x_center, y_center = slot_map[obj.station_name]
            color = self.COLOR_MAP[obj.color_name]
            siblings = sorted(
                [
                    candidate.name
                    for candidate in self.objects.values()
                    if candidate.station_name == obj.station_name and not candidate.collected and not candidate.held
                ]
            )
            offset_index = siblings.index(obj.name)
            x_offset = (offset_index * 14) - 7
            for y in range(y_center - 10, y_center + 10):
                for x in range(x_center - 10 + x_offset, x_center + 10 + x_offset):
                    rgb[y][x] = [color[0], color[1], color[2]]

        return SensorObservation(
            camera_name="warehouse_topdown",
            rgb=rgb,
            station_slots=slot_map,
            resolution=(width, height),
        )

    def navigate_to_waypoint(self, waypoint_name: str) -> bool:
        if waypoint_name in self.stations:
            waypoint = self.stations[waypoint_name].position
        elif waypoint_name in self.zones:
            waypoint = self.zones[waypoint_name].center
        else:
            return False
        self.robot.navigation_node = waypoint_name
        self.robot.base_position = waypoint
        self.robot.end_effector_position = (waypoint[0], waypoint[1], waypoint[2] + 0.45)
        return True

    def pick_object(self, object_name: str, policy_mode: str = "scripted") -> bool:
        if object_name not in self.objects:
            return False
        obj = self.objects[object_name]
        if obj.collected or obj.held or obj.station_name != self.robot.navigation_node:
            return False
        clutter_level = sum(
            1
            for candidate in self.objects.values()
            if candidate.station_name == obj.station_name and not candidate.collected and not candidate.held
        )
        scripted_threshold = 0.85 - (0.1 * max(0, clutter_level - 1))
        score = self.random.random()
        if policy_mode == "rl":
            policy = load_grasp_policy(self.policy_path)
            threshold = policy.success_threshold(clutter_level)
        else:
            threshold = scripted_threshold
        success = score < threshold
        if not success:
            return False
        self.robot.holding_object = object_name
        self.robot.gripper_opening = 0.0
        self.objects[object_name] = obj.model_copy(
            update={
                "held": True,
                "station_name": None,
                "position": (
                    self.robot.end_effector_position[0],
                    self.robot.end_effector_position[1],
                    self.robot.end_effector_position[2] - 0.08,
                ),
            }
        )
        return True

    def deliver_held_object(self, zone_name: str) -> bool:
        held_name = self.robot.holding_object
        if held_name is None or zone_name not in self.zones:
            return False
        zone = self.zones[zone_name]
        held = self.objects[held_name]
        self.objects[held_name] = held.model_copy(
            update={"held": False, "collected": True, "position": zone.center, "station_name": zone_name}
        )
        self.robot.holding_object = None
        self.robot.gripper_opening = 0.08
        return True

    def move_to(self, target_position: tuple[float, float, float], yaw: float = 0.0) -> bool:
        self.robot.end_effector_position = target_position
        return True

    def open_gripper(self) -> bool:
        self.robot.gripper_opening = 0.08
        return True

    def close_gripper(self) -> bool:
        self.robot.gripper_opening = 0.0
        return True

    def attempt_grasp(self, object_name: str, approach_offset: float = 0.04) -> bool:
        return self.pick_object(object_name, policy_mode="scripted")

    def lift_object(self, height: float) -> bool:
        if self.robot.holding_object is None:
            return False
        x, y, z = self.robot.end_effector_position
        self.robot.end_effector_position = (x, y, z + height)
        return True

    def place_held_object(self, target_position: tuple[float, float, float]) -> bool:
        held_name = self.robot.holding_object
        if held_name is None:
            return False
        held = self.objects[held_name]
        self.objects[held_name] = held.model_copy(update={"held": False, "position": target_position})
        self.robot.holding_object = None
        self.robot.gripper_opening = 0.08
        return True

    def resolve_zone_center(self, zone_name: str) -> tuple[float, float, float]:
        return self.zones[zone_name].center

    def resolve_object_position(self, object_name: str) -> tuple[float, float, float]:
        return self.objects[object_name].position

    def resolve_stack_position(self, object_name: str) -> tuple[float, float, float]:
        x, y, z = self.objects[object_name].position
        return (x, y, z + self.objects[object_name].size[2] * 2)

    def close(self) -> None:
        return None
