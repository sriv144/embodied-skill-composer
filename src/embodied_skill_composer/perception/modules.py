from __future__ import annotations

from collections import Counter

from embodied_skill_composer.core.interfaces import SimulationAdapter
from embodied_skill_composer.core.models import ObjectState, PerceptionReport, SensorObservation, TaskSpec, WorldState


class OraclePerception:
    mode = "oracle"

    def build_world(self, adapter: SimulationAdapter, task: TaskSpec) -> tuple[WorldState, PerceptionReport]:
        if hasattr(adapter, "get_oracle_world_state"):
            world = getattr(adapter, "get_oracle_world_state")()
        else:
            world = adapter.get_world_state()
        detected = [name for name, obj in world.objects.items() if not obj.collected]
        report = PerceptionReport(
            mode=self.mode,
            detected_objects=detected,
            missed_targets=[name for name in task.target_objects if name not in detected],
            station_predictions={obj.station_name or "unknown": obj.name for obj in world.objects.values() if obj.station_name},
            confidence_by_object={name: 1.0 for name in detected},
        )
        return world, report


class ClassicalWarehousePerception:
    mode = "classical_cv"
    COLOR_MAP = {
        (220, 64, 64): "red",
        (64, 96, 220): "blue",
        (64, 190, 96): "green",
        (220, 200, 80): "yellow",
    }

    def build_world(self, adapter: SimulationAdapter, task: TaskSpec) -> tuple[WorldState, PerceptionReport]:
        if not hasattr(adapter, "capture_observation") or not hasattr(adapter, "get_oracle_world_state"):
            raise TypeError("classical warehouse perception requires a collection adapter")
        observation: SensorObservation = getattr(adapter, "capture_observation")()
        oracle_world: WorldState = getattr(adapter, "get_oracle_world_state")()

        detected_objects: dict[str, ObjectState] = {}
        station_predictions: dict[str, str | None] = {}
        confidence: dict[str, float] = {}

        for station_name, (x_center, y_center) in observation.station_slots.items():
            sampled: list[tuple[int, int, int]] = []
            for y in range(y_center - 8, y_center + 8):
                for x in range(x_center - 18, x_center + 18):
                    pixel = tuple(observation.rgb[y][x])
                    if pixel != (230, 230, 230) and pixel != (245, 245, 245):
                        sampled.append(pixel)
            if not sampled:
                station_predictions[station_name] = None
                continue
            predicted_names: list[str] = []
            for dominant, count in Counter(sampled).most_common():
                if count < 20:
                    continue
                color_name = self.COLOR_MAP.get(dominant)
                if color_name is None:
                    continue
                matching = [
                    obj
                    for obj in oracle_world.objects.values()
                    if obj.color_name == color_name and obj.station_name == station_name and not obj.collected
                ]
                if not matching:
                    continue
                obj = matching[0]
                predicted_names.append(obj.name)
                detected_objects[obj.name] = obj
                confidence[obj.name] = min(0.95, count / max(1, len(sampled)))
            station_predictions[station_name] = ",".join(predicted_names) if predicted_names else None

        perceived_world = WorldState(
            robot=oracle_world.robot,
            objects={name: obj for name, obj in detected_objects.items()},
            zones=oracle_world.zones,
            stations={name: station for name, station in oracle_world.stations.items()},
        )
        report = PerceptionReport(
            mode=self.mode,
            detected_objects=sorted(detected_objects),
            missed_targets=[name for name in task.target_objects if name not in detected_objects],
            station_predictions=station_predictions,
            confidence_by_object=confidence,
        )
        return perceived_world, report


def build_perception(mode: str):
    if mode == "oracle":
        return OraclePerception()
    if mode in {"classical_cv", "perception"}:
        return ClassicalWarehousePerception()
    raise ValueError(f"Unsupported perception mode: {mode}")
