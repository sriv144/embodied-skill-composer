from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from embodied_skill_composer.assembly.models import (
    VisualObjectEstimate,
    VisualPerceptionConfig,
    VisualPerceptionEvaluation,
    VisualPerceptionFeedback,
    VisualTerminalAssessment,
)


@dataclass(frozen=True)
class AssemblyVisualFrame:
    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    camera_position_m: np.ndarray
    camera_rotation: np.ndarray
    vertical_fov_degrees: float

    @property
    def resolution(self) -> tuple[int, int]:
        return (int(self.rgb.shape[1]), int(self.rgb.shape[0]))


class ClassicalAssemblyPerception:
    _HSV_RANGES = {
        "agent0": ((0, 120, 50), (7, 255, 255)),
        "agent1": ((65, 130, 40), (90, 255, 255)),
        "resource": ((9, 110, 80), (22, 255, 255)),
        "blueprint_cell": ((85, 60, 100), (100, 255, 255)),
    }

    def __init__(self, config: VisualPerceptionConfig) -> None:
        self.config = config

    def estimate(
        self,
        frame: AssemblyVisualFrame,
        sample_index: int,
    ) -> VisualPerceptionFeedback:
        import cv2

        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        estimates: list[VisualObjectEstimate] = []
        for detector_name, (lower, upper) in self._HSV_RANGES.items():
            mask = cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
            components: list[VisualObjectEstimate] = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                minimum_area = self.config.minimum_component_area_px
                if detector_name == "blueprint_cell":
                    minimum_area = max(minimum_area, 100)
                if area < minimum_area:
                    continue
                centroid = (float(centroids[label, 0]), float(centroids[label, 1]))
                component_mask = labels == label
                position = self._backproject_component(frame, component_mask, centroid)
                category = "agent" if detector_name.startswith("agent") else detector_name
                components.append(
                    VisualObjectEstimate(
                        track_id=detector_name,
                        category=category,
                        centroid_px=centroid,
                        position_m=position,
                        bounding_box_xywh=(
                            int(stats[label, cv2.CC_STAT_LEFT]),
                            int(stats[label, cv2.CC_STAT_TOP]),
                            int(stats[label, cv2.CC_STAT_WIDTH]),
                            int(stats[label, cv2.CC_STAT_HEIGHT]),
                        ),
                        pixel_area=area,
                        confidence=self._component_confidence(
                            area,
                            hsv[component_mask],
                        ),
                    )
                )
            components.sort(key=lambda item: (item.position_m[0], item.position_m[1]))
            if len(components) > 1 and detector_name in {"resource", "blueprint_cell"}:
                components = [
                    component.model_copy(
                        update={"track_id": f"{detector_name}_{index}"}
                    )
                    for index, component in enumerate(components)
                ]
            estimates.extend(components)

        detected_counts = {
            category: sum(estimate.category == category for estimate in estimates)
            for category in ("agent", "resource", "blueprint_cell")
        }
        confidence_values = [estimate.confidence for estimate in estimates]
        return VisualPerceptionFeedback(
            camera_name=frame.camera_name,
            resolution=frame.resolution,
            sample_index=sample_index,
            estimates=estimates,
            detected_counts=detected_counts,
            tracked_counts=dict(detected_counts),
            mean_confidence=(
                0.0 if not confidence_values else float(np.mean(confidence_values))
            ),
        )

    @staticmethod
    def _backproject_component(
        frame: AssemblyVisualFrame,
        component_mask: np.ndarray,
        centroid: tuple[float, float],
    ) -> tuple[float, float, float]:
        depth_values = frame.depth_m[component_mask]
        depth_values = depth_values[np.isfinite(depth_values) & (depth_values < 100.0)]
        if depth_values.size == 0:
            return (0.0, 0.0, 0.0)
        depth = float(np.median(depth_values))
        width, height = frame.resolution
        fy = 0.5 * height / np.tan(np.deg2rad(frame.vertical_fov_degrees) / 2.0)
        fx = fy * width / height
        camera_point = np.asarray(
            [
                (centroid[0] + 0.5 - width / 2.0) * depth / fx,
                -(centroid[1] + 0.5 - height / 2.0) * depth / fy,
                -depth,
            ]
        )
        world_point = frame.camera_position_m + frame.camera_rotation @ camera_point
        return tuple(float(value) for value in world_point)

    @staticmethod
    def _component_confidence(area: int, hsv_pixels: np.ndarray) -> float:
        saturation = float(np.mean(hsv_pixels[:, 1])) / 255.0
        area_score = min(1.0, area / 150.0)
        return float(min(1.0, 0.5 * saturation + 0.5 * area_score))


@dataclass
class _VisualTrack:
    estimate: VisualObjectEstimate
    velocity_m_per_sample: np.ndarray


class MultiObjectVisualTracker:
    def __init__(self, config: VisualPerceptionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[str, _VisualTrack] = {}
        self._next_track_index = {
            "agent": 0,
            "resource": 0,
            "blueprint_cell": 0,
        }

    def update(
        self,
        feedback: VisualPerceptionFeedback,
    ) -> VisualPerceptionFeedback:
        if not self.config.tracking_enabled:
            return feedback.model_copy(
                update={
                    "tracked_counts": dict(feedback.detected_counts),
                    "predicted_estimate_count": 0,
                }
            )

        detections_by_category = {
            category: [
                estimate
                for estimate in feedback.estimates
                if estimate.category == category
            ]
            for category in ("agent", "resource", "blueprint_cell")
        }
        updated_track_ids: set[str] = set()
        for category, detections in detections_by_category.items():
            category_track_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if track.estimate.category == category
            ]
            matches = self._match_category(category, category_track_ids, detections)
            matched_detection_indices: set[int] = set()
            for track_id, detection_index in matches:
                self._update_track(track_id, detections[detection_index])
                updated_track_ids.add(track_id)
                matched_detection_indices.add(detection_index)
            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detection_indices:
                    continue
                track_id = self._new_track_id(detection)
                self._tracks[track_id] = _VisualTrack(
                    estimate=detection.model_copy(update={"track_id": track_id}),
                    velocity_m_per_sample=np.zeros(3, dtype=np.float64),
                )
                updated_track_ids.add(track_id)

        expired: list[str] = []
        for track_id, track in self._tracks.items():
            if track_id in updated_track_ids:
                continue
            missed_frames = track.estimate.missed_frames + 1
            if missed_frames > self.config.tracking_max_missed_frames:
                expired.append(track_id)
                continue
            velocity_scale = 0.0
            if track.estimate.category != "blueprint_cell":
                velocity_scale = 0.5**missed_frames
            position = (
                np.asarray(track.estimate.position_m)
                + track.velocity_m_per_sample * velocity_scale
            )
            track.estimate = track.estimate.model_copy(
                update={
                    "position_m": tuple(float(value) for value in position),
                    "confidence": (
                        track.estimate.confidence
                        * self.config.prediction_confidence_decay
                    ),
                    "is_predicted": True,
                    "track_age": track.estimate.track_age + 1,
                    "missed_frames": missed_frames,
                }
            )
        for track_id in expired:
            del self._tracks[track_id]

        estimates = [
            track.estimate
            for _, track in sorted(
                self._tracks.items(),
                key=lambda item: (item[1].estimate.category, item[0]),
            )
        ]
        tracked_counts = {
            category: sum(estimate.category == category for estimate in estimates)
            for category in ("agent", "resource", "blueprint_cell")
        }
        confidence_values = [estimate.confidence for estimate in estimates]
        return feedback.model_copy(
            update={
                "estimates": estimates,
                "tracked_counts": tracked_counts,
                "predicted_estimate_count": sum(
                    estimate.is_predicted for estimate in estimates
                ),
                "mean_confidence": (
                    0.0
                    if not confidence_values
                    else float(np.mean(confidence_values))
                ),
            }
        )

    def _match_category(
        self,
        category: str,
        track_ids: list[str],
        detections: list[VisualObjectEstimate],
    ) -> list[tuple[str, int]]:
        matches: list[tuple[str, int]] = []
        used_tracks: set[str] = set()
        used_detections: set[int] = set()
        if category == "agent":
            for detection_index, detection in enumerate(detections):
                if detection.track_id in track_ids:
                    matches.append((detection.track_id, detection_index))
                    used_tracks.add(detection.track_id)
                    used_detections.add(detection_index)

        max_distance = self.config.tracking_max_match_distance_m
        if category == "blueprint_cell":
            max_distance = min(max_distance, 0.4)
        candidates = []
        for track_id in track_ids:
            if track_id in used_tracks:
                continue
            track_position = np.asarray(self._tracks[track_id].estimate.position_m[:2])
            for detection_index, detection in enumerate(detections):
                if detection_index in used_detections:
                    continue
                distance = float(
                    np.linalg.norm(
                        track_position - np.asarray(detection.position_m[:2])
                    )
                )
                candidates.append((distance, track_id, detection_index))
        for distance, track_id, detection_index in sorted(candidates):
            if distance > max_distance:
                continue
            if track_id in used_tracks or detection_index in used_detections:
                continue
            matches.append((track_id, detection_index))
            used_tracks.add(track_id)
            used_detections.add(detection_index)
        return matches

    def _update_track(
        self,
        track_id: str,
        detection: VisualObjectEstimate,
    ) -> None:
        track = self._tracks[track_id]
        velocity = np.asarray(detection.position_m) - np.asarray(
            track.estimate.position_m
        )
        track.velocity_m_per_sample = velocity
        track.estimate = detection.model_copy(
            update={
                "track_id": track_id,
                "is_predicted": False,
                "track_age": track.estimate.track_age + 1,
                "missed_frames": 0,
            }
        )

    def _new_track_id(self, detection: VisualObjectEstimate) -> str:
        if detection.category == "agent" and detection.track_id not in self._tracks:
            return detection.track_id
        while True:
            index = self._next_track_index[detection.category]
            self._next_track_index[detection.category] += 1
            track_id = f"{detection.category}_{index}"
            if track_id not in self._tracks:
                return track_id


def assess_visual_terminal_readiness(
    feedback: VisualPerceptionFeedback,
    carrying: bool,
    config: VisualPerceptionConfig,
) -> VisualTerminalAssessment | None:
    if not config.estimated_state_control_enabled:
        return None
    phase = "install" if carrying else "grasp"
    eligible = [
        estimate
        for estimate in feedback.estimates
        if estimate.confidence >= config.control_min_track_confidence
    ]
    agents = [estimate for estimate in eligible if estimate.category == "agent"]
    resources = [
        estimate for estimate in eligible if estimate.category == "resource"
    ]
    if len(agents) < 2:
        return VisualTerminalAssessment(
            phase=phase,
            reason="insufficient_agent_tracks",
            agent_track_ids=[estimate.track_id for estimate in agents],
            minimum_track_confidence=_minimum_confidence(agents),
            uses_predicted_tracks=any(estimate.is_predicted for estimate in agents),
        )
    agents = sorted(agents, key=lambda estimate: estimate.track_id)[:2]
    if not resources:
        return VisualTerminalAssessment(
            phase=phase,
            reason="insufficient_resource_tracks",
            agent_track_ids=[estimate.track_id for estimate in agents],
            minimum_track_confidence=_minimum_confidence(agents),
            uses_predicted_tracks=any(estimate.is_predicted for estimate in agents),
        )

    resource_candidates = []
    for resource in resources:
        distances = [
            _planar_distance(agent.position_m, resource.position_m)
            for agent in agents
        ]
        resource_candidates.append((max(distances), resource))
    agent_resource_distance, resource = min(
        resource_candidates,
        key=lambda candidate: candidate[0],
    )
    agent_resource_tolerance = (
        config.install_agent_resource_tolerance_m
        if carrying
        else config.grasp_agent_resource_tolerance_m
    )
    selected = [*agents, resource]
    if agent_resource_distance > agent_resource_tolerance:
        return VisualTerminalAssessment(
            phase=phase,
            reason="agent_resource_misaligned",
            agent_track_ids=[estimate.track_id for estimate in agents],
            resource_track_id=resource.track_id,
            max_agent_resource_distance_m=agent_resource_distance,
            minimum_track_confidence=_minimum_confidence(selected),
            uses_predicted_tracks=any(estimate.is_predicted for estimate in selected),
        )
    if not carrying:
        return VisualTerminalAssessment(
            phase=phase,
            ready=True,
            reason="ready",
            agent_track_ids=[estimate.track_id for estimate in agents],
            resource_track_id=resource.track_id,
            max_agent_resource_distance_m=agent_resource_distance,
            minimum_track_confidence=_minimum_confidence(selected),
            uses_predicted_tracks=any(estimate.is_predicted for estimate in selected),
        )

    blueprint_cells = [
        estimate
        for estimate in eligible
        if estimate.category == "blueprint_cell"
    ]
    if len(blueprint_cells) < 2:
        return VisualTerminalAssessment(
            phase=phase,
            reason="insufficient_blueprint_tracks",
            agent_track_ids=[estimate.track_id for estimate in agents],
            resource_track_id=resource.track_id,
            max_agent_resource_distance_m=agent_resource_distance,
            blueprint_track_ids=[estimate.track_id for estimate in blueprint_cells],
            minimum_track_confidence=_minimum_confidence([*selected, *blueprint_cells]),
            uses_predicted_tracks=any(
                estimate.is_predicted for estimate in [*selected, *blueprint_cells]
            ),
        )
    blueprint_distances = sorted(
        [
            (
                _planar_distance(resource.position_m, blueprint.position_m),
                blueprint,
            )
            for blueprint in blueprint_cells
        ],
        key=lambda item: item[0],
    )
    selected_blueprints = [item[1] for item in blueprint_distances[:2]]
    resource_blueprint_distance = blueprint_distances[1][0]
    selected.extend(selected_blueprints)
    ready = (
        resource_blueprint_distance
        <= config.install_resource_blueprint_tolerance_m
    )
    return VisualTerminalAssessment(
        phase=phase,
        ready=ready,
        reason="ready" if ready else "resource_blueprint_misaligned",
        agent_track_ids=[estimate.track_id for estimate in agents],
        resource_track_id=resource.track_id,
        blueprint_track_ids=[
            estimate.track_id for estimate in selected_blueprints
        ],
        max_agent_resource_distance_m=agent_resource_distance,
        max_resource_blueprint_distance_m=resource_blueprint_distance,
        minimum_track_confidence=_minimum_confidence(selected),
        uses_predicted_tracks=any(estimate.is_predicted for estimate in selected),
    )


def _minimum_confidence(estimates: list[VisualObjectEstimate]) -> float:
    return min((estimate.confidence for estimate in estimates), default=0.0)


def _planar_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return float(np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])))


def evaluate_visual_perception(
    feedback: VisualPerceptionFeedback,
    truth_positions: dict[str, list[tuple[float, float, float]]],
) -> VisualPerceptionEvaluation:
    expected_counts: dict[str, int] = {}
    visible_matched_counts: dict[str, int] = {}
    matched_counts: dict[str, int] = {}
    visible_recall: dict[str, float] = {}
    recall: dict[str, float] = {}
    errors: list[float] = []
    for category in ("agent", "resource", "blueprint_cell"):
        truth = truth_positions.get(category, [])
        estimates = [
            estimate.position_m
            for estimate in feedback.estimates
            if estimate.category == category
        ]
        visible_estimates = [
            estimate.position_m
            for estimate in feedback.estimates
            if estimate.category == category and not estimate.is_predicted
        ]
        expected_counts[category] = len(truth)
        matched = min(len(truth), len(estimates))
        visible_matched = min(len(truth), len(visible_estimates))
        matched_counts[category] = matched
        visible_matched_counts[category] = visible_matched
        visible_recall[category] = (
            1.0 if not truth else visible_matched / len(truth)
        )
        recall[category] = 1.0 if not truth else matched / len(truth)
        errors.extend(_minimum_assignment_errors(estimates, truth))
    return VisualPerceptionEvaluation(
        sample_index=feedback.sample_index,
        expected_counts=expected_counts,
        detected_counts=dict(feedback.detected_counts),
        tracked_counts=dict(feedback.tracked_counts),
        visible_matched_counts=visible_matched_counts,
        matched_counts=matched_counts,
        visible_recall_by_category=visible_recall,
        recall_by_category=recall,
        position_errors_m=errors,
        mean_position_error_m=0.0 if not errors else float(np.mean(errors)),
        max_position_error_m=0.0 if not errors else float(np.max(errors)),
    )


def _minimum_assignment_errors(
    estimates: list[tuple[float, float, float]],
    truth: list[tuple[float, float, float]],
) -> list[float]:
    matched = min(len(estimates), len(truth))
    if matched == 0:
        return []
    if len(estimates) <= len(truth):
        candidates = permutations(truth, matched)
        scored = [
            [
                float(
                    np.linalg.norm(
                        np.asarray(estimate[:2]) - np.asarray(target[:2])
                    )
                )
                for estimate, target in zip(estimates, candidate)
            ]
            for candidate in candidates
        ]
    else:
        candidates = permutations(estimates, matched)
        scored = [
            [
                float(
                    np.linalg.norm(
                        np.asarray(estimate[:2]) - np.asarray(target[:2])
                    )
                )
                for estimate, target in zip(candidate, truth)
            ]
            for candidate in candidates
        ]
    return min(scored, key=sum)
