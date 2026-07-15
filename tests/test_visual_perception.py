from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.brain import (
    HeuristicConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    TeamOption,
    VisualObjectEstimate,
    VisualPerceptionConfig,
    VisualPerceptionFeedback,
    VisualTerminalAssessment,
)
from embodied_skill_composer.assembly.perception import (
    ClassicalAssemblyPerception,
    MultiObjectVisualTracker,
    assess_visual_terminal_readiness,
    evaluate_visual_perception,
)
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
)


mujoco_available = importlib.util.find_spec("mujoco") is not None
workspace = Path(__file__).resolve().parents[1]


def build_env(profile: AssemblyRuntimeProfile | None = None):
    config = load_assembly_scenario(workspace / "configs" / "assembly_env.yaml")
    env = build_assembly_backend(
        config,
        profile or AssemblyRuntimeProfile(name="vision_test", backend="mujoco_local"),
        seed=7,
    )
    env.reset(seed=7)
    return env


@pytest.mark.parametrize(
    "payload",
    [
        {"width": 63},
        {"height": 1025},
        {"minimum_component_area_px": 0},
        {"tracking_max_missed_frames": -1},
        {"tracking_max_match_distance_m": 0.0},
        {"prediction_confidence_decay": 0.0},
        {"control_min_track_confidence": 1.01},
        {"grasp_agent_resource_tolerance_m": 0.0},
        {"install_resource_blueprint_tolerance_m": 0.0},
    ],
)
def test_visual_perception_config_rejects_invalid_values(
    payload: dict[str, int | float],
) -> None:
    with pytest.raises(ValidationError):
        VisualPerceptionConfig.model_validate(payload)


def build_blueprint_feedback(
    sample_index: int,
    positions: list[tuple[float, float, float]],
) -> VisualPerceptionFeedback:
    estimates = [
        VisualObjectEstimate(
            track_id=f"blueprint_cell_{index}",
            category="blueprint_cell",
            centroid_px=(10.0 + index, 20.0),
            position_m=position,
            bounding_box_xywh=(10 + index, 20, 4, 4),
            pixel_area=100,
            confidence=0.8,
        )
        for index, position in enumerate(positions)
    ]
    return VisualPerceptionFeedback(
        camera_name="perception_cam",
        resolution=(256, 256),
        sample_index=sample_index,
        estimates=estimates,
        detected_counts={"agent": 0, "resource": 0, "blueprint_cell": len(estimates)},
        tracked_counts={"agent": 0, "resource": 0, "blueprint_cell": len(estimates)},
        mean_confidence=0.8,
    )


def test_visual_tracker_predicts_occlusion_and_preserves_track_identity() -> None:
    tracker = MultiObjectVisualTracker(
        VisualPerceptionConfig(
            enabled=True,
            tracking_enabled=True,
            tracking_max_missed_frames=2,
            prediction_confidence_decay=0.5,
        )
    )
    first = tracker.update(
        build_blueprint_feedback(1, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    )
    occluded = tracker.update(build_blueprint_feedback(2, [(0.01, 0.0, 0.0)]))
    recovered = tracker.update(
        build_blueprint_feedback(3, [(0.02, 0.0, 0.0), (1.01, 0.0, 0.0)])
    )

    assert first.tracked_counts["blueprint_cell"] == 2
    assert occluded.detected_counts["blueprint_cell"] == 1
    assert occluded.tracked_counts["blueprint_cell"] == 2
    assert occluded.predicted_estimate_count == 1
    predicted = next(item for item in occluded.estimates if item.is_predicted)
    assert predicted.track_id == "blueprint_cell_1"
    assert predicted.missed_frames == 1
    assert predicted.confidence == pytest.approx(0.4)
    assert {item.track_id for item in recovered.estimates} == {
        "blueprint_cell_0",
        "blueprint_cell_1",
    }
    assert recovered.predicted_estimate_count == 0


def test_visual_tracker_expires_predictions_after_missed_frame_limit() -> None:
    tracker = MultiObjectVisualTracker(
        VisualPerceptionConfig(
            enabled=True,
            tracking_enabled=True,
            tracking_max_missed_frames=1,
        )
    )
    tracker.update(build_blueprint_feedback(1, [(0.0, 0.0, 0.0)]))

    predicted = tracker.update(build_blueprint_feedback(2, []))
    expired = tracker.update(build_blueprint_feedback(3, []))

    assert predicted.tracked_counts["blueprint_cell"] == 1
    assert predicted.predicted_estimate_count == 1
    assert expired.tracked_counts["blueprint_cell"] == 0
    assert expired.predicted_estimate_count == 0


def build_terminal_feedback(
    resource_position: tuple[float, float, float],
    blueprint_positions: list[tuple[float, float, float]],
    predicted_blueprint: bool = False,
) -> VisualPerceptionFeedback:
    estimates = [
        VisualObjectEstimate(
            track_id="agent0",
            category="agent",
            centroid_px=(10.0, 10.0),
            position_m=(-0.2, 0.0, 0.2),
            bounding_box_xywh=(8, 8, 4, 4),
            pixel_area=100,
            confidence=0.9,
        ),
        VisualObjectEstimate(
            track_id="agent1",
            category="agent",
            centroid_px=(20.0, 10.0),
            position_m=(0.2, 0.0, 0.2),
            bounding_box_xywh=(18, 8, 4, 4),
            pixel_area=100,
            confidence=0.9,
        ),
        VisualObjectEstimate(
            track_id="resource_0",
            category="resource",
            centroid_px=(15.0, 10.0),
            position_m=resource_position,
            bounding_box_xywh=(13, 8, 4, 4),
            pixel_area=100,
            confidence=0.8,
        ),
    ]
    estimates.extend(
        VisualObjectEstimate(
            track_id=f"blueprint_cell_{index}",
            category="blueprint_cell",
            centroid_px=(15.0 + index, 15.0),
            position_m=position,
            bounding_box_xywh=(13 + index, 13, 4, 4),
            pixel_area=100,
            confidence=0.7,
            is_predicted=predicted_blueprint,
            missed_frames=1 if predicted_blueprint else 0,
        )
        for index, position in enumerate(blueprint_positions)
    )
    counts = {
        "agent": 2,
        "resource": 1,
        "blueprint_cell": len(blueprint_positions),
    }
    return VisualPerceptionFeedback(
        camera_name="perception_cam",
        resolution=(256, 256),
        sample_index=1,
        estimates=estimates,
        detected_counts=counts,
        tracked_counts=counts,
        predicted_estimate_count=sum(item.is_predicted for item in estimates),
        mean_confidence=0.8,
    )


def test_visual_terminal_assessment_uses_track_geometry_and_predictions() -> None:
    config = VisualPerceptionConfig(
        enabled=True,
        tracking_enabled=True,
        estimated_state_control_enabled=True,
    )
    grasp_ready = assess_visual_terminal_readiness(
        build_terminal_feedback((0.0, 0.0, 0.1), []),
        carrying=False,
        config=config,
    )
    grasp_misaligned = assess_visual_terminal_readiness(
        build_terminal_feedback((0.8, 0.0, 0.1), []),
        carrying=False,
        config=config,
    )
    install_ready = assess_visual_terminal_readiness(
        build_terminal_feedback(
            (0.0, 0.0, 0.1),
            [(-0.18, 0.0, 0.0), (0.18, 0.0, 0.0)],
            predicted_blueprint=True,
        ),
        carrying=True,
        config=config,
    )

    assert grasp_ready is not None and grasp_ready.ready is True
    assert grasp_ready.max_agent_resource_distance_m == pytest.approx(0.2)
    assert grasp_misaligned is not None
    assert grasp_misaligned.ready is False
    assert grasp_misaligned.reason == "agent_resource_misaligned"
    assert install_ready is not None and install_ready.ready is True
    assert install_ready.max_resource_blueprint_distance_m == pytest.approx(0.18)
    assert install_ready.uses_predicted_tracks is True


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for vision tests")
def test_mujoco_visual_frame_contains_rgb_metric_depth_and_segmentation() -> None:
    frame = build_env().capture_visual_frame(width=192, height=128)

    assert frame.resolution == (192, 128)
    assert frame.rgb.shape == (128, 192, 3)
    assert frame.depth_m.shape == (128, 192)
    assert frame.segmentation.shape == (128, 192, 2)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_m.dtype == np.float32
    assert np.isfinite(frame.depth_m).all()
    assert float(np.min(frame.depth_m)) > 0.0
    assert np.any(frame.segmentation[:, :, 0] >= 0)


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for vision tests")
def test_classical_assembly_perception_estimates_visible_poses_without_oracle_input() -> None:
    env = build_env()
    frame = env.capture_visual_frame()
    feedback = ClassicalAssemblyPerception(
        VisualPerceptionConfig(enabled=True)
    ).estimate(frame, sample_index=1)
    evaluation = evaluate_visual_perception(feedback, env._visual_truth_positions())

    assert feedback.detected_counts == {
        "agent": 2,
        "resource": 2,
        "blueprint_cell": 4,
    }
    assert feedback.mean_confidence > 0.7
    assert evaluation.recall_by_category == {
        "agent": 1.0,
        "resource": 1.0,
        "blueprint_cell": 1.0,
    }
    assert evaluation.mean_position_error_m < 0.03
    assert evaluation.max_position_error_m < 0.08
    assert not hasattr(feedback, "position_errors_m")


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for vision tests")
def test_vision_profile_populates_brain_observations_and_diagnostics() -> None:
    config = load_assembly_scenario(workspace / "configs" / "assembly_recovery.yaml")
    profile = load_runtime_profile(
        workspace / "configs" / "assembly_profiles" / "mujoco_vision.yaml"
    )
    env = build_assembly_backend(config, profile, seed=7)

    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=7,
    )
    diagnostics = episode.diagnostics["mujoco_visual_perception"]

    assert episode.artifact.metrics.success is True
    assert diagnostics["enabled"] is True
    assert diagnostics["sample_count"] == len(episode.steps) + 2
    assert diagnostics["mean_position_error_m"] < 0.08
    assert diagnostics["tracking_enabled"] is True
    assert diagnostics["estimated_state_control_enabled"] is True
    assert diagnostics["samples_with_predictions"] > 0
    assert diagnostics["latest_feedback"]["detected_counts"]["blueprint_cell"] == 2
    assert diagnostics["latest_feedback"]["tracked_counts"]["blueprint_cell"] == 4
    assert diagnostics["latest_evaluation"]["visible_recall_by_category"][
        "blueprint_cell"
    ] == 0.5
    assert diagnostics["latest_evaluation"]["recall_by_category"][
        "blueprint_cell"
    ] == 1.0
    latest_estimates = diagnostics["latest_feedback"]["estimates"]
    resource_tracks = [
        item for item in latest_estimates if item["category"] == "resource"
    ]
    predicted_blueprints = [
        item
        for item in latest_estimates
        if item["category"] == "blueprint_cell" and item["is_predicted"]
    ]
    assert {item["track_id"] for item in resource_tracks} == {
        "resource_0",
        "resource_1",
    }
    assert all(item["track_age"] == diagnostics["sample_count"] for item in resource_tracks)
    assert len(predicted_blueprints) == 2
    assert all(item["missed_frames"] > 0 for item in predicted_blueprints)
    assert diagnostics["prediction_backed_ready_count"] > 0
    terminal_steps = [
        step
        for step in episode.steps
        if step.decision.option in {TeamOption.GRAB, TeamOption.INSTALL}
    ]
    assert all(
        step.observation.visual_feedback is not None
        and step.observation.visual_feedback.terminal_assessment is not None
        and step.observation.visual_feedback.terminal_assessment.ready
        for step in terminal_steps
    )
    assert all(step.observation.visual_feedback is not None for step in episode.steps)
    assert all(
        "Visual perception sample" in step.decision.rationale
        for step in episode.steps
        if step.decision.safety_hold_reason is None
    )


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for vision tests")
def test_brain_holds_grasp_when_visual_geometry_is_misaligned() -> None:
    profile = load_runtime_profile(
        workspace / "configs" / "assembly_profiles" / "mujoco_vision.yaml"
    )
    env = build_env(profile)
    env.execute_team_option(TeamOption.GO_PICKUP)
    observation = env.get_construction_observation()
    assert observation.visual_feedback is not None
    misaligned_assessment = VisualTerminalAssessment(
        phase="grasp",
        ready=False,
        reason="agent_resource_misaligned",
        agent_track_ids=["agent0", "agent1"],
        resource_track_id="resource_0",
        max_agent_resource_distance_m=0.8,
        minimum_track_confidence=0.8,
    )
    unavailable_visual = observation.visual_feedback.model_copy(
        update={
            "terminal_assessment": misaligned_assessment,
        }
    )
    unavailable = observation.model_copy(
        update={"visual_feedback": unavailable_visual}
    )
    brain = HeuristicConstructionBrain()
    brain.reset(unavailable)

    decision = brain.decide(unavailable)

    assert decision.option == TeamOption.WAIT
    assert decision.safety_hold_reason == "visual_target_unavailable"
    assert "agent_resource_misaligned" in decision.rationale


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for vision tests")
def test_perception_capture_cli_writes_decodable_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(workspace / "scripts" / "capture_assembly_perception.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["feedback"]["detected_counts"] == {
        "agent": 2,
        "resource": 2,
        "blueprint_cell": 4,
    }
    assert report["feedback"]["terminal_assessment"]["phase"] == "grasp"
    assert report["feedback"]["terminal_assessment"]["ready"] is False
    assert (
        report["feedback"]["terminal_assessment"]["reason"]
        == "agent_resource_misaligned"
    )
    assert report["evaluation"]["mean_position_error_m"] < 0.03
    for name in ("rgb.png", "depth.png", "segmentation.png", "overlay.png"):
        import imageio.v3 as imageio

        image = imageio.imread(tmp_path / name)
        assert image.shape[:2] == (256, 256)
        assert float(np.std(image)) > 1.0
