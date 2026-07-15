# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.models import TeamOption
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and evaluate MuJoCo RGB/depth assembly perception."
    )
    parser.add_argument(
        "--env-config",
        default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"),
    )
    parser.add_argument(
        "--runtime-profile",
        default=str(
            PROJECT_ROOT / "configs" / "assembly_profiles" / "mujoco_vision.yaml"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--advance-options",
        default="",
        help="Comma-separated team option names to execute before capture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "assembly_perception",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_assembly_scenario(Path(args.env_config))
    profile = load_runtime_profile(Path(args.runtime_profile))
    if not profile.visual_perception.enabled:
        raise ValueError("runtime profile must enable visual_perception")
    env = build_assembly_backend(config, profile, seed=args.seed)
    if not env.get_backend_status().is_ready:
        raise RuntimeError("MuJoCo backend is not ready")
    env.reset(seed=args.seed)
    visual_config = profile.visual_perception
    estimator = ClassicalAssemblyPerception(visual_config)
    tracker = MultiObjectVisualTracker(visual_config)
    sample_index = 1
    frame = env.capture_visual_frame(
        width=visual_config.width,
        height=visual_config.height,
        camera_name=visual_config.camera_name,
    )
    feedback = tracker.update(estimator.estimate(frame, sample_index=sample_index))
    feedback = feedback.model_copy(
        update={
            "terminal_assessment": assess_visual_terminal_readiness(
                feedback,
                carrying=env.logical_env.state.carrying,
                config=visual_config,
            )
        }
    )
    executed_options: list[str] = []
    for raw_name in filter(None, (item.strip() for item in args.advance_options.split(","))):
        option = TeamOption[raw_name.upper()]
        result = env.execute_team_option(option)
        executed_options.append(option.name.lower())
        if not result.success:
            raise RuntimeError(f"advance option '{option.name.lower()}' failed: {result.info}")
        sample_index += 1
        frame = env.capture_visual_frame(
            width=visual_config.width,
            height=visual_config.height,
            camera_name=visual_config.camera_name,
        )
        feedback = tracker.update(
            estimator.estimate(frame, sample_index=sample_index)
        )
        feedback = feedback.model_copy(
            update={
                "terminal_assessment": assess_visual_terminal_readiness(
                    feedback,
                    carrying=env.logical_env.state.carrying,
                    config=visual_config,
                )
            }
        )
    evaluation = evaluate_visual_perception(feedback, env._visual_truth_positions())

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_dir / "rgb.png", frame.rgb)
    imageio.imwrite(output_dir / "depth.png", _depth_visualization(frame.depth_m))
    imageio.imwrite(
        output_dir / "segmentation.png",
        _segmentation_visualization(frame.segmentation),
    )
    imageio.imwrite(output_dir / "overlay.png", _overlay(frame.rgb, feedback.estimates))
    report = {
        "runtime_profile": profile.model_dump(mode="json"),
        "seed": args.seed,
        "executed_options": executed_options,
        "frame": {
            "camera_name": frame.camera_name,
            "resolution": list(frame.resolution),
            "depth_min_m": float(np.min(frame.depth_m)),
            "depth_max_m": float(np.max(frame.depth_m)),
        },
        "feedback": feedback.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Camera: {frame.camera_name} ({frame.resolution[0]}x{frame.resolution[1]})")
    print(f"Detected: {feedback.detected_counts}")
    print(
        f"Tracked: {feedback.tracked_counts} "
        f"({feedback.predicted_estimate_count} predicted)"
    )
    print(f"Mean confidence: {feedback.mean_confidence:.3f}")
    print(f"Mean position error: {evaluation.mean_position_error_m:.4f} m")
    print(
        "Blueprint recall: "
        f"visible={evaluation.visible_recall_by_category['blueprint_cell']:.3f}, "
        f"tracked={evaluation.recall_by_category['blueprint_cell']:.3f}"
    )
    if feedback.terminal_assessment is not None:
        assessment = feedback.terminal_assessment
        print(
            f"Estimated-state {assessment.phase} readiness: "
            f"{assessment.ready} ({assessment.reason})"
        )
    print(f"Artifacts: {output_dir}")
    return 0


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth < 100.0)
    result = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return result
    near = float(np.min(depth[valid]))
    far = float(np.max(depth[valid]))
    normalized = (depth[valid] - near) / max(1e-9, far - near)
    result[valid] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    return result


def _segmentation_visualization(segmentation: np.ndarray) -> np.ndarray:
    object_ids = segmentation[:, :, 0]
    result = np.zeros((*object_ids.shape, 3), dtype=np.uint8)
    for object_id in np.unique(object_ids):
        if object_id < 0:
            continue
        result[object_ids == object_id] = [
            (int(object_id) * 67) % 255,
            (int(object_id) * 131) % 255,
            (int(object_id) * 193) % 255,
        ]
    return result


def _overlay(rgb: np.ndarray, estimates: list) -> np.ndarray:
    result = rgb.copy()
    occupied_labels: list[tuple[int, int, int, int]] = []
    category_colors = {
        "agent": (255, 40, 40),
        "resource": (255, 170, 30),
        "blueprint_cell": (30, 150, 255),
    }
    for estimate in estimates:
        x, y, width, height = estimate.bounding_box_xywh
        color = category_colors[estimate.category]
        if estimate.is_predicted:
            color = (190, 60, 255)
        cv2.rectangle(result, (x, y), (x + width, y + height), color, 1)
        label = _short_track_label(estimate.track_id)
        if estimate.is_predicted:
            label = f"~{label}"
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            1,
        )
        label_x, label_y, label_rect = _place_label(
            x,
            y,
            width,
            height,
            text_size,
            result.shape[1],
            result.shape[0],
            occupied_labels,
        )
        occupied_labels.append(label_rect)
        cv2.putText(
            result,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def _place_label(
    x: int,
    y: int,
    width: int,
    height: int,
    text_size: tuple[int, int],
    image_width: int,
    image_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, tuple[int, int, int, int]]:
    text_width, text_height = text_size
    candidates = [
        (x, y - 3),
        (x + width + 2, y + text_height),
        (x, y + height + text_height + 2),
        (x - text_width - 2, y + text_height),
    ]
    for candidate_x, candidate_y in candidates:
        label_x = min(max(1, candidate_x), image_width - text_width - 1)
        label_y = min(max(text_height + 1, candidate_y), image_height - 2)
        rect = (label_x, label_y - text_height, label_x + text_width, label_y + 2)
        if not any(_rectangles_overlap(rect, previous) for previous in occupied):
            return label_x, label_y, rect
    label_x = min(max(1, x), image_width - text_width - 1)
    label_y = min(max(text_height + 1, y + height + text_height + 2), image_height - 2)
    return label_x, label_y, (
        label_x,
        label_y - text_height,
        label_x + text_width,
        label_y + 2,
    )


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _short_track_label(track_id: str) -> str:
    if track_id.startswith("agent"):
        return f"a{track_id.removeprefix('agent')}"
    if track_id.startswith("resource_"):
        return f"r{track_id.rsplit('_', 1)[1]}"
    if track_id.startswith("blueprint_cell_"):
        return f"b{track_id.rsplit('_', 1)[1]}"
    return track_id


if __name__ == "__main__":
    raise SystemExit(main())
