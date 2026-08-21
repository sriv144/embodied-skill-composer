from __future__ import annotations

import bisect
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5JobReplay,
    Phase5ProvenanceManifest,
    verify_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotTelemetry,
    ScenarioManifest,
)
from embodied_skill_composer.construction.public_demo_provenance import (
    BundleArtifact,
    EvidenceBundleManifest,
    SourceIdentity,
)


_REPLAY_ADAPTER = TypeAdapter(list[Phase5JobReplay])
_VIDEO_NAME = "evidence_replay.mp4"
_VIDEO_MEDIA_TYPE = "video/mp4"
_MAX_VIDEO_BYTES = 512 * 1024 * 1024
_NATIVE_FILES = {
    "manifest": ("manifest.json", "application/json"),
    "scenario": ("scenario.json", "application/json"),
    "planned_jobs": ("planned_jobs.json", "application/json"),
    "replay": ("planned_vs_measured_replay.json", "application/json"),
    "wheel_commands": ("wheel_commands.jsonl", "application/x-ndjson"),
    "telemetry": ("measured_telemetry.jsonl", "application/x-ndjson"),
    "trace": ("trace.json", "application/json"),
    "metrics": ("metrics.json", "application/json"),
    "report": ("report.md", "text/markdown"),
    "scene": ("construction_intelligence.ttt", "application/octet-stream"),
}
_MODULE_COLORS = {
    "foundation": (94, 129, 172),
    "wall_panel": (223, 214, 190),
    "door_panel": (151, 101, 65),
    "window_panel": (91, 177, 222),
    "interior_panel": (196, 180, 157),
    "roof_panel": (73, 81, 92),
}


def render_coppelia_evidence_video(
    bundle_dir: Path,
    output_path: Path,
    *,
    fps: int = 12,
    duration_s: float = 24.0,
    width: int = 960,
    height: int = 544,
) -> Path:
    """Render a deterministic top-down visualization from attested measurements."""

    if fps < 1 or duration_s <= 0 or width < 320 or height < 240:
        raise ValueError("video dimensions, duration, and frame rate must be positive")
    manifest = verify_phase5_artifact_bundle(bundle_dir)
    return _render_verified_coppelia_video(
        bundle_dir,
        output_path,
        manifest=manifest,
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
    )


def _render_verified_coppelia_video(
    bundle_dir: Path,
    output_path: Path,
    *,
    manifest: Phase5ProvenanceManifest,
    fps: int,
    duration_s: float,
    width: int = 960,
    height: int = 544,
) -> Path:
    if fps < 1 or duration_s <= 0 or width < 320 or height < 240:
        raise ValueError("video dimensions, duration, and frame rate must be positive")
    if (
        manifest.evidence_kind != "live_coppelia"
        or not manifest.live_gate_passed
        or manifest.run_status != "completed"
    ):
        raise ValueError("replay video requires a passing live Coppelia bundle")

    scenario = ScenarioManifest.model_validate_json(
        (bundle_dir / "scenario.json").read_text(encoding="utf-8")
    )
    replay = _REPLAY_ADAPTER.validate_json(
        (bundle_dir / "planned_vs_measured_replay.json").read_text(encoding="utf-8")
    )
    telemetry = _read_telemetry(bundle_dir / "measured_telemetry.jsonl")
    if not telemetry or not replay:
        raise ValueError("replay video requires measured telemetry and installed jobs")

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        _write_video(
            temporary,
            scenario=scenario,
            replay=replay,
            telemetry=telemetry,
            scenario_name=manifest.scenario,
            fps=fps,
            duration_s=duration_s,
            width=width,
            height=height,
        )
        verify_coppelia_replay_video(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def verify_coppelia_replay_video(path: Path) -> None:
    """Reject malformed or unreadable MP4 evidence visualizations."""

    resolved = path.resolve()
    with resolved.open("rb") as handle:
        header = handle.read(32)
    size = resolved.stat().st_size
    if (
        size < 1_024
        or size > _MAX_VIDEO_BYTES
        or len(header) < 12
        or header[4:8] != b"ftyp"
    ):
        raise ValueError("Coppelia evidence replay is not a valid MP4 container")
    try:
        import imageio.v3 as imageio

        frame = imageio.imread(resolved, index=0)
    except Exception as exc:
        raise ValueError("Coppelia evidence replay cannot be decoded") from exc
    if frame.ndim != 3 or frame.shape[0] < 240 or frame.shape[1] < 320:
        raise ValueError("Coppelia evidence replay has an invalid first frame")


def package_coppelia_release_evidence(
    output_dir: Path,
    *,
    nominal_bundle: Path,
    recovery_bundle: Path,
    fps: int = 12,
    duration_s: float = 24.0,
) -> Path:
    """Copy two native bundles, add measured replays, and write a public descriptor."""

    nominal = verify_phase5_artifact_bundle(nominal_bundle)
    recovery = verify_phase5_artifact_bundle(recovery_bundle)
    _require_release_pair(nominal, recovery)

    destination = output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-stage-", dir=destination.parent))
    try:
        for prefix, source, manifest in (
            ("nominal", nominal_bundle.resolve(), nominal),
            ("recovery", recovery_bundle.resolve(), recovery),
        ):
            target = temporary / prefix
            shutil.copytree(source, target)
            _render_verified_coppelia_video(
                target,
                temporary / "videos" / f"{prefix}.mp4",
                manifest=manifest,
                fps=fps,
                duration_s=duration_s,
            )
        descriptor = _write_descriptor(temporary, nominal=nominal, recovery=recovery)
        if destination.exists():
            raise FileExistsError(f"release evidence output already exists: {destination}")
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / descriptor.name


def _write_video(
    path: Path,
    *,
    scenario: ScenarioManifest,
    replay: list[Phase5JobReplay],
    telemetry: list[RobotTelemetry],
    scenario_name: str,
    fps: int,
    duration_s: float,
    width: int,
    height: int,
) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    by_robot: dict[str, list[RobotTelemetry]] = defaultdict(list)
    for sample in telemetry:
        by_robot[sample.robot_id].append(sample)
    for samples in by_robot.values():
        samples.sort(key=lambda item: item.timestamp_s)
    timestamps = {
        robot_id: [sample.timestamp_s for sample in samples]
        for robot_id, samples in by_robot.items()
    }
    start_s = min(sample.timestamp_s for sample in telemetry)
    end_s = max(
        max(item.returned_at_s for item in replay),
        max(timestamps_[-1] for timestamps_ in timestamps.values()),
    )
    frame_count = max(2, round(duration_s * fps))
    grid = scenario.plan.site_grid
    world_bounds = (
        grid.origin.x,
        grid.origin.y,
        grid.origin.x + grid.width * grid.resolution_m,
        grid.origin.y + grid.height * grid.resolution_m,
    )
    imageio_api: Any = imageio
    writer = imageio_api.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        macro_block_size=16,
        ffmpeg_params=[
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            "-metadata",
            "encoder=Construction Intelligence v1",
            "-threads",
            "1",
        ],
    )
    try:
        for index in range(frame_count):
            fraction = index / (frame_count - 1)
            simulation_s = start_s + (end_s - start_s) * fraction
            image = Image.new("RGB", (width, height), (16, 21, 29))
            draw = ImageDraw.Draw(image)
            _draw_frame(
                draw,
                scenario=scenario,
                replay=replay,
                by_robot=by_robot,
                timestamps=timestamps,
                simulation_s=simulation_s,
                world_bounds=world_bounds,
                width=width,
                height=height,
                scenario_name=scenario_name,
            )
            writer.append_data(np.asarray(image))
    finally:
        writer.close()


def _draw_frame(
    draw: Any,
    *,
    scenario: ScenarioManifest,
    replay: list[Phase5JobReplay],
    by_robot: dict[str, list[RobotTelemetry]],
    timestamps: dict[str, list[float]],
    simulation_s: float,
    world_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    scenario_name: str,
) -> None:
    left, top, right, bottom = 42, 58, width - 238, height - 38
    draw.rectangle((left, top, right, bottom), fill=(24, 31, 41), outline=(82, 98, 116), width=2)
    grid = scenario.plan.site_grid
    for cell_x, cell_y in grid.obstacle_cells:
        x0 = grid.origin.x + cell_x * grid.resolution_m
        y0 = grid.origin.y + cell_y * grid.resolution_m
        p0 = _screen(x0, y0, world_bounds, (left, top, right, bottom))
        p1 = _screen(
            x0 + grid.resolution_m, y0 + grid.resolution_m, world_bounds, (left, top, right, bottom)
        )
        draw.rectangle(
            (
                min(p0[0], p1[0]),
                min(p0[1], p1[1]),
                max(p0[0], p1[0]),
                max(p0[1], p1[1]),
            ),
            fill=(73, 79, 88),
        )

    replay_by_module = {item.module_id: item for item in replay}
    installed = 0
    active_job: Phase5JobReplay | None = None
    for module in scenario.plan.modules:
        item = replay_by_module[module.module_id]
        pose = module.staging_pose
        base_color = _MODULE_COLORS[str(module.module_type)]
        fill = (
            max(25, base_color[0] // 2),
            max(25, base_color[1] // 2),
            max(25, base_color[2] // 2),
        )
        if simulation_s >= item.installed_at_s:
            pose = module.target_pose
            fill = _MODULE_COLORS[str(module.module_type)]
            installed += 1
        elif item.pickup_at_s <= simulation_s < item.installed_at_s:
            active_job = item
            points = [
                _sample_at(by_robot[robot_id], timestamps[robot_id], simulation_s)
                for robot_id in item.executed_robot_ids
            ]
            pose = pose.model_copy(
                update={
                    "position": pose.position.model_copy(
                        update={
                            "x": sum(point.measured_pose.position.x for point in points)
                            / len(points)
                            + item.logical_carrier_offset.x,
                            "y": sum(point.measured_pose.position.y for point in points)
                            / len(points)
                            + item.logical_carrier_offset.y,
                        }
                    )
                }
            )
            fill = (238, 168, 69)
        _draw_module(
            draw,
            module,
            pose.position.x,
            pose.position.y,
            pose.rotation_rpy_degrees.z,
            fill,
            world_bounds,
            (left, top, right, bottom),
        )

    palette = [(89, 180, 255), (255, 112, 112), (123, 220, 155), (215, 144, 255)]
    for index, robot in enumerate(scenario.plan.robots):
        sample = _sample_at(by_robot[robot.robot_id], timestamps[robot.robot_id], simulation_s)
        x, y = _screen(
            sample.measured_pose.position.x,
            sample.measured_pose.position.y,
            world_bounds,
            (left, top, right, bottom),
        )
        color = palette[index % len(palette)]
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(245, 248, 252),
            width=1,
        )
        draw.text((x + 9, y - 7), robot.robot_id, fill=color)

    draw.text(
        (42, 20), "Construction Intelligence v1 — measured Coppelia replay", fill=(238, 242, 247)
    )
    draw.text((width - 216, 61), scenario_name.replace("_", " ").title(), fill=(102, 195, 255))
    draw.text((width - 216, 91), f"Simulator time  {simulation_s:8.2f} s", fill=(207, 216, 226))
    draw.text(
        (width - 216, 117),
        f"Installed       {installed:2d} / {len(replay):2d}",
        fill=(207, 216, 226),
    )
    if active_job is not None:
        draw.text((width - 216, 151), "Active logical carry", fill=(238, 168, 69))
        draw.text((width - 216, 173), active_job.module_id, fill=(222, 226, 232))
        draw.text((width - 216, 195), "+".join(active_job.executed_robot_ids), fill=(222, 226, 232))
    draw.multiline_text(
        (width - 216, height - 112),
        "Evidence visualization\nMeasured robot bases\nLogical payload transport\nNo arm/contact claim",
        fill=(157, 170, 185),
        spacing=5,
    )


def _draw_module(
    draw: Any,
    module: Any,
    x: float,
    y: float,
    yaw_degrees: float,
    fill: tuple[int, int, int],
    world_bounds: tuple[float, float, float, float],
    screen_bounds: tuple[int, int, int, int],
) -> None:
    half_width = module.dimensions.width / 2
    half_depth = module.dimensions.depth / 2
    yaw = math.radians(yaw_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners = []
    for local_x, local_y in (
        (-half_width, -half_depth),
        (half_width, -half_depth),
        (half_width, half_depth),
        (-half_width, half_depth),
    ):
        world_x = x + local_x * cos_yaw - local_y * sin_yaw
        world_y = y + local_x * sin_yaw + local_y * cos_yaw
        corners.append(_screen(world_x, world_y, world_bounds, screen_bounds))
    draw.polygon(corners, fill=fill, outline=(226, 231, 237))


def _screen(
    x: float,
    y: float,
    world: tuple[float, float, float, float],
    screen: tuple[int, int, int, int],
) -> tuple[int, int]:
    wx0, wy0, wx1, wy1 = world
    sx0, sy0, sx1, sy1 = screen
    return (
        round(sx0 + (x - wx0) / (wx1 - wx0) * (sx1 - sx0)),
        round(sy1 - (y - wy0) / (wy1 - wy0) * (sy1 - sy0)),
    )


def _sample_at(
    samples: list[RobotTelemetry],
    timestamps: list[float],
    timestamp_s: float,
) -> RobotTelemetry:
    index = max(0, bisect.bisect_right(timestamps, timestamp_s) - 1)
    return samples[index]


def _read_telemetry(path: Path) -> list[RobotTelemetry]:
    return [
        RobotTelemetry.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require_release_pair(
    nominal: Phase5ProvenanceManifest,
    recovery: Phase5ProvenanceManifest,
) -> None:
    if nominal.scenario != "nominal" or recovery.scenario != "unavailable_robot_recovery":
        raise ValueError("release evidence requires nominal and recovery scenarios")
    for manifest in (nominal, recovery):
        if (
            manifest.evidence_kind != "live_coppelia"
            or not manifest.live_gate_passed
            or manifest.run_status != "completed"
            or manifest.source_dirty
        ):
            raise ValueError("release evidence requires clean passing live Coppelia runs")
    if (
        nominal.source_commit != recovery.source_commit
        or nominal.source_tree_digest != recovery.source_tree_digest
    ):
        raise ValueError("nominal and recovery evidence must share a source identity")


def _write_descriptor(
    root: Path,
    *,
    nominal: Phase5ProvenanceManifest,
    recovery: Phase5ProvenanceManifest,
) -> Path:
    artifacts: list[BundleArtifact] = []
    for prefix in ("nominal", "recovery"):
        for role, (file_name, media_type) in {
            **_NATIVE_FILES,
            "video": (_VIDEO_NAME, _VIDEO_MEDIA_TYPE),
        }.items():
            relative = f"videos/{prefix}.mp4" if role == "video" else f"{prefix}/{file_name}"
            source = root / relative
            artifacts.append(
                BundleArtifact(
                    role=f"{prefix}_{role}",
                    path=relative,
                    target=(
                        f"evidence/coppelia/visualizations/{prefix}.mp4"
                        if role == "video"
                        else f"evidence/coppelia/{prefix}/{file_name}"
                    ),
                    sha256=_sha256_file(source),
                    media_type=media_type,
                )
            )
    manifest = EvidenceBundleManifest(
        schema_version="construction-intelligence-public-demo-input-v1",
        kind="simulator",
        evidence_status="canonical",
        source=SourceIdentity(
            commit=nominal.source_commit,
            dirty=False,
            tree_digest=nominal.source_tree_digest,
        ),
        created_at=max(nominal.generated_at, recovery.generated_at),
        configuration_digests=sorted({nominal.configuration_digest, recovery.configuration_digest}),
        artifacts=sorted(artifacts, key=lambda item: item.role),
    )
    path = root / "simulator-bundle.json"
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
