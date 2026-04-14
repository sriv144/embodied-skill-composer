from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from embodied_skill_composer.assembly.models import (
    AssemblyPlaybackFrame,
    AssemblyScenarioConfig,
)


def load_playback_frames(diagnostics: dict[str, object]) -> list[AssemblyPlaybackFrame]:
    raw_frames = diagnostics.get("state_snapshots", [])
    if not isinstance(raw_frames, list):
        return []
    return [AssemblyPlaybackFrame.model_validate(frame) for frame in raw_frames]


def render_playback_frames(
    config: AssemblyScenarioConfig,
    diagnostics: dict[str, object],
    output_dir: Path,
    title_prefix: str = "assembly-playback",
) -> list[Path]:
    frames = load_playback_frames(diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for index, frame in enumerate(frames):
        figure, axis = plt.subplots(figsize=(7, 7))
        _draw_frame(axis, config, frame, diagnostics, title=f"{title_prefix} :: frame {index:03d}")
        output_path = output_dir / f"frame_{index:03d}.png"
        figure.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        written_paths.append(output_path)
    return written_paths


def render_summary_figure(
    config: AssemblyScenarioConfig,
    diagnostics: dict[str, object],
    output_path: Path,
    title: str = "Collaborative Assembly Playback",
) -> Path:
    frames = load_playback_frames(diagnostics)
    if not frames:
        raise ValueError("No playback frames found in diagnostics.")
    figure, axis = plt.subplots(figsize=(8, 8))
    _draw_frame(axis, config, frames[-1], diagnostics, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _draw_frame(
    axis,
    config: AssemblyScenarioConfig,
    frame: AssemblyPlaybackFrame,
    diagnostics: dict[str, object],
    title: str,
) -> None:
    axis.set_xlim(-0.5, config.grid_size - 0.5)
    axis.set_ylim(config.grid_size - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks(range(config.grid_size))
    axis.set_yticks(range(config.grid_size))
    axis.grid(True, color="#d7dde5", linewidth=0.7)
    axis.set_facecolor("#f6f8fb")

    for beam in config.beams:
        for x, y in [beam.pickup_left, beam.pickup_right]:
            axis.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor="#fde68a", edgecolor="#c68a00", alpha=0.35))
        for x, y in [beam.assembly_left, beam.assembly_right]:
            axis.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor="#bfdbfe", edgecolor="#2563eb", alpha=0.35))

    for x, y in frame.pickup_targets:
        axis.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor="#fbbf24", edgecolor="#92400e", alpha=0.65))
    for x, y in frame.assembly_targets:
        axis.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor="#60a5fa", edgecolor="#1d4ed8", alpha=0.65))

    for agent_index, (x, y) in enumerate(frame.agent_positions):
        color = "#dc2626" if agent_index == 0 else "#059669"
        axis.scatter([x], [y], s=240, c=color, edgecolors="white", linewidths=1.8, zorder=3)
        axis.text(x, y, str(agent_index), ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=4)

    selected_options = diagnostics.get("selected_options", [])
    option_text = frame.selected_option or "initial_state"
    if isinstance(selected_options, list) and selected_options:
        option_text = frame.selected_option or str(selected_options[min(frame.current_beam_index, len(selected_options) - 1)])

    axis.set_title(
        (
            f"{title}\n"
            f"beam={frame.current_beam_index + 1} ({frame.current_beam_name or 'complete'}) | "
            f"step={frame.step_count} | carrying={frame.carrying} | option={option_text}"
        ),
        fontsize=11,
        pad=14,
    )
    axis.text(
        1.02,
        0.98,
        "\n".join(
            [
                f"primitive step: {frame.primitive_step_index}",
                f"option reward: {frame.option_reward:.2f}",
                f"option success: {frame.option_success}",
                f"switches: {diagnostics.get('option_switch_count', 0)}",
                f"first beam done: {diagnostics.get('first_beam_completion_step')}",
                f"second pickup: {diagnostics.get('second_beam_pickup_step')}",
                f"second install: {diagnostics.get('second_beam_install_step')}",
            ]
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#d0d7de"},
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
