from __future__ import annotations

from embodied_skill_composer.construction.models import BuildPlan, ConstructionSchedule, ExecutionTrace


def render_research_report(
    plan: BuildPlan,
    traces: dict[str, ExecutionTrace],
) -> str:
    optimized = traces["optimized"]
    sequential = traces["sequential"]
    improvement = 100 * (
        1 - optimized.metrics.makespan_s / max(sequential.metrics.makespan_s, 1)
    )
    lines = [
        f"# Construction Research Report: {plan.design.title}",
        "",
        "## Architectural Compilation",
        "",
        f"- Design: `{plan.design.design_id}`",
        f"- Build modules: `{len(plan.modules)}`",
        f"- Robot fleet: `{len(plan.robots)}`",
        f"- Footprint: `{plan.design.footprint_width_m:.1f} x {plan.design.footprint_depth_m:.1f} m`",
        f"- Roof: `{plan.design.roof.style}` at `{plan.design.roof.pitch_degrees:.0f} deg`",
        "",
        "## Controller Comparison",
        "",
        "| Controller | Makespan | Travel | Energy | Idle robot time |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("sequential", "greedy", "optimized"):
        metrics = traces[name].metrics
        lines.append(
            f"| {name} | {metrics.makespan_s} s | {metrics.total_travel_m:.1f} m | "
            f"{metrics.total_energy_wh:.1f} Wh | {metrics.idle_robot_seconds} s |"
        )
    lines.extend(
        [
            "",
            f"The optimized schedule reduces fixture makespan by **{improvement:.1f}%** "
            "relative to deterministic sequential construction.",
            "",
            "## Optimized Critical Path",
            "",
            " -> ".join(f"`{item}`" for item in optimized.schedule.critical_path),
            "",
            "## AI Brain Decisions",
            "",
        ]
    )
    for event in optimized.brain_events:
        if event.event_type == "assignment":
            lines.append(
                f"- `{event.timestamp_s:03d}s` assign `{event.module_id}` to "
                f"{', '.join(event.robot_ids)}: {event.reason}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The CP-SAT controller is an explainable scheduling oracle, not a learned policy. "
            "It demonstrates where multi-robot coordination creates measurable value before MARL "
            "is introduced. Browser and simulator playback consume the same execution trace.",
            "",
            "## Limitations",
            "",
            "- Geometry is modular and architectural, not structurally certified.",
            "- Transport uses metric task-level motion rather than wheel/gripper dynamics.",
            "- Floor-plan interpretation requires human approval before compilation.",
            "- CoppeliaSim and OpenAI services are optional validation and assistance layers.",
            "",
        ]
    )
    return "\n".join(lines)


def comparison_summary(schedules: dict[str, ConstructionSchedule]) -> dict[str, object]:
    sequential = schedules["sequential"].makespan_s
    optimized = schedules["optimized"].makespan_s
    return {
        "controllers": {name: item.model_dump(mode="json") for name, item in schedules.items()},
        "optimized_improvement_percent": round(100 * (1 - optimized / sequential), 2),
    }
