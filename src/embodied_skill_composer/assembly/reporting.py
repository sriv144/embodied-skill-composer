from __future__ import annotations

from embodied_skill_composer.assembly.models import (
    CompiledBlueprint,
    ConstructionBrainEpisode,
)


def render_construction_lab_report(
    compiled: CompiledBlueprint,
    episode: ConstructionBrainEpisode,
) -> str:
    metrics = episode.artifact.metrics
    diagnostics = episode.diagnostics
    backend_status = diagnostics.get("backend_status", {})
    readiness_notes = (
        backend_status.get("readiness_notes", [])
        if isinstance(backend_status, dict)
        else []
    )
    lines = [
        f"# Construction Lab Report: {compiled.blueprint.title}",
        "",
        "## Result",
        "",
        f"- Blueprint: `{compiled.blueprint.blueprint_id}`",
        f"- Brain: `{episode.brain_name}`",
        f"- Backend: `{episode.backend}`",
        f"- Success: `{metrics.success}`",
        (
            "- Structure completion: "
            f"`{metrics.structure_completion_rate:.3f}` "
            f"({metrics.beams_installed}/{metrics.total_beams} components)"
        ),
        f"- Resource delivery accuracy: `{metrics.resource_delivery_accuracy:.3f}`",
        f"- Total reward: `{metrics.total_reward:.3f}`",
        f"- Logical steps: `{metrics.step_count}`",
        f"- Energy cost: `{metrics.energy_cost:.3f}`",
        f"- Idle / wasted steps: `{metrics.idle_step_count}` / `{metrics.wasted_step_count}`",
        f"- Collisions: `{metrics.collision_count}`",
        (
            "- Manipulation failures / recoveries: "
            f"`{metrics.manipulation_failure_count}` / "
            f"`{metrics.manipulation_recovery_count}`"
        ),
        "",
        "## Installation Graph",
        "",
    ]
    components = {
        component.component_id: component for component in compiled.blueprint.components
    }
    for index, component_id in enumerate(compiled.installation_order, start=1):
        dependencies = components[component_id].depends_on
        dependency_text = ", ".join(dependencies) if dependencies else "foundation"
        lines.append(f"{index}. `{component_id}` after {dependency_text}")

    lines.extend(["", "## Assignments", ""])
    for assignment in episode.assignments:
        lines.append(
            "- "
            f"`{assignment.resource_id}` -> `{assignment.component_id or assignment.slot_id}`; "
            f"robots {assignment.assigned_robot_ids}; status `{assignment.status}`; "
            f"estimated grid cost `{assignment.estimated_cost:.1f}`"
        )

    lines.extend(["", "## Decision Timeline", ""])
    for step in episode.steps:
        assignment = step.decision.assignment
        component = (
            "none"
            if assignment is None
            else assignment.component_id or assignment.resource_id
        )
        lines.append(
            f"- `{step.decision_index:02d}` "
            f"component `{component}`: `{step.decision.option.name.lower()}` - "
            f"{step.decision.rationale}"
        )

    lines.extend(["", "## Backend Notes", ""])
    if readiness_notes:
        lines.extend(f"- {note}" for note in readiness_notes)
    else:
        lines.append("- No backend readiness notes were recorded.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The precedence controller is the deterministic oracle for this blueprint. "
                "Future graph-based hierarchical RL/MARL policies should be compared against "
                "this installation order, completion rate, coordination cost, and recovery behavior."
            ),
            "",
        ]
    )
    return "\n".join(lines)
