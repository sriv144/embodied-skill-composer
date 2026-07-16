from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from embodied_skill_composer.assembly.backends import AssemblyTaskBackend
from embodied_skill_composer.assembly.models import (
    BlueprintSlotState,
    ConstructionAssignment,
    ConstructionBrainDecision,
    ConstructionBrainEpisode,
    ConstructionBrainObservation,
    ConstructionBrainStep,
    ConstructionResourceState,
    TeamOption,
)


@runtime_checkable
class ConstructionBrain(Protocol):
    name: str

    def reset(self, observation: ConstructionBrainObservation) -> None: ...

    def decide(self, observation: ConstructionBrainObservation) -> ConstructionBrainDecision: ...

    def assignments(
        self,
        observation: ConstructionBrainObservation,
    ) -> list[ConstructionAssignment]: ...


@dataclass
class ScriptedConstructionBrain:
    name: str = "scripted_construction_brain"
    _assignments: list[ConstructionAssignment] = field(default_factory=list, init=False)

    def reset(self, observation: ConstructionBrainObservation) -> None:
        self._assignments = _allocate_resources(observation, prefer_declared_assignment=True)

    def decide(self, observation: ConstructionBrainObservation) -> ConstructionBrainDecision:
        option = _select_structured_option(observation)
        assignment = _active_assignment(self.assignments(observation), observation)
        return ConstructionBrainDecision(
            option=option,
            rationale=_option_rationale(option, observation),
            assignment=assignment,
            safety_hold_reason=_terminal_safety_hold_reason(observation),
        )

    def assignments(
        self,
        observation: ConstructionBrainObservation,
    ) -> list[ConstructionAssignment]:
        return [_with_live_status(item, observation) for item in self._assignments]


@dataclass
class HeuristicConstructionBrain:
    name: str = "heuristic_allocator_brain"
    _assignments: list[ConstructionAssignment] = field(default_factory=list, init=False)

    def reset(self, observation: ConstructionBrainObservation) -> None:
        self._assignments = _allocate_resources(observation, prefer_declared_assignment=False)

    def decide(self, observation: ConstructionBrainObservation) -> ConstructionBrainDecision:
        option = _select_structured_option(observation)
        assignment = _active_assignment(self.assignments(observation), observation)
        rationale = _option_rationale(option, observation)
        if assignment is not None:
            rationale = (
                f"{rationale} Active allocation: {assignment.resource_id} -> "
                f"{assignment.slot_id} (estimated cost {assignment.estimated_cost:.1f})."
            )
        return ConstructionBrainDecision(
            option=option,
            rationale=rationale,
            assignment=assignment,
            safety_hold_reason=_terminal_safety_hold_reason(observation),
        )

    def assignments(
        self,
        observation: ConstructionBrainObservation,
    ) -> list[ConstructionAssignment]:
        return [_with_live_status(item, observation) for item in self._assignments]


@dataclass
class PrecedenceConstructionBrain:
    name: str = "precedence_construction_brain"
    _assignments: list[ConstructionAssignment] = field(default_factory=list, init=False)

    def reset(self, observation: ConstructionBrainObservation) -> None:
        self._assignments = _allocate_resources(
            observation,
            prefer_declared_assignment=True,
        )

    def decide(self, observation: ConstructionBrainObservation) -> ConstructionBrainDecision:
        assignment = _active_assignment(self.assignments(observation), observation)
        blocked = _blocked_prerequisites(assignment, observation)
        if blocked:
            return ConstructionBrainDecision(
                option=TeamOption.WAIT,
                rationale=(
                    f"Hold {assignment.component_id or assignment.resource_id}; "
                    f"prerequisites are incomplete: {', '.join(blocked)}."
                ),
                assignment=assignment,
                safety_hold_reason="blocked_dependency",
            )
        option = _select_structured_option(observation)
        rationale = _option_rationale(option, observation)
        if assignment is not None:
            rationale = (
                f"{rationale} Precedence-ready component: "
                f"{assignment.component_id or assignment.resource_id}; robots "
                f"{assignment.assigned_robot_ids}."
            )
        return ConstructionBrainDecision(
            option=option,
            rationale=rationale,
            assignment=assignment,
            safety_hold_reason=_terminal_safety_hold_reason(observation),
        )

    def assignments(
        self,
        observation: ConstructionBrainObservation,
    ) -> list[ConstructionAssignment]:
        return [_with_live_status(item, observation) for item in self._assignments]


def run_construction_brain_episode(
    env: AssemblyTaskBackend,
    brain: ConstructionBrain,
    seed: int = 7,
    max_decisions: int | None = None,
) -> ConstructionBrainEpisode:
    env.set_curriculum_stage(None)
    env.reset(seed=seed)
    observation = env.get_construction_observation()
    brain.reset(observation)
    steps: list[ConstructionBrainStep] = []
    decision_limit = max_decisions or env.config.max_steps
    done = False

    while not done:
        if len(steps) >= decision_limit:
            raise RuntimeError(
                f"Construction brain '{brain.name}' exceeded {decision_limit} decisions."
            )
        observation = env.get_construction_observation()
        decision = brain.decide(observation)
        if decision.option not in observation.available_options:
            raise RuntimeError(
                f"Construction brain '{brain.name}' selected unavailable option "
                f"'{decision.option.name.lower()}'."
            )
        execution = env.execute_team_option(
            decision.option,
            max_primitive_steps=env.config.option_max_primitive_steps,
        )
        steps.append(
            ConstructionBrainStep(
                decision_index=len(steps),
                observation=observation,
                decision=decision,
                execution=execution,
            )
        )
        done = execution.done

    final_observation = env.get_construction_observation()
    diagnostics = env.get_option_episode_diagnostics()
    diagnostics["construction_brain"] = {
        "name": brain.name,
        "decision_count": len(steps),
        "terminal_safety_hold_count": sum(
            step.decision.safety_hold_reason is not None for step in steps
        ),
        "sensor_safety_hold_count": sum(
            step.decision.safety_hold_reason
            in {"sensor_unavailable", "alignment_error"}
            for step in steps
        ),
        "visual_safety_hold_count": sum(
            step.decision.safety_hold_reason == "visual_target_unavailable"
            for step in steps
        ),
        "dependency_hold_count": sum(
            step.decision.safety_hold_reason == "blocked_dependency"
            for step in steps
        ),
        "installation_graph": {
            slot.component_id or slot.slot_id: list(slot.depends_on)
            for slot in final_observation.blueprint_slots
        },
        "assignments": [
            item.model_dump(mode="json") for item in brain.assignments(final_observation)
        ],
    }
    return ConstructionBrainEpisode(
        brain_name=brain.name,
        seed=seed,
        backend=env.backend_name,
        assignments=brain.assignments(final_observation),
        steps=steps,
        artifact=env.build_artifact(policy_mode="brain"),
        diagnostics=diagnostics,
    )


def _select_structured_option(observation: ConstructionBrainObservation) -> TeamOption:
    if _terminal_safety_hold_reason(observation) is not None:
        return TeamOption.WAIT
    if observation.carrying:
        priority = [
            TeamOption.INSTALL,
            TeamOption.ALIGN_FOR_TERMINAL_ACTION,
            TeamOption.GO_ASSEMBLY,
            TeamOption.WAIT,
        ]
    else:
        priority = [
            TeamOption.REPOSITION_AFTER_INSTALL,
            TeamOption.RESET_TO_PICKUP_ROUTE,
            TeamOption.GRAB,
            TeamOption.ALIGN_FOR_TERMINAL_ACTION,
            TeamOption.GO_PICKUP,
            TeamOption.WAIT,
        ]
    for option in priority:
        if option in observation.available_options:
            return option
    raise RuntimeError("Construction observation exposes no supported team option.")


def _option_rationale(
    option: TeamOption,
    observation: ConstructionBrainObservation,
) -> str:
    beam = observation.current_beam_name or "completed structure"
    hold_reason = _terminal_safety_hold_reason(observation)
    if option == TeamOption.WAIT and hold_reason is not None:
        if hold_reason == "visual_target_unavailable":
            visual = observation.visual_feedback
            if visual is None:
                return "Hold the terminal action until visual feedback is available."
            assessment = visual.terminal_assessment
            if assessment is not None:
                return (
                    "Hold the terminal action because estimated visual geometry is "
                    f"not ready ({assessment.reason})."
                )
            return (
                "Hold the terminal action until visual tracking recovers the required "
                f"targets. Tracked counts: {visual.tracked_counts}."
            )
        feedback = observation.physical_feedback
        if feedback is None:
            return "Hold the terminal action until physical feedback is available."
        if not feedback.sensor_fresh or feedback.current_alignment_error_m is None:
            return (
                "Hold the terminal action after a dropped or stale physical sensor "
                f"sample (age {feedback.sensor_age_physics_steps} physics steps)."
            )
        return (
            "Hold position while the low-level controller reduces sensed alignment "
            f"error from {feedback.current_alignment_error_m:.4f} m to within "
            f"{feedback.alignment_tolerance_m:.4f} m."
        )
    rationales = {
        TeamOption.GO_PICKUP: f"Move both robots to the resource cells for {beam}.",
        TeamOption.GRAB: f"Both robots are aligned; grasp {beam} together.",
        TeamOption.GO_ASSEMBLY: f"Transport {beam} toward its blueprint slot.",
        TeamOption.INSTALL: f"Both robots are aligned; install {beam} into the blueprint.",
        TeamOption.RESET_TO_PICKUP_ROUTE: "Clear the previous build area and return to pickup staging.",
        TeamOption.REPOSITION_AFTER_INSTALL: "Move away from the completed slot before the next task.",
        TeamOption.WAIT: "Hold position while no productive transition is available.",
        TeamOption.ALIGN_FOR_TERMINAL_ACTION: "Finish joint alignment before grasp or installation.",
    }
    rationale = rationales[option]
    if observation.last_manipulation_failure and option in {
        TeamOption.GRAB,
        TeamOption.INSTALL,
    }:
        rationale = (
            f"Retry the terminal action after a recoverable failure. "
            f"Previous result: {observation.last_manipulation_failure}. {rationale}"
        )
    return (
        f"{rationale}{_physical_feedback_rationale(option, observation)}"
        f"{_visual_feedback_rationale(observation)}"
    )


def _physical_feedback_rationale(
    option: TeamOption,
    observation: ConstructionBrainObservation,
) -> str:
    feedback = observation.physical_feedback
    if feedback is None or option not in {TeamOption.GRAB, TeamOption.INSTALL}:
        return ""
    alignment = (
        "unavailable"
        if feedback.current_alignment_error_m is None
        else f"{feedback.current_alignment_error_m:.4f} m"
    )
    summary = (
        f" Physical feedback: {feedback.sensor_mode} sample "
        f"#{feedback.sensor_sample_index}, alignment error {alignment} within "
        f"{feedback.alignment_tolerance_m:.4f} m tolerance; grippers "
        f"{feedback.gripper_state}."
    )
    if option == TeamOption.GRAB:
        summary += (
            f" Required per-robot grip force is "
            f"{feedback.required_minimum_grip_force_n:.1f} N."
        )
        if feedback.last_check_phase == "grasp" and feedback.last_contact_forces_n:
            forces = ", ".join(
                f"{agent}={force:.2f} N"
                for agent, force in sorted(feedback.last_contact_forces_n.items())
            )
            summary += f" Last grasp forces: {forces}."
    elif feedback.active_attachment_beam:
        summary += f" Active attachment: {feedback.active_attachment_beam}."
    return summary


def _physical_safety_hold_reason(
    observation: ConstructionBrainObservation,
) -> str | None:
    terminal_option = TeamOption.INSTALL if observation.carrying else TeamOption.GRAB
    if terminal_option not in observation.available_options:
        return None
    feedback = observation.physical_feedback
    if feedback is None:
        return None
    if not feedback.sensor_fresh or feedback.current_alignment_error_m is None:
        return "sensor_unavailable"
    if feedback.current_alignment_error_m > feedback.alignment_tolerance_m:
        return "alignment_error"
    return None


def _visual_safety_hold_reason(
    observation: ConstructionBrainObservation,
) -> str | None:
    terminal_option = TeamOption.INSTALL if observation.carrying else TeamOption.GRAB
    if terminal_option not in observation.available_options:
        return None
    feedback = observation.visual_feedback
    if feedback is None:
        return None
    if feedback.terminal_assessment is not None:
        if not feedback.terminal_assessment.ready:
            return "visual_target_unavailable"
        return None
    if feedback.tracked_counts.get("agent", 0) < 2:
        return "visual_target_unavailable"
    target_category = "blueprint_cell" if observation.carrying else "resource"
    required_count = 2 if observation.carrying else 1
    if feedback.tracked_counts.get(target_category, 0) < required_count:
        return "visual_target_unavailable"
    return None


def _terminal_safety_hold_reason(
    observation: ConstructionBrainObservation,
) -> str | None:
    return _physical_safety_hold_reason(observation) or _visual_safety_hold_reason(
        observation
    )


def _visual_feedback_rationale(
    observation: ConstructionBrainObservation,
) -> str:
    feedback = observation.visual_feedback
    if feedback is None:
        return ""
    agents = feedback.detected_counts.get("agent", 0)
    resources = feedback.detected_counts.get("resource", 0)
    blueprint_cells = feedback.detected_counts.get("blueprint_cell", 0)
    summary = (
        f" Visual perception sample #{feedback.sample_index}: visible {agents} agents, "
        f"{resources} resources, {blueprint_cells} blueprint cells; tracked "
        f"{feedback.tracked_counts} with {feedback.predicted_estimate_count} predicted "
        f"(mean confidence {feedback.mean_confidence:.2f})."
    )
    assessment = feedback.terminal_assessment
    if assessment is not None:
        summary += (
            f" Estimated-state {assessment.phase} readiness: {assessment.ready} "
            f"({assessment.reason}), agent-resource error "
            f"{_format_optional_distance(assessment.max_agent_resource_distance_m)}"
        )
        if assessment.phase == "install":
            summary += (
                ", resource-blueprint error "
                f"{_format_optional_distance(assessment.max_resource_blueprint_distance_m)}"
            )
        summary += (
            f", minimum confidence {assessment.minimum_track_confidence:.2f}, "
            f"prediction-backed {assessment.uses_predicted_tracks}."
        )
    return summary


def _format_optional_distance(distance: float | None) -> str:
    return "unavailable" if distance is None else f"{distance:.3f} m"


def _allocate_resources(
    observation: ConstructionBrainObservation,
    prefer_declared_assignment: bool,
) -> list[ConstructionAssignment]:
    available = list(observation.resources)
    assignments: list[ConstructionAssignment] = []
    for slot in observation.blueprint_slots:
        candidates = [
            resource
            for resource in available
            if resource.resource_type == slot.resource_type
            and (slot.required_resource_id is None or resource.resource_id == slot.required_resource_id)
        ]
        if prefer_declared_assignment:
            declared = [resource for resource in candidates if resource.assigned_slot_id == slot.slot_id]
            if declared:
                candidates = declared
        if not candidates:
            continue
        resource = min(candidates, key=lambda item: _assignment_cost(item, slot))
        available.remove(resource)
        assignments.append(
            ConstructionAssignment(
                resource_id=resource.resource_id,
                slot_id=slot.slot_id,
                beam_name=slot.required_resource_id or resource.resource_id,
                estimated_cost=_assignment_cost(resource, slot),
                component_id=slot.component_id or resource.component_id,
                prerequisites=list(slot.depends_on),
                assigned_robot_ids=(
                    list(resource.assigned_robot_ids)
                    if resource.assigned_robot_ids
                    else list(range(len(observation.agent_positions)))
                ),
            )
        )
    return assignments


def _assignment_cost(
    resource: ConstructionResourceState,
    slot: BlueprintSlotState,
) -> float:
    if not resource.source_cells or not slot.target_cells:
        return 0.0
    count = min(len(resource.source_cells), len(slot.target_cells))
    return float(
        sum(
            abs(resource.source_cells[index][0] - slot.target_cells[index][0])
            + abs(resource.source_cells[index][1] - slot.target_cells[index][1])
            for index in range(count)
        )
    )


def _with_live_status(
    assignment: ConstructionAssignment,
    observation: ConstructionBrainObservation,
) -> ConstructionAssignment:
    slots = {slot.slot_id: slot for slot in observation.blueprint_slots}
    if slots.get(assignment.slot_id) and slots[assignment.slot_id].completed:
        status = "completed"
    elif assignment.beam_name == observation.current_beam_name:
        status = "active"
    else:
        status = "pending"
    return assignment.model_copy(update={"status": status})


def _active_assignment(
    assignments: list[ConstructionAssignment],
    observation: ConstructionBrainObservation,
) -> ConstructionAssignment | None:
    for assignment in assignments:
        if assignment.status == "active":
            return assignment
    for assignment in assignments:
        if assignment.status == "pending":
            return assignment
    return None


def _blocked_prerequisites(
    assignment: ConstructionAssignment | None,
    observation: ConstructionBrainObservation,
) -> list[str]:
    if assignment is None:
        return []
    completed_components = {
        slot.component_id or slot.slot_id
        for slot in observation.blueprint_slots
        if slot.completed
    }
    return [
        prerequisite
        for prerequisite in assignment.prerequisites
        if prerequisite not in completed_components
    ]
