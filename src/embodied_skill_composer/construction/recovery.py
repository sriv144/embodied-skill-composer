from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.models import (
    BrainEvent,
    BuildPlan,
    ConstructionSchedule,
    ExecutionTrace,
)
from embodied_skill_composer.construction.trace import build_execution_trace


FailureType = Literal["obstacle", "robot_unavailable", "dropped_resource"]


class Disruption(BaseModel):
    failure_type: FailureType
    timestamp_s: int = Field(ge=0)
    robot_id: str | None = None


RECOVERY_DELAY_SECONDS: dict[FailureType, int] = {
    "obstacle": 14,
    "robot_unavailable": 24,
    "dropped_resource": 12,
}


def inject_disruption(
    plan: BuildPlan,
    schedule: ConstructionSchedule,
    disruption: Disruption,
) -> ExecutionTrace:
    if disruption.timestamp_s >= schedule.makespan_s:
        raise ValueError("disruption must occur before construction completes")
    delay = RECOVERY_DELAY_SECONDS[disruption.failure_type]
    future_starts = sorted(
        {job.start_s for job in schedule.jobs if job.start_s >= disruption.timestamp_s}
    )
    shift_from = future_starts[0] if future_starts else disruption.timestamp_s
    pivot = min(
        (job for job in schedule.jobs if job.end_s > disruption.timestamp_s),
        key=lambda job: (max(job.start_s, disruption.timestamp_s), job.module_id),
    )
    shifted_jobs = [
        job.model_copy(
            update={
                "start_s": job.start_s + delay,
                "pickup_s": job.pickup_s + delay,
                "end_s": job.end_s + delay,
            }
        )
        if job.start_s >= shift_from
        else job.model_copy(deep=True)
        for job in schedule.jobs
    ]
    new_schedule = schedule.model_copy(
        update={
            "jobs": shifted_jobs,
            "makespan_s": schedule.makespan_s + delay,
            "idle_robot_seconds": schedule.idle_robot_seconds + delay * len(plan.robots),
            "total_energy_wh": round(schedule.total_energy_wh + delay * 0.035, 3),
            "solver_status": f"{schedule.solver_status}+recovered_{disruption.failure_type}",
        }
    )
    trace = build_execution_trace(plan, new_schedule)
    trace.metrics.recovery_cost_s = delay
    trace.metrics.wasted_work_s = 6 if disruption.failure_type == "dropped_resource" else 0
    trace.brain_events.append(
        BrainEvent(
            timestamp_s=disruption.timestamp_s,
            event_type="recovery",
            module_id=pivot.module_id,
            robot_ids=(
                [disruption.robot_id]
                if disruption.robot_id is not None
                else list(pivot.robot_ids)
            ),
            candidates=[pivot.module_id],
            reason=_recovery_reason(disruption.failure_type, delay),
            predicted_remaining_s=new_schedule.makespan_s - disruption.timestamp_s,
        )
    )
    trace.brain_events.sort(
        key=lambda item: (item.timestamp_s, 0 if item.event_type == "recovery" else 1)
    )
    return trace


def _recovery_reason(failure_type: FailureType, delay: int) -> str:
    reasons = {
        "obstacle": "Site obstacle invalidated the active route; reserved paths were cleared and replanned.",
        "robot_unavailable": "Robot heartbeat was lost; work paused for a bounded health recovery window.",
        "dropped_resource": "Grip loss was detected; the affected module was inspected and returned to pickup state.",
    }
    return f"{reasons[failure_type]} Recovery adds {delay} seconds to predicted makespan."
