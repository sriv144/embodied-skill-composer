from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from embodied_skill_composer.core.interfaces import SimulationAdapter
from embodied_skill_composer.core.logging_utils import write_execution_report
from embodied_skill_composer.core.models import ExecutionEvent, ExecutionReport, SkillStep, TaskSpec
from embodied_skill_composer.core.skills import build_skill_registry


class TaskExecutor:
    def __init__(self, adapter: SimulationAdapter, log_dir: Path) -> None:
        self.adapter = adapter
        self.log_dir = log_dir
        self.skills = build_skill_registry()

    def run(self, task: TaskSpec, plan: list[SkillStep]) -> ExecutionReport:
        events: list[ExecutionEvent] = []
        failure_step: str | None = None

        for step in plan:
            skill = self.skills[step.name]
            for attempt in range(step.max_retries + 1):
                world = self.adapter.get_world_state()
                candidate_step = self._step_for_attempt(step, attempt)
                precheck = skill.check_preconditions(world, candidate_step)
                if not precheck.success:
                    events.append(
                        ExecutionEvent(
                            step_name=step.name,
                            attempt=attempt + 1,
                            success=False,
                            message=precheck.message,
                            params=candidate_step.params,
                            error_code=precheck.error_code,
                        )
                    )
                    failure_step = step.name
                    return self._finalize(task, plan, events, failure_step)

                result = skill.execute(self.adapter, world, candidate_step)
                events.append(
                    ExecutionEvent(
                        step_name=step.name,
                        attempt=attempt + 1,
                        success=result.success,
                        message=result.message,
                        params=candidate_step.params,
                        error_code=result.error_code,
                    )
                )
                if result.success:
                    break
                if attempt == step.max_retries:
                    failure_step = step.name
                    return self._finalize(task, plan, events, failure_step)

        return self._finalize(task, plan, events, failure_step)

    def _step_for_attempt(self, step: SkillStep, attempt: int) -> SkillStep:
        adjusted = deepcopy(step)
        if adjusted.name == "grasp_object":
            base_offset = float(adjusted.params.get("approach_offset", 0.04))
            adjusted.params["approach_offset"] = max(0.015, base_offset - (attempt * 0.005))
        return adjusted

    def _finalize(
        self,
        task: TaskSpec,
        plan: list[SkillStep],
        events: list[ExecutionEvent],
        failure_step: str | None,
    ) -> ExecutionReport:
        report = ExecutionReport(
            task_name=task.name,
            success=failure_step is None,
            plan=plan,
            events=events,
            failure_step=failure_step,
            final_world=self.adapter.get_world_state(),
        )
        log_path = write_execution_report(report, self.log_dir)
        report.log_path = str(log_path)
        return report

