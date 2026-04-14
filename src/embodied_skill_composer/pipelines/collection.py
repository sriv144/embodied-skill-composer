from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embodied_skill_composer.core.executor import TaskExecutor
from embodied_skill_composer.core.logging_utils import write_execution_report
from embodied_skill_composer.core.models import ExecutionEvent, ExecutionReport, PerceptionReport, TaskSpec
from embodied_skill_composer.core.planner import RuleBasedPlanner
from embodied_skill_composer.perception.modules import build_perception


@dataclass
class CollectionEpisodeResult:
    report: ExecutionReport
    perception_reports: list[PerceptionReport]
    objects_collected: int
    target_completion_rate: float
    grasp_retry_rate: float
    perception_miss_rate: float
    action_count: int


class CollectionEpisodeRunner:
    def __init__(self, adapter, log_dir: Path, max_cycles: int = 3) -> None:
        self.adapter = adapter
        self.log_dir = log_dir
        self.max_cycles = max_cycles
        self.planner = RuleBasedPlanner()

    def run(self, task: TaskSpec) -> CollectionEpisodeResult:
        executor = TaskExecutor(adapter=self.adapter, log_dir=self.log_dir)
        all_events: list[ExecutionEvent] = []
        perception_reports: list[PerceptionReport] = []
        last_report: ExecutionReport | None = None

        for _ in range(self.max_cycles):
            perception = build_perception(task.perception_mode)
            perceived_world, perception_report = perception.build_world(self.adapter, task)
            perception_reports.append(perception_report)
            pending_targets = [
                name for name in task.target_objects if name in perceived_world.objects and not perceived_world.objects[name].collected
            ]
            if not pending_targets:
                break
            cycle_task = task.model_copy(update={"target_objects": pending_targets})
            plan = self.planner.plan(cycle_task, perceived_world)
            last_report = executor.run(cycle_task, plan)
            all_events.extend(last_report.events)
            final_world = self.adapter.get_world_state()
            if all(final_world.objects[name].collected for name in task.target_objects if name in final_world.objects):
                break

        final_world = self.adapter.get_world_state()
        if last_report is None:
            empty_report = ExecutionReport(
                task_name=task.name,
                success=False,
                plan=[],
                events=[],
                failure_step="perception",
                final_world=final_world,
            )
            empty_report.log_path = str(write_execution_report(empty_report, self.log_dir))
            last_report = empty_report
        else:
            aggregate = ExecutionReport(
                task_name=task.name,
                success=all(
                    final_world.objects[name].collected
                    for name in task.target_objects
                    if name in final_world.objects
                ),
                plan=last_report.plan,
                events=all_events,
                failure_step=None if all(
                    final_world.objects[name].collected
                    for name in task.target_objects
                    if name in final_world.objects
                ) else last_report.failure_step,
                final_world=final_world,
            )
            aggregate.log_path = str(write_execution_report(aggregate, self.log_dir))
            last_report = aggregate

        collected_targets = sum(
            1
            for name in task.target_objects
            if name in final_world.objects and final_world.objects[name].collected
        )
        target_completion_rate = collected_targets / max(1, len(task.target_objects))
        grasp_events = [event for event in all_events if event.step_name == "pick_object"]
        grasp_failures = sum(1 for event in grasp_events if not event.success)
        total_missed = sum(len(report.missed_targets) for report in perception_reports)
        total_expected = max(1, len(task.target_objects) * max(1, len(perception_reports)))

        return CollectionEpisodeResult(
            report=last_report,
            perception_reports=perception_reports,
            objects_collected=collected_targets,
            target_completion_rate=target_completion_rate,
            grasp_retry_rate=grasp_failures / max(1, len(grasp_events)),
            perception_miss_rate=total_missed / total_expected,
            action_count=len(all_events),
        )
