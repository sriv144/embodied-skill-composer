from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from embodied_skill_composer.core.models import TaskSpec
from embodied_skill_composer.pipelines.collection import CollectionEpisodeRunner


@dataclass
class BenchmarkSummary:
    task_name: str
    episodes: int
    success_rate: float
    collection_completion_rate: float
    average_objects_collected: float
    grasp_retry_rate: float
    perception_miss_rate: float
    average_action_count: float
    policy_mode: str
    perception_mode: str


class BenchmarkRunner:
    def __init__(self, adapter_factory, log_dir: Path) -> None:
        self.adapter_factory = adapter_factory
        self.log_dir = log_dir

    def run(self, task: TaskSpec, seeds: list[int]) -> BenchmarkSummary:
        successes = 0
        objects_collected = 0
        completion_rate = 0.0
        grasp_retry_rate = 0.0
        perception_miss_rate = 0.0
        action_count = 0

        for seed in seeds:
            adapter = self.adapter_factory(seed)
            try:
                runner = CollectionEpisodeRunner(adapter=adapter, log_dir=self.log_dir)
                result = runner.run(task)
                successes += int(result.report.success)
                objects_collected += result.objects_collected
                completion_rate += result.target_completion_rate
                grasp_retry_rate += result.grasp_retry_rate
                perception_miss_rate += result.perception_miss_rate
                action_count += result.action_count
            finally:
                adapter.close()

        episodes = len(seeds)
        summary = BenchmarkSummary(
            task_name=task.name,
            episodes=episodes,
            success_rate=successes / max(1, episodes),
            collection_completion_rate=completion_rate / max(1, episodes),
            average_objects_collected=objects_collected / max(1, episodes),
            grasp_retry_rate=grasp_retry_rate / max(1, episodes),
            perception_miss_rate=perception_miss_rate / max(1, episodes),
            average_action_count=action_count / max(1, episodes),
            policy_mode=task.policy_mode,
            perception_mode=task.perception_mode,
        )
        self._write_summary(summary)
        return summary

    def _write_summary(self, summary: BenchmarkSummary) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"benchmark-{summary.task_name}-{summary.policy_mode}-{summary.perception_mode}.json"
        path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
