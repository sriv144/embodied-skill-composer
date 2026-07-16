# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaExecutor,
)
from embodied_skill_composer.construction.marl_env_v1 import (
    TemporalConstructionCoordinationEnv,
    auction_temporal_actions,
    scripted_temporal_actions,
)
from embodied_skill_composer.construction.policy import (
    load_policy_checkpoint,
    policy_actions,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import generate_cottage_scenario
from embodied_skill_composer.construction.scheduler import schedule_build
from embodied_skill_composer.construction.training import cp_sat_expert_actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online Construction Intelligence through wheel-driven CoppeliaSim YouBots.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=WORKSPACE / "configs" / "construction" / "cottage_v1.yaml",
    )
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument(
        "--controller",
        choices=("greedy", "auction", "cp_sat", "mappo", "ippo"),
        default="greedy",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "coppelia",
    )
    parser.add_argument("--yes", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.yes:
        print("Coppelia launch is approval-gated. Re-run with --yes after starting CoppeliaSim.")
        return 2
    if args.controller in {"mappo", "ippo"} and not args.checkpoint:
        raise SystemExit("Learned controllers require --checkpoint")

    scenario = generate_cottage_scenario(args.seed, load_house_design(args.design))
    env = TemporalConstructionCoordinationEnv(scenario)
    bundle = (
        load_policy_checkpoint(args.checkpoint, device=args.device) if args.checkpoint else None
    )
    priority = None
    if args.controller == "cp_sat":
        schedule = schedule_build(scenario.plan, "optimized")
        priority = {
            job.module_id: (job.start_s, job.end_s, tuple(job.robot_ids))
            for job in schedule.jobs
        }

    def action_provider(active_env, observations):
        if args.controller == "greedy":
            return scripted_temporal_actions(active_env)
        if args.controller == "auction":
            return auction_temporal_actions(active_env)
        if args.controller == "cp_sat":
            assert priority is not None
            return cp_sat_expert_actions(active_env, priority)
        assert bundle is not None
        return policy_actions(
            bundle.actor_model,
            observations,
            active_env.possible_agents,
            device=args.device,
            deterministic=True,
        )

    executor = DynamicCoppeliaExecutor(scenario.plan)
    executor.connect()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / f"{timestamp}-{args.controller}-s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    failure: Exception | None = None
    try:
        diagnostics = executor.execute_online(
            env,
            action_provider,
            max_decisions=args.max_decisions,
        )
        diagnostics["status"] = "completed"
    except Exception as exc:  # Persist measured evidence before returning a non-zero status.
        failure = exc
        diagnostics = executor.diagnostics(logical_metrics=env.metrics())
        diagnostics.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    (run_dir / "scenario.json").write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "planned_trace.json").write_text(
        json.dumps(env.event_log, indent=2),
        encoding="utf-8",
    )
    (run_dir / "brain_events.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in env.brain_events], indent=2),
        encoding="utf-8",
    )
    (run_dir / "wheel_commands.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in executor.commands], indent=2),
        encoding="utf-8",
    )
    (run_dir / "measured_telemetry.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in executor.telemetry], indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    scene_path = run_dir / "construction_intelligence.ttt"
    executor.sim.saveScene(str(scene_path))
    (run_dir / "report.md").write_text(_report(args.controller, diagnostics), encoding="utf-8")
    print(f"Dynamic Coppelia run written to {run_dir}")
    return 1 if failure else 0


def _report(controller: str, diagnostics: dict[str, object]) -> str:
    raw_logical = diagnostics.get("logical_metrics")
    logical = raw_logical if isinstance(raw_logical, dict) else {}
    measured = _numeric_value(diagnostics.get("measured_duration_s"), "measured_duration_s")
    planned = _numeric_value(logical.get("makespan_s", 0), "makespan_s")
    return "\n".join(
        [
            "# Dynamic Coppelia Construction Run",
            "",
            f"- Status: `{diagnostics.get('status', 'unknown')}`",
            f"- Controller: `{controller}`",
            f"- Base controller: `{diagnostics['controller']}`",
            f"- Logical completion: `{logical.get('structure_completion_rate', 0):.3f}`",
            f"- Planned event duration: `{planned:.2f} s`",
            f"- Measured simulator duration: `{measured:.2f} s`",
            f"- Measured/planned ratio: `{measured / planned:.2f}`" if planned else "- Measured/planned ratio: `n/a`",
            f"- Wheel commands: `{diagnostics['wheel_command_count']}`",
            f"- Measured pose samples: `{diagnostics['telemetry_sample_count']}`",
            f"- Post-start robot pose writes: `{diagnostics['post_start_robot_pose_writes']}`",
            "",
            "Payload transport is a logical carrier constraint. This run does not claim arm or "
            "gripper contact dynamics.",
            f"\nFailure: `{diagnostics['error']}`" if diagnostics.get("error") else "",
            "",
        ]
    )


def _numeric_value(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
