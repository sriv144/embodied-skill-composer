from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_skill_composer.copilot.agent import run_agent_prompt
from embodied_skill_composer.copilot.nvidia import run_nvidia_readiness_check
from embodied_skill_composer.copilot.registry import default_registry
from embodied_skill_composer.copilot.reports import write_report
from embodied_skill_composer.copilot.runner import (
    ConfirmationRequired,
    run_benchmark,
    run_eval_options,
    run_sweep,
    run_train_marl,
    run_train_options,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first robotics experiment copilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask the OpenAI Agents SDK copilot.")
    ask.add_argument("prompt")
    ask.add_argument("--model", default=None)
    ask.add_argument("--yes", action="store_true", help="Allow training/sweep tools during the agent run.")

    benchmark = subparsers.add_parser("benchmark", help="Run policy benchmark and record artifacts.")
    benchmark.add_argument("--episodes", type=int, default=5)
    benchmark.add_argument("--runtime-profile", default="configs/assembly_profiles/local_dev.yaml")

    eval_options = subparsers.add_parser("eval-options", help="Run option-policy evaluation.")
    eval_options.add_argument("--policy", choices=["scripted", "learned"], default="scripted")
    eval_options.add_argument("--episodes", type=int, default=1)
    eval_options.add_argument("--runtime-profile", default="configs/assembly_profiles/local_dev.yaml")

    train_options = subparsers.add_parser("train-options", help="Train hierarchical options into a run folder.")
    train_options.add_argument("--runtime-profile", default="configs/assembly_profiles/local_dev.yaml")
    train_options.add_argument("--yes", action="store_true")

    train_marl = subparsers.add_parser("train-marl", help="Train low-level MARL into a run folder.")
    train_marl.add_argument("--runtime-profile", default="configs/assembly_profiles/local_dev.yaml")
    train_marl.add_argument("--yes", action="store_true")

    sweep = subparsers.add_parser("sweep", help="Run generated scenario experiment sweep.")
    sweep.add_argument("--scenarios", type=int, default=5)
    sweep.add_argument("--seeds", default="7,8,9")
    sweep.add_argument("--beam-count", type=int, default=2)
    sweep.add_argument("--runtime-profile", default="configs/assembly_profiles/local_dev.yaml")
    sweep.add_argument("--yes", action="store_true")

    nvidia = subparsers.add_parser("nvidia-check", help="Run NVIDIA/CUDA/Isaac/AI-Q readiness checks.")
    nvidia.add_argument("--runtime-profile", default="configs/assembly_profiles/local_gpu.yaml")
    nvidia.add_argument("--aiq-url", default="http://localhost:8000")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ask":
            print(run_agent_prompt(args.prompt, model=args.model, allow_training=args.yes))
            return 0
        if args.command == "benchmark":
            return _print_result(run_benchmark(args.episodes, args.runtime_profile))
        if args.command == "eval-options":
            return _print_result(run_eval_options(args.policy, args.episodes, args.runtime_profile))
        if args.command == "train-options":
            return _print_result(run_train_options(args.runtime_profile, yes=args.yes))
        if args.command == "train-marl":
            return _print_result(run_train_marl(args.runtime_profile, yes=args.yes))
        if args.command == "sweep":
            return _print_result(run_sweep(args.scenarios, args.seeds, args.beam_count, args.runtime_profile, yes=args.yes))
        if args.command == "nvidia-check":
            return _run_nvidia_check(args.runtime_profile, args.aiq_url)
    except ConfirmationRequired as exc:
        print(f"Confirmation required: {exc}")
        return 2
    except RuntimeError as exc:
        print(f"Copilot unavailable: {exc}")
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


def _print_result(result) -> int:
    print(f"Run ID: {result.run_id}")
    print(f"Run dir: {result.run_dir}")
    print(f"Report: {result.report_path}")
    print(f"Exit code: {result.exit_code}")
    return result.exit_code


def _run_nvidia_check(runtime_profile: str, aiq_url: str) -> int:
    registry = default_registry()
    record = registry.create_run("nvidia-check", command=[], runtime_profile=runtime_profile)
    summary = run_nvidia_readiness_check(Path(runtime_profile), aiq_url=aiq_url)
    summary_path = record.run_dir / "nvidia_readiness.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_report(record.run_dir, "NVIDIA Readiness Check", [], 0, summary)
    registry.add_artifact(record.id, "summary", summary_path, "CUDA, Isaac, and AI-Q readiness JSON.")
    registry.add_artifact(record.id, "report", report_path, "Markdown readiness report.")
    registry.complete_run(record.id, "completed", 0, report_path)
    print(json.dumps(summary, indent=2))
    print(f"Run ID: {record.id}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
