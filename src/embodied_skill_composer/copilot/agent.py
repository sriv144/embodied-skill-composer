from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from embodied_skill_composer.copilot.nvidia import run_nvidia_readiness_check
from embodied_skill_composer.copilot.registry import default_registry
from embodied_skill_composer.copilot.runner import (
    ConfirmationRequired,
    run_benchmark,
    run_sweep,
    run_train_marl,
    run_train_options,
)


def load_dotenv_if_needed(path: Path) -> None:
    if os.environ.get("OPENAI_API_KEY") or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def run_agent_prompt(prompt: str, model: str | None = None, allow_training: bool = False) -> str:
    load_dotenv_if_needed(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured. Save it in .env or set it in the environment.")
    try:
        from agents import Agent, RunConfig, Runner, function_tool
    except ImportError as exc:
        raise RuntimeError("Install the Agents SDK with `pip install -r requirements.txt`.") from exc
    logging.getLogger("openai.agents").setLevel(logging.CRITICAL)

    registry = default_registry()

    @function_tool
    def list_recent_runs(limit: int = 5) -> str:
        return json.dumps(registry.recent_runs(limit=limit), indent=2)

    @function_tool
    def benchmark(episodes: int = 3) -> str:
        result = run_benchmark(episodes=episodes, registry=registry)
        return _result_json(result)

    @function_tool
    def nvidia_check() -> str:
        return json.dumps(run_nvidia_readiness_check(), indent=2)

    @function_tool
    def train_options() -> str:
        try:
            result = run_train_options(yes=allow_training, registry=registry)
        except ConfirmationRequired as exc:
            return str(exc)
        return _result_json(result)

    @function_tool
    def train_marl() -> str:
        try:
            result = run_train_marl(yes=allow_training, registry=registry)
        except ConfirmationRequired as exc:
            return str(exc)
        return _result_json(result)

    @function_tool
    def sweep(scenarios: int = 5, seeds: str = "7,8,9") -> str:
        try:
            result = run_sweep(scenarios=scenarios, seeds=seeds, yes=allow_training, registry=registry)
        except ConfirmationRequired as exc:
            return str(exc)
        return _result_json(result)

    agent = Agent(
        name="Embodied Skill Composer Experiment Copilot",
        instructions=(
            "You are a research/debug copilot for a two-robot assembly simulation repo. "
            "Use tools to ground claims in current metrics. Prefer concise lab-note style: "
            "what was run, what happened, likely cause, and next experiment. "
            "Do not claim NVIDIA AI-Q or Isaac infrastructure is running unless nvidia_check proves it."
        ),
        tools=[list_recent_runs, benchmark, nvidia_check, train_options, train_marl, sweep],
    )
    run_config = RunConfig(model=model) if model else None
    try:
        result = Runner.run_sync(agent, prompt, run_config=run_config)
    except Exception as exc:
        raise RuntimeError(f"OpenAI agent run failed: {exc}") from exc
    return str(result.final_output)


def _result_json(result: Any) -> str:
    return json.dumps(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "report_path": str(result.report_path),
            "exit_code": result.exit_code,
            "summary": result.summary,
        },
        indent=2,
    )
