from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(
    run_dir: Path,
    title: str,
    command: list[str],
    exit_code: int,
    summary: dict[str, Any] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> Path:
    lines = [
        f"# {title}",
        "",
        "## Command",
        "",
        "```text",
        " ".join(command) if command else "(no subprocess command)",
        "```",
        "",
        "## Result",
        "",
        f"- Exit code: `{exit_code}`",
    ]
    if summary:
        lines.extend(["", "## Research Debug Summary", ""])
        lines.extend(_summarize_payload(summary))
    if stdout_path:
        lines.append(f"- Stdout: `{stdout_path}`")
    if stderr_path:
        lines.append(f"- Stderr: `{stderr_path}`")
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"notes": [f"Expected artifact not found: {path}"]}
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"notes": [f"Could not parse JSON artifact {path}: {exc}"]}
    if not isinstance(payload, dict):
        return {"notes": [f"Expected a JSON object in artifact {path}"]}
    return {str(key): value for key, value in payload.items()}


def _summarize_payload(payload: dict[str, Any]) -> list[str]:
    if {"scripted_options", "learned_options", "low_level_learned"}.issubset(payload):
        rows = [
            payload["scripted_options"],
            payload["learned_options"],
            payload["low_level_learned"],
        ]
        lines = [
            "| Policy | Success | Return | Beams | Steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {policy_name} | {success_rate:.3f} | {mean_return:.2f} | "
                "{mean_beams_installed:.2f} | {mean_step_count:.1f} |".format(
                    policy_name=row.get("policy_name", "unknown"),
                    success_rate=float(row.get("success_rate", 0.0)),
                    mean_return=float(row.get("mean_return", 0.0)),
                    mean_beams_installed=float(row.get("mean_beams_installed", 0.0)),
                    mean_step_count=float(row.get("mean_step_count", 0.0)),
                )
            )
        learned = payload["learned_options"]
        low_level = payload["low_level_learned"]
        lines.extend(
            [
                "",
                "- Hierarchical options are the main policy to watch because they learn over task-level coordination choices.",
                "- Low-level MARL remains a retained baseline; poor beam completion usually points to coordination and credit-assignment difficulty.",
                f"- Backend: `{payload.get('backend', 'unknown')}`, profile: `{payload.get('runtime_profile', 'unknown')}`.",
                f"- Success gap: `{float(learned.get('success_rate', 0.0)) - float(low_level.get('success_rate', 0.0)):.3f}`.",
            ]
        )
        return lines
    if "runtime" in payload and "aiq" in payload:
        return [
            f"- CUDA available: `{payload['runtime'].get('cuda_available')}`",
            f"- Tensor allocation OK: `{payload['runtime'].get('tensor_allocation_ok')}`",
            f"- Isaac ready: `{payload['isaac_backend'].get('is_ready')}`",
            f"- AI-Q reachable: `{payload['aiq'].get('reachable')}`",
            "- NVIDIA v1 remains a readiness/check track; no AI-Q or Isaac infrastructure was started.",
        ]
    if "notes" in payload:
        return [f"- {note}" for note in payload["notes"]]
    return ["```json", json.dumps(payload, indent=2), "```"]
