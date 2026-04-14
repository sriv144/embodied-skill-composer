from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from embodied_skill_composer.core.models import ExecutionReport


def write_execution_report(report: ExecutionReport, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{report.task_name}-{timestamp}.json"
    log_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return log_path

