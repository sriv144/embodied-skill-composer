from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from embodied_skill_composer.construction.phase3_verification import (
    DEFAULT_OUTPUT_ROOT,
    run_phase3_end_to_end_verification,
)


@pytest.mark.construction_phase3_e2e
@pytest.mark.skipif(
    os.environ.get("RUN_CONSTRUCTION_PHASE3_E2E") != "1",
    reason="set RUN_CONSTRUCTION_PHASE3_E2E=1 on a clean worktree",
)
def test_canonical_unit_matrix_runs_real_recovery_selection_and_heldout() -> None:
    configured_root = os.environ.get("CONSTRUCTION_PHASE3_E2E_ROOT")
    output_root = (
        Path(configured_root)
        if configured_root
        else DEFAULT_OUTPUT_ROOT
        / (
            "pytest-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + f"-{uuid4().hex[:8]}"
        )
    )
    timeout_seconds = float(
        os.environ.get("CONSTRUCTION_PHASE3_E2E_TIMEOUT_SECONDS", "3600")
    )
    result = run_phase3_end_to_end_verification(
        output_root,
        timeout_seconds=timeout_seconds,
    )

    assert result.training_run_count == 20
    assert result.interruption.completed_attempt >= 2
    assert (
        result.interruption.resumed_transitions
        >= result.interruption.checkpoint_transitions
    )
    assert result.fractional_policy_export_count == 100
    assert result.selection_count == 20
    assert result.validation_candidate_count == 100
    assert result.validation_episode_count == 1000
    assert result.heldout_episode_count == 240
    assert result.learned_episode_count == 200
    assert result.baseline_episode_count == 40
    assert result.verification_path.is_file()
