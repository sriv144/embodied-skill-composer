from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from embodied_skill_composer.construction.experiment_protocol import (
    DEFAULT_EXPERIMENT_PROTOCOL_PATH,
    AblationEvidence,
    CheckpointValidationResult,
    ReleaseEvidence,
    assert_split_isolation,
    decide_ablation_support,
    decide_release_completeness,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
    select_validation_checkpoint,
)

DIGEST = "a" * 64


def _candidate(
    checkpoint_id: str,
    *,
    fraction: float,
    completion: float,
    makespan: float,
    transitions: int,
    seeds: list[int] | None = None,
    configuration_digest: str = DIGEST,
    source_commit: str = "abc123",
) -> CheckpointValidationResult:
    return CheckpointValidationResult(
        checkpoint_id=checkpoint_id,
        experiment_id="construction_intelligence_v1",
        experiment_variant="mappo_full",
        training_seed=7,
        checkpoint_fraction=fraction,
        transition_count=transitions,
        split="validation",
        scenario_seeds=seeds or [800, 801, 802, 803, 804],
        mean_completion_rate=completion,
        mean_makespan_s=makespan,
        checkpoint_path=f"artifacts/{checkpoint_id}.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_lineage=[f"snapshot-{fraction:.2f}"],
        configuration_digest=configuration_digest,
        source_commit=source_commit,
        resume_provenance={"attempt": 2, "resumed": True},
    )


def _complete_candidates() -> list[CheckpointValidationResult]:
    return [
        _candidate(
            f"checkpoint-{index}",
            fraction=fraction,
            completion=0.90,
            makespan=110.0,
            transitions=transitions,
        )
        for index, (fraction, transitions) in enumerate(
            [
                (0.1, 150_000),
                (0.25, 375_000),
                (0.5, 750_000),
                (0.75, 1_125_000),
                (1.0, 1_500_000),
            ],
            start=1,
        )
    ]


def test_load_frozen_protocol_and_stable_digest() -> None:
    protocol = load_experiment_protocol()

    assert protocol.experiment_id == "construction_intelligence_v1"
    assert protocol.profiles["research"].transitions == 1_500_000
    assert protocol.scenario_splits.validation_seeds == [800, 801, 802, 803, 804]
    assert protocol.scenario_splits.heldout_seeds == [900, 901, 902, 903, 904]
    assert protocol.selection.failure_modes == [False, True]
    assert len(protocol_digest(protocol)) == 64
    assert protocol_digest(protocol) == protocol_digest(load_experiment_protocol())
    assert_split_isolation(protocol)


def test_loader_rejects_unknown_fields_and_split_leakage(tmp_path: Path) -> None:
    payload = yaml.safe_load(DEFAULT_EXPERIMENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["unregistered"] = True
    bad_extra = tmp_path / "extra.yaml"
    bad_extra.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_experiment_protocol(bad_extra)

    payload.pop("unregistered")
    payload["selection"]["scenario_seeds"] = [900, 901, 902, 903, 904]
    leaked = tmp_path / "leaked.yaml"
    leaked.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="registered validation seeds"):
        load_experiment_protocol(leaked)


@pytest.mark.parametrize("profile", ["unit", "smoke", "research"])
def test_every_profile_expands_to_exact_twenty_run_matrix(profile: str) -> None:
    protocol = load_experiment_protocol()
    runs = expand_experiment_matrix(protocol, profile)  # type: ignore[arg-type]

    assert len(runs) == 20
    assert len({run.run_id for run in runs}) == 20
    assert {run.training_seed for run in runs} == {7, 8, 9, 10, 11}
    assert {run.experiment_variant for run in runs} == {
        "mappo_full",
        "ippo_full",
        "mappo_no_bc",
        "mappo_no_failure_curriculum",
    }
    assert all(run.seed == run.training_seed for run in runs)
    assert all(run.checkpoint_fractions == [0.1, 0.25, 0.5, 0.75, 1.0] for run in runs)
    assert all(len(run.protocol_run_digest) == 64 for run in runs)
    assert all(
        run.training_config_payload()["protocol_run_digest"]
        == run.protocol_run_digest
        for run in runs
    )


def test_matrix_applies_only_registered_ablation_overrides() -> None:
    runs = expand_experiment_matrix(load_experiment_protocol(), "research")
    by_variant = {run.experiment_variant: run for run in runs if run.training_seed == 7}

    assert by_variant["mappo_full"].expert_episodes == 128
    assert by_variant["mappo_full"].behavior_clone_epochs == 40
    assert by_variant["mappo_full"].include_training_failures
    assert by_variant["ippo_full"].algorithm == "ippo"
    assert by_variant["mappo_no_bc"].expert_episodes == 0
    assert by_variant["mappo_no_bc"].behavior_clone_epochs == 0
    assert not by_variant["mappo_no_failure_curriculum"].include_training_failures
    assert by_variant["mappo_full"].training_config_payload()["experiment_variant"] == (
        "mappo_full"
    )
    assert "configuration_digest" not in by_variant["mappo_full"].training_config_payload()


def test_validation_checkpoint_selection_uses_registered_order() -> None:
    candidates = _complete_candidates()
    candidates[2] = _candidate(
        "late",
        fraction=0.5,
        completion=0.96,
        makespan=105.0,
        transitions=750_000,
    )
    candidates[3] = _candidate(
        "fast",
        fraction=0.75,
        completion=0.96,
        makespan=100.0,
        transitions=1_125_000,
    )
    candidates[4] = _candidate(
        "slower",
        fraction=1.0,
        completion=0.96,
        makespan=100.0,
        transitions=1_500_000,
    )

    selected = select_validation_checkpoint(
        candidates,
        validation_seeds=[800, 801, 802, 803, 804],
    )

    assert selected.checkpoint_id == "fast"
    assert selected.split == "validation"
    assert selected.selection_rule.startswith("completion_rate_desc")
    assert selected.checkpoint_path == "artifacts/fast.pt"
    assert selected.checkpoint_sha256 == "b" * 64
    assert selected.checkpoint_lineage == ["snapshot-0.75"]
    assert selected.resume_provenance == {"attempt": 2, "resumed": True}
    assert selected.required_checkpoint_fractions == [0.1, 0.25, 0.5, 0.75, 1.0]
    assert selected.candidate_ranking[0] == "fast"


def test_validation_checkpoint_selection_is_deterministic_at_ties() -> None:
    candidates = _complete_candidates()
    candidates[2] = _candidate(
        "z-checkpoint",
        fraction=0.5,
        completion=0.95,
        makespan=100.0,
        transitions=750_000,
    )
    candidates[3] = _candidate(
        "a-checkpoint",
        fraction=0.75,
        completion=0.95,
        makespan=100.0,
        transitions=750_000,
    )

    forward = select_validation_checkpoint(
        candidates,
        validation_seeds=[800, 801, 802, 803, 804],
    )
    reverse = select_validation_checkpoint(
        list(reversed(candidates)),
        validation_seeds=[800, 801, 802, 803, 804],
    )

    assert forward.checkpoint_id == reverse.checkpoint_id == "a-checkpoint"


def test_checkpoint_selection_rejects_incomplete_or_heldout_provenance() -> None:
    with pytest.raises(ValueError, match="not evaluated on exactly"):
        candidates = _complete_candidates()
        candidates[0] = _candidate(
            "partial",
            fraction=0.1,
            completion=0.9,
            makespan=100.0,
            transitions=150_000,
            seeds=[800],
        )
        select_validation_checkpoint(
            candidates,
            validation_seeds=[800, 801, 802, 803, 804],
        )
    with pytest.raises(ValidationError, match="validation"):
        CheckpointValidationResult.model_validate(
            {
                **_candidate(
                    "leaked",
                    fraction=0.5,
                    completion=0.9,
                    makespan=100.0,
                    transitions=750_000,
                ).model_dump(),
                "split": "test",
                "scenario_seeds": [900, 901, 902, 903, 904],
            }
        )


def test_checkpoint_selection_requires_complete_unique_fraction_grid() -> None:
    complete = _complete_candidates()
    with pytest.raises(ValueError, match="exactly the required"):
        select_validation_checkpoint(
            complete[:-1],
            validation_seeds=[800, 801, 802, 803, 804],
        )

    duplicate = list(complete)
    duplicate[-1] = _candidate(
        "duplicate-fraction",
        fraction=0.75,
        completion=0.90,
        makespan=110.0,
        transitions=1_500_000,
    )
    with pytest.raises(ValueError, match="duplicate fraction"):
        select_validation_checkpoint(
            duplicate,
            validation_seeds=[800, 801, 802, 803, 804],
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("configuration_digest", "configuration digest"),
        ("source_commit", "source commit"),
    ],
)
def test_checkpoint_selection_requires_shared_provenance(field: str, message: str) -> None:
    candidates = _complete_candidates()
    replacement = candidates[-1].model_dump()
    replacement[field] = "b" * 64 if field == "configuration_digest" else "other-commit"
    candidates[-1] = CheckpointValidationResult.model_validate(replacement)

    with pytest.raises(ValueError, match=message):
        select_validation_checkpoint(
            candidates,
            validation_seeds=[800, 801, 802, 803, 804],
        )


def test_behavior_cloning_inclusive_boundaries_and_either_rule() -> None:
    protocol = load_experiment_protocol()
    completion_boundary = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="behavior_cloning",
            full_final_completion=0.95,
            ablated_final_completion=0.90,
        ),
    )
    transition_boundary = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="behavior_cloning",
            full_final_completion=0.94,
            ablated_final_completion=0.90,
            full_transitions_to_95=800,
            ablated_transitions_to_95=1000,
        ),
    )

    assert completion_boundary.supported
    assert completion_boundary.completion_boundary_met
    assert transition_boundary.supported
    assert transition_boundary.transition_boundary_met


def test_failure_curriculum_requires_gain_and_no_failure_safety() -> None:
    protocol = load_experiment_protocol()
    supported = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="failure_curriculum",
            full_failure_completion=0.90,
            ablated_failure_completion=0.80,
            full_no_failure_completion=0.94,
            ablated_no_failure_completion=0.96,
        ),
    )
    unsafe = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="failure_curriculum",
            full_failure_completion=0.90,
            ablated_failure_completion=0.80,
            full_no_failure_completion=0.93,
            ablated_no_failure_completion=0.96,
        ),
    )

    assert supported.supported
    assert supported.no_failure_safety_boundary_met
    assert not unsafe.supported
    assert not unsafe.no_failure_safety_boundary_met


def test_unsupported_ablation_does_not_block_complete_release() -> None:
    protocol = load_experiment_protocol()
    expected = [run.run_id for run in expand_experiment_matrix(protocol, "research")]
    unsupported_bc = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="behavior_cloning",
            full_final_completion=0.91,
            ablated_final_completion=0.90,
        ),
    )
    supported_curriculum = decide_ablation_support(
        protocol,
        AblationEvidence(
            hypothesis="failure_curriculum",
            full_failure_completion=0.90,
            ablated_failure_completion=0.80,
            full_no_failure_completion=0.95,
            ablated_no_failure_completion=0.96,
        ),
    )

    decision = decide_release_completeness(
        protocol,
        ReleaseEvidence(
            completed_training_run_ids=expected,
            selected_policy_run_ids=expected,
            heldout_evaluated_run_ids=expected,
            ablation_decisions=[unsupported_bc, supported_curriculum],
            primary_acceptance_passed=True,
            report_generated=True,
            reproducibility_artifacts_complete=True,
        ),
    )

    assert decision.complete
    assert decision.blocking_reasons == []
    assert decision.unsupported_ablation_hypotheses == ["behavior_cloning"]


def test_incomplete_matrix_and_heldout_selection_are_release_blockers() -> None:
    protocol = load_experiment_protocol()
    expected = [run.run_id for run in expand_experiment_matrix(protocol, "research")]
    decisions = [
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="behavior_cloning",
                full_final_completion=0.95,
                ablated_final_completion=0.90,
            ),
        ),
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="failure_curriculum",
                full_failure_completion=0.90,
                ablated_failure_completion=0.80,
                full_no_failure_completion=0.95,
                ablated_no_failure_completion=0.96,
            ),
        ),
    ]

    decision = decide_release_completeness(
        protocol,
        ReleaseEvidence(
            completed_training_run_ids=expected[:-1],
            selected_policy_run_ids=expected,
            heldout_evaluated_run_ids=expected,
            ablation_decisions=decisions,
            primary_acceptance_passed=True,
            report_generated=True,
            reproducibility_artifacts_complete=True,
            heldout_used_for_selection=True,
        ),
    )

    assert not decision.complete
    assert decision.missing_training_run_ids == [expected[-1]]
    assert any("held-out" in reason for reason in decision.blocking_reasons)
