from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ExperimentProfile = Literal["unit", "smoke", "research"]
Algorithm = Literal["mappo", "ippo"]
AblationHypothesis = Literal["behavior_cloning", "failure_curriculum"]
_OverrideValue = TypeVar("_OverrideValue", int, bool)

DEFAULT_EXPERIMENT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "construction"
    / "experiments"
    / "construction_intelligence_v1.yaml"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProfileSpec(_FrozenModel):
    transitions: int = Field(gt=0)
    expert_episodes: int = Field(ge=0)
    behavior_clone_epochs: int = Field(ge=0)
    rollout_decisions: int = Field(gt=0)
    ppo_epochs: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    hidden_dim: int = Field(ge=32, le=512)
    device: Literal["auto", "cpu", "cuda"]


class VariantOverrides(_FrozenModel):
    expert_episodes: int | None = Field(default=None, ge=0)
    behavior_clone_epochs: int | None = Field(default=None, ge=0)
    include_training_failures: bool | None = None


class ExperimentVariant(_FrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    algorithm: Algorithm
    role: Literal["primary", "ablation"]
    description: str = Field(min_length=1)
    hypothesis: AblationHypothesis | None
    overrides: VariantOverrides

    @model_validator(mode="after")
    def validate_role(self) -> ExperimentVariant:
        if self.role == "primary" and self.hypothesis is not None:
            raise ValueError("primary variants cannot declare an ablation hypothesis")
        if self.role == "ablation" and self.hypothesis is None:
            raise ValueError("ablation variants must declare their hypothesis")
        return self


class SeedRange(_FrozenModel):
    start: int = Field(ge=0, le=999)
    end: int = Field(ge=0, le=999)

    @model_validator(mode="after")
    def validate_order(self) -> SeedRange:
        if self.start > self.end:
            raise ValueError("seed range is inverted")
        return self


class ScenarioSplitProtocol(_FrozenModel):
    training: SeedRange
    validation_seeds: list[int] = Field(min_length=1)
    heldout_seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_isolation(self) -> ScenarioSplitProtocol:
        _require_unique_sorted(self.validation_seeds, "validation_seeds")
        _require_unique_sorted(self.heldout_seeds, "heldout_seeds")
        training = set(range(self.training.start, self.training.end + 1))
        validation = set(self.validation_seeds)
        heldout = set(self.heldout_seeds)
        if training & validation or training & heldout or validation & heldout:
            raise ValueError("training, validation, and held-out scenario seeds must be disjoint")
        if any(not 800 <= seed <= 899 for seed in validation):
            raise ValueError("validation scenario seeds must be in the reserved 800-899 range")
        if any(not 900 <= seed <= 999 for seed in heldout):
            raise ValueError("held-out scenario seeds must be in the reserved 900-999 range")
        if self.training.start != 0 or self.training.end != 799:
            raise ValueError("the frozen training scenario range must be 0-799")
        return self


class SelectionProtocol(_FrozenModel):
    split: Literal["validation"]
    scenario_seeds: list[int] = Field(min_length=1)
    failure_modes: list[bool]
    deterministic: Literal[True]
    metric_order: list[
        Literal[
            "completion_rate_desc",
            "makespan_s_asc",
            "transition_count_asc",
            "checkpoint_id_asc",
        ]
    ]
    forbidden_splits: list[Literal["train", "test"]]

    @model_validator(mode="after")
    def validate_selection_rule(self) -> SelectionProtocol:
        expected_order = [
            "completion_rate_desc",
            "makespan_s_asc",
            "transition_count_asc",
            "checkpoint_id_asc",
        ]
        if self.metric_order != expected_order:
            raise ValueError(f"checkpoint selection metric_order must be {expected_order}")
        if "test" not in self.forbidden_splits:
            raise ValueError("held-out test results must be forbidden during checkpoint selection")
        if self.failure_modes != [False, True]:
            raise ValueError(
                "checkpoint selection must evaluate no-failure and failure validation suites"
            )
        _require_unique_sorted(self.scenario_seeds, "selection scenario_seeds")
        return self


class EvaluationProtocol(_FrozenModel):
    split: Literal["test"]
    scenario_seeds: list[int] = Field(min_length=1)
    deterministic: Literal[True]
    failure_modes: list[bool]
    controllers: list[
        Literal["sequential", "greedy", "auction", "cp_sat", "ippo", "mappo"]
    ]
    aggregation: Literal["hierarchical_bootstrap"]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_samples: Literal[2000]

    @model_validator(mode="after")
    def validate_evaluation_matrix(self) -> EvaluationProtocol:
        _require_unique_sorted(self.scenario_seeds, "evaluation scenario_seeds")
        if self.failure_modes != [False, True]:
            raise ValueError("evaluation must run without and with failures, in that order")
        expected_controllers = {"sequential", "greedy", "auction", "cp_sat", "ippo", "mappo"}
        if set(self.controllers) != expected_controllers or len(self.controllers) != 6:
            raise ValueError("evaluation controllers must contain every frozen learned and baseline controller")
        return self


class AcceptanceThresholds(_FrozenModel):
    mappo_no_failure_mean_completion_min: float = Field(ge=0, le=1)
    ippo_no_failure_mean_completion_min: float = Field(ge=0, le=1)
    mappo_median_makespan_cp_sat_ratio_max: float = Field(gt=0)
    mappo_failure_mean_completion_min: float = Field(ge=0, le=1)


class BehaviorCloningThresholds(_FrozenModel):
    final_completion_gain_min: float = Field(ge=0, le=1)
    transitions_to_95_reduction_min: float = Field(ge=0, le=1)


class FailureCurriculumThresholds(_FrozenModel):
    failure_completion_gain_min: float = Field(ge=0, le=1)
    no_failure_completion_delta_min: float = Field(ge=-1, le=1)


class AblationThresholds(_FrozenModel):
    behavior_cloning: BehaviorCloningThresholds
    failure_curriculum: FailureCurriculumThresholds


class ExperimentProtocol(_FrozenModel):
    schema_version: Literal["construction-intelligence-experiment-v1"]
    experiment_id: Literal["construction_intelligence_v1"]
    protocol_version: Literal["1.0.0"]
    description: str = Field(min_length=1)
    training_seeds: list[int]
    checkpoint_fractions: list[float]
    profiles: dict[ExperimentProfile, ProfileSpec]
    variants: list[ExperimentVariant]
    scenario_splits: ScenarioSplitProtocol
    selection: SelectionProtocol
    evaluation: EvaluationProtocol
    acceptance: AcceptanceThresholds
    ablations: AblationThresholds

    @model_validator(mode="after")
    def validate_frozen_protocol(self) -> ExperimentProtocol:
        if self.training_seeds != [7, 8, 9, 10, 11]:
            raise ValueError("the frozen training seeds must be 7-11")
        if self.checkpoint_fractions != [0.1, 0.25, 0.5, 0.75, 1.0]:
            raise ValueError("the frozen checkpoint fractions must be 10%, 25%, 50%, 75%, 100%")
        if set(self.profiles) != {"unit", "smoke", "research"}:
            raise ValueError("unit, smoke, and research profiles are required")
        if self.profiles["research"].transitions != 1_500_000:
            raise ValueError("the research profile must use 1.5M transitions per run")
        _validate_frozen_variants(self.variants)
        if self.selection.scenario_seeds != self.scenario_splits.validation_seeds:
            raise ValueError("checkpoint selection must use exactly the registered validation seeds")
        if self.evaluation.scenario_seeds != self.scenario_splits.heldout_seeds:
            raise ValueError("final evaluation must use exactly the registered held-out seeds")
        return self


class ExperimentRunSpec(_FrozenModel):
    run_id: str
    experiment_id: str
    protocol_version: str
    protocol_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: ExperimentProfile
    algorithm: Algorithm
    role: Literal["primary", "ablation"]
    hypothesis: AblationHypothesis | None
    experiment_variant: str
    seed: int
    training_seed: int
    transitions: int = Field(gt=0)
    expert_episodes: int = Field(ge=0)
    behavior_clone_epochs: int = Field(ge=0)
    rollout_decisions: int = Field(gt=0)
    ppo_epochs: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    hidden_dim: int = Field(ge=32, le=512)
    include_training_failures: bool
    device: Literal["auto", "cpu", "cuda"]
    checkpoint_fractions: list[float]
    protocol_run_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    def training_config_payload(self) -> dict[str, object]:
        """Return only fields accepted by ``TrainingConfig``."""
        return {
            "algorithm": self.algorithm,
            "profile": self.profile,
            "seed": self.seed,
            "experiment_id": self.experiment_id,
            "experiment_variant": self.experiment_variant,
            "training_seed": self.training_seed,
            "protocol_run_digest": self.protocol_run_digest,
            "transitions": self.transitions,
            "expert_episodes": self.expert_episodes,
            "behavior_clone_epochs": self.behavior_clone_epochs,
            "rollout_decisions": self.rollout_decisions,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "hidden_dim": self.hidden_dim,
            "include_training_failures": self.include_training_failures,
            "device": self.device,
            "checkpoint_fractions": list(self.checkpoint_fractions),
        }


class CheckpointValidationResult(_FrozenModel):
    checkpoint_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    experiment_variant: str = Field(min_length=1)
    training_seed: int
    checkpoint_fraction: float = Field(gt=0, le=1)
    transition_count: int = Field(gt=0)
    split: Literal["validation"]
    scenario_seeds: list[int] = Field(min_length=1)
    mean_completion_rate: float = Field(ge=0, le=1)
    mean_makespan_s: float = Field(gt=0)
    checkpoint_path: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_lineage: list[str]
    configuration_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(min_length=1)
    resume_provenance: dict[str, object]

    @model_validator(mode="after")
    def validate_seed_provenance(self) -> CheckpointValidationResult:
        _require_unique_sorted(self.scenario_seeds, "checkpoint validation scenario_seeds")
        if any(not 800 <= seed <= 899 for seed in self.scenario_seeds):
            raise ValueError("checkpoint selection results may only contain validation seeds")
        return self


class SelectedCheckpoint(_FrozenModel):
    checkpoint_id: str
    experiment_id: str
    experiment_variant: str
    training_seed: int
    checkpoint_fraction: float
    transition_count: int
    split: Literal["validation"]
    scenario_seeds: list[int]
    mean_completion_rate: float
    mean_makespan_s: float
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_lineage: list[str]
    configuration_digest: str
    source_commit: str
    resume_provenance: dict[str, object]
    selection_rule: Literal[
        "completion_rate_desc,makespan_s_asc,transition_count_asc,checkpoint_id_asc"
    ] = "completion_rate_desc,makespan_s_asc,transition_count_asc,checkpoint_id_asc"
    required_checkpoint_fractions: list[float]
    candidate_ranking: list[str]

    @model_validator(mode="after")
    def validate_selection_evidence(self) -> SelectedCheckpoint:
        if self.required_checkpoint_fractions != sorted(
            set(self.required_checkpoint_fractions)
        ):
            raise ValueError(
                "required checkpoint fractions must be unique and sorted"
            )
        if len(self.candidate_ranking) != len(
            self.required_checkpoint_fractions
        ):
            raise ValueError(
                "candidate ranking must cover every required checkpoint fraction"
            )
        if len(self.candidate_ranking) != len(set(self.candidate_ranking)):
            raise ValueError("candidate ranking must contain unique checkpoint IDs")
        if self.checkpoint_id not in self.candidate_ranking:
            raise ValueError("selected checkpoint must appear in candidate ranking")
        return self


class AblationEvidence(_FrozenModel):
    hypothesis: AblationHypothesis
    full_final_completion: float | None = Field(default=None, ge=0, le=1)
    ablated_final_completion: float | None = Field(default=None, ge=0, le=1)
    full_transitions_to_95: int | None = Field(default=None, gt=0)
    ablated_transitions_to_95: int | None = Field(default=None, gt=0)
    full_failure_completion: float | None = Field(default=None, ge=0, le=1)
    ablated_failure_completion: float | None = Field(default=None, ge=0, le=1)
    full_no_failure_completion: float | None = Field(default=None, ge=0, le=1)
    ablated_no_failure_completion: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_hypothesis_evidence(self) -> AblationEvidence:
        if self.hypothesis == "behavior_cloning":
            if self.full_final_completion is None or self.ablated_final_completion is None:
                raise ValueError("behavior-cloning evidence requires both final completion rates")
        elif any(
            value is None
            for value in (
                self.full_failure_completion,
                self.ablated_failure_completion,
                self.full_no_failure_completion,
                self.ablated_no_failure_completion,
            )
        ):
            raise ValueError("failure-curriculum evidence requires failure and no-failure completion")
        return self


class AblationDecision(_FrozenModel):
    hypothesis: AblationHypothesis
    supported: bool
    completion_gain: float | None = None
    transitions_to_95_reduction: float | None = None
    failure_completion_gain: float | None = None
    no_failure_completion_delta: float | None = None
    completion_boundary_met: bool = False
    transition_boundary_met: bool = False
    failure_boundary_met: bool = False
    no_failure_safety_boundary_met: bool = False
    interpretation: str


class ReleaseEvidence(_FrozenModel):
    completed_training_run_ids: list[str]
    selected_policy_run_ids: list[str]
    heldout_evaluated_run_ids: list[str]
    ablation_decisions: list[AblationDecision]
    primary_acceptance_passed: bool
    report_generated: bool
    reproducibility_artifacts_complete: bool
    heldout_used_for_selection: bool = False


class ReleaseCompletenessDecision(_FrozenModel):
    complete: bool
    missing_training_run_ids: list[str]
    missing_selected_policy_run_ids: list[str]
    missing_heldout_evaluation_run_ids: list[str]
    missing_ablation_hypotheses: list[AblationHypothesis]
    unsupported_ablation_hypotheses: list[AblationHypothesis]
    blocking_reasons: list[str]


def load_experiment_protocol(
    path: Path | str = DEFAULT_EXPERIMENT_PROTOCOL_PATH,
) -> ExperimentProtocol:
    """Load and strictly validate the frozen Construction Intelligence v1 manifest."""
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment protocol must be a YAML mapping: {manifest_path}")
    return ExperimentProtocol.model_validate(payload)


def protocol_digest(protocol: ExperimentProtocol) -> str:
    """Return a stable SHA-256 over the fully validated protocol."""
    return _digest_json(protocol.model_dump(mode="json"))


def assert_split_isolation(protocol: ExperimentProtocol) -> None:
    """Fail closed if selection or evaluation crosses a frozen data boundary."""
    training = set(range(protocol.scenario_splits.training.start, protocol.scenario_splits.training.end + 1))
    validation = set(protocol.scenario_splits.validation_seeds)
    heldout = set(protocol.scenario_splits.heldout_seeds)
    if training & validation or training & heldout or validation & heldout:
        raise ValueError("scenario seed splits overlap")
    if protocol.selection.split != "validation":
        raise ValueError("checkpoint selection must be validation-only")
    if set(protocol.selection.scenario_seeds) != validation:
        raise ValueError("checkpoint selection seeds do not match the validation split")
    if protocol.evaluation.split != "test":
        raise ValueError("final evaluation must use the held-out test split")
    if set(protocol.evaluation.scenario_seeds) != heldout:
        raise ValueError("final evaluation seeds do not match the held-out split")


def expand_experiment_matrix(
    protocol: ExperimentProtocol,
    profile: ExperimentProfile,
) -> list[ExperimentRunSpec]:
    """Expand one frozen profile into exactly four variants by five seeds."""
    assert_split_isolation(protocol)
    profile_spec = protocol.profiles[profile]
    manifest_digest = protocol_digest(protocol)
    runs: list[ExperimentRunSpec] = []
    for variant in protocol.variants:
        for training_seed in protocol.training_seeds:
            expert_episodes = _override(
                profile_spec.expert_episodes,
                variant.overrides.expert_episodes,
            )
            behavior_clone_epochs = _override(
                profile_spec.behavior_clone_epochs,
                variant.overrides.behavior_clone_epochs,
            )
            include_training_failures = _override(
                True,
                variant.overrides.include_training_failures,
            )
            run_id = (
                f"{protocol.experiment_id}-{profile}-{variant.name}-seed-{training_seed}"
            )
            digest_payload = {
                "protocol_digest": manifest_digest,
                "run_id": run_id,
                "algorithm": variant.algorithm,
                "profile": profile,
                "seed": training_seed,
                "experiment_id": protocol.experiment_id,
                "experiment_variant": variant.name,
                "transitions": profile_spec.transitions,
                "expert_episodes": expert_episodes,
                "behavior_clone_epochs": behavior_clone_epochs,
                "rollout_decisions": profile_spec.rollout_decisions,
                "ppo_epochs": profile_spec.ppo_epochs,
                "minibatch_size": profile_spec.minibatch_size,
                "hidden_dim": profile_spec.hidden_dim,
                "include_training_failures": include_training_failures,
                "device": profile_spec.device,
                "checkpoint_fractions": protocol.checkpoint_fractions,
            }
            runs.append(
                ExperimentRunSpec(
                    run_id=run_id,
                    experiment_id=protocol.experiment_id,
                    protocol_version=protocol.protocol_version,
                    protocol_digest=manifest_digest,
                    profile=profile,
                    algorithm=variant.algorithm,
                    role=variant.role,
                    hypothesis=variant.hypothesis,
                    experiment_variant=variant.name,
                    seed=training_seed,
                    training_seed=training_seed,
                    transitions=profile_spec.transitions,
                    expert_episodes=expert_episodes,
                    behavior_clone_epochs=behavior_clone_epochs,
                    rollout_decisions=profile_spec.rollout_decisions,
                    ppo_epochs=profile_spec.ppo_epochs,
                    minibatch_size=profile_spec.minibatch_size,
                    hidden_dim=profile_spec.hidden_dim,
                    include_training_failures=include_training_failures,
                    device=profile_spec.device,
                    checkpoint_fractions=list(protocol.checkpoint_fractions),
                    protocol_run_digest=_digest_json(digest_payload),
                )
            )
    if len(runs) != 20 or len({run.run_id for run in runs}) != 20:
        raise ValueError("the frozen experiment matrix must expand to exactly 20 unique runs")
    return runs


def select_validation_checkpoint(
    candidates: Sequence[CheckpointValidationResult],
    *,
    validation_seeds: Sequence[int],
    required_fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> SelectedCheckpoint:
    """Select deterministically without accepting training or held-out evidence."""
    if not candidates:
        raise ValueError("at least one validation checkpoint result is required")
    expected_seeds = list(validation_seeds)
    _require_unique_sorted(expected_seeds, "expected validation_seeds")
    if any(not 800 <= seed <= 899 for seed in expected_seeds):
        raise ValueError("checkpoint selection can only use validation seeds 800-899")
    expected_fractions = list(required_fractions)
    if expected_fractions != sorted(set(expected_fractions)):
        raise ValueError("required checkpoint fractions must be unique and sorted")
    candidate_fractions = [candidate.checkpoint_fraction for candidate in candidates]
    if len(candidate_fractions) != len(set(candidate_fractions)):
        raise ValueError("checkpoint candidates contain a duplicate fraction cell")
    if candidate_fractions and sorted(candidate_fractions) != expected_fractions:
        raise ValueError(
            "checkpoint candidates must cover exactly the required checkpoint fractions"
        )
    checkpoint_ids = [candidate.checkpoint_id for candidate in candidates]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("checkpoint candidate IDs must be unique")
    identity = {
        (candidate.experiment_id, candidate.experiment_variant, candidate.training_seed)
        for candidate in candidates
    }
    if len(identity) != 1:
        raise ValueError("checkpoint candidates must belong to one experiment variant and seed")
    if len({candidate.configuration_digest for candidate in candidates}) != 1:
        raise ValueError("checkpoint candidates must share one configuration digest")
    if len({candidate.source_commit for candidate in candidates}) != 1:
        raise ValueError("checkpoint candidates must share one source commit")
    for candidate in candidates:
        if candidate.scenario_seeds != expected_seeds:
            raise ValueError(
                f"checkpoint {candidate.checkpoint_id} was not evaluated on exactly "
                "the registered validation seeds"
            )
    ranking = sorted(
        candidates,
        key=lambda item: (
            -item.mean_completion_rate,
            item.mean_makespan_s,
            item.transition_count,
            item.checkpoint_id,
        ),
    )
    selected = ranking[0]
    return SelectedCheckpoint(
        **selected.model_dump(),
        required_checkpoint_fractions=expected_fractions,
        candidate_ranking=[candidate.checkpoint_id for candidate in ranking],
    )


def decide_ablation_support(
    protocol: ExperimentProtocol,
    evidence: AblationEvidence,
) -> AblationDecision:
    """Apply the pre-registered inclusive ablation boundaries."""
    if evidence.hypothesis == "behavior_cloning":
        assert evidence.full_final_completion is not None
        assert evidence.ablated_final_completion is not None
        completion_gain = evidence.full_final_completion - evidence.ablated_final_completion
        completion_met = _at_least(
            completion_gain,
            protocol.ablations.behavior_cloning.final_completion_gain_min,
        )
        transition_reduction = _transition_reduction(evidence)
        transition_met = transition_reduction is not None and _at_least(
            transition_reduction,
            protocol.ablations.behavior_cloning.transitions_to_95_reduction_min,
        )
        supported = completion_met or transition_met
        return AblationDecision(
            hypothesis=evidence.hypothesis,
            supported=supported,
            completion_gain=completion_gain,
            transitions_to_95_reduction=transition_reduction,
            completion_boundary_met=completion_met,
            transition_boundary_met=transition_met,
            interpretation=(
                "Behavior cloning is supported by the pre-registered boundary."
                if supported
                else "Behavior cloning is not supported by the pre-registered boundary."
            ),
        )

    assert evidence.full_failure_completion is not None
    assert evidence.ablated_failure_completion is not None
    assert evidence.full_no_failure_completion is not None
    assert evidence.ablated_no_failure_completion is not None
    failure_gain = evidence.full_failure_completion - evidence.ablated_failure_completion
    no_failure_delta = (
        evidence.full_no_failure_completion - evidence.ablated_no_failure_completion
    )
    failure_met = _at_least(
        failure_gain,
        protocol.ablations.failure_curriculum.failure_completion_gain_min,
    )
    safety_met = _at_least(
        no_failure_delta,
        protocol.ablations.failure_curriculum.no_failure_completion_delta_min,
    )
    supported = failure_met and safety_met
    return AblationDecision(
        hypothesis=evidence.hypothesis,
        supported=supported,
        failure_completion_gain=failure_gain,
        no_failure_completion_delta=no_failure_delta,
        failure_boundary_met=failure_met,
        no_failure_safety_boundary_met=safety_met,
        interpretation=(
            "Failure curriculum is supported by the pre-registered boundaries."
            if supported
            else "Failure curriculum is not supported by the pre-registered boundaries."
        ),
    )


def decide_release_completeness(
    protocol: ExperimentProtocol,
    evidence: ReleaseEvidence,
) -> ReleaseCompletenessDecision:
    """Decide whether research evidence is complete; unsupported hypotheses do not block."""
    expected = {
        run.run_id
        for run in expand_experiment_matrix(protocol, "research")
    }
    completed = set(evidence.completed_training_run_ids)
    selected = set(evidence.selected_policy_run_ids)
    evaluated = set(evidence.heldout_evaluated_run_ids)
    missing_training = sorted(expected - completed)
    missing_selected = sorted(expected - selected)
    missing_evaluated = sorted(expected - evaluated)
    decisions = {decision.hypothesis: decision for decision in evidence.ablation_decisions}
    expected_hypotheses: set[AblationHypothesis] = {
        "behavior_cloning",
        "failure_curriculum",
    }
    missing_hypotheses = sorted(expected_hypotheses - decisions.keys())
    unsupported = sorted(
        hypothesis for hypothesis, decision in decisions.items() if not decision.supported
    )
    blockers: list[str] = []
    if missing_training:
        blockers.append("one or more research training runs are incomplete")
    if missing_selected:
        blockers.append("one or more research runs lack a validation-selected checkpoint")
    if missing_evaluated:
        blockers.append("one or more selected policies lack held-out evaluation")
    if missing_hypotheses:
        blockers.append("one or more ablation hypotheses lack a decision")
    if not evidence.primary_acceptance_passed:
        blockers.append("one or more primary acceptance thresholds failed")
    if not evidence.report_generated:
        blockers.append("the reproducible research report is missing")
    if not evidence.reproducibility_artifacts_complete:
        blockers.append("one or more reproducibility artifacts are missing")
    if evidence.heldout_used_for_selection:
        blockers.append("held-out results were used during checkpoint selection")
    return ReleaseCompletenessDecision(
        complete=not blockers,
        missing_training_run_ids=missing_training,
        missing_selected_policy_run_ids=missing_selected,
        missing_heldout_evaluation_run_ids=missing_evaluated,
        missing_ablation_hypotheses=missing_hypotheses,
        unsupported_ablation_hypotheses=unsupported,
        blocking_reasons=blockers,
    )


def _validate_frozen_variants(variants: Sequence[ExperimentVariant]) -> None:
    by_name = {variant.name: variant for variant in variants}
    expected = {
        "mappo_full",
        "ippo_full",
        "mappo_no_bc",
        "mappo_no_failure_curriculum",
    }
    if set(by_name) != expected or len(variants) != 4:
        raise ValueError(f"the frozen variants must be {sorted(expected)}")
    mappo = by_name["mappo_full"]
    ippo = by_name["ippo_full"]
    no_bc = by_name["mappo_no_bc"]
    no_failure = by_name["mappo_no_failure_curriculum"]
    if (mappo.algorithm, mappo.role, mappo.hypothesis) != ("mappo", "primary", None):
        raise ValueError("mappo_full must be the primary full MAPPO variant")
    if (ippo.algorithm, ippo.role, ippo.hypothesis) != ("ippo", "primary", None):
        raise ValueError("ippo_full must be the primary full IPPO variant")
    if (
        no_bc.algorithm,
        no_bc.role,
        no_bc.hypothesis,
        no_bc.overrides.expert_episodes,
        no_bc.overrides.behavior_clone_epochs,
    ) != ("mappo", "ablation", "behavior_cloning", 0, 0):
        raise ValueError("mappo_no_bc must disable expert collection and behavior cloning")
    if (
        no_failure.algorithm,
        no_failure.role,
        no_failure.hypothesis,
        no_failure.overrides.include_training_failures,
    ) != ("mappo", "ablation", "failure_curriculum", False):
        raise ValueError("mappo_no_failure_curriculum must disable training failures")


def _require_unique_sorted(values: Sequence[int], label: str) -> None:
    if list(values) != sorted(set(values)):
        raise ValueError(f"{label} must be unique and sorted")


def _override(
    default: _OverrideValue,
    override: _OverrideValue | None,
) -> _OverrideValue:
    return default if override is None else override


def _digest_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transition_reduction(evidence: AblationEvidence) -> float | None:
    full = evidence.full_transitions_to_95
    ablated = evidence.ablated_transitions_to_95
    if full is None:
        return None
    if ablated is None:
        return 1.0
    return (ablated - full) / ablated


def _at_least(value: float, boundary: float) -> bool:
    return value > boundary or abs(value - boundary) <= 1e-12
