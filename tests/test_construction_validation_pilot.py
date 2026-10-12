from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import run_construction_validation_pilot as pilot


SOURCE = {"commit": "a" * 40, "dirty": False, "tree_digest": "b" * 64}


class Model:
    def __init__(self, payload):
        self.payload = deepcopy(payload)

    def model_dump(self, **_kwargs):
        return deepcopy(self.payload)


def config_digest(config):
    return pilot._digest({
        key: value for key, value in config.model_dump().items()
        if key not in {
            "configuration_digest", "checkpoint_lineage", "output_root", "resume_checkpoint",
            "resume_provenance", "source_commit", "source_dirty", "source_tree_digest",
        }
    })


class Registry:
    def __init__(self):
        self.matrix = None
        self.evaluations = []
        self.foreign_matrices = []
        self.foreign_runs = []

    def get_experiment_matrix(self, matrix_id):
        if self.matrix and self.matrix["id"] == matrix_id:
            return deepcopy(self.matrix)
        return None

    def list_experiment_matrices(self):
        return ([deepcopy(self.matrix)] if self.matrix else []) + self.foreign_matrices

    def list_runs(self, **_kwargs):
        return (deepcopy(self.matrix["runs"]) if self.matrix else []) + self.foreign_runs

    def upsert_evaluation(self, evaluation_id, **kwargs):
        self.evaluations.append({"id": evaluation_id, **kwargs})


class Service:
    def __init__(self, registry):
        self.registry = registry
        self.launches = 0
        self.resumes = []
        self.shutdowns = 0

    def launch_training_matrix(self, design, **kwargs):
        self.launches += 1
        assert kwargs["execution_profile"] == "validation_pilot"
        assert len(kwargs["runs"]) == 4
        self.registry.matrix = {
            "id": kwargs["matrix_id"], "protocol_digest": kwargs["protocol_digest"],
            "protocol": deepcopy(kwargs["protocol"]), "execution_profile": "validation_pilot",
            "expected_run_count": 4,
            "runs": [{
                "id": f"{kwargs['matrix_id']}-{key}", "run_key": key,
                "config": config.model_dump(), "config_digest": config_digest(config),
                "source_commit": SOURCE["commit"], "input": {"design": design.model_dump()},
                "status": "completed", "attempt": 1, "progress": 1,
                "latest_checkpoint": None,
            } for key, config in kwargs["runs"]],
        }

    def resume(self, run_id):
        self.resumes.append(run_id)
        run = next(row for row in self.registry.matrix["runs"] if row["id"] == run_id)
        run.update(status="completed", attempt=run["attempt"] + 1, progress=1)
        return True

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    registry = Registry()
    service = Service(registry)
    calls = {"baseline": [], "candidates": [], "service": 0}
    parent = SimpleNamespace(
        protocol_version="1.1.0", checkpoint_fractions=list(pilot.FRACTIONS),
        selection=Model({
            "split": "validation", "scenario_seeds": list(pilot.VALIDATION_SEEDS),
            "failure_modes": [False, True], "deterministic": True,
        }),
    )

    def episode(_design, **kwargs):
        calls["baseline"].append(kwargs)
        assert kwargs["controller"] == "cp_sat"
        assert kwargs["expected_split"] == "validation"
        assert kwargs["seed"] in range(800, 805)
        return Model({
            "seed": kwargs["seed"], "split": "validation", "controller": "cp_sat",
            "failure_enabled": kwargs["failure_enabled"], "structure_completion_rate": 1,
            "makespan_s": 100,
        })

    def candidates(_design, run, _parent, **kwargs):
        calls["candidates"].append(kwargs)
        config = run["config"]
        candidates = []
        for index, fraction in enumerate(pilot.FRACTIONS):
            result = {
                "checkpoint_id": f"{run['run_key']}-{fraction}",
                "checkpoint_fraction": fraction, "transition_count": int(150_000 * fraction),
                "scenario_seeds": list(pilot.VALIDATION_SEEDS), "split": "validation",
                "experiment_id": pilot.EXPERIMENT_ID,
                "experiment_variant": config["experiment_variant"], "training_seed": 7,
                "configuration_digest": run["config_digest"], "source_commit": SOURCE["commit"],
                "checkpoint_sha256": str(index) * 64,
            }
            rows = []
            for failure in (False, True):
                for seed in pilot.VALIDATION_SEEDS:
                    rows.append({
                        **{field: result[field] for field in (
                            "experiment_id", "experiment_variant", "training_seed",
                            "configuration_digest", "source_commit", "checkpoint_sha256",
                        )},
                        "seed": seed, "split": "validation", "controller": config["algorithm"],
                        "failure_enabled": failure,
                        "structure_completion_rate": 1.0 if index == 0 else 0.96,
                        "makespan_s": (
                            110 if config["reward_time_coefficient"] == 0.15 else 105
                        ) if "reward_time_coefficient" in config else (
                            110 if config["ppo_epochs"] == 6 else 105
                        ),
                    })
            candidates.append({"result": result, "episodes": rows})
        return Model({
            "matrix_id": kwargs["matrix_id"], "run_key": run["run_key"],
            "candidates": candidates, "selected": candidates[0]["result"],
        })

    def service_factory(_registry):
        calls["service"] += 1
        return service

    runtime = pilot.PilotRuntime(
        source=lambda: deepcopy(SOURCE), environment=lambda: {"packages": {"fake": "1"}},
        load_design=lambda _path: Model({"design": "fixture"}),
        design_digest=lambda _design: "c" * 64,
        load_protocol=lambda: parent, protocol_digest=lambda _parent: "d" * 64,
        config=lambda payload: Model(payload), config_digest=config_digest,
        registry=lambda _path: registry, service=service_factory,
        candidates=candidates, episode=episode,
    )
    args = pilot.build_parser().parse_args([
        "execute", "--registry", str(tmp_path / "pilot.sqlite"),
        "--evidence-root", str(tmp_path / "evidence"), "--device", "cpu",
        "--poll-seconds", "0.01",
    ])
    monkeypatch.setattr(pilot.time, "sleep", lambda _seconds: None)
    return args, runtime, registry, service, calls, parent


def test_pilot_protocol_has_four_distinct_one_factor_runs_and_research_prefix():
    matrix_id, digest, protocol = pilot.build_pilot_protocol(
        SOURCE, design_digest="c" * 64, parent_protocol_digest="d" * 64, device="cuda",
    )
    assert "validation-pilot" in matrix_id
    assert len(digest) == 64
    assert protocol["schema_version"] != "construction-intelligence-experiment-v1"
    assert protocol["release_eligible"] is False
    assert protocol["total_training_transitions"] == 600_000
    assert len({row["run_key"] for row in protocol["run_specs"]}) == 4
    assert len({row["config"]["protocol_run_digest"] for row in protocol["run_specs"]}) == 4
    for algorithm in ("mappo", "ippo"):
        configs = [row["config"] for row in protocol["run_specs"]
                   if row["config"]["algorithm"] == algorithm]
        assert {config["ppo_epochs"] for config in configs} == {2, 6}
        assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {
            "ppo_epochs", "experiment_variant", "protocol_run_digest",
        }
        for config in configs:
            assert config["seed"] == config["training_seed"] == 7
            assert config["transitions"] == 150_000
            assert config["expert_episodes"] == 128
            assert config["behavior_clone_epochs"] == 40
            assert config["failure_curriculum_mode"] == "linear"
            assert config["failure_curriculum_reference_transitions"] == 1_500_000
            assert config["checkpoint_fractions"] == pilot.FRACTIONS


def test_identity_is_stable_and_bound_to_source_design_and_device():
    def identity(**updates):
        fields = {"source": SOURCE, "design_digest": "c" * 64,
                  "parent_protocol_digest": "d" * 64, "device": "cuda", **updates}
        return pilot.build_pilot_protocol(**fields)[:2]

    assert identity() == identity()
    assert identity() != identity(source={**SOURCE, "commit": "e" * 40})
    assert identity() != identity(design_digest="f" * 64)
    assert identity() != identity(device="cpu")


def test_default_ppo_comparison_preserves_pre_generalization_identity():
    fields = dict(source=SOURCE, design_digest="c" * 64, parent_protocol_digest="d" * 64, device="cuda")
    omitted = pilot.build_pilot_protocol(**fields)
    explicit = pilot.build_pilot_protocol(**fields, comparison="ppo-epochs")
    assert omitted == explicit
    # Captured from the original committed pilot builder before adding reward mode.
    assert omitted[:2] == (
        "validation-pilot-aaaaaaaaaaaa-9785be8368bc",
        "9785be8368bcf3c0ac89ba8ef6d574733624db89bbffc1b297d07c7b94a78742",
    )
    assert "comparison_mode" not in omitted[2]
    assert pilot.build_parser().parse_args([]).comparison == "ppo-epochs"


def test_reward_time_protocol_is_exact_one_factor_with_research_prefix():
    fields = dict(source=SOURCE, design_digest="c" * 64, parent_protocol_digest="d" * 64, device="cuda")
    matrix_id, digest, protocol = pilot.build_pilot_protocol(**fields, comparison="reward-time")
    assert (matrix_id, digest) != pilot.build_pilot_protocol(**fields)[:2]
    assert protocol["comparison_parameter"] == "reward_time_coefficient"
    assert protocol["comparison_protocol_version"] == "reward-time-v1"
    assert protocol["winner_rule"]["lexicographic_order"][-1] == "reward_time_coefficient_asc"
    assert protocol["release_eligible"] is protocol["research_launch_authorized"] is False
    assert protocol["total_training_transitions"] == 600_000
    assert {row["run_key"] for row in protocol["run_specs"]} == {
        "mappo-time0p15-ppo6-seed7", "mappo-time1p5-ppo6-seed7",
        "ippo-time0p15-ppo6-seed7", "ippo-time1p5-ppo6-seed7",
    }
    assert len({row["config"]["protocol_run_digest"] for row in protocol["run_specs"]}) == 4
    for algorithm in ("mappo", "ippo"):
        configs = [row["config"] for row in protocol["run_specs"] if row["config"]["algorithm"] == algorithm]
        assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {
            "reward_time_coefficient", "experiment_variant", "protocol_run_digest",
        }
        assert {config["reward_time_coefficient"] for config in configs} == {0.15, 1.5}
        for config in configs:
            assert config["ppo_epochs"] == 6
            assert config["seed"] == config["training_seed"] == 7
            assert config["transitions"] == 150_000
            assert config["expert_episodes"] == 128
            assert config["behavior_clone_epochs"] == 40
            assert config["failure_curriculum_reference_transitions"] == 1_500_000
            assert config["failure_curriculum_mode"] == "linear"
            assert config["checkpoint_fractions"] == pilot.FRACTIONS


def test_unknown_comparison_fails_without_building_runs():
    with pytest.raises(ValueError, match="comparison"):
        pilot.build_pilot_protocol(
            SOURCE, design_digest="c" * 64, parent_protocol_digest="d" * 64,
            device="cuda", comparison="unregistered",
        )


def test_fresh_workflow_launches_once_and_persists_only_validation_evidence(workflow):
    args, runtime, registry, service, calls, _parent = workflow
    assert pilot.execute(args, runtime) == 0
    assert service.launches == 1
    assert service.shutdowns == 1
    assert len(calls["baseline"]) == 10
    assert len(calls["candidates"]) == 4
    root = args.evidence_root / registry.matrix["id"]
    import json

    evidence = json.loads((root / "pilot_evidence.json").read_text())
    summary = json.loads((root / "validation_pilot_summary.json").read_text())
    assert evidence["release_eligible"] is False
    assert len(evidence["selections"]) == 4
    assert sum(len(candidate["episodes"]) for selection in evidence["selections"]
               for candidate in selection["candidates"]) == 200
    assert summary["release_eligible"] is False
    assert summary["research_launch_authorized"] is False
    assert summary["final_checkpoint_decision"]["winner_ppo_epochs"] == 2
    assert all(row["selected"]["checkpoint_fraction"] == 0.1 for row in summary["runs"])
    assert all(row["final"]["checkpoint_fraction"] == 1 for row in summary["runs"])
    assert len(registry.evaluations) == 1
    assert registry.evaluations[0]["split"] == "validation"


def test_completed_reattachment_never_enqueues_duplicate_training(workflow):
    args, runtime, _registry, service, _calls, _parent = workflow
    assert pilot.execute(args, runtime) == 0
    assert pilot.execute(args, runtime) == 0
    assert service.launches == 1


def test_interrupted_compatible_checkpoint_is_resumed(workflow, tmp_path):
    args, runtime, registry, service, _calls, _parent = workflow
    assert pilot.execute(args, runtime) == 0
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    run = registry.matrix["runs"][0]
    run.update(status="interrupted", progress=0.5, latest_checkpoint=str(checkpoint))
    assert pilot.execute(args, runtime) == 0
    assert service.resumes == [run["id"]]
    assert service.launches == 1


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_zero_progress_without_checkpoint_restarts_through_resume(workflow, status):
    args, runtime, registry, service, _calls, _parent = workflow
    pilot.execute(args, runtime)
    run = registry.matrix["runs"][0]
    run.update(status=status, progress=0, latest_checkpoint=None)
    assert pilot.execute(args, runtime) == 0
    assert service.resumes == [run["id"]]
    assert service.launches == 1


def test_keyboard_interrupt_detaches_without_cancelling_running_job(workflow, monkeypatch):
    args, runtime, registry, service, calls, _parent = workflow
    pilot.execute(args, runtime)
    run = registry.matrix["runs"][0]
    run.update(status="running", progress=0.5, latest_checkpoint="untouched.pt")
    before = deepcopy(calls)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot.time, "sleep", interrupt)
    assert pilot.execute(args, runtime) == 130
    assert run["status"] == "running"
    assert run["latest_checkpoint"] == "untouched.pt"
    assert service.resumes == []
    assert service.shutdowns == 2
    assert calls["baseline"] == before["baseline"]
    assert calls["candidates"] == before["candidates"]


@pytest.mark.parametrize("comparison", ["ppo-epochs", "reward-time"])
def test_source_change_before_dispatch_is_rejected(workflow, comparison):
    args, runtime, _registry, _service, calls, _parent = workflow
    args.comparison = comparison
    fingerprints = iter([SOURCE, {**SOURCE, "dirty": True}])
    with pytest.raises(ValueError, match="before dispatch"):
        pilot.execute(args, replace(runtime, source=lambda: next(fingerprints)))
    assert calls["service"] == 0


@pytest.mark.parametrize("comparison", ["ppo-epochs", "reward-time"])
def test_source_change_while_polling_detaches_before_evaluation(workflow, comparison):
    args, runtime, _registry, service, calls, _parent = workflow
    args.comparison = comparison
    fingerprints = iter([SOURCE, SOURCE, {**SOURCE, "dirty": True}])
    with pytest.raises(ValueError, match="during execution"):
        pilot.execute(args, replace(runtime, source=lambda: next(fingerprints)))
    assert service.shutdowns == 1
    assert calls["baseline"] == calls["candidates"] == []


@pytest.mark.parametrize("comparison", ["ppo-epochs", "reward-time"])
@pytest.mark.parametrize("mutation", ["epochs", "reward", "digest", "source", "design", "grid"])
def test_reattachment_refuses_changed_records_before_service_start(workflow, mutation, comparison):
    args, runtime, registry, _service, calls, _parent = workflow
    args.comparison = comparison
    pilot.execute(args, runtime)
    run = registry.matrix["runs"][0]
    if mutation == "epochs":
        run["config"]["ppo_epochs"] = 8
    elif mutation == "reward":
        run["config"]["reward_time_coefficient"] = 2.0
    elif mutation == "digest":
        run["config_digest"] = "wrong"
    elif mutation == "source":
        run["source_commit"] = "e" * 40
    elif mutation == "design":
        run["input"]["design"] = {"different": "design"}
    else:
        registry.matrix["runs"].pop()
    before = calls["service"]
    with pytest.raises(ValueError, match="pilot|fingerprint"):
        pilot.execute(args, runtime)
    assert calls["service"] == before


@pytest.mark.parametrize("status", ["cancelled", "cancel_requested"])
def test_cancelled_pilot_does_not_resume(workflow, status):
    args, runtime, registry, service, _calls, _parent = workflow
    pilot.execute(args, runtime)
    registry.matrix["runs"][0]["status"] = status
    with pytest.raises(RuntimeError, match="cancellation"):
        pilot.execute(args, runtime)
    assert service.resumes == []


@pytest.mark.parametrize("reason", ["missing", "limit", "lost-progress"])
def test_unsafe_resume_fails_closed(workflow, reason, tmp_path):
    args, runtime, registry, service, _calls, _parent = workflow
    pilot.execute(args, runtime)
    run = registry.matrix["runs"][0]
    run["status"] = "interrupted"
    if reason == "missing":
        run["latest_checkpoint"] = str(tmp_path / "absent.pt")
    elif reason == "limit":
        run["attempt"] = args.max_resume_attempts
    else:
        run.update(latest_checkpoint=None, progress=0.5)
    with pytest.raises(RuntimeError, match="safely resume"):
        pilot.execute(args, runtime)
    assert service.resumes == []


def test_foreign_research_registry_is_rejected_without_dispatch(workflow):
    args, runtime, registry, _service, calls, _parent = workflow
    registry.foreign_matrices = [{"protocol": {"schema_version": "research"}}]
    with pytest.raises(ValueError, match="dedicated registry"):
        pilot.execute(args, runtime)
    assert calls["service"] == 0


def test_foreign_pending_job_is_rejected_without_dispatch(workflow):
    args, runtime, registry, _service, calls, _parent = workflow
    registry.foreign_runs = [{"id": "other", "status": "queued"}]
    with pytest.raises(ValueError, match="another matrix"):
        pilot.execute(args, runtime)
    assert calls["service"] == 0


def test_dirty_source_and_heldout_selection_are_rejected(workflow):
    args, runtime, _registry, _service, calls, parent = workflow
    with pytest.raises(ValueError, match="clean committed"):
        pilot.execute(args, replace(runtime, source=lambda: {**SOURCE, "dirty": True}))
    parent.selection.payload["scenario_seeds"] = [900, 901, 902, 903, 904]
    with pytest.raises(ValueError, match="validation-only"):
        pilot.execute(args, runtime)
    assert calls["service"] == 0


def test_evidence_split_guard_rejects_tampered_candidate(workflow):
    args, runtime, _registry, _service, _calls, _parent = workflow
    real = runtime.candidates

    def tampered(*args, **kwargs):
        result = real(*args, **kwargs)
        result.payload["candidates"][0]["episodes"][0]["seed"] = 900
        return result

    with pytest.raises(ValueError, match="validation seeds"):
        pilot.execute(args, replace(runtime, candidates=tampered))


@pytest.mark.parametrize("comparison", ["ppo-epochs", "reward-time"])
def test_source_change_during_last_candidate_refuses_summary_and_registration(workflow, comparison):
    args, runtime, registry, _service, _calls, _parent = workflow
    args.comparison = comparison
    real = runtime.candidates
    state = {"calls": 0, "changed": False}

    def candidates(*args, **kwargs):
        result = real(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == 4:
            state["changed"] = True
        return result

    guarded = replace(
        runtime, candidates=candidates,
        source=lambda: {**SOURCE, "dirty": bool(state["changed"])},
    )
    with pytest.raises(ValueError, match="before validation evidence persistence"):
        pilot.execute(args, guarded)
    root = args.evidence_root / registry.matrix["id"]
    assert not (root / "pilot_evidence.json").exists()
    assert not (root / "validation_pilot_summary.json").exists()
    assert registry.evaluations == []


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("missing", [False, True])
def test_every_checkpoint_must_have_exact_pilot_transition_budget(workflow, index, missing):
    args, runtime, _registry, _service, _calls, _parent = workflow
    real = runtime.candidates

    def tampered(*args, **kwargs):
        result = real(*args, **kwargs)
        candidate = result.payload["candidates"][index]["result"]
        if missing:
            candidate.pop("transition_count")
        else:
            candidate["transition_count"] -= 1
        return result

    with pytest.raises(ValueError, match="provenance"):
        pilot.execute(args, replace(runtime, candidates=tampered))


def test_kernel_executor_lock_prevents_duplicate_owner_and_releases(tmp_path):
    path = tmp_path / "owner.lock"
    with pilot.executor_lock(path):
        with pytest.raises(RuntimeError, match="already owns"):
            with pilot.executor_lock(path):
                raise AssertionError("a second executor acquired the lock")
    with pilot.executor_lock(path):
        pass


def _summary(algorithm, epochs, nominal=0.96, failure=0.9, ratio=1.1, makespan=110):
    return {
        "algorithm": algorithm, "ppo_epochs": epochs,
        "selected": {"nominal": {"mean_completion": 1}},
        "final": {
            "nominal": {"mean_completion": nominal,
                        "median_makespan_cp_sat_ratio": ratio, "mean_makespan_s": makespan},
            "failure": {"mean_completion": failure},
        },
    }


def test_common_winner_uses_final_not_optimistic_selected_checkpoint():
    rows = [_summary(algorithm, epochs, nominal=0.8 if epochs == 6 else 0.95)
            for algorithm in ("mappo", "ippo") for epochs in (6, 2)]
    decision = pilot.choose_common_epochs(rows)
    assert decision["winner_ppo_epochs"] == 2
    assert decision["research_launch_authorized"] is False
    assert decision["release_eligible"] is False


def test_no_eligible_common_winner_is_null_despite_per_algorithm_preferences():
    rows = [_summary(algorithm, epochs,
                     nominal=0.8 if (algorithm, epochs) in {("mappo", 6), ("ippo", 2)} else 0.96)
            for algorithm in ("mappo", "ippo") for epochs in (6, 2)]
    decision = pilot.choose_common_epochs(rows)
    assert decision["winner_ppo_epochs"] is None
    assert decision["per_algorithm_preferences"] == {"mappo": 2, "ippo": 6}


@pytest.mark.parametrize("nominal,failure,ratio,eligible", [
    (0.95, 0.85, 1.15, True), (0.949, 0.85, 1.15, False),
    (0.95, 0.849, 1.15, False), (0.95, 0.85, 1.151, False),
])
def test_pilot_eligibility_boundaries_are_inclusive(nominal, failure, ratio, eligible):
    rows = [_summary(algorithm, epochs, nominal, failure, ratio)
            for algorithm in ("mappo", "ippo") for epochs in (6, 2)]
    decision = pilot.choose_common_epochs(rows)
    assert all(option["validation_pilot_eligible"] is eligible
               for option in decision["epoch_options"])
    assert decision["winner_ppo_epochs"] == (2 if eligible else None)


@pytest.mark.parametrize("field", ["minimum", "mean", "failure", "ratio", "makespan", "tie"])
def test_preregistered_lexicographic_tiebreaks(field):
    rows = [_summary(algorithm, epochs) for algorithm in ("mappo", "ippo") for epochs in (6, 2)]
    control = [row for row in rows if row["ppo_epochs"] == 6]
    if field == "minimum":
        for row in control:
            row["final"]["nominal"]["mean_completion"] = 0.97
    elif field == "mean":
        control[0]["final"]["nominal"]["mean_completion"] = 0.97
    elif field == "failure":
        control[0]["final"]["failure"]["mean_completion"] = 0.91
    elif field == "ratio":
        control[0]["final"]["nominal"]["median_makespan_cp_sat_ratio"] = 1.09
    elif field == "makespan":
        for row in control:
            row["final"]["nominal"]["mean_makespan_s"] = 109
    assert pilot.choose_common_epochs(rows)["winner_ppo_epochs"] == (2 if field == "tie" else 6)


def test_default_command_is_nonlaunching_plan(workflow):
    args, runtime, _registry, _service, calls, _parent = workflow
    args.command = "plan"
    assert pilot.execute(args, runtime) == 0
    assert calls["service"] == 0
    assert calls["baseline"] == calls["candidates"] == []


def _reward_summary(algorithm, coefficient, **kwargs):
    return {**_summary(algorithm, 6, **kwargs), "reward_time_coefficient": coefficient}


def test_reward_time_winner_uses_final_checkpoint_not_best_selected():
    rows = [_reward_summary(algorithm, coefficient, nominal=0.8 if coefficient == 1.5 else 0.95)
            for algorithm in ("mappo", "ippo") for coefficient in (0.15, 1.5)]
    decision = pilot.choose_common_reward_time_coefficient(rows)
    assert decision["winner_reward_time_coefficient"] == 0.15
    assert "winner_ppo_epochs" not in decision
    assert "epoch_options" not in decision
    assert decision["release_eligible"] is decision["research_launch_authorized"] is False


def test_no_shared_eligible_reward_winner_is_null():
    rows = [_reward_summary(algorithm, coefficient,
                           nominal=0.8 if (algorithm, coefficient) in {("mappo", 0.15), ("ippo", 1.5)} else 0.96)
            for algorithm in ("mappo", "ippo") for coefficient in (0.15, 1.5)]
    decision = pilot.choose_common_reward_time_coefficient(rows)
    assert decision["winner_reward_time_coefficient"] is None
    assert decision["per_algorithm_preferences"] == {"mappo": 1.5, "ippo": 0.15}


@pytest.mark.parametrize("nominal,failure,ratio,eligible", [
    (0.95, 0.85, 1.15, True), (0.949, 0.85, 1.15, False),
    (0.95, 0.849, 1.15, False), (0.95, 0.85, 1.151, False),
])
def test_reward_eligibility_is_unchanged_and_ties_prefer_control(nominal, failure, ratio, eligible):
    rows = [_reward_summary(algorithm, coefficient, nominal=nominal, failure=failure, ratio=ratio)
            for algorithm in ("mappo", "ippo") for coefficient in (0.15, 1.5)]
    decision = pilot.choose_common_reward_time_coefficient(rows)
    assert all(row["validation_pilot_eligible"] is eligible for row in decision["coefficient_options"])
    assert decision["winner_reward_time_coefficient"] == (0.15 if eligible else None)


@pytest.mark.parametrize("mutation", ["epochs", "missing", "duplicate"])
def test_reward_winner_requires_fixed_epochs_and_complete_distinct_grid(mutation):
    rows = [_reward_summary(algorithm, coefficient)
            for algorithm in ("mappo", "ippo") for coefficient in (0.15, 1.5)]
    if mutation == "epochs":
        rows[0]["ppo_epochs"] = 2
    elif mutation == "missing":
        rows.pop()
    else:
        rows[-1]["reward_time_coefficient"] = 0.15
    with pytest.raises(ValueError, match="PPO epochs 6|both time coefficients"):
        pilot.choose_common_reward_time_coefficient(rows)


def test_reward_workflow_launches_four_and_persists_validation_only_decision(workflow):
    import json

    args, runtime, registry, service, calls, _parent = workflow
    args.comparison = "reward-time"
    assert pilot.execute(args, runtime) == 0
    assert service.launches == 1
    assert len(calls["baseline"]) == 10
    assert len(calls["candidates"]) == 4
    root = args.evidence_root / registry.matrix["id"]
    summary = json.loads((root / "validation_pilot_summary.json").read_text())
    evidence = json.loads((root / "pilot_evidence.json").read_text())
    assert summary["comparison_parameter"] == "reward_time_coefficient"
    assert summary["final_checkpoint_decision"]["winner_reward_time_coefficient"] == 1.5
    assert "winner_ppo_epochs" not in summary["final_checkpoint_decision"]
    assert summary["release_eligible"] is summary["research_launch_authorized"] is False
    assert sum(len(candidate["episodes"]) for selection in evidence["selections"]
               for candidate in selection["candidates"]) == 200
    assert all(row["selected"]["checkpoint_fraction"] == 0.1 for row in summary["runs"])
    assert all(row["final"]["transition_count"] == 150_000 and row["ppo_epochs"] == 6 for row in summary["runs"])
    assert {row["reward_time_coefficient"] for row in summary["runs"]} == {0.15, 1.5}
    assert registry.evaluations[0]["split"] == "validation"


def test_reward_completed_reattachment_and_interrupted_resume_do_not_duplicate_queue(workflow, tmp_path):
    args, runtime, registry, service, _calls, _parent = workflow
    args.comparison = "reward-time"
    assert pilot.execute(args, runtime) == 0
    assert pilot.execute(args, runtime) == 0
    checkpoint = tmp_path / "reward-latest.pt"
    checkpoint.write_bytes(b"synthetic-workflow-checkpoint")
    run = registry.matrix["runs"][0]
    run.update(status="interrupted", progress=0.5, latest_checkpoint=str(checkpoint))
    assert pilot.execute(args, runtime) == 0
    assert service.launches == 1
    assert service.resumes == [run["id"]]


@pytest.mark.parametrize("mutation", ["heldout_seed", "test_split", "config_digest"])
def test_reward_validation_evidence_rejects_wrong_split_or_reward_config_provenance(workflow, mutation):
    args, runtime, registry, _service, _calls, _parent = workflow
    args.comparison = "reward-time"
    original = runtime.candidates

    def tampered(*args, **kwargs):
        result = original(*args, **kwargs)
        first = result.payload["candidates"][0]
        if mutation == "heldout_seed":
            first["episodes"][0]["seed"] = 900
        elif mutation == "test_split":
            first["episodes"][0]["split"] = "test"
        else:
            first["result"]["configuration_digest"] = "0" * 64
        return result

    with pytest.raises(ValueError, match="validation seeds|non-validation|provenance"):
        pilot.execute(args, replace(runtime, candidates=tampered))
    assert registry.evaluations == []


def test_reward_runtime_must_not_silently_drop_reward_configuration(workflow):
    args, runtime, _registry, _service, calls, _parent = workflow
    args.comparison = "reward-time"

    def ignoring_config(payload):
        return Model({key: value for key, value in payload.items() if key != "reward_time_coefficient"})

    with pytest.raises(ValueError, match="preserve the reward-time comparison"):
        pilot.execute(args, replace(runtime, config=ignoring_config))
    assert calls["service"] == 0


def test_reward_plan_does_not_start_services_or_evaluate(workflow):
    args, runtime, _registry, _service, calls, _parent = workflow
    args.command = "plan"
    args.comparison = "reward-time"
    assert pilot.execute(args, runtime) == 0
    assert calls["service"] == 0
    assert calls["baseline"] == calls["candidates"] == []
