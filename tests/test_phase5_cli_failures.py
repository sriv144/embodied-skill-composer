from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import embodied_skill_composer.construction.coppelia_phase5 as phase5
from scripts import run_construction_phase5_coppelia as cli


class _CountOnlyStream:
    def __len__(self):
        return 123_456

    def __iter__(self):
        raise AssertionError("failure persistence must not materialize evidence streams")


def _result(*, passed: bool):
    return SimpleNamespace(
        status="completed" if passed else "failed",
        live_gate_passed=passed,
        error_type=None if passed else "RoutingError",
        error=None if passed else "dispatch return is unreachable after foundation_0_0",
        acceptance={"every_module_installed": passed, "zero_post_start_pose_writes": True},
        installed_module_ids=["foundation_0_0"],
        expected_module_count=26,
        replay=[1],
        trace=[
            {
                "timestamp_s": 12.5,
                "event": "simulator_stop_failed",
                "error_type": "DynamicCoppeliaError",
                "error": "shutdown deadline",
            },
            {
                "timestamp_s": 12.5,
                "event": "live_runtime_attestation_failed",
                "error_type": "ValueError",
                "error": "stopped state was not observed",
            },
        ],
    )


def _invoke(tmp_path, monkeypatch, *, result, writer=None):
    monkeypatch.setattr(
        cli,
        "Phase5FullCottageRunner",
        lambda *_args, **_kwargs: SimpleNamespace(run=lambda _: result),
    )
    monkeypatch.setattr(cli, "_simulator_version", lambda _sim: "4.10.0")
    if writer is not None:
        monkeypatch.setattr(cli, "write_phase5_artifact_bundle", writer)
    executor = SimpleNamespace(
        commands=_CountOnlyStream(), telemetry=_CountOnlyStream(), sim=object()
    )
    return cli._run_phase5_evidence(
        executor,
        SimpleNamespace(seed=900),
        object(),
        "nominal",
        run_dir=tmp_path / "run",
        run_id="run",
        source_commit="a" * 40,
        source_tree_digest="b" * 64,
    )


def test_attested_runner_failure_retains_native_artifacts_but_never_passes_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def retain_native(run_dir, *, result, **_kwargs):
        assert result.status == "failed"
        assert result.live_gate_passed is False
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("measured failed attempt", encoding="utf-8")
        (run_dir / "manifest.json").write_text("already verified native inventory", encoding="utf-8")
        return SimpleNamespace(live_evidence=True)

    code = _invoke(
        tmp_path,
        monkeypatch,
        result=_result(passed=False),
        writer=retain_native,
    )
    path = tmp_path / "run-diagnostics" / "gate_failure.json"
    payload = json.loads(path.read_bytes())
    assert code == 1
    assert payload["error_type"] == "RoutingError"
    assert "dispatch return" in payload["error"]
    assert payload["failed_gates"] == ["every_module_installed"]
    assert payload["acceptance"]["zero_post_start_pose_writes"] is True
    assert payload["command_count"] == payload["telemetry_count"] == 123_456
    assert payload["source_commit"] == "a" * 40
    assert payload["source_tree_digest"] == "b" * 64
    assert payload["source_dirty"] is False
    assert payload["runner_error_type"] == "RoutingError"
    assert [event["event"] for event in payload["trace_errors"]] == [
        "simulator_stop_failed",
        "live_runtime_attestation_failed",
    ]
    assert payload["trace_error_count"] == 2
    assert payload["trace_errors_truncated"] is False
    assert (
        payload["live_gate_passed"]
        is payload["live_evidence"]
        is payload["release_eligible"]
        is False
    )
    assert {path.name for path in (tmp_path / "run").iterdir()} == {
        "manifest.json",
        "metrics.json",
    }
    assert (tmp_path / "run" / "manifest.json").read_text() == "already verified native inventory"
    assert (tmp_path / "run" / "metrics.json").read_text() == "measured failed attempt"
    assert b"\r\n" not in path.read_bytes()


def test_attestation_writer_failure_is_persisted_even_after_runner_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*_args, **_kwargs):
        raise ValueError("live Coppelia evidence lacks a finalized runtime-origin attestation")

    assert _invoke(tmp_path, monkeypatch, result=_result(passed=True), writer=reject) == 1
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["runner_status"] == "completed"
    assert payload["phase"] == "artifact_write_or_verification"
    assert "finalized runtime-origin" in payload["error"]
    assert payload["runner_error"] is None
    assert payload["live_gate_passed"] is False


def test_failed_runner_and_failed_attestation_keep_distinct_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*_args, **_kwargs):
        raise ValueError("live Coppelia evidence lacks a finalized runtime-origin attestation")

    assert _invoke(tmp_path, monkeypatch, result=_result(passed=False), writer=reject) == 1
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["runner_error_type"] == "RoutingError"
    assert "dispatch return" in payload["runner_error"]
    assert payload["error_type"] == "ValueError"
    assert "finalized runtime-origin" in payload["error"]
    assert payload["failed_gates"] == ["every_module_installed"]
    assert payload["live_evidence"] is payload["live_gate_passed"] is False


def test_partial_writer_failure_keeps_nonrelease_diagnostic_and_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(run_dir, **_kwargs):
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("partial", encoding="utf-8")
        raise OSError("scene save failed")

    assert _invoke(tmp_path, monkeypatch, result=_result(passed=True), writer=reject) == 1
    assert (tmp_path / "run" / "metrics.json").read_text(encoding="utf-8") == "partial"
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["error_type"] == "OSError"
    assert payload["release_eligible"] is False


@pytest.mark.parametrize("phase", ["runner_initialization", "runner_execution"])
def test_uncaught_runner_errors_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    def reject(*_args, **_kwargs):
        raise RuntimeError("runtime-origin challenge failed")

    monkeypatch.setattr(
        cli,
        "Phase5FullCottageRunner",
        reject
        if phase == "runner_initialization"
        else lambda *_a, **_kw: SimpleNamespace(run=reject),
    )
    executor = SimpleNamespace(commands=[], telemetry=[])
    assert (
        cli._run_phase5_evidence(
            executor,
            SimpleNamespace(seed=900),
            object(),
            "nominal",
            run_dir=tmp_path / "run",
            run_id="run",
            source_commit="a" * 40,
            source_tree_digest="b" * 64,
        )
        == 1
    )
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["phase"] == phase
    assert payload["runner_status"] is None
    assert payload["error"] == "runtime-origin challenge failed"


def test_successful_bundle_does_not_emit_a_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        _invoke(
            tmp_path,
            monkeypatch,
            result=_result(passed=True),
            writer=lambda *_args, **_kwargs: SimpleNamespace(live_evidence=True),
        )
        == 0
    )
    assert not (tmp_path / "run" / "gate_failure.json").exists()


def test_nonlive_writer_result_is_rejected_and_diagnosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        _invoke(
            tmp_path,
            monkeypatch,
            result=_result(passed=True),
            writer=lambda *_args, **_kwargs: SimpleNamespace(live_evidence=False),
        )
        == 1
    )
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["phase"] == "artifact_write_or_verification"
    assert payload["release_eligible"] is False
    assert "did not accept" in payload["error"]


def test_diagnostic_error_records_are_bounded_and_original_cause_is_not_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result(passed=False)
    result.error_type = result.error = None
    result.trace = [
        {
            "timestamp_s": index,
            "event": "simulator_stop_failed",
            "error_type": "DynamicCoppeliaError",
            "error": "x" * 20_000,
            "large_unrelated_payload": _CountOnlyStream(),
        }
        for index in range(100)
    ]
    assert (
        _invoke(
            tmp_path,
            monkeypatch,
            result=result,
            writer=lambda *_args, **_kwargs: SimpleNamespace(live_evidence=True),
        )
        == 1
    )
    payload = json.loads((tmp_path / "run" / "gate_failure.json").read_bytes())
    assert payload["error"] is payload["runner_error"] is payload["error_type"] is None
    assert payload["trace_error_count"] == 100
    assert payload["trace_errors_truncated"] is True
    assert len(payload["trace_errors"]) == 32
    assert payload["trace_errors"][0]["timestamp_s"] == 68
    assert all(len(event["error"]) == 8192 for event in payload["trace_errors"])
    assert all("large_unrelated_payload" not in event for event in payload["trace_errors"])


def test_connection_failure_diagnostic_is_nonrelease_and_does_not_overwrite(
    tmp_path: Path,
) -> None:
    kwargs = {
        "status": "offline_or_connection_failed",
        "phase": "connection",
        "scenario_name": "nominal",
        "scenario_seed": 900,
        "source_commit": "a" * 40,
        "source_tree_digest": "b" * 64,
        "executor": SimpleNamespace(commands=[], telemetry=[]),
        "error": ConnectionError("remote API is unavailable"),
        "required_resource": "A running CoppeliaSim instance",
    }
    path = cli._persist_gate_failure(tmp_path / "run", **kwargs)
    before = path.read_bytes()
    payload = json.loads(before)
    assert payload["runner_status"] is None
    assert payload["trace_errors"] == []
    assert payload["error_type"] == "ConnectionError"
    assert payload["release_eligible"] is False
    assert payload["required_resource"] == "A running CoppeliaSim instance"
    with pytest.raises(FileExistsError):
        cli._persist_gate_failure(tmp_path / "run", **kwargs)
    assert path.read_bytes() == before


@pytest.mark.parametrize("execution_fails", [False, True])
def test_runner_keeps_execution_error_primary_and_records_secondary_shutdown_failures(
    monkeypatch: pytest.MonkeyPatch, execution_fails: bool
) -> None:
    def start():
        if execution_fails:
            raise LookupError("original execution failure")

    def stop():
        raise TimeoutError("shutdown deadline")

    def finalize(_executor):
        raise ValueError("runtime attestation could not finalize")

    monkeypatch.setattr(phase5, "schedule_build", lambda *_args: SimpleNamespace(jobs=[]))
    monkeypatch.setattr(phase5, "_finalize_live_runtime_attestation", finalize)
    executor = SimpleNamespace(
        start=start,
        stop=stop,
        simulation_time_s=12.5,
        installed_modules=[],
        diagnostics=lambda: {},
        runtime_events=[],
    )
    runner = SimpleNamespace(
        executor=executor,
        plan=SimpleNamespace(modules=[SimpleNamespace(module_id="foundation_0_0")]),
        scenario=SimpleNamespace(scenario_id="controlled-error-fixture", seed=900),
        evidence_kind="live_coppelia",
        replay=[],
        trace=[],
        recovery=None,
        physical_yard=SimpleNamespace(
            maximum_preflight_formation_offset_m=0.0,
            maximum_preflight_transport_height_m=0.0,
            model_dump=lambda **_kwargs: {},
        ),
        _acceptance=lambda *_args: {"runtime_completed": False},
    )
    # Exercise only terminal error handling: no simulator or evidence writer.
    result = phase5.Phase5FullCottageRunner.run(runner, "nominal")
    assert result.status == "failed"
    assert result.live_gate_passed is False
    assert result.error_type == ("LookupError" if execution_fails else "TimeoutError")
    assert result.error == ("original execution failure" if execution_fails else "shutdown deadline")
    assert {(event["event"], event["error_type"]) for event in result.trace} == {
        ("simulator_stop_failed", "TimeoutError"),
        ("live_runtime_attestation_failed", "ValueError"),
    }
