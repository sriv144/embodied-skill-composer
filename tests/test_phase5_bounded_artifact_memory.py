import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from embodied_skill_composer.construction.coppelia_phase5 import (
    _intern_telemetry_samples,
    _iter_json_array,
    _load_json_array_models,
    _load_jsonl_models,
    _sha256_json,
    _write_json,
    _write_jsonl,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.models import Pose3D, Vec2, Vec3


class EvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    label: str


def test_lazy_models_preserve_canonical_hash_and_formatted_json(tmp_path: Path) -> None:
    rows = [EvidenceRow(count=index, label='café \\"[x],{y}') for index in range(25)]
    plain = [row.model_dump(mode="json") for row in rows]
    expected_hash = hashlib.sha256(
        json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _sha256_json(rows) == expected_hash
    path = tmp_path / "replay.json"
    _write_json(path, rows)
    assert path.read_text(encoding="utf-8") == (
        json.dumps(plain, indent=2, sort_keys=True) + "\n"
    )


def test_model_hashing_bounds_live_dump_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    live = maximum_live = 0

    class CountedDump(dict):
        def __init__(self, value: dict) -> None:
            nonlocal live, maximum_live
            super().__init__(value)
            live += 1
            maximum_live = max(maximum_live, live)

        def __del__(self) -> None:
            nonlocal live
            live -= 1

    original_dump = EvidenceRow.model_dump

    def tracked_dump(self: EvidenceRow, **kwargs) -> dict:
        return CountedDump(original_dump(self, **kwargs))

    monkeypatch.setattr(EvidenceRow, "model_dump", tracked_dump)
    rows = [EvidenceRow(count=index, label="sample") for index in range(1000)]
    _sha256_json(rows)
    assert maximum_live <= 2
    assert live == 0


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 19, 65536])
@pytest.mark.parametrize(
    "payload",
    [
        [],
        [None, True, False, 1, -1.25, "", "café"],
        [
            {"nested": [{"brackets": "[x],{y}"}, [1, 2]], "escape": '\\"\\\\'},
            ["\\", '\"', "\n\t", {"unicode": "\u2603"}],
            "\\\\\\\"[]{}",
        ],
    ],
)
def test_array_reader_preserves_nested_escaped_values_across_chunks(
    tmp_path: Path, chunk_size: int, payload: list
) -> None:
    path = tmp_path / "array.json"
    path.write_text(" \n" + json.dumps(payload, indent=2) + "\t\n", encoding="utf-8")
    assert list(_iter_json_array(path, chunk_size=chunk_size)) == payload


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
@pytest.mark.parametrize(
    "payload",
    [
        "",
        "null",
        "{}",
        "[",
        "[1",
        "[1,]",
        "[,]",
        "[1,,2]",
        "[1] null",
        "[1][]",
        '[{"x":1]]',
        '["unterminated]',
        '["bad\\q"]',
        '[1 2]',
        '[}]',
        '\v[1]',
        '[1]\u00a0',
    ],
)
def test_array_reader_rejects_malformed_or_trailing_input(
    tmp_path: Path, chunk_size: int, payload: str
) -> None:
    path = tmp_path / "array.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        list(_iter_json_array(path, chunk_size=chunk_size))


@pytest.mark.parametrize("jsonl", [False, True])
def test_verified_source_reuse_preserves_identity(tmp_path: Path, jsonl: bool) -> None:
    rows = [EvidenceRow(count=index, label="sample") for index in range(3)]
    path = tmp_path / ("rows.jsonl" if jsonl else "rows.json")
    if jsonl:
        _write_jsonl(path, (row.model_dump(mode="json") for row in rows))
        loaded = _load_jsonl_models(path, EvidenceRow, source_rows=rows)
    else:
        _write_json(path, rows)
        loaded = _load_json_array_models(path, EvidenceRow, source_rows=rows)
    assert all(actual is original for actual, original in zip(loaded, rows, strict=True))


@pytest.mark.parametrize("jsonl", [False, True])
@pytest.mark.parametrize("mismatch", ["value", "missing", "extra", "schema"])
def test_source_reuse_still_parses_validates_and_compares_every_row(
    tmp_path: Path, jsonl: bool, mismatch: str
) -> None:
    rows = [EvidenceRow(count=index, label="sample") for index in range(3)]
    disk_rows = [row.model_dump(mode="json") for row in rows]
    if mismatch == "value":
        disk_rows[1]["label"] = "modified"
    elif mismatch == "missing":
        disk_rows.pop()
    elif mismatch == "extra":
        disk_rows.append({"count": 3, "label": "extra"})
    else:
        disk_rows[1]["unexpected"] = True
    path = tmp_path / ("rows.jsonl" if jsonl else "rows.json")
    if jsonl:
        _write_jsonl(path, iter(disk_rows))
        loader = _load_jsonl_models
    else:
        _write_json(path, disk_rows)
        loader = _load_json_array_models
    with pytest.raises((ValueError, ValidationError)):
        loader(path, EvidenceRow, source_rows=rows)


@pytest.mark.parametrize("bad_line", ["\n", "{bad}\n"])
def test_jsonl_loader_reports_invalid_line_with_source_reuse(
    tmp_path: Path, bad_line: str
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"count":0,"label":"sample"}\n' + bad_line, encoding="utf-8")
    with pytest.raises(ValueError, match="rows.jsonl:2"):
        _load_jsonl_models(
            path, EvidenceRow, source_rows=[EvidenceRow(count=0, label="sample")]
        )


def test_source_reuse_distinguishes_signed_zero_in_canonical_evidence(tmp_path: Path) -> None:
    class Measurement(BaseModel):
        value: float

    source = [Measurement(value=-0.0)]
    path = tmp_path / "measurement.json"
    _write_json(path, [{"value": 0.0}])
    assert Measurement(value=0.0) == source[0]
    with pytest.raises(ValueError, match="differs from source evidence"):
        _load_json_array_models(path, Measurement, source_rows=source)


@pytest.mark.parametrize("jsonl", [False, True])
def test_command_roundtrip_preserves_declared_2d_wire_projection(
    tmp_path: Path, jsonl: bool
) -> None:
    source = RobotCommand(
        timestamp_s=0.0,
        robot_id="robot_1",
        linear_velocity_mps=0.0,
        angular_velocity_rps=0.0,
        wheel_target_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
        target_position=Vec3(x=1.0, y=2.0, z=3.0),
        source="path_follower",
    )
    path = tmp_path / ("commands.jsonl" if jsonl else "commands.json")
    if jsonl:
        _write_jsonl(path, [source.model_dump(mode="json")])
        loaded = _load_jsonl_models(path, RobotCommand, source_rows=[source])
    else:
        _write_json(path, [source])
        loaded = _load_json_array_models(path, RobotCommand, source_rows=[source])
    assert type(source.target_position) is Vec3
    assert type(loaded[0].target_position) is Vec2
    assert loaded[0] is not source
    assert loaded[0].model_dump(mode="json") == source.model_dump(mode="json")
    assert _sha256_json(loaded[0]) == _sha256_json(source)


def test_command_projection_still_rejects_modified_wire_coordinate(tmp_path: Path) -> None:
    source = RobotCommand(
        timestamp_s=0.0,
        robot_id="robot_1",
        linear_velocity_mps=0.0,
        angular_velocity_rps=0.0,
        wheel_target_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
        target_position=Vec3(x=1.0, y=2.0, z=3.0),
        source="path_follower",
    )
    modified = source.model_dump(mode="json")
    modified["target_position"]["x"] += 0.05
    path = tmp_path / "commands.jsonl"
    _write_jsonl(path, [modified])
    with pytest.raises(ValueError, match="differs from source evidence"):
        _load_jsonl_models(path, RobotCommand, source_rows=[source])


def _measurement(velocity: float, *, timestamp_s: float = 0.0) -> RobotTelemetry:
    return RobotTelemetry(
        timestamp_s=timestamp_s,
        robot_id="robot_1",
        measured_pose=Pose3D(position=Vec3(x=1.0, y=2.0, z=0.0)),
        linear_velocity_mps=velocity,
        angular_velocity_rps=0.0,
        battery_remaining_wh=100.0,
    )


def test_telemetry_interning_retains_duplicate_timestamps_and_nonmatching_rows() -> None:
    source = [_measurement(0.0), _measurement(0.2)]
    matching = [RobotTelemetry.model_validate(item.model_dump()) for item in source]
    nonmatching = _measurement(0.3)
    samples = [*matching, nonmatching]
    lookup = {_sha256_json(item): item for item in source}
    _intern_telemetry_samples(samples, lookup)
    assert samples[0] is source[0]
    assert samples[1] is source[1]
    assert samples[2] is nonmatching
    assert len(samples) == 3


def test_telemetry_interning_checks_canonical_value_after_lookup() -> None:
    sample = _measurement(0.0)
    signed_zero_candidate = _measurement(-0.0)
    assert sample == signed_zero_candidate
    samples = [sample]
    _intern_telemetry_samples(
        samples, {_sha256_json(sample): signed_zero_candidate}
    )
    assert samples[0] is sample


def test_array_model_transform_runs_only_after_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    _write_json(path, [{"count": 1, "label": "valid"}, {"count": 2, "bad": True}])
    seen: list[EvidenceRow] = []

    def transform(row: EvidenceRow) -> EvidenceRow:
        seen.append(row)
        return row

    with pytest.raises(ValidationError):
        _load_json_array_models(path, EvidenceRow, transform=transform)
    assert seen == [EvidenceRow(count=1, label="valid")]
