import hashlib
import json
from pathlib import Path

import pytest

from embodied_skill_composer.construction.coppelia_phase5 import (
    _sha256_json,
    _write_json,
    _write_jsonl,
)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"unicode": "café", "nested": [1, -0.0, True]},
        [{"t": i / 20, "pose": [i, 0, 0]} for i in range(1000)],
    ],
)
def test_streaming_digest_preserves_canonical_evidence_hash(payload: object) -> None:
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _sha256_json(payload) == expected


def test_streaming_jsonl_preserves_rows(tmp_path: Path) -> None:
    rows = [{"t": 0.05, "label": "café"}, {"t": 0.1, "pose": [1, 2, 3]}]
    path = tmp_path / "telemetry.jsonl"
    _write_jsonl(path, rows)
    assert path.read_text(encoding="utf-8") == "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )


def test_streaming_replay_preserves_formatted_artifact(tmp_path: Path) -> None:
    payload = {"jobs": [{"module": "café", "poses": [[1, 2, 3], [4, 5, 6]]}]}
    path = tmp_path / "replay.json"
    _write_json(path, payload)
    assert path.read_text(encoding="utf-8") == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
