"""Cheap ordering guards; early archive rejection must not bypass native validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_skill_composer.construction import public_demo_provenance as provenance
from tests.test_construction_public_demo_provenance import SOURCE


@pytest.mark.parametrize("mismatch", ["source", "descriptor"])
def test_mismatched_archive_fails_before_expensive_evidence_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    inputs = {
        kind: SimpleNamespace(
            sha256="a" * 64,
            manifest=SimpleNamespace(source=SOURCE),
        )
        for kind in ("deterministic", "research", "simulator")
    }
    archive = SimpleNamespace(
        source=SOURCE.model_copy(update={"commit": "b" * 40}) if mismatch == "source" else SOURCE,
        descriptor_sha256="b" * 64 if mismatch == "descriptor" else "a" * 64,
    )
    loaded: list[str] = []

    def load_bundle(_path: Path, *, expected_kind: str) -> object:
        loaded.append(expected_kind)
        return inputs[expected_kind]

    def must_not_validate(_bundle: object) -> None:
        raise AssertionError("wrong archive identity must fail before scientific/native validation")

    monkeypatch.setattr(provenance, "_load_bundle", load_bundle)
    monkeypatch.setattr(provenance, "load_coppelia_archive_index", lambda _path: archive)
    monkeypatch.setattr(provenance, "_validate_research", must_not_validate)
    monkeypatch.setattr(provenance, "_validate_simulator", must_not_validate)
    output = tmp_path / "release"
    output.mkdir()
    retained = output / "existing.txt"
    retained.write_text("retain previous verified output", encoding="utf-8")

    with pytest.raises(provenance.PublicDemoExportError, match="source or descriptor identity differs"):
        provenance.export_public_demo_bundle(
            output,
            deterministic_bundle=tmp_path / "deterministic.json",
            research_bundle=tmp_path / "research.json",
            simulator_bundle=tmp_path / "simulator.json",
            source=SOURCE,
            channel="release",
            release_version="1.0.0",
            release_tag="v1.0.0",
            coppelia_archive_index=tmp_path / "archive-index.json",
        )

    assert loaded == ["deterministic", "research", "simulator"]
    assert retained.read_text(encoding="utf-8") == "retain previous verified output"
    assert list(output.iterdir()) == [retained]
    assert not list(tmp_path.glob(".release-*"))


def test_bound_archive_still_requires_full_native_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def bind_archive(_path: Path, _bundle: object) -> object:
        calls.append("archive")
        return object()

    def validate_research(_bundle: object) -> object:
        calls.append("research")
        return object()

    def reject_native(_bundle: object) -> None:
        calls.append("native")
        raise provenance.PublicDemoExportError("native measured evidence failed its gate")

    monkeypatch.setattr(provenance, "_load_bundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(provenance, "_validate_simulator_archive", bind_archive)
    monkeypatch.setattr(provenance, "_validate_research", validate_research)
    monkeypatch.setattr(provenance, "_validate_simulator", reject_native)
    output = tmp_path / "release"

    with pytest.raises(provenance.PublicDemoExportError, match="native measured evidence failed"):
        provenance.export_public_demo_bundle(
            output,
            deterministic_bundle=tmp_path / "deterministic.json",
            research_bundle=tmp_path / "research.json",
            simulator_bundle=tmp_path / "simulator.json",
            source=SOURCE,
            channel="release",
            release_version="1.0.0",
            release_tag="v1.0.0",
            coppelia_archive_index=tmp_path / "archive-index.json",
        )

    assert calls == ["archive", "research", "native"]
    assert not output.exists()
