from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def api_cli(monkeypatch: pytest.MonkeyPatch):
    # Parse and wire the CLI without importing the API, starting its services,
    # connecting to a real registry, or opening a listener.
    api = ModuleType("embodied_skill_composer.construction.api")
    create_app = Mock(return_value=object())
    api.create_app = create_app  # type: ignore[attr-defined]
    uvicorn = ModuleType("uvicorn")
    run_server = Mock()
    uvicorn.run = run_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setitem(sys.modules, uvicorn.__name__, uvicorn)
    script = Path(__file__).resolve().parents[1] / "scripts/run_construction_api.py"
    entry = runpy.run_path(str(script))["main"]
    return entry, create_app, run_server


def test_default_api_cli_preserves_reload_and_loopback(api_cli, monkeypatch):
    entry, create_app, run_server = api_cli
    monkeypatch.setattr(sys, "argv", ["api", "--reload", "--port", "8010"])
    entry()
    create_app.assert_not_called()
    run_server.assert_called_once_with(
        "embodied_skill_composer.construction.api:app",
        host="127.0.0.1",
        port=8010,
        reload=True,
    )


@pytest.mark.parametrize("registry,evidence", [(True, False), (False, True), (True, True)])
def test_api_cli_passes_custom_paths(api_cli, monkeypatch, tmp_path, registry, evidence):
    entry, create_app, run_server = api_cli
    registry_path = tmp_path / "lab" / "lab.sqlite"
    evidence_root = tmp_path / "native-bundles"
    evidence_root.mkdir()
    args = ["api"]
    if registry:
        args += ["--registry-path", str(registry_path)]
    if evidence:
        args += ["--coppelia-evidence-root", str(evidence_root)]
    monkeypatch.setattr(sys, "argv", args)
    entry()
    create_app.assert_called_once_with(
        registry_path=registry_path.resolve() if registry else None,
        coppelia_evidence_root=evidence_root.resolve() if evidence else None,
    )
    run_server.assert_called_once_with(
        create_app.return_value, host="127.0.0.1", port=8008, reload=False
    )
    assert registry_path.parent.exists() is registry
    assert not registry_path.exists()


@pytest.mark.parametrize("custom", ["--registry-path", "--coppelia-evidence-root"])
def test_custom_paths_refuse_reload_before_starting(api_cli, monkeypatch, tmp_path, custom):
    entry, create_app, run_server = api_cli
    monkeypatch.setattr(sys, "argv", ["api", "--reload", custom, str(tmp_path)])
    with pytest.raises(SystemExit, match="2"):
        entry()
    create_app.assert_not_called()
    run_server.assert_not_called()


@pytest.mark.parametrize("is_file", [False, True])
def test_evidence_root_must_exist_as_directory(api_cli, monkeypatch, tmp_path, is_file):
    entry, create_app, run_server = api_cli
    invalid = tmp_path / "invalid-evidence"
    if is_file:
        invalid.touch()
    monkeypatch.setattr(sys, "argv", ["api", "--coppelia-evidence-root", str(invalid)])
    with pytest.raises(SystemExit, match="2"):
        entry()
    create_app.assert_not_called()
    run_server.assert_not_called()
