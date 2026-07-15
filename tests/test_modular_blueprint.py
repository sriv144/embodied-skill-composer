from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from embodied_skill_composer.assembly.blueprint import (
    BlueprintCompilationError,
    compile_modular_blueprint,
)
from embodied_skill_composer.assembly.brain import (
    PrecedenceConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import AssemblyScenarioConfig, BeamTask
from embodied_skill_composer.assembly.models import TeamOption
from embodied_skill_composer.assembly.reporting import render_construction_lab_report
from embodied_skill_composer.assembly.runtime import (
    load_asset_catalog,
    load_modular_blueprint,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def load_inputs():
    blueprint = load_modular_blueprint(
        WORKSPACE / "configs" / "blueprints" / "modular_room_v0.yaml"
    )
    catalog = load_asset_catalog(
        WORKSPACE / "configs" / "construction_asset_catalog.yaml"
    )
    return blueprint, catalog


def compile_room(blueprint=None):
    default_blueprint, catalog = load_inputs()
    return compile_modular_blueprint(
        blueprint or default_blueprint,
        catalog,
        workspace_root=WORKSPACE,
    )


def test_legacy_assembly_scenario_remains_valid() -> None:
    config = AssemblyScenarioConfig(
        beams=[
            BeamTask(
                name="beam",
                pickup_left=(1, 1),
                pickup_right=(1, 2),
                assembly_left=(4, 4),
                assembly_right=(4, 5),
            )
        ]
    )

    assert config.blueprint_id is None
    assert config.installation_order == []
    assert config.derived_resources()[0].asset_key is None
    assert config.derived_blueprint_slots()[0].target_pose is None


def test_modular_room_compiles_in_stable_dependency_order() -> None:
    blueprint, _ = load_inputs()
    original = blueprint.model_dump(mode="json")

    compiled = compile_room(blueprint)

    assert compiled.installation_order == [
        "column_nw",
        "column_ne",
        "column_se",
        "column_sw",
        "wall_north",
        "wall_east",
        "wall_south",
        "wall_west",
        "roof_left",
        "roof_right",
    ]
    assert len(compiled.scenario.beams) == 10
    assert len(compiled.scenario.resources) == 10
    assert len(compiled.scenario.blueprint_slots) == 10
    assert compiled.scenario.resources[0].component_id == "column_nw"
    assert compiled.scenario.blueprint_slots[-1].depends_on == [
        "wall_north",
        "wall_east",
        "wall_south",
        "wall_west",
    ]
    assert compiled.component_to_resource["roof_right"] == "material_roof_right"
    assert blueprint.model_dump(mode="json") == original


def test_blueprint_rejects_duplicate_component_ids() -> None:
    blueprint, _ = load_inputs()
    blueprint.components[1].component_id = blueprint.components[0].component_id

    with pytest.raises(BlueprintCompilationError, match="component IDs must be unique"):
        compile_room(blueprint)


def test_blueprint_rejects_unknown_asset() -> None:
    blueprint, _ = load_inputs()
    blueprint.components[0].asset_key = "missing_asset"

    with pytest.raises(BlueprintCompilationError, match="unknown asset"):
        compile_room(blueprint)


def test_blueprint_rejects_material_type_mismatch() -> None:
    blueprint, _ = load_inputs()
    blueprint.materials[0].component_type = "wall_panel"

    with pytest.raises(BlueprintCompilationError, match="does not match material type"):
        compile_room(blueprint)


def test_blueprint_rejects_unknown_dependency() -> None:
    blueprint, _ = load_inputs()
    blueprint.components[-1].depends_on = ["missing_component"]

    with pytest.raises(BlueprintCompilationError, match="unknown dependencies"):
        compile_room(blueprint)


def test_blueprint_rejects_dependency_cycle() -> None:
    blueprint, _ = load_inputs()
    blueprint.components[0].depends_on = ["roof_right"]

    with pytest.raises(BlueprintCompilationError, match="contains a cycle"):
        compile_room(blueprint)


def test_blueprint_rejects_out_of_bounds_cell() -> None:
    blueprint, _ = load_inputs()
    blueprint.materials[0].source_cells[0] = (blueprint.grid_size, 0)

    with pytest.raises(BlueprintCompilationError, match="outside grid"):
        compile_room(blueprint)


def test_blueprint_rejects_wrong_team_size() -> None:
    blueprint, _ = load_inputs()
    blueprint.components[0].required_team_size = 1

    with pytest.raises(BlueprintCompilationError, match="requires team size 1"):
        compile_room(blueprint)


def test_precedence_brain_completes_modular_room() -> None:
    compiled = compile_room()
    episode = run_construction_brain_episode(
        CollaborativeAssemblyEnv(compiled.scenario, seed=7),
        PrecedenceConstructionBrain(),
        seed=7,
    )

    assert episode.artifact.metrics.success is True
    assert episode.artifact.metrics.beams_installed == 10
    assert episode.artifact.metrics.structure_completion_rate == 1.0
    assert episode.artifact.metrics.resource_delivery_accuracy == 1.0
    assert episode.diagnostics["construction_brain"]["dependency_hold_count"] == 0
    assert {assignment.status for assignment in episode.assignments} == {"completed"}
    for step in episode.steps:
        if step.decision.option != TeamOption.INSTALL:
            continue
        assignment = step.decision.assignment
        assert assignment is not None
        completed = {
            slot.component_id
            for slot in step.observation.blueprint_slots
            if slot.completed
        }
        assert set(assignment.prerequisites).issubset(completed)


def test_precedence_brain_reports_blocked_dependency() -> None:
    compiled = compile_room()
    env = CollaborativeAssemblyEnv(compiled.scenario, seed=7)
    env.reset(seed=7)
    observation = env.get_construction_observation().model_copy(
        update={"current_beam_name": "material_wall_north"},
        deep=True,
    )
    brain = PrecedenceConstructionBrain()
    brain.reset(observation)

    decision = brain.decide(observation)

    assert decision.option == TeamOption.WAIT
    assert decision.safety_hold_reason == "blocked_dependency"
    assert "column_nw" in decision.rationale
    assert "column_ne" in decision.rationale


def test_modular_room_lab_report_contains_plan_and_metrics() -> None:
    compiled = compile_room()
    episode = run_construction_brain_episode(
        CollaborativeAssemblyEnv(compiled.scenario, seed=7),
        PrecedenceConstructionBrain(),
        seed=7,
    )

    report = render_construction_lab_report(compiled, episode)

    assert "Cooperative Modular Room v0" in report
    assert "Structure completion: `1.000`" in report
    assert "`wall_north` after column_nw, column_ne" in report
    assert "robots [0, 1]" in report
    assert "deterministic oracle" in report


def test_modular_construction_cli_writes_local_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(WORKSPACE / "scripts" / "run_modular_construction.py"),
            "--runtime-profile",
            str(WORKSPACE / "configs" / "assembly_profiles" / "local_dev.yaml"),
            "--run-id",
            "test-room",
            "--output-root",
            str(tmp_path),
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )

    run_dir = tmp_path / "logs" / "construction_runs" / "test-room"
    assert result.returncode == 0, result.stderr
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "blueprint.json").is_file()
    payload = json.loads((run_dir / "episode.json").read_text(encoding="utf-8"))
    assert payload["artifact"]["metrics"]["success"] is True
    assert payload["artifact"]["metrics"]["beams_installed"] == 10
