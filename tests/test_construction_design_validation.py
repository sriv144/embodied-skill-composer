from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import embodied_skill_composer.construction.api as api_module
from embodied_skill_composer.construction.api import WorkbenchState, create_app
from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.design_validation import (
    DesignValidationError,
    validate_house_design,
)
from embodied_skill_composer.construction.models import (
    HouseDesign,
    ModuleType,
    Opening,
    Vec2,
    WallSegment,
)
from embodied_skill_composer.construction.runtime import load_house_design


WORKSPACE = Path(__file__).resolve().parents[1]
DESIGN_PATH = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"


@pytest.fixture
def design() -> HouseDesign:
    return load_house_design(DESIGN_PATH)


def _small_positive_design(design: HouseDesign) -> HouseDesign:
    candidate = design.model_copy(deep=True)
    candidate.design_id = "small-positive-footprint"
    candidate.footprint_width_m = 0.2
    candidate.footprint_depth_m = 0.15
    candidate.floor_plan.openings = []
    candidate.floor_plan.rooms = []
    candidate.floor_plan.walls = [
        WallSegment(
            wall_id="north",
            start=Vec2(x=-0.1, y=0.075),
            end=Vec2(x=0.1, y=0.075),
            thickness_m=0.01,
            height_m=0.2,
        ),
        WallSegment(
            wall_id="east",
            start=Vec2(x=0.1, y=0.075),
            end=Vec2(x=0.1, y=-0.075),
            thickness_m=0.01,
            height_m=0.2,
        ),
        WallSegment(
            wall_id="south",
            start=Vec2(x=0.1, y=-0.075),
            end=Vec2(x=-0.1, y=-0.075),
            thickness_m=0.01,
            height_m=0.2,
        ),
        WallSegment(
            wall_id="west",
            start=Vec2(x=-0.1, y=-0.075),
            end=Vec2(x=-0.1, y=0.075),
            thickness_m=0.01,
            height_m=0.2,
        ),
    ]
    return candidate


def test_fixture_design_passes_shared_validation(design: HouseDesign) -> None:
    result = validate_house_design(design)

    assert result.valid is True
    assert result.issues == []


def test_validation_reports_stable_unique_id_and_numeric_issues(design: HouseDesign) -> None:
    invalid = design.model_copy(deep=True)
    invalid.design_id = " "
    invalid.footprint_width_m = float("inf")
    invalid.floor_plan.walls[1].wall_id = invalid.floor_plan.walls[0].wall_id
    invalid.floor_plan.walls[2].height_m = 0
    invalid.floor_plan.openings[1].opening_id = invalid.floor_plan.openings[0].opening_id
    invalid.floor_plan.rooms[1].room_id = invalid.floor_plan.rooms[0].room_id

    result = validate_house_design(invalid)

    assert result.valid is False
    assert {
        (issue.code, issue.path)
        for issue in result.issues
    } >= {
        ("invalid_id", "design_id"),
        ("non_finite_dimension", "footprint_width_m"),
        ("duplicate_id", "floor_plan.walls[1].wall_id"),
        ("non_positive_dimension", "floor_plan.walls[2].height_m"),
        ("duplicate_id", "floor_plan.openings[1].opening_id"),
        ("duplicate_id", "floor_plan.rooms[1].room_id"),
    }
    assert result.issues == sorted(
        result.issues,
        key=lambda issue: (issue.path, issue.code, issue.message),
    )


@pytest.mark.parametrize(
    ("wall", "expected_code"),
    [
        (
            WallSegment(
                wall_id="diagonal",
                start=Vec2(x=-2, y=-2),
                end=Vec2(x=2, y=2),
            ),
            "wall_not_axis_aligned",
        ),
        (
            WallSegment(
                wall_id="zero",
                start=Vec2(x=0, y=0),
                end=Vec2(x=0, y=0),
            ),
            "wall_zero_length",
        ),
        (
            WallSegment(
                wall_id="outside",
                start=Vec2(x=-5, y=0),
                end=Vec2(x=0, y=0),
            ),
            "wall_out_of_bounds",
        ),
        (
            WallSegment(
                wall_id="crossing",
                start=Vec2(x=0, y=-3),
                end=Vec2(x=0, y=3),
            ),
            "wall_intersection",
        ),
        (
            WallSegment(
                wall_id="overlap",
                start=Vec2(x=-2, y=3),
                end=Vec2(x=2, y=3),
            ),
            "wall_overlap",
        ),
    ],
)
def test_validation_rejects_unsafe_wall_geometry(
    design: HouseDesign,
    wall: WallSegment,
    expected_code: str,
) -> None:
    invalid = design.model_copy(deep=True)
    invalid.floor_plan.walls.append(wall)

    result = validate_house_design(invalid)

    assert expected_code in {issue.code for issue in result.issues}


def test_validation_allows_only_legal_shared_wall_endpoints(design: HouseDesign) -> None:
    result = validate_house_design(design)

    assert "wall_intersection" not in {issue.code for issue in result.issues}
    assert "wall_overlap" not in {issue.code for issue in result.issues}


def test_validation_rejects_opening_reference_extents_overlap_and_vertical_fit(
    design: HouseDesign,
) -> None:
    invalid = design.model_copy(deep=True)
    invalid.floor_plan.openings[0].wall_id = "missing"
    invalid.floor_plan.openings[2].height_m = 2.0
    invalid.floor_plan.openings[2].sill_height_m = 1.0
    invalid.floor_plan.openings.append(
        Opening(
            opening_id="overlapping_window",
            wall_id="north",
            kind="window",
            offset_m=4.2,
            width_m=1.0,
            height_m=1.0,
            sill_height_m=1.0,
        )
    )
    invalid.floor_plan.openings.append(
        Opening(
            opening_id="outside_window",
            wall_id="north",
            kind="window",
            offset_m=0.2,
            width_m=1.0,
            height_m=1.0,
            sill_height_m=1.0,
        )
    )

    result = validate_house_design(invalid)
    codes = {issue.code for issue in result.issues}

    assert codes >= {
        "opening_unknown_wall",
        "opening_out_of_bounds",
        "opening_overlap",
        "opening_vertical_overflow",
    }


def test_validation_allows_openings_to_touch_without_overlap(design: HouseDesign) -> None:
    candidate = design.model_copy(deep=True)
    candidate.floor_plan.openings = [
        Opening(
            opening_id="left",
            wall_id="north",
            kind="window",
            offset_m=0.5,
            width_m=1.0,
            height_m=1.0,
            sill_height_m=1.0,
        ),
        Opening(
            opening_id="right",
            wall_id="north",
            kind="window",
            offset_m=1.5,
            width_m=1.0,
            height_m=1.0,
            sill_height_m=1.0,
        ),
    ]

    result = validate_house_design(candidate)

    assert result.valid is True


def test_compiler_preserves_multiple_openings_and_metric_semantics(
    design: HouseDesign,
) -> None:
    candidate = design.model_copy(deep=True)
    candidate.floor_plan.openings.append(
        Opening(
            opening_id="north_narrow",
            wall_id="north",
            kind="window",
            offset_m=1.0,
            width_m=0.8,
            height_m=0.7,
            sill_height_m=1.5,
        )
    )

    plan = compile_house_design(candidate)
    north_openings = sorted(
        (
            module
            for module in plan.modules
            if module.module_type == ModuleType.WINDOW
            and module.target_pose.position.y == pytest.approx(3.0)
        ),
        key=lambda module: module.target_pose.position.x,
    )

    assert len(north_openings) == 2
    assert north_openings[0].dimensions.height == pytest.approx(2.8)
    assert north_openings[0].target_pose.position.z == pytest.approx(1.6)
    assert north_openings[0].architectural_opening is not None
    assert north_openings[0].architectural_opening.width_m == pytest.approx(0.8)
    assert north_openings[0].architectural_opening.height_m == pytest.approx(0.7)
    assert north_openings[0].architectural_opening.sill_height_m == pytest.approx(1.5)
    assert north_openings[0].architectural_opening_local_offset_m is not None
    assert (
        north_openings[0].target_pose.position.x
        + north_openings[0].architectural_opening_local_offset_m
    ) == pytest.approx(-3.0)
    assert north_openings[1].dimensions.height == pytest.approx(2.8)
    assert north_openings[1].target_pose.position.z == pytest.approx(1.6)
    assert north_openings[1].architectural_opening is not None
    assert north_openings[1].architectural_opening.width_m == pytest.approx(1.8)
    assert north_openings[1].architectural_opening.height_m == pytest.approx(1.4)
    assert north_openings[1].architectural_opening.sill_height_m == pytest.approx(0.8)
    assert north_openings[1].architectural_opening_local_offset_m is not None
    assert (
        north_openings[1].target_pose.position.x
        + north_openings[1].architectural_opening_local_offset_m
    ) == pytest.approx(0.0)


def test_compiler_validates_before_approval_gate(design: HouseDesign) -> None:
    invalid = design.model_copy(deep=True)
    invalid.floor_plan.approved = False
    invalid.floor_plan.walls[0].end.y = 2.5

    with pytest.raises(DesignValidationError) as error:
        compile_house_design(invalid)

    assert "wall_not_axis_aligned" in {
        issue.code for issue in error.value.result.issues
    }


def test_every_valid_small_positive_footprint_compiles_with_positive_modules(
    design: HouseDesign,
) -> None:
    candidate = _small_positive_design(design)

    assert validate_house_design(candidate).valid is True
    plan = compile_house_design(candidate)

    assert plan.modules
    assert all(
        module.dimensions.width > 0
        and module.dimensions.depth > 0
        and module.dimensions.height > 0
        for module in plan.modules
    )
    interior_modules = [
        module
        for module in plan.modules
        if module.module_type == ModuleType.INTERIOR
    ]
    assert interior_modules
    assert all(
        abs(module.target_pose.position.x) < candidate.footprint_width_m / 2
        for module in interior_modules
    )


def test_small_positive_footprint_validates_and_rebuilds_through_api(
    design: HouseDesign,
    tmp_path: Path,
) -> None:
    candidate = _small_positive_design(design)
    app = create_app(registry_path=tmp_path / "small-api.sqlite")

    with TestClient(app) as client:
        validation = client.post(
            "/api/design/validate",
            json={"design": candidate.model_dump(mode="json")},
        )
        assert validation.status_code == 200
        assert validation.json() == {"valid": True, "issues": []}

        rebuilt = client.post(
            "/api/design/rebuild",
            json={"design": candidate.model_dump(mode="json")},
        )
        assert rebuilt.status_code == 200
        assert rebuilt.json()["design"]["footprint_width_m"] == pytest.approx(0.2)
        assert all(
            module["dimensions"]["width"] > 0
            and module["dimensions"]["depth"] > 0
            for module in rebuilt.json()["plan"]["modules"]
        )


def test_workbench_replace_is_atomic_when_downstream_rebuild_fails(
    design: HouseDesign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkbenchState(design)
    before = {
        "design": state.design.model_dump(mode="json"),
        "plan": state.plan.model_dump(mode="json"),
        "schedules": {
            key: value.model_dump(mode="json") for key, value in state.schedules.items()
        },
        "traces": {
            key: value.model_dump(mode="json") for key, value in state.traces.items()
        },
    }
    candidate = design.model_copy(deep=True)
    candidate.design_id = "replacement"
    candidate.title = "Replacement"

    def fail_after_compilation(_plan: object) -> None:
        raise ValueError("synthetic scheduler rejection")

    monkeypatch.setattr(api_module, "compare_controllers", fail_after_compilation)
    with pytest.raises(ValueError, match="synthetic scheduler rejection"):
        state.replace_design(candidate)

    assert state.design.model_dump(mode="json") == before["design"]
    assert state.plan.model_dump(mode="json") == before["plan"]
    assert {
        key: value.model_dump(mode="json") for key, value in state.schedules.items()
    } == before["schedules"]
    assert {
        key: value.model_dump(mode="json") for key, value in state.traces.items()
    } == before["traces"]


def test_design_validation_api_and_failed_rebuild_keep_current_project(
    design: HouseDesign,
    tmp_path: Path,
) -> None:
    app = create_app(registry_path=tmp_path / "api.sqlite")
    with TestClient(app) as client:
        original = client.get("/api/project").json()
        invalid = design.model_dump(mode="json")
        invalid["design_id"] = "invalid_replacement"
        invalid["floor_plan"]["walls"][0]["end"]["y"] = 2.5

        validation = client.post("/api/design/validate", json={"design": invalid})
        assert validation.status_code == 200
        assert validation.json()["valid"] is False
        assert {
            issue["code"] for issue in validation.json()["issues"]
        } >= {"wall_not_axis_aligned"}

        rejected = client.post("/api/design/rebuild", json={"design": invalid})
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["valid"] is False
        assert client.get("/api/project").json() == original


def test_design_validation_api_returns_structured_editable_dimension_errors(
    design: HouseDesign,
    tmp_path: Path,
) -> None:
    app = create_app(registry_path=tmp_path / "api.sqlite")
    with TestClient(app) as client:
        invalid = design.model_dump(mode="json")
        invalid["footprint_width_m"] = 0
        invalid["floor_plan"]["walls"] = invalid["floor_plan"]["walls"][:3]
        invalid["floor_plan"]["walls"][0]["thickness_m"] = -0.1

        response = client.post("/api/design/validate", json={"design": invalid})

        assert response.status_code == 200
        assert {
            issue["code"] for issue in response.json()["issues"]
        } >= {"insufficient_walls", "non_positive_dimension"}
