from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import DynamicCoppeliaError
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5PhysicalYardConfig,
    _base_world_clearance_minima,
    _phase5_carry_handoff_tolerance,
    _plan_phase5_dispatch_return,
)
from embodied_skill_composer.construction.models import (
    BuildModule,
    Dimensions3D,
    Pose3D,
    SiteGrid,
    Vec2,
    Vec3,
)
from embodied_skill_composer.construction.routing import RoutingError, create_routing_adapter
from embodied_skill_composer.construction.runtime import load_house_design


@pytest.fixture
def geometry():
    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs/construction/cottage_v1.yaml"
    )
    plan = compile_house_design(design)
    # Exact module shape and commanded endpoints from the failed b890 live job.
    # This small offline geometry fixture does not replace native run evidence.
    module = BuildModule(
        module_id="foundation_0_1", module_type="foundation",
        dimensions=Dimensions3D(width=5.96, depth=1.96, height=0.2),
        mass_kg=85, material="foundation_concrete", required_team_size=2,
        staging_pose=Pose3D(position=Vec3(x=-11.08, y=1.94, z=0.1)),
        target_pose=Pose3D(position=Vec3(x=3, y=-1, z=0.1)),
        install_duration_s=16, mesh_node="module__foundation_0_1",
        grip_points=[Vec3(x=-1.49, y=0, z=0), Vec3(x=1.49, y=0, z=0)],
    )
    robots = [
        robot.model_copy(update={"start_pose": Pose3D(
            position=Vec3(x=-23.6, y=-1.5 + index, z=0)
        )}) for index, robot in enumerate(plan.robots)
    ]
    plan = plan.model_copy(update={
        "modules": [module], "robots": robots,
        "site_grid": SiteGrid(
            width=110, height=85, resolution_m=0.5,
            origin=Vec2(x=-30, y=-15), obstacle_cells=[(105, 2)],
        ),
    })
    terminals = {
        "robot_1": Vec2(x=3.045446625525692, y=-2.4819612015861408),
        "robot_3": Vec2(x=3.0204410182487607, y=0.5233149151976768),
    }
    positions = {
        "robot_1": terminals["robot_1"], "robot_3": terminals["robot_3"],
        "robot_2": Vec2(x=-23.501910352357292, y=-0.47751084465125626),
        "robot_4": Vec2(x=-23.591753135526186, y=1.6861975605897528),
    }
    return plan, module, terminals, positions, Phase5PhysicalYardConfig()


def handoff(geometry, *, terminals=None, positions=None, disabled=(), configured=0.12):
    plan, module, original_terminals, original_positions, config = geometry
    return _phase5_carry_handoff_tolerance(
        plan, module=module,
        terminal_positions=original_terminals if terminals is None else terminals,
        installed_module_ids=set(),
        robot_positions=original_positions if positions is None else positions,
        disabled_robot_ids=set(disabled), waypoint_tolerance_m=configured, config=config,
    )


def return_plan(geometry, terminals, *, positions=None, disabled=()):
    plan, module, _terminals, original_positions, config = geometry
    return _plan_phase5_dispatch_return(
        create_routing_adapter(seed=900, prefer_w9=False), plan,
        module=module, starts=terminals, installed_module_ids={module.module_id},
        robot_positions={**(positions or original_positions), **terminals},
        active_robot_ids=set(terminals), disabled_robot_ids=set(disabled), config=config,
    )


def test_recorded_carry_error_consumes_return_reserve(geometry) -> None:
    plan, module, terminals, positions, config = geometry
    actual = {
        "robot_1": Vec2(x=3.022073695878018, y=-2.3658579534777333),
        "robot_3": Vec2(x=2.9876470206782626, y=0.6355644630936129),
    }
    assert math.hypot(
        actual["robot_1"].x-terminals["robot_1"].x,
        actual["robot_1"].y-terminals["robot_1"].y,
    ) == pytest.approx(0.11843250424455914)
    _, _, clearances = return_plan(geometry, terminals)
    assert clearances[0].minimum_surface_clearance_m == pytest.approx(0.3819612015861408)
    assert clearances[0].required_clearance_m == pytest.approx(0.28)
    with pytest.raises(DynamicCoppeliaError, match="installed_module is 0.266 m") as failure:
        return_plan(geometry, actual)
    assert isinstance(failure.value.__cause__, RoutingError)
    measured = _base_world_clearance_minima(
        plan, phase="return", source="measured", routes={rid: [p] for rid, p in actual.items()},
        active_module_id=module.module_id, installed_module_ids={module.module_id},
        robot_positions=positions, active_robot_ids=set(actual), disabled_robot_ids=set(),
        active_module_is_payload=False, config=config,
    )
    assert measured[0].required_clearance_m == pytest.approx(0.16)
    assert measured[0].minimum_surface_clearance_m == pytest.approx(0.26585795347773333)


def test_bounded_tolerance_retains_full_planned_return_reserve(geometry) -> None:
    _plan, _module, terminals, _positions, _config = geometry
    tolerance = handoff(geometry)
    assert tolerance == pytest.approx(0.05)
    # Worst inward errors for each base, not just the recorded error direction.
    inward = {
        "robot_1": Vec2(x=terminals["robot_1"].x, y=terminals["robot_1"].y+tolerance),
        "robot_3": Vec2(x=terminals["robot_3"].x, y=terminals["robot_3"].y-tolerance),
    }
    _, _, clearances = return_plan(geometry, inward)
    assert clearances[0].minimum_surface_clearance_m == pytest.approx(0.3319612015861408)
    assert all(
        item.minimum_surface_clearance_m is None
        or item.minimum_surface_clearance_m >= item.required_clearance_m
        for item in clearances
    )
    assert handoff(geometry, configured=0.01) == 0.01


def test_newly_installed_module_limits_handoff_slack(geometry) -> None:
    _plan, _module, terminals, _positions, _config = geometry
    terminals = {**terminals, "robot_1": Vec2(x=3, y=-2.4)}
    assert handoff(geometry, terminals=terminals) == pytest.approx(0.01)


@pytest.mark.parametrize("y", [-2.38, -2.36])
def test_zero_or_negative_terminal_slack_fails_closed(geometry, y) -> None:
    terminals = {**geometry[2], "robot_1": Vec2(x=3, y=y)}
    with pytest.raises((DynamicCoppeliaError, RoutingError), match="slack|clearance"):
        handoff(geometry, terminals=terminals)


@pytest.mark.parametrize("disabled", [(), ("robot_2",)])
def test_pairwise_idle_or_disabled_clearance_is_not_treated_as_static(geometry, disabled) -> None:
    terminals = {"robot_1": Vec2(x=-10, y=-10), "robot_3": Vec2(x=-10, y=-8)}
    positions = {**geometry[3], **terminals, "robot_2": Vec2(x=-9.32, y=-10)}
    # Center separation .68 minus two radii .24 minus planned reserve .40
    # leaves .04 slack: two possible endpoint errors must share that slack.
    tolerance = handoff(geometry, terminals=terminals, positions=positions, disabled=disabled)
    assert tolerance == pytest.approx(0.02)
    assert .68 - 2 * tolerance - .24 >= .4 - 1e-9


def test_active_pair_separation_has_its_own_slack_bound(geometry) -> None:
    terminals = {"robot_1": Vec2(x=-10, y=-10), "robot_3": Vec2(x=-10, y=-9.34)}
    assert handoff(geometry, terminals=terminals) == pytest.approx(0.01)
    terminals["robot_3"] = Vec2(x=-10, y=-9.37)
    with pytest.raises(DynamicCoppeliaError, match="positive clearance slack"):
        handoff(geometry, terminals=terminals)


@pytest.mark.parametrize("configured", [0, -1, math.nan, math.inf, -math.inf])
def test_invalid_handoff_tolerance_fails_closed(geometry, configured) -> None:
    with pytest.raises(DynamicCoppeliaError, match="finite and positive"):
        handoff(geometry, configured=configured)


@pytest.mark.parametrize("terminals", [{}, {"robot_1": Vec2(x=0, y=0)}])
def test_handoff_requires_a_full_two_base_team(geometry, terminals) -> None:
    with pytest.raises(DynamicCoppeliaError, match="exactly two"):
        handoff(geometry, terminals=terminals)


def test_late_passive_drift_still_fails_fresh_return_preflight(geometry) -> None:
    terminals = {"robot_1": Vec2(x=-10, y=-10), "robot_3": Vec2(x=-10, y=-8)}
    before = {**geometry[3], **terminals, "robot_2": Vec2(x=-15, y=-10)}
    assert handoff(geometry, terminals=terminals, positions=before, disabled=("robot_2",)) > 0
    after = {**before, "robot_2": Vec2(x=-9.5, y=-10)}
    with pytest.raises(DynamicCoppeliaError, match="disabled_robot"):
        return_plan(geometry, terminals, positions=after, disabled=("robot_2",))
