from __future__ import annotations

from pathlib import Path
from random import Random

import pytest

from embodied_skill_composer.construction.intelligence_models import FailureKind
from embodied_skill_composer.construction.marl_env_v1 import (
    TemporalConstructionCoordinationEnv,
    _formation_point,
)
from embodied_skill_composer.construction.models import Pose3D, SiteGrid, Vec2, Vec3
from embodied_skill_composer.construction.routing import (
    PrioritizedRoutingAdapter,
    RoutingError,
    world_to_cell,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    _required_cells_connected,
    _required_interaction_cells,
    _sample_failures,
    _sample_obstacle_cells,
    generate_cottage_scenario,
)


@pytest.fixture(scope="module")
def base_design():
    return load_house_design(
        Path(__file__).resolve().parents[1] / "configs/construction/cottage_v1.yaml"
    )


@pytest.fixture(autouse=True)
def prohibit_rollouts(monkeypatch):
    def prohibited(*_args, **_kwargs):
        raise AssertionError("scenario feasibility tests must not execute rollouts")

    monkeypatch.setattr(TemporalConstructionCoordinationEnv, "step", prohibited)


@pytest.mark.parametrize("seed", [7, 8, 11, 800, 801, 802, 803, 804])
def test_required_interaction_cells_match_temporal_formation_contract(base_design, seed):
    scenario = generate_cottage_scenario(seed, base_design)
    assert scenario.split.value in {"train", "validation"}
    plan = scenario.plan
    expected = {
        world_to_cell(robot.start_pose.position, plan.site_grid)
        for robot in plan.robots
    }
    for module in plan.modules:
        for pose in (module.staging_pose, module.target_pose):
            expected.add(world_to_cell(pose.position, plan.site_grid))
            expected.update(
                world_to_cell(
                    _formation_point(pose.position, index, module.required_team_size),
                    plan.site_grid,
                )
                for index in range(module.required_team_size)
            )
    assert _required_interaction_cells(plan) == expected
    assert not expected.intersection(plan.site_grid.obstacle_cells)


def test_seed_801_is_retained_with_unblocked_foundation_endpoints(base_design):
    scenario = generate_cottage_scenario(801, base_design)
    plan = scenario.plan
    foundation = next(module for module in plan.modules if module.module_id == "foundation_0_0")
    cells = {
        world_to_cell(_formation_point(foundation.staging_pose.position, index, 2), plan.site_grid)
        for index in range(2)
    }
    assert scenario.seed == 801
    assert scenario.split.value == "validation"
    assert len(plan.modules) == 18
    assert len(plan.site_grid.obstacle_cells) == 4
    assert cells == {(3, 3), (3, 4)}
    assert cells.isdisjoint(plan.site_grid.obstacle_cells)
    assert _required_cells_connected(
        plan.site_grid, _required_interaction_cells(plan), set(plan.site_grid.obstacle_cells)
    )


@pytest.mark.parametrize(
    ("seed", "count", "kind", "trigger", "duration"),
    [
        (7, 1, "dropped_module", 58, 15),
        (8, 3, "dropped_module", 73, 24),
        (11, 4, "dropped_module", 57, 25),
        (800, 0, "robot_unavailable", 46, 13),
        (801, 4, "obstacle", 60, 21),
        (802, 0, "robot_unavailable", 54, 27),
        (803, 3, "dropped_module", 80, 25),
        (804, 3, "obstacle", 92, 24),
    ],
)
def test_endpoint_correction_preserves_counts_and_failure_mixture(
    base_design, seed, count, kind, trigger, duration,
):
    config = CottageScenarioConfig(include_failures=True, failure_probability=1.0)
    before = base_design.model_dump(mode="json")
    scenario = generate_cottage_scenario(seed, base_design, config=config)
    repeated = generate_cottage_scenario(seed, base_design, config=config)
    assert scenario.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert base_design.model_dump(mode="json") == before
    assert len(scenario.plan.site_grid.obstacle_cells) == count
    assert len(scenario.failures) == 1
    failure = scenario.failures[0]
    assert failure.kind.value == kind
    assert failure.trigger_time_s == trigger
    assert failure.duration_s == duration
    obstacles = set(scenario.plan.site_grid.obstacle_cells)
    if failure.kind == FailureKind.OBSTACLE:
        assert failure.obstacle_cell is not None
        assert failure.obstacle_cell not in obstacles
        obstacles.add(failure.obstacle_cell)
    required = _required_interaction_cells(scenario.plan)
    assert required.isdisjoint(obstacles)
    assert _required_cells_connected(scenario.plan.site_grid, required, obstacles)


class _OrderedObstacleRandom(Random):
    def __init__(self, prefix):
        super().__init__(7)
        self.prefix = prefix

    def shuffle(self, values):
        front = [value for value in self.prefix if value in values]
        values[:] = front + [value for value in values if value not in front]

    def choice(self, values):
        assert values == list(FailureKind)
        return FailureKind.OBSTACLE


@pytest.fixture
def small_plan(base_design):
    plan = generate_cottage_scenario(7, base_design).plan
    module = plan.modules[0].model_copy(update={
        "required_team_size": 1,
        "staging_pose": Pose3D(position=Vec3(x=3, y=3, z=0)),
        "target_pose": Pose3D(position=Vec3(x=6, y=6, z=0)),
    })
    robots = [robot.model_copy(update={
        "start_pose": Pose3D(position=Vec3(x=3, y=3, z=0)),
    }) for robot in plan.robots]
    return plan.model_copy(update={
        "modules": [module], "robots": robots,
        "site_grid": SiteGrid(width=10, height=10, resolution_m=1, origin=Vec2(x=0, y=0)),
    })


def test_nominal_sampler_skips_a_disconnect_without_dropping_obstacle_count(small_plan):
    ring = [(2, 3), (3, 2), (4, 3), (3, 4)]
    sampled = set(_sample_obstacle_cells(small_plan, 4, _OrderedObstacleRandom(ring)))
    assert len(sampled) == 4
    assert not set(ring) <= sampled
    assert _required_cells_connected(
        small_plan.site_grid, _required_interaction_cells(small_plan), sampled
    )


def test_failure_obstacle_sampler_preserves_endpoint_connectivity(small_plan):
    small_plan.site_grid.obstacle_cells = [(2, 3), (3, 2), (4, 3)]
    failures = _sample_failures(
        small_plan, _OrderedObstacleRandom([(3, 4), (4, 4)]),
        CottageScenarioConfig(include_failures=True, failure_probability=1.0),
    )
    assert len(failures) == 1
    assert failures[0].kind == FailureKind.OBSTACLE
    assert failures[0].obstacle_cell == (4, 4)
    assert _required_cells_connected(
        small_plan.site_grid,
        _required_interaction_cells(small_plan),
        {*small_plan.site_grid.obstacle_cells, failures[0].obstacle_cell},
    )


@pytest.mark.parametrize("count", [-1, 10_000])
def test_nominal_sampler_fails_explicitly_on_impossible_count(small_plan, count):
    with pytest.raises(ValueError, match="negative|eligible cells"):
        _sample_obstacle_cells(small_plan, count, Random(7))


def test_failure_obstacle_sampler_fails_explicitly_when_no_cell_exists(small_plan):
    small_plan.site_grid = SiteGrid(
        width=6, height=6, resolution_m=1, origin=Vec2(x=0, y=0)
    )
    with pytest.raises(ValueError, match="no feasible cell"):
        _sample_failures(
            small_plan, _OrderedObstacleRandom([]),
            CottageScenarioConfig(include_failures=True, failure_probability=1.0),
        )


def test_connectivity_handles_empty_required_set_and_blocked_endpoint(small_plan):
    assert _required_cells_connected(small_plan.site_grid, set(), {(3, 3)})
    assert not _required_cells_connected(small_plan.site_grid, {(3, 3)}, {(3, 3)})


def test_router_still_rejects_blocked_endpoints(small_plan):
    small_plan.site_grid.obstacle_cells = [(3, 3)]
    with pytest.raises(RoutingError, match="endpoint is blocked"):
        PrioritizedRoutingAdapter().route_many(
            small_plan.site_grid,
            {"robot": Vec2(x=6, y=6)},
            {"robot": Vec2(x=3, y=3)},
        )


@pytest.mark.parametrize("seed", range(800, 805))
def test_frozen_validation_seed_set_is_feasible_without_rollouts(base_design, seed):
    scenario = generate_cottage_scenario(
        seed, base_design,
        config=CottageScenarioConfig(include_failures=True, failure_probability=1.0),
    )
    assert scenario.seed == seed
    assert scenario.split.value == "validation"
    required = _required_interaction_cells(scenario.plan)
    obstacles = set(scenario.plan.site_grid.obstacle_cells)
    for failure in scenario.failures:
        if failure.kind == FailureKind.OBSTACLE:
            assert failure.obstacle_cell is not None
            obstacles.add(failure.obstacle_cell)
    assert required.isdisjoint(obstacles)
    assert _required_cells_connected(scenario.plan.site_grid, required, obstacles)


def test_all_training_and_validation_seeds_retain_connected_interactions(base_design):
    # This is only a static geometry audit. It deliberately excludes every
    # held-out seed and cannot collect actions, rewards, or policy outcomes.
    failure_kinds = set()
    for seed in range(900):
        scenario = generate_cottage_scenario(
            seed, base_design,
            config=CottageScenarioConfig(include_failures=True, failure_probability=1.0),
        )
        assert scenario.seed == seed
        assert scenario.split.value in {"train", "validation"}
        plan = scenario.plan
        required = {
            world_to_cell(robot.start_pose.position, plan.site_grid)
            for robot in plan.robots
        }
        for module in plan.modules:
            for pose in (module.staging_pose, module.target_pose):
                required.add(world_to_cell(pose.position, plan.site_grid))
                required.update(
                    world_to_cell(
                        _formation_point(pose.position, index, module.required_team_size),
                        plan.site_grid,
                    )
                    for index in range(module.required_team_size)
                )
        obstacles = set(plan.site_grid.obstacle_cells)
        assert len(obstacles) <= 4
        assert len(scenario.failures) == 1
        failure = scenario.failures[0]
        failure_kinds.add(failure.kind)
        if failure.kind == FailureKind.OBSTACLE:
            assert failure.obstacle_cell is not None
            obstacles.add(failure.obstacle_cell)
        assert required.isdisjoint(obstacles), seed
        # An independent flood-fill checks the production connectivity helper's
        # result using exact router cells and the actual temporal endpoints.
        reached = {min(required)}
        frontier = list(reached)
        while frontier:
            x, y = frontier.pop()
            for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (
                    0 <= cell[0] < plan.site_grid.width
                    and 0 <= cell[1] < plan.site_grid.height
                    and cell not in obstacles
                    and cell not in reached
                ):
                    reached.add(cell)
                    frontier.append(cell)
        assert required <= reached, seed
    assert failure_kinds == set(FailureKind)
