from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embodied_skill_composer.construction.intelligence_models import (
    ConstructionFailure,
    FailureKind,
)
from embodied_skill_composer.construction.marl_env_v1 import (
    MAX_MODULES,
    TemporalConstructionCoordinationEnv,
    auction_temporal_actions,
    scripted_temporal_actions,
)
from embodied_skill_composer.construction.models import SiteGrid, Vec2
from embodied_skill_composer.construction.routing import (
    PrioritizedRoutingAdapter,
    count_path_conflicts,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
    scenario_split_for_seed,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DESIGN_PATH = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"


@pytest.fixture(scope="module")
def base_design():
    return load_house_design(DESIGN_PATH)


def test_seed_partitions_are_explicit_and_disjoint() -> None:
    assert scenario_split_for_seed(0).value == "train"
    assert scenario_split_for_seed(799).value == "train"
    assert scenario_split_for_seed(800).value == "validation"
    assert scenario_split_for_seed(899).value == "validation"
    assert scenario_split_for_seed(900).value == "test"
    assert scenario_split_for_seed(999).value == "test"
    with pytest.raises(ValueError, match="between 0 and 999"):
        scenario_split_for_seed(1000)


def test_cottage_family_is_deterministic_and_does_not_mutate_source(base_design) -> None:
    source = base_design.model_dump(mode="json")
    left = generate_cottage_scenario(7, base_design)
    right = generate_cottage_scenario(7, base_design)
    assert left.model_dump(mode="json") == right.model_dump(mode="json")
    assert base_design.model_dump(mode="json") == source
    assert 16 <= len(left.plan.modules) <= MAX_MODULES
    assert len(left.plan.robots) == 4


@pytest.mark.parametrize(
    ("width", "depth", "interiors", "expected_modules"),
    [(4.0, 4.0, 0, 16), (12.0, 8.0, 4, 32)],
)
def test_cottage_family_spans_supported_module_range(
    base_design,
    width: float,
    depth: float,
    interiors: int,
    expected_modules: int,
) -> None:
    scenario = generate_cottage_scenario(
        11,
        base_design,
        config=CottageScenarioConfig(
            widths_m=(width,),
            depths_m=(depth,),
            interior_panel_range=(interiors, interiors),
            obstacle_count_range=(0, 0),
        ),
    )
    assert len(scenario.plan.modules) == expected_modules


def test_prioritized_router_avoids_static_and_agent_conflicts() -> None:
    grid = SiteGrid(
        width=10,
        height=8,
        resolution_m=1.0,
        origin=Vec2(x=0, y=0),
        obstacle_cells=[(4, 1), (4, 2), (4, 3), (4, 4), (4, 5)],
    )
    route = PrioritizedRoutingAdapter().route_many(
        grid,
        {"robot_1": Vec2(x=0, y=1), "robot_2": Vec2(x=0, y=2)},
        {"robot_1": Vec2(x=9, y=2), "robot_2": Vec2(x=9, y=1)},
    )
    assert route.backend == "prioritized_astar"
    assert route.conflict_count == 0
    assert count_path_conflicts(route.cell_paths) == 0
    assert all(
        cell not in grid.obstacle_cells
        for path in route.cell_paths.values()
        for cell in path
    )


def test_temporal_environment_advances_time_and_enforces_busy_wait(base_design) -> None:
    scenario = generate_cottage_scenario(
        7,
        base_design,
        config=CottageScenarioConfig(obstacle_count_range=(0, 0)),
    )
    env = TemporalConstructionCoordinationEnv(scenario)
    observations, infos = env.reset(seed=7)
    assert set(observations) == set(env.possible_agents)
    assert all(observation["modules"].shape == (MAX_MODULES, 14) for observation in observations.values())
    assert all(info["sim_time_s"] == 0 for info in infos.values())

    for _ in range(8):
        actions = scripted_temporal_actions(env)
        _, _, _, _, infos = env.step(actions)
        if env.active_jobs:
            break
    assert env.sim_time_s > 0
    assert env.active_jobs
    for agent, runtime in env.robot_runtime.items():
        if runtime.status == "busy":
            assert np.array_equal(env.action_mask(agent), np.array([1] + [0] * MAX_MODULES))
    assert next(iter(infos.values()))["assignments"]


@pytest.mark.parametrize("controller", [scripted_temporal_actions, auction_temporal_actions])
def test_temporal_controllers_complete_in_dependency_order(base_design, controller) -> None:
    scenario = generate_cottage_scenario(
        19,
        base_design,
        config=CottageScenarioConfig(obstacle_count_range=(0, 0)),
    )
    env = TemporalConstructionCoordinationEnv(scenario)
    env.reset(seed=19)
    modules = {module.module_id: module for module in scenario.plan.modules}
    while env.agents:
        completed_before = set(env.completed)
        _, _, _, _, infos = env.step(controller(env))
        assignments = next(iter(infos.values()))["assignments"]
        for assignment in assignments:
            assert set(modules[assignment["module_id"]].dependencies) <= completed_before
    assert len(env.completed) == len(scenario.plan.modules)
    assert env.metrics()["structure_completion_rate"] == 1.0
    assert env.sim_time_s > sum(module.install_duration_s for module in scenario.plan.modules) / 4


def test_robot_failure_cancels_work_then_recovers(base_design) -> None:
    scenario = generate_cottage_scenario(
        31,
        base_design,
        config=CottageScenarioConfig(obstacle_count_range=(0, 0)),
    )
    scenario.failures = [
        ConstructionFailure(
            failure_id="fixture-unavailable",
            kind=FailureKind.ROBOT_UNAVAILABLE,
            trigger_time_s=5,
            duration_s=12,
            robot_id="robot_1",
        )
    ]
    env = TemporalConstructionCoordinationEnv(scenario)
    env.reset(seed=31)
    while env.agents:
        env.step(scripted_temporal_actions(env))
    assert env.metrics()["structure_completion_rate"] == 1.0
    assert env.wasted_work_s > 0
    assert any(item["event"] == "failure" for item in env.event_log)
    assert any(item["event"] == "robot_recovered" for item in env.event_log)


def test_temporal_environment_passes_pettingzoo_parallel_api(base_design) -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    scenario = generate_cottage_scenario(
        5,
        base_design,
        config=CottageScenarioConfig(obstacle_count_range=(0, 0)),
    )
    parallel_api_test(
        TemporalConstructionCoordinationEnv(scenario, max_decisions=24),
        num_cycles=28,
    )


def test_pointer_actor_masks_padded_and_blocked_jobs() -> None:
    torch = pytest.importorskip("torch")
    from embodied_skill_composer.construction.policy import SwarmPointerActor

    actor = SwarmPointerActor(hidden_dim=32)
    action_mask = torch.zeros(2, 4, 33, dtype=torch.bool)
    action_mask[..., 0] = True
    logits = actor(
        torch.zeros(2, 4, 12),
        torch.zeros(2, 4, 4, 11),
        torch.zeros(2, 4, 32, 14),
        torch.zeros(2, 4, 32, 32),
        action_mask,
    )
    assert logits.shape == (2, 4, 33)
    assert torch.equal(logits.argmax(dim=-1), torch.zeros(2, 4, dtype=torch.long))
    assert torch.all(logits[..., 1:] < -1e8)


@pytest.mark.parametrize("algorithm", ["mappo", "ippo"])
def test_torchrl_ppo_losses_accept_temporal_rollouts(base_design, algorithm: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchrl")
    from embodied_skill_composer.construction.policy import build_torchrl_policy
    from embodied_skill_composer.construction.training import (
        TrainingConfig,
        build_ppo_batch,
        build_ppo_loss,
        collect_policy_rollouts,
        optimize_ppo_batch,
    )

    config = TrainingConfig.for_profile("unit", algorithm=algorithm, seed=47)
    bundle = build_torchrl_policy(algorithm, hidden_dim=config.hidden_dim)
    episodes, _ = collect_policy_rollouts(
        bundle,
        base_design,
        config,
        decision_target=4,
        episode_cursor=0,
        device="cpu",
    )
    batch = build_ppo_batch(
        episodes,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device="cpu",
    )
    loss = build_ppo_loss(bundle, config)
    optimizer = torch.optim.Adam(loss.parameters(), lr=config.learning_rate)
    metrics = optimize_ppo_batch(
        loss,
        optimizer,
        batch,
        epochs=1,
        minibatch_size=4,
        max_grad_norm=1.0,
    )
    assert batch.batch_size[0] == 4
    assert all(np.isfinite(value) for value in metrics.values())


def test_checkpoint_and_onnx_exports_are_loadable(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("torchrl")
    from embodied_skill_composer.construction.policy import (
        build_torchrl_policy,
        export_actor_onnx,
        load_policy_checkpoint,
        save_policy_checkpoint,
    )

    bundle = build_torchrl_policy("mappo", hidden_dim=32)
    checkpoint = tmp_path / "policy.pt"
    checksum = save_policy_checkpoint(
        bundle,
        checkpoint,
        metadata={"hidden_dim": 32},
    )
    loaded = load_policy_checkpoint(checkpoint)
    exported = export_actor_onnx(loaded.actor_model, tmp_path / "actor.onnx")
    onnx.checker.check_model(onnx.load(exported))
    assert checkpoint.is_file()
    assert exported.is_file()
    assert len(checksum) == 64
