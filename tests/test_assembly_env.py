from embodied_skill_composer.assembly.baseline import scripted_joint_policy
from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import AssemblyScenarioConfig, BeamTask, TeamOption


def build_env() -> CollaborativeAssemblyEnv:
    config = AssemblyScenarioConfig(
        grid_size=10,
        max_steps=60,
        agent_starts=[(0, 2), (0, 3)],
        beams=[
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(7, 6),
                assembly_right=(7, 7),
            )
        ],
    )
    return CollaborativeAssemblyEnv(config=config, seed=7)


def build_two_beam_env() -> CollaborativeAssemblyEnv:
    config = AssemblyScenarioConfig(
        grid_size=12,
        max_steps=120,
        agent_starts=[(0, 2), (0, 3)],
        beams=[
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(8, 7),
                assembly_right=(8, 8),
            ),
            BeamTask(
                name="beam_beta",
                pickup_left=(2, 6),
                pickup_right=(2, 7),
                assembly_left=(9, 7),
                assembly_right=(9, 8),
            ),
        ],
        curriculum_stage_beams=[
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                )
            ],
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                ),
                BeamTask(
                    name="beam_beta_easy",
                    pickup_left=(2, 5),
                    pickup_right=(2, 6),
                    assembly_left=(8, 9),
                    assembly_right=(8, 10),
                ),
            ],
            [
                BeamTask(
                    name="beam_alpha",
                    pickup_left=(2, 2),
                    pickup_right=(2, 3),
                    assembly_left=(8, 7),
                    assembly_right=(8, 8),
                ),
                BeamTask(
                    name="beam_beta",
                    pickup_left=(2, 6),
                    pickup_right=(2, 7),
                    assembly_left=(9, 7),
                    assembly_right=(9, 8),
                ),
            ],
        ],
    )
    return CollaborativeAssemblyEnv(config=config, seed=7)


def test_assembly_env_reset_shapes() -> None:
    env = build_env()
    observations, state = env.reset(seed=7)

    assert observations.shape[0] == 2
    assert state.ndim == 1
    assert env.obs_dim == observations.shape[1]


def test_scripted_policy_solves_single_beam_task() -> None:
    env = build_env()
    env.reset(seed=7)
    done = False
    while not done:
        _, _, _, done, _ = env.step(scripted_joint_policy(env))

    artifact = env.build_artifact(policy_mode="scripted")
    assert artifact.metrics.success is True
    assert artifact.metrics.beams_installed == 1


def test_action_mask_unlocks_grab_only_at_pickup() -> None:
    env = build_env()
    env.reset(seed=7)

    initial_masks = env.get_action_masks()
    assert initial_masks[0][5] == 0.0
    assert initial_masks[1][5] == 0.0

    env.state.agent_positions = [(2, 2), (2, 3)]
    pickup_masks = env.get_action_masks()
    assert pickup_masks[0][5] == 1.0
    assert pickup_masks[1][5] == 1.0


def test_curriculum_limits_active_beam_count() -> None:
    env = build_env()
    env.set_curriculum_stage(1)
    _, state = env.reset(seed=7)

    assert env.active_beam_count == 1
    assert env._current_beam().name == "beam_alpha"


def test_stage_specific_curriculum_beams_override_default_beams() -> None:
    env = build_env()
    env.config.curriculum_stage_beams = [
        [env.config.beams[0]],
        [
            env.config.beams[0],
            env.config.beams[0].model_copy(
                update={
                    "name": "beam_beta_easy",
                    "pickup_left": (2, 4),
                    "pickup_right": (2, 5),
                    "assembly_left": (6, 6),
                    "assembly_right": (6, 7),
                }
            ),
        ],
    ]
    env.set_curriculum_stage(stage_index=1)

    assert env.active_beam_count == 2
    assert env._available_beams()[1].name == "beam_beta_easy"


def test_team_option_mask_unlocks_grab_only_at_pickup() -> None:
    env = build_env()
    env.reset(seed=7)

    initial_mask = env.get_team_option_mask()
    assert initial_mask[int(TeamOption.GRAB)] == 0.0

    env.state.agent_positions = [(2, 2), (2, 3)]
    pickup_mask = env.get_team_option_mask()
    assert pickup_mask[int(TeamOption.GRAB)] == 1.0
    assert pickup_mask[int(TeamOption.INSTALL)] == 0.0


def test_go_pickup_and_go_assembly_options_reach_targets() -> None:
    env = build_env()
    env.reset(seed=7)

    pickup_result = env.execute_team_option(TeamOption.GO_PICKUP)
    assert pickup_result.success is True
    assert env.state.agent_positions == [(2, 2), (2, 3)]

    grab_result = env.execute_team_option(TeamOption.GRAB)
    assert grab_result.success is True
    assembly_result = env.execute_team_option(TeamOption.GO_ASSEMBLY)
    assert assembly_result.success is True
    assert env.state.agent_positions == [(7, 6), (7, 7)]


def test_scripted_team_option_solves_single_beam() -> None:
    env = build_env()
    env.reset(seed=7)
    selected_options: list[str] = []

    done = False
    while not done:
        option = env.scripted_team_option()
        selected_options.append(option.name.lower())
        result = env.execute_team_option(option)
        done = result.done

    artifact = env.build_artifact(policy_mode="scripted")
    diagnostics = env.get_option_episode_diagnostics()
    assert artifact.metrics.success is True
    assert diagnostics["selected_options"] == selected_options
    assert selected_options == ["go_pickup", "grab", "go_assembly", "install"]


def test_recovery_options_activate_after_first_beam_install() -> None:
    env = build_two_beam_env()
    env.reset(seed=7)

    while env.state.current_beam_index < 1:
        env.execute_team_option(env.scripted_team_option())

    assert env.scripted_team_option() == TeamOption.REPOSITION_AFTER_INSTALL
    reposition_result = env.execute_team_option(TeamOption.REPOSITION_AFTER_INSTALL)
    assert reposition_result.success is True
    assert env.recovery_option_usage["reposition_after_install"] == 1

    assert env.scripted_team_option() == TeamOption.RESET_TO_PICKUP_ROUTE
    reset_result = env.execute_team_option(TeamOption.RESET_TO_PICKUP_ROUTE)
    assert reset_result.success is True
    assert env.recovery_option_usage["reset_to_pickup_route"] == 1
