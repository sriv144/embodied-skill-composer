from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BeamTask,
    BlueprintSlot,
    ConstructionResource,
    GridCoord,
)


class GeneratedAssemblyScenario(BaseModel):
    scenario_id: str
    difficulty: str
    seed: int
    config: AssemblyScenarioConfig


class ScenarioGenerationConfig(BaseModel):
    scenario_count: int = 5
    base_seed: int = 7
    beam_count: int = 2
    obstacle_count: int = Field(default=0, ge=0)
    max_attempts: int = 250
    difficulty: str = "generated_v1"


def generate_assembly_scenarios(
    base_config: AssemblyScenarioConfig,
    generation_config: ScenarioGenerationConfig | None = None,
    runtime_profile: AssemblyRuntimeProfile | None = None,
) -> list[GeneratedAssemblyScenario]:
    generation_config = generation_config or ScenarioGenerationConfig()
    runtime_profile = runtime_profile or AssemblyRuntimeProfile()
    scenarios: list[GeneratedAssemblyScenario] = []
    attempts = 0
    rng = random.Random(generation_config.base_seed)

    while len(scenarios) < generation_config.scenario_count and attempts < generation_config.max_attempts:
        attempts += 1
        scenario_seed = rng.randrange(1_000_000)
        candidate = _build_candidate_config(
            base_config,
            generation_config.beam_count,
            generation_config.obstacle_count,
            scenario_seed,
        )
        if not _scripted_options_solve(candidate, runtime_profile, scenario_seed):
            continue
        scenarios.append(
            GeneratedAssemblyScenario(
                scenario_id=f"scenario_{len(scenarios) + 1:03d}",
                difficulty=generation_config.difficulty,
                seed=scenario_seed,
                config=candidate,
            )
        )

    if len(scenarios) < generation_config.scenario_count:
        raise RuntimeError(
            f"Generated {len(scenarios)} valid scenarios after {attempts} attempts; "
            f"requested {generation_config.scenario_count}."
        )
    return scenarios


def write_generated_scenario(scenario: GeneratedAssemblyScenario, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(scenario.config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def scenario_occupied_cells(config: AssemblyScenarioConfig) -> set[GridCoord]:
    cells = set(config.agent_starts) | set(config.obstacle_cells)
    for beam in config.beams:
        cells.update([beam.pickup_left, beam.pickup_right, beam.assembly_left, beam.assembly_right])
    return cells


def _build_candidate_config(
    base_config: AssemblyScenarioConfig,
    beam_count: int,
    obstacle_count: int,
    seed: int,
) -> AssemblyScenarioConfig:
    rng = random.Random(seed)
    grid_size = max(10, base_config.grid_size)
    beam_count = max(1, min(3, beam_count))
    y_pairs = [(y, y + 1) for y in range(1, grid_size - 2)]
    rng.shuffle(y_pairs)

    start_y = y_pairs.pop()[0]
    agent_starts = [(0, start_y), (0, start_y + 1)]
    occupied: set[GridCoord] = set(agent_starts)
    beams: list[BeamTask] = []
    pickup_x_choices = list(range(2, max(3, grid_size // 2 - 1)))
    assembly_x_choices = list(range(max(grid_size // 2 + 2, 6), grid_size - 1))

    for index in range(beam_count):
        pickup = _pick_pair(rng, y_pairs, pickup_x_choices, occupied)
        assembly = _pick_pair(rng, y_pairs, assembly_x_choices, occupied)
        beams.append(
            BeamTask(
                name=f"beam_{index + 1:02d}",
                pickup_left=pickup[0],
                pickup_right=pickup[1],
                assembly_left=assembly[0],
                assembly_right=assembly[1],
            )
        )
        occupied.update([pickup[0], pickup[1], assembly[0], assembly[1]])

    available_obstacles = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) not in occupied
    ]
    rng.shuffle(available_obstacles)
    if obstacle_count > len(available_obstacles):
        raise ValueError(
            f"requested {obstacle_count} obstacles but only {len(available_obstacles)} cells are available"
        )
    obstacles = available_obstacles[:obstacle_count]

    payload = base_config.model_dump(mode="python")
    payload.update(
        {
            "grid_size": grid_size,
            "agent_starts": agent_starts,
            "beams": beams,
            "obstacle_cells": obstacles,
            "manipulation_failures": [],
            "resources": [ConstructionResource.from_beam(beam) for beam in beams],
            "blueprint_slots": [BlueprintSlot.from_beam(beam) for beam in beams],
            "curriculum_beam_stages": list(range(1, beam_count + 1)),
            "curriculum_stage_beams": [beams[: index + 1] for index in range(beam_count)],
        }
    )
    return AssemblyScenarioConfig.model_validate(payload)


def _pick_pair(
    rng: random.Random,
    y_pairs: list[tuple[int, int]],
    x_choices: list[int],
    occupied: set[GridCoord],
) -> tuple[GridCoord, GridCoord]:
    candidates: list[tuple[GridCoord, GridCoord]] = []
    for y0, y1 in y_pairs:
        for x in x_choices:
            pair = ((x, y0), (x, y1))
            if pair[0] not in occupied and pair[1] not in occupied:
                candidates.append(pair)
    if not candidates:
        raise RuntimeError("Could not place a non-overlapping beam pair.")
    return rng.choice(candidates)


def _scripted_options_solve(
    config: AssemblyScenarioConfig,
    runtime_profile: AssemblyRuntimeProfile,
    seed: int,
) -> bool:
    env = build_assembly_backend(config, runtime_profile, seed=seed)
    env.reset(seed=seed)
    done = False
    while not done:
        result = env.execute_team_option(env.scripted_team_option())
        done = result.done
    artifact = env.build_artifact(policy_mode="scripted")
    return artifact.metrics.success and artifact.metrics.beams_installed == len(config.beams)
