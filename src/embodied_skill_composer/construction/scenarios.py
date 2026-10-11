from __future__ import annotations

from collections import deque
from math import hypot
from random import Random

from pydantic import BaseModel, Field, model_validator

from embodied_skill_composer.construction.compiler import (
    ConstructionCompileSettings,
    compile_house_design,
)
from embodied_skill_composer.construction.intelligence_models import (
    ConstructionFailure,
    FailureKind,
    ScenarioManifest,
    ScenarioSplit,
)
from embodied_skill_composer.construction.models import BuildPlan, HouseDesign, SiteGrid, Vec2


class CottageScenarioConfig(BaseModel):
    widths_m: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)
    depths_m: tuple[float, ...] = (4.0, 6.0, 8.0)
    interior_panel_range: tuple[int, int] = (0, 4)
    obstacle_count_range: tuple[int, int] = (0, 4)
    include_failures: bool = False
    failure_probability: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> "CottageScenarioConfig":
        if not self.widths_m or not self.depths_m:
            raise ValueError("scenario dimensions cannot be empty")
        if self.interior_panel_range[0] < 0 or self.interior_panel_range[1] > 4:
            raise ValueError("interior panel count must stay between 0 and 4")
        if self.interior_panel_range[0] > self.interior_panel_range[1]:
            raise ValueError("interior panel range is inverted")
        if self.obstacle_count_range[0] < 0:
            raise ValueError("obstacle count cannot be negative")
        if self.obstacle_count_range[0] > self.obstacle_count_range[1]:
            raise ValueError("obstacle count range is inverted")
        return self


def scenario_split_for_seed(seed: int) -> ScenarioSplit:
    if not 0 <= seed <= 999:
        raise ValueError("construction scenario seeds must be between 0 and 999")
    if seed <= 799:
        return ScenarioSplit.TRAIN
    if seed <= 899:
        return ScenarioSplit.VALIDATION
    return ScenarioSplit.TEST


def generate_cottage_scenario(
    seed: int,
    base_design: HouseDesign,
    *,
    config: CottageScenarioConfig | None = None,
) -> ScenarioManifest:
    """Generate one deterministic, validated member of the cottage task family."""
    split = scenario_split_for_seed(seed)
    config = config or CottageScenarioConfig()
    rng = Random(seed)

    width_m = rng.choice(config.widths_m)
    depth_m = rng.choice(config.depths_m)
    interior_count = rng.randint(*config.interior_panel_range)
    design = _scaled_design(base_design, seed, width_m, depth_m)
    plan = compile_house_design(
        design,
        ConstructionCompileSettings(interior_panel_count=interior_count),
    )
    if not 16 <= len(plan.modules) <= 32:
        raise ValueError(f"generated plan has {len(plan.modules)} modules; expected 16-32")

    _randomize_yard_and_fleet(plan, rng)
    obstacle_count = rng.randint(*config.obstacle_count_range)
    plan.site_grid.obstacle_cells = _sample_obstacle_cells(plan, obstacle_count, rng)
    failures = _sample_failures(plan, rng, config)
    scenario_id = f"cottage-{split.value}-{seed:03d}-{len(plan.modules):02d}m"
    plan.plan_id = f"{scenario_id}-plan"
    return ScenarioManifest(
        scenario_id=scenario_id,
        seed=seed,
        split=split,
        plan=plan,
        failures=failures,
        tags=[
            "single_story",
            "orthogonal",
            f"{len(plan.modules)}_modules",
            f"{obstacle_count}_obstacles",
        ],
    )


def _scaled_design(
    base_design: HouseDesign,
    seed: int,
    width_m: float,
    depth_m: float,
) -> HouseDesign:
    design = base_design.model_copy(deep=True)
    old_width = design.footprint_width_m
    old_depth = design.footprint_depth_m
    scale_x = width_m / old_width
    scale_y = depth_m / old_depth
    original_wall_lengths = {
        wall.wall_id: hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
        for wall in design.floor_plan.walls
    }
    for wall in design.floor_plan.walls:
        wall.start.x *= scale_x
        wall.end.x *= scale_x
        wall.start.y *= scale_y
        wall.end.y *= scale_y
    scaled_wall_lengths = {
        wall.wall_id: hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
        for wall in design.floor_plan.walls
    }
    for opening in design.floor_plan.openings:
        old_length = original_wall_lengths[opening.wall_id]
        opening.offset_m *= scaled_wall_lengths[opening.wall_id] / old_length
        opening.width_m = min(opening.width_m, scaled_wall_lengths[opening.wall_id] * 0.4)
    for room in design.floor_plan.rooms:
        for point in room.polygon:
            point.x *= scale_x
            point.y *= scale_y
    design.design_id = f"cottage_seed_{seed:03d}"
    design.title = f"Procedural Cottage {seed:03d}"
    design.footprint_width_m = width_m
    design.footprint_depth_m = depth_m
    return design


def _randomize_yard_and_fleet(plan: BuildPlan, rng: Random) -> None:
    yard_dx = rng.uniform(-0.35, 0.35)
    yard_dy = rng.uniform(-0.25, 0.45)
    for module in plan.modules:
        module.staging_pose.position.x += yard_dx
        module.staging_pose.position.y += yard_dy
    for index, robot in enumerate(plan.robots):
        if robot.role == "heavy":
            robot.payload_capacity_kg = rng.uniform(50.0, 58.0)
        else:
            robot.payload_capacity_kg = rng.uniform(36.0, 42.0)
        robot.speed_mps *= rng.uniform(0.9, 1.1)
        robot.battery_capacity_wh *= rng.uniform(0.9, 1.1)
        robot.start_pose.position.y += rng.uniform(-0.15, 0.15)
        robot.start_pose.position.x += 0.08 * index
    BuildPlan.model_validate(plan.model_dump(mode="json"))


def _sample_obstacle_cells(plan: BuildPlan, count: int, rng: Random) -> list[tuple[int, int]]:
    if count < 0:
        raise ValueError("obstacle count cannot be negative")
    # Keep the original permutation and RNG consumption so the subsequent
    # failure occurrence/type remain unchanged for a seed. Apply the stronger
    # interaction-cell exclusions after shuffling, not to its input population.
    excluded: set[tuple[int, int]] = set()
    for robot in plan.robots:
        excluded.add(_world_to_cell(robot.start_pose.position, plan.site_grid))
    for module in plan.modules:
        excluded.add(_world_to_cell(module.staging_pose.position, plan.site_grid))
        excluded.add(_world_to_cell(module.target_pose.position, plan.site_grid))

    candidates = [
        (x, y)
        for x in range(2, plan.site_grid.width - 2)
        for y in range(2, plan.site_grid.height - 2)
        if (x, y) not in excluded
    ]
    rng.shuffle(candidates)
    required = _required_interaction_cells(plan)
    candidates = [cell for cell in candidates if cell not in required]
    if count > len(candidates):
        raise ValueError(
            f"requested {count} obstacles but only {len(candidates)} eligible cells exist"
        )
    selected: set[tuple[int, int]] = set()
    if count == 0:
        return []
    for candidate in candidates:
        if not _required_cells_connected(plan.site_grid, required, selected | {candidate}):
            continue
        selected.add(candidate)
        if len(selected) == count:
            return sorted(selected)
    raise ValueError(f"cannot place {count} obstacles while keeping required endpoints connected")


def _required_interaction_cells(plan: BuildPlan) -> set[tuple[int, int]]:
    """Cells used by temporal carrier centers and all possible base endpoints.

    The temporal environment's two-base formation is y +/- 0.35 m at both
    pickup and installation. Installation endpoints become the next route's
    starts, so reserving staging or target centers alone is insufficient.
    Keep this lightweight: scenario generation must not import the optional
    PettingZoo environment. Contract tests compare these offsets to its helper.
    """
    required = {
        _world_to_cell(robot.start_pose.position, plan.site_grid)
        for robot in plan.robots
    }
    for module in plan.modules:
        offsets = (0.0,) if module.required_team_size == 1 else (0.0, -0.35, 0.35)
        for pose in (module.staging_pose, module.target_pose):
            required.update(
                _world_to_cell(
                    Vec2(x=pose.position.x, y=pose.position.y + offset),
                    plan.site_grid,
                )
                for offset in offsets
            )
    return required


def _required_cells_connected(
    grid: SiteGrid,
    required: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> bool:
    """Check static endpoint reachability without changing router semantics."""
    if not required:
        return True
    if required & obstacles:
        return False
    start = min(required)
    remaining = required - {start}
    frontier = deque([start])
    visited = {start}
    while frontier and remaining:
        x, y = frontier.popleft()
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if (
                0 <= neighbor[0] < grid.width
                and 0 <= neighbor[1] < grid.height
                and neighbor not in obstacles
                and neighbor not in visited
            ):
                visited.add(neighbor)
                remaining.discard(neighbor)
                frontier.append(neighbor)
    return not remaining


def _sample_failures(
    plan: BuildPlan,
    rng: Random,
    config: CottageScenarioConfig,
) -> list[ConstructionFailure]:
    if not config.include_failures or rng.random() >= config.failure_probability:
        return []
    kind = rng.choice(list(FailureKind))
    common = {
        "failure_id": f"failure-{kind.value}",
        "kind": kind,
        "trigger_time_s": float(rng.randint(35, 95)),
        "duration_s": float(rng.randint(12, 30)),
    }
    if kind == FailureKind.ROBOT_UNAVAILABLE:
        common["robot_id"] = rng.choice(plan.robots).robot_id
    elif kind == FailureKind.DROPPED_MODULE:
        common["module_id"] = rng.choice(plan.modules).module_id
    else:
        occupied = set(plan.site_grid.obstacle_cells)
        required = _required_interaction_cells(plan)
        candidates = [
            (x, y)
            for x in range(3, plan.site_grid.width - 3)
            for y in range(3, plan.site_grid.height - 3)
            if (x, y) not in occupied and (x, y) not in required
        ]
        rng.shuffle(candidates)
        for candidate in candidates:
            if _required_cells_connected(plan.site_grid, required, occupied | {candidate}):
                common["obstacle_cell"] = candidate
                break
        else:
            raise ValueError("no feasible cell is available for the obstacle failure")
    return [ConstructionFailure.model_validate(common)]


def _world_to_cell(point: Vec2, grid: SiteGrid) -> tuple[int, int]:
    x = round((point.x - grid.origin.x) / grid.resolution_m)
    y = round((point.y - grid.origin.y) / grid.resolution_m)
    return (
        min(max(x, 0), grid.width - 1),
        min(max(y, 0), grid.height - 1),
    )
