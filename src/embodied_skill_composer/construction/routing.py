from __future__ import annotations

from collections.abc import Mapping
from heapq import heappop, heappush
from importlib.util import find_spec
from itertools import count
from math import hypot

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.models import SiteGrid, Vec2


Cell = tuple[int, int]


class RoutingError(RuntimeError):
    pass


class RoutePlan(BaseModel):
    backend: str
    cell_paths: dict[str, list[Cell]]
    world_paths: dict[str, list[Vec2]]
    total_distance_m: float = Field(ge=0)
    conflict_count: int = Field(ge=0)
    fallback_reason: str | None = None


class RoutingAdapter:
    def route_many(
        self,
        grid: SiteGrid,
        starts: Mapping[str, Vec2],
        goals: Mapping[str, Vec2],
    ) -> RoutePlan:
        raise NotImplementedError


class PrioritizedRoutingAdapter(RoutingAdapter):
    """Deterministic reservation-table planner used when w9 is unavailable."""

    def __init__(self, *, backend_name: str = "prioritized_astar") -> None:
        self.backend_name = backend_name

    def route_many(
        self,
        grid: SiteGrid,
        starts: Mapping[str, Vec2],
        goals: Mapping[str, Vec2],
    ) -> RoutePlan:
        _validate_route_request(starts, goals)
        obstacles = set(grid.obstacle_cells)
        vertex_reservations: set[tuple[Cell, int]] = set()
        edge_reservations: set[tuple[Cell, Cell, int]] = set()
        paths: dict[str, list[Cell]] = {}
        max_length = max(grid.width * grid.height * 2, 64)
        for agent_id in sorted(starts):
            start = world_to_cell(starts[agent_id], grid)
            goal = world_to_cell(goals[agent_id], grid)
            path = _reserved_astar(
                grid,
                start,
                goal,
                obstacles,
                vertex_reservations,
                edge_reservations,
                max_length=max_length,
            )
            paths[agent_id] = path
            _reserve_path(
                path,
                vertex_reservations,
                edge_reservations,
                hold_until=max_length,
            )
        return _route_plan(self.backend_name, grid, paths)


class W9RoutingAdapter(RoutingAdapter):
    """Conflict-based routing through the optional w9-pathfinding extension."""

    def __init__(self, *, seed: int = 0, max_time_s: float = 2.0) -> None:
        self.seed = seed
        self.max_time_s = max_time_s

    def route_many(
        self,
        grid: SiteGrid,
        starts: Mapping[str, Vec2],
        goals: Mapping[str, Vec2],
    ) -> RoutePlan:
        _validate_route_request(starts, goals)
        try:
            from w9_pathfinding.envs import Grid
            from w9_pathfinding.mapf import CBS
        except ImportError as exc:
            raise RoutingError("w9-pathfinding is not installed") from exc

        agent_ids = sorted(starts)
        environment = Grid(width=grid.width, height=grid.height, edge_collision=True)
        for cell in grid.obstacle_cells:
            environment.add_obstacle(cell)
        solver = CBS(environment, seed=self.seed)
        paths = solver.mapf(
            [world_to_cell(starts[item], grid) for item in agent_ids],
            [world_to_cell(goals[item], grid) for item in agent_ids],
            max_length=max(grid.width * grid.height * 2, 64),
            max_time=self.max_time_s,
            disjoint_splitting=True,
        )
        if not paths or len(paths) != len(agent_ids):
            raise RoutingError("w9 CBS did not find a complete route set")
        mapped = {
            agent_id: [tuple(int(value) for value in cell) for cell in path]
            for agent_id, path in zip(agent_ids, paths, strict=True)
        }
        return _route_plan("w9_cbs", grid, mapped)


class PreferredRoutingAdapter(RoutingAdapter):
    def __init__(self, *, seed: int = 0) -> None:
        self.preferred = W9RoutingAdapter(seed=seed)
        self.fallback = PrioritizedRoutingAdapter()

    def route_many(
        self,
        grid: SiteGrid,
        starts: Mapping[str, Vec2],
        goals: Mapping[str, Vec2],
    ) -> RoutePlan:
        if find_spec("w9_pathfinding") is not None:
            try:
                return self.preferred.route_many(grid, starts, goals)
            except (RoutingError, RuntimeError, ValueError) as exc:
                result = self.fallback.route_many(grid, starts, goals)
                result.fallback_reason = f"w9 routing failed: {exc}"
                return result
        result = self.fallback.route_many(grid, starts, goals)
        result.fallback_reason = (
            "w9-pathfinding 0.1.3 is unavailable; Windows source builds require "
            "Visual Studio 2022 Build Tools with MSVC and the Windows SDK"
        )
        return result


def create_routing_adapter(*, seed: int = 0, prefer_w9: bool = True) -> RoutingAdapter:
    if prefer_w9:
        return PreferredRoutingAdapter(seed=seed)
    return PrioritizedRoutingAdapter()


def world_to_cell(point: Vec2, grid: SiteGrid) -> Cell:
    cell = (
        round((point.x - grid.origin.x) / grid.resolution_m),
        round((point.y - grid.origin.y) / grid.resolution_m),
    )
    if not 0 <= cell[0] < grid.width or not 0 <= cell[1] < grid.height:
        raise RoutingError(f"world point ({point.x}, {point.y}) is outside the site grid")
    return cell


def cell_to_world(cell: Cell, grid: SiteGrid) -> Vec2:
    return Vec2(
        x=grid.origin.x + cell[0] * grid.resolution_m,
        y=grid.origin.y + cell[1] * grid.resolution_m,
    )


def count_path_conflicts(paths: Mapping[str, list[Cell]]) -> int:
    if not paths:
        return 0
    horizon = max(len(path) for path in paths.values())
    conflicts = 0
    agent_ids = sorted(paths)
    for step in range(horizon):
        positions = {
            agent: paths[agent][min(step, len(paths[agent]) - 1)] for agent in agent_ids
        }
        for left_index, left in enumerate(agent_ids):
            for right in agent_ids[left_index + 1 :]:
                if positions[left] == positions[right]:
                    conflicts += 1
                if step > 0:
                    left_previous = paths[left][min(step - 1, len(paths[left]) - 1)]
                    right_previous = paths[right][min(step - 1, len(paths[right]) - 1)]
                    if left_previous == positions[right] and right_previous == positions[left]:
                        conflicts += 1
    return conflicts


def _validate_route_request(
    starts: Mapping[str, Vec2],
    goals: Mapping[str, Vec2],
) -> None:
    if not starts:
        raise RoutingError("at least one route is required")
    if set(starts) != set(goals):
        raise RoutingError("route starts and goals must contain the same agent IDs")


def _reserved_astar(
    grid: SiteGrid,
    start: Cell,
    goal: Cell,
    obstacles: set[Cell],
    vertex_reservations: set[tuple[Cell, int]],
    edge_reservations: set[tuple[Cell, Cell, int]],
    *,
    max_length: int,
) -> list[Cell]:
    if start in obstacles or goal in obstacles:
        raise RoutingError(f"route endpoint is blocked: {start} -> {goal}")
    serial = count()
    frontier: list[tuple[int, int, int, Cell, list[Cell]]] = []
    heappush(frontier, (_manhattan(start, goal), 0, next(serial), start, [start]))
    best: dict[tuple[Cell, int], int] = {(start, 0): 0}
    while frontier:
        _, elapsed, _, cell, path = heappop(frontier)
        if cell == goal and not any(
            (goal, future_time) in vertex_reservations
            for future_time in range(elapsed, max_length + 1)
        ):
            return path
        if elapsed >= max_length:
            continue
        next_time = elapsed + 1
        neighbors = [cell, *_neighbors(cell, grid)]
        for neighbor in neighbors:
            if neighbor in obstacles:
                continue
            if (neighbor, next_time) in vertex_reservations:
                continue
            if (neighbor, cell, next_time) in edge_reservations:
                continue
            state = (neighbor, next_time)
            if best.get(state, max_length + 1) <= next_time:
                continue
            best[state] = next_time
            priority = next_time + _manhattan(neighbor, goal)
            heappush(
                frontier,
                (priority, next_time, next(serial), neighbor, [*path, neighbor]),
            )
    raise RoutingError(f"no reserved path found: {start} -> {goal}")


def _neighbors(cell: Cell, grid: SiteGrid) -> list[Cell]:
    x, y = cell
    return [
        item
        for item in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= item[0] < grid.width and 0 <= item[1] < grid.height
    ]


def _reserve_path(
    path: list[Cell],
    vertex_reservations: set[tuple[Cell, int]],
    edge_reservations: set[tuple[Cell, Cell, int]],
    *,
    hold_until: int,
) -> None:
    for time_index, cell in enumerate(path):
        vertex_reservations.add((cell, time_index))
        if time_index:
            edge_reservations.add((path[time_index - 1], cell, time_index))
    for time_index in range(len(path), hold_until + 1):
        vertex_reservations.add((path[-1], time_index))


def _route_plan(backend: str, grid: SiteGrid, paths: dict[str, list[Cell]]) -> RoutePlan:
    world_paths = {
        agent_id: [cell_to_world(cell, grid) for cell in path]
        for agent_id, path in paths.items()
    }
    total_distance = sum(
        hypot(right.x - left.x, right.y - left.y)
        for path in world_paths.values()
        for left, right in zip(path, path[1:], strict=False)
    )
    return RoutePlan(
        backend=backend,
        cell_paths=paths,
        world_paths=world_paths,
        total_distance_m=total_distance,
        conflict_count=count_path_conflicts(paths),
    )


def _manhattan(left: Cell, right: Cell) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
