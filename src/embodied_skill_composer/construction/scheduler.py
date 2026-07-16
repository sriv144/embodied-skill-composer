from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import ceil, hypot

from embodied_skill_composer.construction.models import (
    BuildModule,
    BuildPlan,
    ConstructionSchedule,
    Pose3D,
    ScheduledJob,
    Vec2,
)


def schedule_build(plan: BuildPlan, controller: str) -> ConstructionSchedule:
    if controller == "sequential":
        return _schedule_sequential(plan)
    if controller == "greedy":
        return _schedule_greedy(plan)
    if controller == "optimized":
        return _schedule_optimized(plan)
    raise ValueError(f"unknown construction controller: {controller}")


def compare_controllers(plan: BuildPlan) -> dict[str, ConstructionSchedule]:
    return {
        name: schedule_build(plan, name)
        for name in ("sequential", "greedy", "optimized")
    }


def _schedule_sequential(plan: BuildPlan) -> ConstructionSchedule:
    modules = {item.module_id: item for item in plan.modules}
    end_by_module: dict[str, int] = {}
    jobs: list[ScheduledJob] = []
    cursor = 0
    for module_id in _topological_order(plan.modules):
        module = modules[module_id]
        robots = _eligible_groups(plan, module)[0]
        start = max(cursor, *(end_by_module[item] for item in module.dependencies), 0)
        duration, distance = _duration_and_distance(module, robots)
        job = _job(module, robots, start, duration, distance)
        jobs.append(job)
        cursor = job.end_s
        end_by_module[module_id] = job.end_s
    return _finish_schedule(plan, "sequential", jobs, "deterministic_topological")


def _schedule_greedy(plan: BuildPlan) -> ConstructionSchedule:
    pending = {item.module_id: item for item in plan.modules}
    completed: set[str] = set()
    robot_free = {robot.robot_id: 0 for robot in plan.robots}
    jobs: list[ScheduledJob] = []
    now = 0
    while pending:
        for job in jobs:
            if job.end_s <= now:
                completed.add(job.module_id)
        ready = sorted(
            (item for item in pending.values() if set(item.dependencies) <= completed),
            key=lambda item: (-_downstream_depth(item.module_id, plan.modules), item.module_id),
        )
        assigned = False
        for module in ready:
            group = next(
                (
                    group
                    for group in _eligible_groups(plan, module)
                    if all(robot_free[item.robot_id] <= now for item in group)
                ),
                None,
            )
            if group is None:
                continue
            duration, distance = _duration_and_distance(module, group)
            job = _job(module, group, now, duration, distance)
            jobs.append(job)
            for robot in group:
                robot_free[robot.robot_id] = job.end_s
            pending.pop(module.module_id)
            assigned = True
        if not assigned:
            future = [value for value in robot_free.values() if value > now]
            if not future:
                raise RuntimeError("greedy scheduler reached an infeasible state")
            now = min(future)
    return _finish_schedule(plan, "greedy", jobs, "critical_ready_list")


def _schedule_optimized(plan: BuildPlan) -> ConstructionSchedule:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        fallback = _schedule_greedy(plan)
        return fallback.model_copy(
            update={"controller": "optimized", "solver_status": "greedy_fallback_ortools_missing"}
        )

    model = cp_model.CpModel()
    horizon = sum(
        max(_duration_and_distance(module, group)[0] for group in _eligible_groups(plan, module))
        for module in plan.modules
    )
    starts: dict[str, object] = {}
    ends: dict[str, object] = {}
    selections: dict[tuple[str, int], object] = {}
    group_data: dict[str, list[tuple[list[object], int, float]]] = {}
    intervals_by_robot: dict[str, list[object]] = defaultdict(list)

    for module in plan.modules:
        starts[module.module_id] = model.new_int_var(0, horizon, f"start_{module.module_id}")
        ends[module.module_id] = model.new_int_var(0, horizon, f"end_{module.module_id}")
        options: list[tuple[list[object], int, float]] = []
        choice_vars = []
        for index, group in enumerate(_eligible_groups(plan, module)):
            duration, distance = _duration_and_distance(module, group)
            selected = model.new_bool_var(f"select_{module.module_id}_{index}")
            interval = model.new_optional_interval_var(
                starts[module.module_id],
                duration,
                ends[module.module_id],
                selected,
                f"interval_{module.module_id}_{index}",
            )
            selections[(module.module_id, index)] = selected
            choice_vars.append(selected)
            options.append((list(group), duration, distance))
            for robot in group:
                intervals_by_robot[robot.robot_id].append(interval)
        model.add_exactly_one(choice_vars)
        group_data[module.module_id] = options

    for module in plan.modules:
        for dependency in module.dependencies:
            model.add(starts[module.module_id] >= ends[dependency])
    for intervals in intervals_by_robot.values():
        model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(ends.values()))
    assignment_cost = sum(
        int(distance * 10) * selections[(module_id, index)]
        for module_id, options in group_data.items()
        for index, (_, _, distance) in enumerate(options)
    )
    model.minimize(makespan * 100_000 + assignment_cost)
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 1
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT could not schedule build: {solver.status_name(status)}")

    modules = {item.module_id: item for item in plan.modules}
    jobs: list[ScheduledJob] = []
    for module_id, options in group_data.items():
        selected_index = next(
            index
            for index in range(len(options))
            if solver.boolean_value(selections[(module_id, index)])
        )
        group, duration, distance = options[selected_index]
        jobs.append(
            _job(
                modules[module_id],
                group,
                solver.value(starts[module_id]),
                duration,
                distance,
            )
        )
    return _finish_schedule(
        plan,
        "optimized",
        jobs,
        f"cp_sat_{solver.status_name(status).lower()}",
    )


def _eligible_groups(plan: BuildPlan, module: BuildModule) -> list[list[object]]:
    groups = []
    for group in combinations(plan.robots, module.required_team_size):
        if sum(robot.payload_capacity_kg for robot in group) >= module.mass_kg:
            groups.append(list(group))
    groups.sort(
        key=lambda group: (
            -sum(robot.speed_mps for robot in group),
            tuple(robot.robot_id for robot in group),
        )
    )
    if not groups:
        raise ValueError(f"no eligible robot group for {module.module_id}")
    return groups


def _duration_and_distance(module: BuildModule, robots: list[object]) -> tuple[int, float]:
    staging = module.staging_pose.position
    target = module.target_pose.position
    distance = hypot(target.x - staging.x, target.y - staging.y)
    speed = min(robot.speed_mps for robot in robots)
    return 4 + ceil(distance / speed) + module.install_duration_s, distance


def _job(
    module: BuildModule,
    robots: list[object],
    start: int,
    duration: int,
    distance: float,
) -> ScheduledJob:
    route = _orthogonal_route(module.staging_pose, module.target_pose)
    return ScheduledJob(
        module_id=module.module_id,
        robot_ids=[item.robot_id for item in robots],
        start_s=start,
        pickup_s=start + 4,
        end_s=start + duration,
        travel_distance_m=round(distance, 3),
        route=route,
    )


def _orthogonal_route(source: Pose3D, target: Pose3D) -> list[Vec2]:
    return [
        Vec2(x=source.position.x, y=source.position.y),
        Vec2(x=target.position.x, y=source.position.y),
        Vec2(x=target.position.x, y=target.position.y),
    ]


def _finish_schedule(
    plan: BuildPlan,
    controller: str,
    jobs: list[ScheduledJob],
    solver_status: str,
) -> ConstructionSchedule:
    jobs.sort(key=lambda item: (item.start_s, item.module_id))
    critical_path = _critical_path(plan.modules)
    jobs = [
        item.model_copy(update={"critical": item.module_id in critical_path}) for item in jobs
    ]
    makespan = max(item.end_s for item in jobs)
    busy = sum((item.end_s - item.start_s) * len(item.robot_ids) for item in jobs)
    total_travel = sum(item.travel_distance_m * len(item.robot_ids) for item in jobs)
    return ConstructionSchedule(
        controller=controller,
        jobs=jobs,
        makespan_s=makespan,
        total_travel_m=round(total_travel, 3),
        total_energy_wh=round(busy * 0.065 + total_travel * 0.12, 3),
        idle_robot_seconds=makespan * len(plan.robots) - busy,
        solver_status=solver_status,
        critical_path=critical_path,
    )


def _topological_order(modules: list[BuildModule]) -> list[str]:
    remaining = {item.module_id: set(item.dependencies) for item in modules}
    order: list[str] = []
    while remaining:
        ready = sorted(key for key, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError("cyclic module graph")
        order.extend(ready)
        for key in ready:
            remaining.pop(key)
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def _downstream_depth(module_id: str, modules: list[BuildModule]) -> int:
    children = [item.module_id for item in modules if module_id in item.dependencies]
    return 1 + max((_downstream_depth(item, modules) for item in children), default=0)


def _critical_path(modules: list[BuildModule]) -> list[str]:
    by_id = {item.module_id: item for item in modules}
    score: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    for module_id in _topological_order(modules):
        module = by_id[module_id]
        if not module.dependencies:
            score[module_id] = module.install_duration_s
            parent[module_id] = None
            continue
        predecessor = max(module.dependencies, key=lambda item: score[item])
        score[module_id] = score[predecessor] + module.install_duration_s
        parent[module_id] = predecessor
    current = max(score, key=score.get)
    path: list[str] = []
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))
