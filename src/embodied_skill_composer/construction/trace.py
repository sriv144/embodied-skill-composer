from __future__ import annotations

from embodied_skill_composer.construction.models import (
    BrainEvent,
    BuildPlan,
    ConstructionMetrics,
    ConstructionSchedule,
    ExecutionFrame,
    ExecutionTrace,
    ModuleTraceState,
    RobotTraceState,
    Vec3,
)


def build_execution_trace(
    plan: BuildPlan,
    schedule: ConstructionSchedule,
    *,
    frame_interval_s: int = 2,
) -> ExecutionTrace:
    modules = {item.module_id: item for item in plan.modules}
    jobs = {item.module_id: item for item in schedule.jobs}
    frames: list[ExecutionFrame] = []
    for timestamp in range(0, schedule.makespan_s + frame_interval_s, frame_interval_s):
        completed = sorted(item.module_id for item in schedule.jobs if item.end_s <= timestamp)
        robot_states: list[RobotTraceState] = []
        for robot in plan.robots:
            active = next(
                (
                    item
                    for item in schedule.jobs
                    if robot.robot_id in item.robot_ids and item.start_s <= timestamp < item.end_s
                ),
                None,
            )
            if active is None:
                robot_states.append(
                    RobotTraceState(
                        robot_id=robot.robot_id,
                        position=robot.start_pose.position,
                        status="idle",
                    )
                )
                continue
            module = modules[active.module_id]
            progress = (timestamp - active.start_s) / max(active.end_s - active.start_s, 1)
            if timestamp < active.pickup_s:
                position = module.staging_pose.position
                status = "moving"
            elif progress < 0.78:
                position = _interpolate_route(active.route, (progress - 0.2) / 0.58)
                status = "carrying"
            else:
                position = module.target_pose.position
                status = "installing"
            robot_states.append(
                RobotTraceState(
                    robot_id=robot.robot_id,
                    position=position,
                    status=status,
                    module_id=module.module_id,
                )
            )

        module_states = []
        for module in plan.modules:
            job = jobs[module.module_id]
            if timestamp >= job.end_s:
                status = "installed"
                position = module.target_pose.position
            elif timestamp >= job.pickup_s:
                status = "in_transit"
                position = _interpolate_route(
                    job.route,
                    (timestamp - job.pickup_s) / max(job.end_s - job.pickup_s, 1),
                )
            else:
                status = "staged"
                position = module.staging_pose.position
            module_states.append(
                ModuleTraceState(module_id=module.module_id, status=status, position=position)
            )
        frames.append(
            ExecutionFrame(
                timestamp_s=min(timestamp, schedule.makespan_s),
                robots=robot_states,
                modules=module_states,
                completed_module_ids=completed,
            )
        )

    events = _brain_events(plan, schedule)
    busy_by_robot = {
        robot.robot_id: sum(
            job.end_s - job.start_s
            for job in schedule.jobs
            if robot.robot_id in job.robot_ids
        )
        for robot in plan.robots
    }
    metrics = ConstructionMetrics(
        controller=schedule.controller,
        structure_completion_rate=1.0,
        makespan_s=schedule.makespan_s,
        total_travel_m=schedule.total_travel_m,
        total_energy_wh=schedule.total_energy_wh,
        idle_robot_seconds=schedule.idle_robot_seconds,
        robot_utilization={
            key: round(value / schedule.makespan_s, 4) for key, value in busy_by_robot.items()
        },
    )
    return ExecutionTrace(
        plan_id=plan.plan_id,
        schedule=schedule,
        frames=frames,
        brain_events=events,
        metrics=metrics,
    )


def _brain_events(plan: BuildPlan, schedule: ConstructionSchedule) -> list[BrainEvent]:
    modules = {item.module_id: item for item in plan.modules}
    ordered = sorted(schedule.jobs, key=lambda item: (item.start_s, item.module_id))
    events: list[BrainEvent] = []
    for job in ordered:
        completed = {item.module_id for item in ordered if item.end_s <= job.start_s}
        candidates = sorted(
            module.module_id
            for module in plan.modules
            if module.module_id not in completed
            and set(module.dependencies) <= completed
        )
        events.append(
            BrainEvent(
                timestamp_s=job.start_s,
                event_type="assignment",
                module_id=job.module_id,
                robot_ids=job.robot_ids,
                candidates=candidates,
                reason=(
                    "Selected from the precedence-ready set to protect the critical path."
                    if job.critical
                    else "Selected because prerequisites and robot capacity are satisfied."
                ),
                predicted_remaining_s=schedule.makespan_s - job.start_s,
            )
        )
        blocked = sorted(
            item.module_id
            for item in plan.modules
            if item.module_id not in completed
            and item.module_id != job.module_id
            and not set(item.dependencies) <= completed
        )
        if blocked:
            rejected = blocked[0]
            missing = sorted(set(modules[rejected].dependencies) - completed)
            events.append(
                BrainEvent(
                    timestamp_s=job.start_s,
                    event_type="rejection",
                    module_id=rejected,
                    candidates=candidates,
                    reason=f"Blocked by incomplete prerequisites: {', '.join(missing)}.",
                    predicted_remaining_s=schedule.makespan_s - job.start_s,
                )
            )
    events.append(
        BrainEvent(
            timestamp_s=schedule.makespan_s,
            event_type="completion",
            reason="All modules reached their approved metric target poses.",
            predicted_remaining_s=0,
        )
    )
    return events


def _interpolate_route(route: list[object], progress: float) -> Vec3:
    progress = max(0.0, min(1.0, progress))
    if len(route) < 2:
        point = route[0]
        return Vec3(x=point.x, y=point.y, z=0.25)
    segment_progress = progress * (len(route) - 1)
    index = min(int(segment_progress), len(route) - 2)
    local = segment_progress - index
    start, end = route[index], route[index + 1]
    return Vec3(
        x=start.x + (end.x - start.x) * local,
        y=start.y + (end.y - start.y) * local,
        z=0.25,
    )
