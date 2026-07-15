from __future__ import annotations

from math import atan2, degrees, hypot

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.models import (
    BuildModule,
    BuildPlan,
    Dimensions3D,
    HouseDesign,
    ModuleType,
    Pose3D,
    RobotSpec,
    SiteGrid,
    Vec3,
)


class ConstructionCompileSettings(BaseModel):
    """Deterministic decomposition controls for Construction v2 plans."""

    wall_panel_target_width_m: float = Field(default=2.0, gt=0.5, le=4.0)
    interior_panel_count: int = Field(default=2, ge=0, le=4)
    roof_rows: int = Field(default=2, ge=1, le=4)


def compile_house_design(
    design: HouseDesign,
    settings: ConstructionCompileSettings | None = None,
) -> BuildPlan:
    """Compile an approved metric house into independently transportable modules."""
    if not design.floor_plan.approved:
        raise ValueError("floor plan must be reviewed and approved before compilation")
    settings = settings or ConstructionCompileSettings()

    modules: list[BuildModule] = []
    foundation_ids: list[str] = []
    tile_w = design.footprint_width_m / 2
    tile_d = design.footprint_depth_m / 2
    for row in range(2):
        for col in range(2):
            module_id = f"foundation_{row}_{col}"
            foundation_ids.append(module_id)
            modules.append(
                _module(
                    module_id,
                    ModuleType.FOUNDATION,
                    x=-design.footprint_width_m / 2 + tile_w * (col + 0.5),
                    y=-design.footprint_depth_m / 2 + tile_d * (row + 0.5),
                    z=0.1,
                    width=tile_w - 0.04,
                    depth=tile_d - 0.04,
                    height=0.2,
                    mass=85,
                    team=2,
                    duration=16,
                    dependencies=[],
                    material="foundation_concrete",
                    staging_index=len(modules),
                )
            )

    exterior_ids: list[str] = []
    opening_by_wall = {item.wall_id: item for item in design.floor_plan.openings}
    for wall_index, wall in enumerate(design.floor_plan.walls):
        length = hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
        segment_count = max(1, round(length / settings.wall_panel_target_width_m))
        angle = degrees(atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x))
        for segment in range(segment_count):
            ratio = (segment + 0.5) / segment_count
            x = wall.start.x + (wall.end.x - wall.start.x) * ratio
            y = wall.start.y + (wall.end.y - wall.start.y) * ratio
            opening = opening_by_wall.get(wall.wall_id)
            module_type = ModuleType.WALL
            if opening and segment == segment_count // 2:
                module_type = ModuleType.DOOR if opening.kind == "door" else ModuleType.WINDOW
            module_id = f"{wall.wall_id}_{segment:02d}"
            exterior_ids.append(module_id)
            modules.append(
                _module(
                    module_id,
                    module_type,
                    x=x,
                    y=y,
                    z=wall.height_m / 2 + 0.2,
                    width=length / segment_count - 0.04,
                    depth=wall.thickness_m,
                    height=wall.height_m,
                    mass=32 if module_type == ModuleType.WALL else 26,
                    team=1,
                    duration=11,
                    dependencies=[foundation_ids[wall_index % len(foundation_ids)]],
                    material="plaster_white",
                    staging_index=len(modules),
                    yaw=angle,
                )
            )

    interior_ids: list[str] = []
    interior_x_positions = _interior_x_positions(
        design.footprint_width_m,
        settings.interior_panel_count,
    )
    for index, x in enumerate(interior_x_positions):
        module_id = f"interior_panel_{index}"
        interior_ids.append(module_id)
        modules.append(
            _module(
                module_id,
                ModuleType.INTERIOR,
                x=x,
                y=0,
                z=1.6,
                width=design.footprint_depth_m - 1.0,
                depth=0.14,
                height=2.8,
                mass=28,
                team=1,
                duration=10,
                dependencies=[
                    foundation_ids[index % len(foundation_ids)],
                    foundation_ids[(index + 1) % len(foundation_ids)],
                ],
                material="interior_white",
                staging_index=len(modules),
                yaw=90,
            )
        )

    support_ids = exterior_ids + interior_ids
    roof_width = design.footprint_width_m / 2 + design.roof.overhang_m
    roof_row_depth = design.footprint_depth_m / settings.roof_rows
    roof_y_positions = [
        -design.footprint_depth_m / 2 + roof_row_depth * (index + 0.5)
        for index in range(settings.roof_rows)
    ]
    for index, y in enumerate(roof_y_positions):
        for side, x in enumerate((-design.footprint_width_m / 4, design.footprint_width_m / 4)):
            module_id = f"roof_{index}_{side}"
            modules.append(
                _module(
                    module_id,
                    ModuleType.ROOF,
                    x=x,
                    y=y,
                    z=3.35,
                    width=roof_width,
                    depth=roof_row_depth + design.roof.overhang_m,
                    height=0.18,
                    mass=62,
                    team=2,
                    duration=18,
                    dependencies=support_ids,
                    material=design.roof_material,
                    staging_index=len(modules),
                    pitch=(-design.roof.pitch_degrees if side == 0 else design.roof.pitch_degrees),
                )
            )

    robots = [
        RobotSpec(
            robot_id=f"robot_{index + 1}",
            role="heavy" if index < 2 else "precision",
            payload_capacity_kg=55 if index < 2 else 38,
            speed_mps=1.0 + index * 0.05,
            battery_capacity_wh=900,
            start_pose=Pose3D(position=Vec3(x=-7, y=-2.4 + index * 1.6, z=0)),
        )
        for index in range(4)
    ]
    return BuildPlan(
        plan_id=f"{design.design_id}_build_plan_v2",
        design=design.model_copy(deep=True),
        modules=modules,
        robots=robots,
        site_grid=SiteGrid(width=34, height=26),
    )


def _interior_x_positions(width_m: float, count: int) -> list[float]:
    if count == 0:
        return []
    if count == 2 and abs(width_m - 8.0) < 1e-9:
        return [-1.35, 1.35]
    usable_width = max(width_m - 2.0, 0.5)
    spacing = usable_width / (count + 1)
    return [-usable_width / 2 + spacing * (index + 1) for index in range(count)]


def _module(
    module_id: str,
    module_type: ModuleType,
    *,
    x: float,
    y: float,
    z: float,
    width: float,
    depth: float,
    height: float,
    mass: float,
    team: int,
    duration: int,
    dependencies: list[str],
    material: str,
    staging_index: int,
    yaw: float = 0,
    pitch: float = 0,
) -> BuildModule:
    staging_row, staging_col = divmod(staging_index, 7)
    return BuildModule(
        module_id=module_id,
        module_type=module_type,
        mesh_node=f"module__{module_id}",
        target_pose=Pose3D(
            position=Vec3(x=x, y=y, z=z),
            rotation_rpy_degrees=Vec3(x=0, y=pitch, z=yaw),
        ),
        staging_pose=Pose3D(
            position=Vec3(x=-6.2 + staging_col * 0.65, y=-4.5 + staging_row * 0.8, z=height / 2),
        ),
        dimensions=Dimensions3D(width=width, depth=depth, height=height),
        mass_kg=mass,
        grip_points=[Vec3(x=-width / 4, y=0, z=0), Vec3(x=width / 4, y=0, z=0)],
        required_team_size=team,
        install_duration_s=duration,
        dependencies=list(dependencies),
        material=material,
    )
