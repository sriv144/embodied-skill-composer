from __future__ import annotations

from pathlib import Path

from embodied_skill_composer.assembly.models import (
    AssemblyScenarioConfig,
    AssetCatalog,
    BeamTask,
    BlueprintSlot,
    CompiledBlueprint,
    ConstructionResource,
    GridCoord,
    ModularBlueprint,
)


class BlueprintCompilationError(ValueError):
    """Raised when a modular blueprint cannot become an executable scenario."""


def validate_asset_catalog(catalog: AssetCatalog, workspace_root: Path) -> None:
    unknown_sources = {
        item.source for item in catalog.components.values() if item.source not in catalog.sources
    }
    unknown_sources.update(
        item.source for item in catalog.robots.values() if item.source not in catalog.sources
    )
    if unknown_sources:
        raise BlueprintCompilationError(
            f"asset catalog references unknown sources: {sorted(unknown_sources)}"
        )

    for asset_key, component in catalog.components.items():
        mesh_path = (workspace_root / component.visual_mesh).resolve()
        if not mesh_path.is_file():
            raise BlueprintCompilationError(
                f"asset '{asset_key}' mesh does not exist: {mesh_path}"
            )
        if mesh_path.suffix.lower() == ".obj" and not mesh_path.with_suffix(".mtl").is_file():
            raise BlueprintCompilationError(
                f"asset '{asset_key}' is missing its MTL file: {mesh_path.with_suffix('.mtl')}"
            )


def compile_modular_blueprint(
    blueprint: ModularBlueprint,
    catalog: AssetCatalog,
    *,
    workspace_root: Path,
) -> CompiledBlueprint:
    validate_asset_catalog(catalog, workspace_root)
    _validate_unique_ids(blueprint)
    _validate_geometry(blueprint)

    materials = {item.resource_id: item for item in blueprint.materials}
    components = {item.component_id: item for item in blueprint.components}
    for component in blueprint.components:
        material = materials.get(component.required_material_id)
        if material is None:
            raise BlueprintCompilationError(
                f"component '{component.component_id}' requires unknown material "
                f"'{component.required_material_id}'"
            )
        if component.asset_key not in catalog.components:
            raise BlueprintCompilationError(
                f"component '{component.component_id}' references unknown asset "
                f"'{component.asset_key}'"
            )
        if material.asset_key not in catalog.components:
            raise BlueprintCompilationError(
                f"material '{material.resource_id}' references unknown asset "
                f"'{material.asset_key}'"
            )
        if material.component_type != component.component_type:
            raise BlueprintCompilationError(
                f"component '{component.component_id}' type '{component.component_type}' "
                f"does not match material type '{material.component_type}'"
            )
        if material.asset_key != component.asset_key:
            raise BlueprintCompilationError(
                f"component '{component.component_id}' and material "
                f"'{material.resource_id}' must use the same asset"
            )
        if component.required_team_size != len(blueprint.agent_starts):
            raise BlueprintCompilationError(
                f"component '{component.component_id}' requires team size "
                f"{component.required_team_size}; Modular Room v0 requires "
                f"{len(blueprint.agent_starts)}"
            )
        unknown_dependencies = sorted(set(component.depends_on) - components.keys())
        if unknown_dependencies:
            raise BlueprintCompilationError(
                f"component '{component.component_id}' has unknown dependencies: "
                f"{unknown_dependencies}"
            )

    installation_order = _stable_topological_order(blueprint)
    ordered_components = [components[item] for item in installation_order]
    beams: list[BeamTask] = []
    resources: list[ConstructionResource] = []
    slots: list[BlueprintSlot] = []
    component_to_resource: dict[str, str] = {}
    for component in ordered_components:
        material = materials[component.required_material_id]
        component_to_resource[component.component_id] = material.resource_id
        beams.append(
            BeamTask(
                name=material.resource_id,
                pickup_left=material.source_cells[0],
                pickup_right=material.source_cells[1],
                assembly_left=component.target_cells[0],
                assembly_right=component.target_cells[1],
            )
        )
        slot_id = f"{component.component_id}_slot"
        resources.append(
            ConstructionResource(
                resource_id=material.resource_id,
                resource_type=material.component_type,
                source_cells=list(material.source_cells),
                assigned_slot_id=slot_id,
                component_id=component.component_id,
                asset_key=material.asset_key,
                source_pose=material.source_pose,
                assigned_robot_ids=list(range(len(blueprint.agent_starts))),
            )
        )
        slots.append(
            BlueprintSlot(
                slot_id=slot_id,
                resource_type=component.component_type,
                target_cells=list(component.target_cells),
                required_resource_id=material.resource_id,
                component_id=component.component_id,
                asset_key=component.asset_key,
                target_pose=component.target_pose,
                depends_on=list(component.depends_on),
                required_team_size=component.required_team_size,
            )
        )

    scenario = AssemblyScenarioConfig(
        blueprint_id=blueprint.blueprint_id,
        installation_order=installation_order,
        grid_size=blueprint.grid_size,
        max_steps=blueprint.max_steps,
        agent_starts=list(blueprint.agent_starts),
        beams=beams,
        obstacle_cells=list(blueprint.obstacle_cells),
        resources=resources,
        blueprint_slots=slots,
        curriculum_beam_stages=[len(beams)],
        option_max_primitive_steps=blueprint.option_max_primitive_steps,
    )
    return CompiledBlueprint(
        blueprint=blueprint.model_copy(deep=True),
        scenario=scenario,
        installation_order=installation_order,
        component_to_resource=component_to_resource,
    )


def _validate_unique_ids(blueprint: ModularBlueprint) -> None:
    material_ids = [item.resource_id for item in blueprint.materials]
    component_ids = [item.component_id for item in blueprint.components]
    if len(material_ids) != len(set(material_ids)):
        raise BlueprintCompilationError("blueprint material IDs must be unique")
    if len(component_ids) != len(set(component_ids)):
        raise BlueprintCompilationError("blueprint component IDs must be unique")
    required_material_ids = [item.required_material_id for item in blueprint.components]
    if len(required_material_ids) != len(set(required_material_ids)):
        raise BlueprintCompilationError(
            "Modular Room v0 requires one distinct material per component"
        )
    unused_materials = sorted(set(material_ids) - set(required_material_ids))
    if unused_materials:
        raise BlueprintCompilationError(
            f"blueprint contains unused materials: {unused_materials}"
        )


def _validate_geometry(blueprint: ModularBlueprint) -> None:
    groups: list[tuple[str, list[GridCoord]]] = [
        ("agent starts", blueprint.agent_starts),
        ("obstacles", blueprint.obstacle_cells),
    ]
    groups.extend(
        (f"material '{item.resource_id}'", item.source_cells)
        for item in blueprint.materials
    )
    groups.extend(
        (f"component '{item.component_id}'", item.target_cells)
        for item in blueprint.components
    )
    occupied: dict[GridCoord, str] = {}
    for label, cells in groups:
        if len(cells) != len(set(cells)):
            raise BlueprintCompilationError(f"{label} contains duplicate cells")
        for cell in cells:
            if not (0 <= cell[0] < blueprint.grid_size and 0 <= cell[1] < blueprint.grid_size):
                raise BlueprintCompilationError(
                    f"{label} cell {cell} is outside grid size {blueprint.grid_size}"
                )
            previous = occupied.get(cell)
            if previous is not None:
                raise BlueprintCompilationError(
                    f"{label} cell {cell} overlaps {previous}"
                )
            occupied[cell] = label


def _stable_topological_order(blueprint: ModularBlueprint) -> list[str]:
    declaration_order = [item.component_id for item in blueprint.components]
    dependencies = {
        item.component_id: set(item.depends_on) for item in blueprint.components
    }
    order: list[str] = []
    completed: set[str] = set()
    while len(order) < len(declaration_order):
        ready = [
            component_id
            for component_id in declaration_order
            if component_id not in completed
            and dependencies[component_id].issubset(completed)
        ]
        if not ready:
            unresolved = [item for item in declaration_order if item not in completed]
            raise BlueprintCompilationError(
                f"blueprint dependency graph contains a cycle: {unresolved}"
            )
        next_component = ready[0]
        order.append(next_component)
        completed.add(next_component)
    return order
