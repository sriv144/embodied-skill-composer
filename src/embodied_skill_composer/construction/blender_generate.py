"""Blender-side deterministic modular cottage generator.

Run only through Blender's Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


COLORS = {
    "foundation_concrete": (0.24, 0.27, 0.28, 1),
    "plaster_white": (0.82, 0.84, 0.80, 1),
    "interior_white": (0.91, 0.90, 0.86, 1),
    "standing_seam_charcoal": (0.09, 0.12, 0.13, 1),
    "timber": (0.42, 0.20, 0.08, 1),
    "glass": (0.08, 0.40, 0.48, 0.42),
    "accent": (0.94, 0.30, 0.08, 1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    materials = {key: _material(key, value) for key, value in COLORS.items()}
    roots = []
    for module in plan["modules"]:
        roots.append(_build_module(module, materials))
    _add_site(plan, materials)
    camera = _add_camera(plan)
    _add_lighting()
    _render(camera, args.output / "assembled_preview.png")
    assembled_positions = {root.name: root.location.copy() for root in roots}
    for index, root in enumerate(roots):
        angle = index * 2.39996
        distance = 0.7 + 0.22 * index
        root.location.x += math.cos(angle) * distance
        root.location.y += math.sin(angle) * distance
        root.location.z += 0.09 * (index % 5)
    _render(camera, args.output / "exploded_modules.png")
    for root in roots:
        root.location = assembled_positions[root.name]
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "house.blend"))
    bpy.ops.export_scene.gltf(
        filepath=str(args.output / "house.glb"),
        export_format="GLB",
        export_yup=True,
    )
    manifest = {
        "generator": "embodied_skill_composer.construction.blender_generate",
        "module_nodes": [root.name for root in roots],
        "module_count": len(roots),
        "units": "meters",
    }
    (args.output / "geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    bpy.context.scene.render.engine = (
        "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    )
    bpy.context.scene.render.resolution_x = 1120
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.world.color = (0.035, 0.045, 0.045)


def _material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.58
    if color[3] < 1:
        principled.inputs["Alpha"].default_value = color[3]
        principled.inputs["Transmission Weight"].default_value = 0.25
        material.surface_render_method = "DITHERED"
    return material


def _build_module(module: dict, materials: dict):
    root = bpy.data.objects.new(module["mesh_node"], None)
    bpy.context.collection.objects.link(root)
    position = module["target_pose"]["position"]
    rotation = module["target_pose"]["rotation_rpy_degrees"]
    root.location = (position["x"], position["y"], position["z"])
    root.rotation_euler = tuple(
        math.radians(rotation[key]) for key in ("x", "y", "z")
    )
    dimensions = module["dimensions"]
    kind = module["module_type"]
    material = materials.get(module["material"], materials["plaster_white"])
    if kind == "door_panel":
        _opening_panel(root, dimensions, materials, "door")
    elif kind == "window_panel":
        _opening_panel(root, dimensions, materials, "window")
    else:
        _cube_child(root, "body", dimensions, material)
        if kind == "roof_panel":
            _roof_ribs(root, dimensions, materials["accent"])
    root["module_id"] = module["module_id"]
    root["module_type"] = kind
    root["required_team_size"] = module["required_team_size"]
    root["mass_kg"] = module["mass_kg"]
    return root


def _opening_panel(root, dimensions, materials, opening_kind: str) -> None:
    width, depth, height = dimensions["width"], dimensions["depth"], dimensions["height"]
    opening_w = min(width * 0.56, 1.25)
    opening_h = min(height * (0.72 if opening_kind == "door" else 0.48), 2.15)
    sill = 0 if opening_kind == "door" else height * 0.23
    side_w = max((width - opening_w) / 2, 0.12)
    for sign in (-1, 1):
        _cube_child(
            root,
            f"jamb_{sign}",
            {"width": side_w, "depth": depth, "height": height},
            materials["plaster_white"],
            x=sign * (opening_w / 2 + side_w / 2),
        )
    top_h = max(height - sill - opening_h, 0.15)
    _cube_child(
        root,
        "lintel",
        {"width": opening_w, "depth": depth, "height": top_h},
        materials["plaster_white"],
        z=height / 2 - top_h / 2,
    )
    if sill:
        _cube_child(
            root,
            "sill",
            {"width": opening_w, "depth": depth, "height": sill},
            materials["plaster_white"],
            z=-height / 2 + sill / 2,
        )
    insert = materials["timber"] if opening_kind == "door" else materials["glass"]
    _cube_child(
        root,
        opening_kind,
        {"width": opening_w * 0.92, "depth": depth * 0.35, "height": opening_h * 0.96},
        insert,
        z=-height / 2 + sill + opening_h / 2,
    )


def _cube_child(root, name, dimensions, material, *, x=0, y=0, z=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = f"{root.name}__{name}"
    obj.dimensions = (dimensions["width"], dimensions["depth"], dimensions["height"])
    obj.location = (x, y, z)
    obj.data.materials.append(material)
    obj.parent = root
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = obj.modifiers.new("edge_softening", "BEVEL")
    bevel.width = 0.025
    bevel.segments = 2
    return obj


def _roof_ribs(root, dimensions, material) -> None:
    for index in range(5):
        x = -dimensions["width"] / 2 + dimensions["width"] * index / 4
        _cube_child(
            root,
            f"roof_seam_{index}",
            {"width": 0.025, "depth": dimensions["depth"] * 0.98, "height": 0.04},
            material,
            x=x,
            z=dimensions["height"] / 2 + 0.015,
        )


def _add_site(plan, materials) -> None:
    bpy.ops.mesh.primitive_plane_add(size=28, location=(-1.5, 0, -0.015))
    ground = bpy.context.object
    ground.name = "construction_site"
    ground.data.materials.append(materials["foundation_concrete"])
    bpy.ops.mesh.primitive_plane_add(size=11, location=(-6.2, -1.2, -0.005))
    yard = bpy.context.object
    yard.name = "material_yard"
    yard.scale.y = 0.55
    yard.data.materials.append(materials["timber"])


def _add_camera(plan):
    bpy.ops.object.camera_add(location=(13.5, -15.5, 11.0))
    camera = bpy.context.object
    camera.name = "overview_camera"
    _look_at(camera, (0, 0, 1.2))
    camera.data.lens = 48
    bpy.context.scene.camera = camera
    return camera


def _look_at(obj, target) -> None:
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(1, -4, 12))
    key = bpy.context.object
    key.data.energy = 1800
    key.data.shape = "DISK"
    key.data.size = 7
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 8))
    sun = bpy.context.object
    sun.rotation_euler = (math.radians(25), math.radians(-20), math.radians(135))
    sun.data.energy = 2.2


def _render(camera, path: Path) -> None:
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    import mathutils

    main()
