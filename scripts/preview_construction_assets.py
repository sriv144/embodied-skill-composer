# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.blueprint import compile_modular_blueprint
from embodied_skill_composer.assembly.coppelia_backend import (
    CoppeliaSimAssemblyBackend,
)
from embodied_skill_composer.assembly.models import (
    BlueprintComponent,
    BlueprintMaterial,
    ModularBlueprint,
    ScenePose,
)
from embodied_skill_composer.assembly.runtime import (
    load_asset_catalog,
    load_runtime_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and preview every construction asset in CoppeliaSim."
    )
    parser.add_argument(
        "--asset-catalog",
        type=Path,
        default=PROJECT_ROOT / "configs" / "construction_asset_catalog.yaml",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=(
            PROJECT_ROOT / "configs" / "assembly_profiles" / "coppelia_local.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "construction_assets",
    )
    return parser.parse_args()


def build_preview_blueprint(asset_keys: list[str]) -> ModularBlueprint:
    materials: list[BlueprintMaterial] = []
    components: list[BlueprintComponent] = []
    for index, asset_key in enumerate(asset_keys):
        row = index // 4
        column = index % 4
        source_x = 4 + row * 4
        source_y = 2 + column * 4
        target_x = 14 + row * 2
        target_y = source_y
        resource_id = f"preview_material_{asset_key}"
        materials.append(
            BlueprintMaterial(
                resource_id=resource_id,
                component_type=asset_key,
                asset_key=asset_key,
                source_cells=[(source_x, source_y), (source_x, source_y + 1)],
            )
        )
        components.append(
            BlueprintComponent(
                component_id=f"preview_{asset_key}",
                component_type=asset_key,
                asset_key=asset_key,
                required_material_id=resource_id,
                target_cells=[(target_x, target_y), (target_x, target_y + 1)],
                target_pose=ScenePose(
                    position_m=(float(column), float(row), 0.5)
                ),
            )
        )
    return ModularBlueprint(
        blueprint_id="construction_asset_preview",
        title="Construction Asset Catalog Preview",
        grid_size=20,
        max_steps=100,
        agent_starts=[(1, 9), (1, 10)],
        materials=materials,
        components=components,
    )


def add_asset_legend(image_path: Path, asset_keys: list[str]) -> None:
    image = Image.open(image_path).convert("RGB")
    line_height = 20
    legend_height = 34 + line_height * ((len(asset_keys) + 1) // 2)
    output = Image.new(
        "RGB",
        (image.width, image.height + legend_height),
        color=(24, 28, 33),
    )
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.text((12, image.height + 10), "Imported construction assets", fill="white")
    for index, asset_key in enumerate(asset_keys):
        column = index % 2
        row = index // 2
        draw.text(
            (12 + column * image.width // 2, image.height + 32 + row * line_height),
            f"{index + 1}. {asset_key}",
            fill=(218, 224, 231),
        )
    output.save(image_path)


def main() -> int:
    args = parse_args()
    catalog = load_asset_catalog(args.asset_catalog.resolve())
    asset_keys = list(catalog.components)
    compiled = compile_modular_blueprint(
        build_preview_blueprint(asset_keys),
        catalog,
        workspace_root=PROJECT_ROOT,
    )
    profile = load_runtime_profile(args.runtime_profile.resolve())
    backend = build_assembly_backend(compiled.scenario, profile, seed=7)
    if not isinstance(backend, CoppeliaSimAssemblyBackend):
        print("Asset preview requires a CoppeliaSim runtime profile.")
        return 1
    if not backend.is_ready:
        print("CoppeliaSim backend is not ready.")
        for note in backend.readiness_notes:
            print(f"- {note}")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        backend.reset(seed=7)
        image_path = backend.capture_camera(
            output_dir / "contact_sheet.png",
            camera_name="topdown",
        )
        add_asset_legend(image_path, asset_keys)
        scene_path = backend.save_scene(output_dir / "asset_preview.ttt")
        diagnostics = backend.get_option_episode_diagnostics()["coppelia_sim"]
    finally:
        backend.close()

    print(f"Catalog assets: {len(asset_keys)}")
    print(f"Imported meshes: {diagnostics['loaded_asset_meshes']}")
    print(f"Primitive fallbacks: {len(diagnostics['asset_fallbacks'])}")
    print(f"Contact sheet: {image_path}")
    print(f"Scene: {scene_path}")
    return 0 if not diagnostics["asset_fallbacks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
