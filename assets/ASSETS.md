# Asset Provenance

Third-party assets are kept small, explicit, and replaceable. Physics collision geometry remains
primitive unless a component explicitly requires a convex hull.

## Kenney Modular Buildings 2.1

- Source: https://kenney.nl/assets/modular-buildings
- Retrieved: 2026-07-14
- License: Creative Commons Zero 1.0 (`CC0-1.0`)
- Local license: `assets/third_party/kenney_modular_buildings/LICENSE.txt`
- Included subset: wall, corner, door, window, roof, and one sample-house OBJ/MTL pair plus the shared
  color-map texture.

The upstream pack contains 100 models. Only the files referenced by
`configs/construction_asset_catalog.yaml` and the sample-house reference are vendored here.
Catalog entries record metric scale, OBJ-to-Coppelia orientation correction, expected dimensions,
collision proxy, source, and attribution. `scripts/preview_construction_assets.py` imports every entry
and creates a contact sheet so scale or axis regressions are visible before a construction run.

## CoppeliaSim Bundled Robot Models

The CoppeliaSim adapter can load the installed `KUKA YouBot.ttm` model at runtime. The model is not
copied into this repository. Its use is governed by the installed CoppeliaSim Edu educational license.
The runtime path and role are recorded in `configs/construction_asset_catalog.yaml`.
