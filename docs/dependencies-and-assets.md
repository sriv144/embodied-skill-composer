# Dependencies And Asset Provenance

This project is Apache-2.0. Dependencies keep their own licenses. Optional desktop tools are
invoked through subprocess or remote APIs and are not redistributed with this repository.

## Core Research Dependencies

| Project | License | Use |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | neural policy training and CUDA execution |
| [TorchRL](https://github.com/pytorch/rl) | MIT | MAPPO/IPPO losses and masked policy primitives |
| [TensorDict](https://github.com/pytorch/tensordict) | MIT | structured rollout tensors |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | MIT | public multi-agent environment contract |
| [OR-Tools](https://github.com/google/or-tools) | Apache-2.0 | deterministic CP-SAT scheduling expert |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | physical skill calibration for retained assembly experiments |
| [w9-pathfinding 0.1.3](https://github.com/andreyd41/w9-pathfinding) | Apache-2.0 | optional CBS/WHCA* routing backend |
| OpenCV | Apache-2.0 | deterministic floor-plan and vision pipelines |
| Shapely | BSD-3-Clause | geometry validation and repair |

`w9-pathfinding` is optional. The routing adapter uses a deterministic reservation-table A* fallback
when its native extension is unavailable. On this Windows machine, installing it requires Visual Studio
2022 Build Tools with MSVC and a Windows SDK.

## Browser Workbench

React, React Three Fiber, Three.js, React Flow, Framer Motion, and Lucide are MIT-licensed. ECharts is
Apache-2.0. The generated cottage and `construction_robot.glb` are original procedural assets exported
by this project and released under Apache-2.0 with the source code.

## Installed-Only Tools

- **Blender** is GPL-licensed and runs as a separate headless geometry process. Blender itself is not
  bundled, linked, or required for CI.
- **CoppeliaSim Edu** is installed separately under its own license. Its KUKA YouBot model is loaded from
  the local Coppelia installation and is never copied into the repository or public demo.
- **OpenAI services** are optional assistants. No API key or hosted service is required by the simulator,
  learned policy, tests, or public demo.

## Referenced Or Isolated Research Projects

Kenney CC0 meshes remain available to the legacy modular-room asset catalog. Poly Haven CC0 materials
are a future visual option and are not currently bundled. CubiCasa5K and FloorplanToBlender3d remain
isolated comparison references because of dataset or GPL constraints; their source is not copied into
the canonical package.
