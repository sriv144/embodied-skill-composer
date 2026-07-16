# Windows + VS Code Workflow

## Recommended Local Setup

This repository should use the workspace-local `.venv` as the default interpreter in VS Code:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-sim-mujoco.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-sim-coppelia.txt
```

## Main Commands

Run these from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_assembly_policies.py --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\eval_assembly_options.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\eval_assembly_policy.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\run_construction_brain.py --brain heuristic --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 1
.\.venv\Scripts\python.exe scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_obstacles.yaml --runtime-profile configs\assembly_profiles\mujoco_local.yaml --episodes 1
.\.venv\Scripts\python.exe scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_local.yaml --episodes 1
.\.venv\Scripts\python.exe scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_sensing.yaml --episodes 20
.\.venv\Scripts\python.exe scripts\capture_assembly_perception.py --output-dir artifacts\assembly_perception\initial
.\.venv\Scripts\python.exe scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_vision.yaml --episodes 20
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
.\.venv\Scripts\python.exe scripts\visualize_assembly_episode.py --policy scripted --runtime-profile configs\assembly_profiles\local_dev.yaml
.\.venv\Scripts\python.exe scripts\run_mujoco_assembly.py --policy scripted --runtime-profile configs\assembly_profiles\mujoco_local.yaml --record artifacts\mujoco_scripted.mp4
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
.\.venv\Scripts\python.exe scripts\check_coppelia_runtime.py
.\.venv\Scripts\python.exe scripts\run_coppelia_assembly.py
```

## Fix Local CUDA PyTorch

If the GPU check says `torch` is installed but CUDA is unavailable, the environment probably has a CPU-only torch wheel.

Use the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_torch_windows.ps1
```

Or install the pinned CUDA wheel through pip:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rl-cuda-cu130.txt
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
```

The helper script uses Windows BITS because the CUDA torch wheel is large and regular pip downloads can time out or leave partial files.

## What To Expect

- `benchmark_assembly_policies.py` should keep the flagship comparison intact:
  - `scripted_options`: success `1.000`
  - `learned_options`: success `1.000`
  - `low_level_learned`: success `0.000`
- `visualize_assembly_episode.py` writes per-frame PNGs plus a `summary.png` into `artifacts/assembly_playback/`
- `run_mujoco_assembly.py` uses physics-stepped pose tracking, opens the scene with `--gui`, and writes the physical trajectory with `--record`
- `mujoco_sensing.yaml` adds reproducible physical sensor noise, filtering, dropout, and brain safety holds
- `mujoco_vision.yaml` adds RGB/depth pose estimation and oracle-only perception evaluation
- `check_gpu_runtime.py` confirms whether the active Python environment can actually see CUDA
- `check_coppelia_runtime.py` verifies the CoppeliaSim executable and live ZeroMQ endpoint
- `run_coppelia_assembly.py` generates and saves the first CoppeliaSim construction scene

## VS Code

The repository now includes:

- `.vscode/settings.json` for interpreter and pytest discovery
- `.vscode/tasks.json` for one-click benchmark, eval, visualizer, and test runs
- `.vscode/launch.json` for debugger entrypoints

Open the command palette and use:

- `Tasks: Run Task`
- `Python: Select Interpreter`
- `Run and Debug`
