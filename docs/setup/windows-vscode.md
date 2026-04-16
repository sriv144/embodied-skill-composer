# Windows + VS Code Workflow

## Recommended Local Setup

This repository should use the workspace-local `.venv` as the default interpreter in VS Code:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

## Main Commands

Run these from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_assembly_policies.py --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\eval_assembly_options.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\eval_assembly_policy.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
.\.venv\Scripts\python.exe scripts\visualize_assembly_episode.py --policy scripted --runtime-profile configs\assembly_profiles\local_dev.yaml
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
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
- `check_gpu_runtime.py` confirms whether the active Python environment can actually see CUDA

## VS Code

The repository now includes:

- `.vscode/settings.json` for interpreter and pytest discovery
- `.vscode/tasks.json` for one-click benchmark, eval, visualizer, and test runs
- `.vscode/launch.json` for debugger entrypoints

Open the command palette and use:

- `Tasks: Run Task`
- `Python: Select Interpreter`
- `Run and Debug`
