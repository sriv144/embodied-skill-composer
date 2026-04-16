# Contributing

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
```

## Main Checks

Use these commands before committing:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
.\.venv\Scripts\python.exe -m compileall src scripts tests
.\.venv\Scripts\python.exe scripts\benchmark_assembly_policies.py --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
.\.venv\Scripts\python.exe scripts\visualize_assembly_episode.py --policy scripted --runtime-profile configs\assembly_profiles\local_dev.yaml
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
```

If the GPU runtime check reports a CPU-only torch wheel, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_torch_windows.ps1
```

Expected benchmark summary:

- `scripted_options`: success `1.000`
- `learned_options`: success `1.000`
- `low_level_learned`: success `0.000`

## Git / Publishing

This repository is intended to live in a **private GitHub repo** during active development.

If `gh auth status` fails, refresh GitHub CLI authentication first:

```powershell
gh auth login -h github.com
```

Once authenticated, create and push the private repo:

```powershell
gh repo create embodied-skill-composer --private --source=. --remote=origin --push
```

## Local Tooling

- Windows + VS Code notes: `docs/setup/windows-vscode.md`
- Linux + NVIDIA + Isaac planning notes: `docs/setup/linux-nvidia-isaac.md`
