# Embodied Skill Composer

Embodied Skill Composer is a **hybrid robotics research project** that now centers on **two-robot collaborative assembly** in a local sandbox, with the older tabletop and warehouse flows retained as supporting baselines.

It now includes the first local-sandbox slice of a bigger research direction:
- **two-robot collaborative assembly**
- **hierarchical team-options learning with imitation warm-start + PPO fine-tuning**
- **state-based multi-agent RL with centralized training / decentralized execution as a retained baseline**
- a simulator-agnostic path toward larger Isaac Lab experiments later

The flagship workflow is:
- coordinate two agents around a shared assembly task,
- choose team-level options instead of raw actions,
- execute deterministic routing and recovery,
- learn the high-level sequence with PPO,
- benchmark the learned policy against scripted and low-level baselines.

The original tabletop manipulation demos are still included as milestone-0 baselines and regression coverage.

## Flagship Result

The current verified result is:

- **hierarchical learned options** solve the default `2/2 beams` assembly task
- the retained **low-level learned MARL** baseline still stalls at `1/2 beams`

Most recent comparison:

| Policy | Success Rate | Mean Return | Mean Beams Installed |
| --- | ---: | ---: | ---: |
| Scripted options | 1.000 | 10.52 | 2.00 |
| Learned hierarchical options | 1.000 | 10.52 | 2.00 |
| Low-level learned MARL | 0.000 | -0.02 | 1.00 |

See [assembly-hierarchical-options.md](docs/results/assembly-hierarchical-options.md) for the short write-up and [isaac-prep.md](docs/isaac-prep.md) for the next backend milestone assumptions.
For current local setup and debugging workflows, see [windows-vscode.md](docs/setup/windows-vscode.md) and [linux-nvidia-isaac.md](docs/setup/linux-nvidia-isaac.md).

Private-repo publishing is intended to happen through GitHub CLI once authentication is valid:

```powershell
gh auth status
gh auth login -h github.com
gh repo create embodied-skill-composer --private --source=. --remote=origin --push
```

## What It Does

- Runs a perception-driven warehouse collection task with reusable skills
- Supports **oracle** and **classical CV** perception modes
- Supports **scripted** and **RL-policy** pickup modes
- Logs execution events, retries, and final world state as JSON
- Benchmarks success rate, completion rate, grasp retry rate, perception miss rate, and action count
- Keeps the assembly task contract separate from backend selection so richer simulators, including Isaac Lab, can be added later

## Current Project Modes

### 1. Tabletop baseline
- `pick_and_place_red_to_tray`
- `sort_blue_to_zone`
- `stack_red_on_green`

Run it with:

```powershell
python scripts\run_demo.py --task pick_and_place_red_to_tray
```

### 2. Warehouse flagship task
- `warehouse_multi_object_collection`

Run a single perception-driven episode:

```powershell
python scripts\run_collection.py --perception classical_cv --policy scripted
```

Compare against oracle perception or RL pickup:

```powershell
python scripts\run_collection.py --perception oracle --policy scripted
python scripts\run_collection.py --perception classical_cv --policy rl
```

### 3. Benchmark suite

```powershell
python scripts\run_benchmark.py --perception classical_cv --policy scripted
python scripts\run_benchmark.py --perception classical_cv --policy rl
```

### 4. RL pickup-policy training

```powershell
python scripts\train_grasp_policy.py --episodes 2000
```

### 5. Collaborative assembly hierarchical-options sandbox

Install the RL dependency if needed:

```powershell
pip install -r requirements-rl.txt
```

Train the hierarchical team-options policy:

```powershell
python scripts\train_assembly_options.py --runtime-profile configs\assembly_profiles\local_dev.yaml
```

Evaluate the scripted option oracle:

```powershell
python scripts\eval_assembly_options.py --policy scripted
```

Evaluate the learned hierarchical policy:

```powershell
python scripts\eval_assembly_options.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml
```

### 6. Low-level MARL baseline

Train the retained low-level action policy baseline:

```powershell
python scripts\train_assembly_marl.py --runtime-profile configs\assembly_profiles\local_dev.yaml
```

Evaluate the scripted coordination baseline:

```powershell
python scripts\eval_assembly_policy.py --policy scripted
```

Evaluate the learned policy:

```powershell
python scripts\eval_assembly_policy.py --policy learned --runtime-profile configs\assembly_profiles\local_dev.yaml
```

### 7. Policy benchmark comparison

```powershell
python scripts\benchmark_assembly_policies.py --runtime-profile configs\assembly_profiles\local_dev.yaml
```

### 8. Assembly playback visualizer

Render the local sandbox episode as a sequence of 2D frames:

```powershell
python scripts\visualize_assembly_episode.py --policy scripted --runtime-profile configs\assembly_profiles\local_dev.yaml
```

This writes:

- `artifacts/assembly_playback/frames/frame_*.png`
- `artifacts/assembly_playback/summary.png`

You can also replay a previously saved diagnostics JSON:

```powershell
python scripts\visualize_assembly_episode.py --diagnostics-json artifacts\assembly_playback\diagnostics_scripted.json
```

### 9. MuJoCo 3D assembly simulation

Install the optional MuJoCo simulation dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-sim-mujoco.txt
```

Run the scripted 3D assembly episode and record a video:

```powershell
python scripts\run_mujoco_assembly.py --policy scripted --gui --record artifacts\mujoco_scripted.mp4
```

Run the learned hierarchical policy in the MuJoCo backend:

```powershell
python scripts\run_mujoco_assembly.py --policy learned --runtime-profile configs\assembly_profiles\mujoco_local.yaml --gui --record artifacts\mujoco_learned.mp4
```

The first MuJoCo backend is intentionally a hybrid step-up: simple mobile agent bodies and beam visuals in 3D, with the same team-option task semantics and metrics as the local sandbox.

### 10. GPU runtime check

Validate whether the active Python environment can really see torch and CUDA:

```powershell
python scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
```

If CUDA is not available even though the NVIDIA driver is installed, replace the CPU-only torch wheel:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_torch_windows.ps1
```

Recommended profiles:

- `configs/assembly_profiles/local_dev.yaml`: CPU regression baseline
- `configs/assembly_profiles/local_gpu.yaml`: local RTX 4060 torch/CUDA validation
- `configs/assembly_profiles/mujoco_local.yaml`: Windows-friendly MuJoCo 3D visual simulation
- `configs/assembly_profiles/isaac_gpu.yaml`: planned Linux + NVIDIA Isaac profile

## Architecture

- `src/embodied_skill_composer/core/`: planner, executor, interfaces, shared models, skills, logging
- `src/embodied_skill_composer/assembly/`: collaborative assembly task contract, backend selection, scripted baseline, benchmark helpers, and learners
- `src/embodied_skill_composer/perception/`: oracle and classical-CV world-state builders
- `src/embodied_skill_composer/sim/`: tabletop adapters plus warehouse adapters
- `src/embodied_skill_composer/pipelines/`: higher-level collection orchestration
- `src/embodied_skill_composer/rl/`: lightweight learned pickup-policy scaffolding
- `src/embodied_skill_composer/tasks/`: YAML task loading
- `configs/`: tabletop and warehouse configs
- `tests/`: tabletop regression tests plus warehouse perception/planner/benchmark coverage

## Setup Notes

Minimal local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp
```

Repository guidance:

- `logs/`, checkpoints, rendered images, and simulator artifacts are generated outputs and are ignored by git
- the intended default runtime profile is `configs/assembly_profiles/local_dev.yaml`
- the intended local GPU validation profile is `configs/assembly_profiles/local_gpu.yaml`
- the intended local 3D simulation profile is `configs/assembly_profiles/mujoco_local.yaml`
- the planned future profile is `configs/assembly_profiles/isaac_gpu.yaml`
- contributor notes live in `CONTRIBUTING.md`
- Windows/VS Code notes live in `docs/setup/windows-vscode.md`
- Linux/NVIDIA bring-up notes for Isaac live in `docs/setup/linux-nvidia-isaac.md`

## Why This Is Hybrid

This project intentionally does **not** use RL for everything.

- **Perception** converts sensor observations into a planning world state.
- **Explicit planning** decides which targets remain, where the robot should go next, and how to recover.
- **Structured option execution** keeps the collaborative assembly task inspectable and benchmarkable.
- **RL** is used selectively where learning adds value instead of forcing the whole stack into one reactive policy.

## Roadmap

- port the assembly task contract into an Isaac-compatible backend stub
- scale the assembly benchmark into an Isaac Lab backend with the same task contract
- add camera-based evaluation for the assembly task before moving to full visual MARL
- compare scripted, local-MARL, and future Isaac-scale policies on coordination metrics
- richer PyBullet warehouse dynamics and rendering
- local obstacle-aware motion as a second learned policy
- ROS 2 bridge for real-robot or higher-fidelity simulator integration
- stronger perception with learned detectors or RGB-D estimation
