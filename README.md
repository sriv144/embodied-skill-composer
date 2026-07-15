# Embodied Skill Composer

Embodied Skill Composer is a **Physical AI construction research workbench**. Its current flagship converts an approved architectural design into 24 transportable modules, schedules four robots, exposes the AI brain's decisions, and replays construction in a browser digital twin. The earlier two-robot assembly, MuJoCo, CoppeliaSim, tabletop, and warehouse experiments remain as research baselines.

## Construction v2 Flagship

The verified cottage fixture demonstrates the complete deterministic path:

```text
reviewed house design -> 24 build modules -> dependency graph
                      -> four-robot CP-SAT schedule -> execution trace
                      -> Three.js replay + metrics + research report
```

Current benchmark:

| Controller | Makespan | Idle Robot Time |
| --- | ---: | ---: |
| Sequential precedence | 551 s | 1433 s |
| Greedy ready-list | 204 s | 35 s |
| CP-SAT optimized | 196 s | 6 s |

The optimized fixture is **64.4% faster** than sequential construction. Light panels use one robot while heavy foundation and roof modules require synchronized two-robot teams.

Install and launch the local workbench:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-construction.txt
cd workbench
npm install
cd ..
powershell -ExecutionPolicy Bypass -File scripts\start_construction_workbench.ps1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The five views cover architectural review, module inspection, dependency planning, trace-driven simulation, failure recovery, and controller results.

Generate the standalone experiment artifacts or the Blender geometry package:

```powershell
.\.venv\Scripts\python.exe scripts\run_construction_workbench.py
.\.venv\Scripts\python.exe scripts\generate_construction_assets.py
```

Construction run records are written under `logs/construction_v2/runs/`. Blender writes the named-node `.glb`, `.blend`, geometry manifest, and assembled/exploded previews under `artifacts/construction_v2/`.

Validate the optional robotics and MARL surfaces:

```powershell
# With CoppeliaSim running and its ZeroMQ Remote API available:
.\.venv\Scripts\python.exe -m pip install -r requirements-sim-coppelia.txt
.\.venv\Scripts\python.exe scripts\run_construction_coppelia.py --max-frames 30

# High-level PettingZoo team-formation environment and scripted oracle:
.\.venv\Scripts\python.exe -m pip install -r requirements-construction-rl.txt
.\.venv\Scripts\python.exe scripts\run_construction_marl_env.py
```

The CoppeliaSim replay is currently kinematic and trace-driven. The PettingZoo environment is a validated MARL research interface with assignment actions, dependency masks, cooperative teams, and rewards; it does not yet include a trained policy.

The retained assembly research includes:
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

## Vision

The long-term direction is an **AI construction swarm simulator**: multiple autonomous robots coordinating to collect resources and assemble modular structures in simulation, with a path from the current local sandbox toward MuJoCo, NVIDIA Isaac Lab, and later ROS 2/sim-to-real ideas.

Read the north-star document in [vision.md](docs/vision.md) and the milestone plan in [roadmap.md](docs/roadmap.md).

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

The MuJoCo backend is a hybrid physical-control step: robot and beam bodies follow interpolated mocap
targets through weld constraints and real `mj_step` dynamics. Each robot has two articulated slide
fingers driven by position actuators. Grasp acquisition closes all four fingers and requires simultaneous
force-bearing contacts from both robots; a runtime site-based weld then carries the beam until installation
opens the fingers and releases it back to placement tracking. Diagnostics report finger commands and
positions, contacts, attachment events, physics steps, trajectory frames, measured normal grip forces,
and pose-tracking error alongside the same construction metrics as the local sandbox. The runtime profile
controls the minimum accepted force through `manipulation_min_grip_force_n`; the calibrated local profile
uses a `25 N` per-robot threshold.

### 10. GPU runtime check

Validate whether the active Python environment can really see torch and CUDA:

```powershell
python scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
```

If CUDA is not available even though the NVIDIA driver is installed, replace the CPU-only torch wheel:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_torch_windows.ps1
```

### 11. Experiment copilot

The local-first copilot records experiment runs under `logs/copilot/`, stores run metadata in `logs/copilot/experiments.sqlite`, and writes timestamped reports under `logs/copilot/runs/<run_id>/`.

Run a one-episode benchmark:

```powershell
python scripts\run_copilot.py benchmark --episodes 1
```

Run a scripted option evaluation:

```powershell
python scripts\run_copilot.py eval-options --policy scripted --episodes 1
```

Check NVIDIA/CUDA/Isaac/AI-Q readiness without starting infrastructure:

```powershell
python scripts\run_copilot.py nvidia-check
```

Ask mode is wired through the OpenAI Agents SDK, but it requires a configured `OPENAI_API_KEY` with available API quota:

```powershell
python scripts\run_copilot.py ask "Summarize the latest benchmark run."
```

### 12. ConstructionBrain v0

Run the deterministic layered AI brain with heuristic resource-to-blueprint allocation:

```powershell
python scripts\run_construction_brain.py --brain heuristic --episodes 1
```

Run the oracle-equivalent brain for regression comparison:

```powershell
python scripts\run_construction_brain.py --brain scripted --episodes 1
```

Run the obstacle-aware construction scenario through real MuJoCo physics:

```powershell
python scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_obstacles.yaml --runtime-profile configs\assembly_profiles\mujoco_local.yaml --episodes 1
```

The detour scenario contains a six-cell wall. Independent robots use conflict-aware shortest paths,
and a carried beam is routed as a rigid two-cell formation. Obstacles are collidable MuJoCo geometry
and are included in typed brain observations and episode diagnostics.

Run the failure-and-recovery scenario:

```powershell
python scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_local.yaml --episodes 1
```

This scenario injects a grasp slip and a rejected placement. The brain observes each retryable failure,
repeats the terminal action with an explicit recovery rationale, and records attempts and successful
recoveries. MuJoCo independently rejects grasp/install actions when physical body alignment exceeds the
profile's `manipulation_alignment_tolerance_m`, when both grippers do not contact the beam, or when an
install is attempted without an active attachment. Weak dual contacts below
`manipulation_min_grip_force_n` are also rejected and handled through the same retry path. MuJoCo brain
observations include live alignment error, gripper state and joint positions, the latest measured contact
forces, and active attachment state. Grasp/install rationales record this feedback while the local backend
keeps the same contract with physical feedback omitted.

Run the same recovery experiment with deterministic sensor noise, filtering, and dropout:

```powershell
python scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_sensing.yaml --episodes 20
```

The sensing profile perturbs alignment, force, and finger-position measurements, filters fresh samples,
and marks dropped samples with their age. Before grasp or installation, the brain uses a typed safety hold
when feedback is unavailable or measured alignment exceeds tolerance. Experiment JSON includes sensor
sample/dropout diagnostics, a typed hold reason on each decision, and the episode safety-hold count. This
is physical-feedback partial observability; robot/task positions and blueprint state are still privileged.

Capture Visual Perception v0 artifacts from the calibrated top-down MuJoCo camera:

```powershell
python scripts\capture_assembly_perception.py --output-dir artifacts\assembly_perception\initial
```

Run ConstructionBrain with RGB/depth perception and noisy physical sensing:

```powershell
python scripts\run_construction_brain.py --brain heuristic --env-config configs\assembly_recovery.yaml --runtime-profile configs\assembly_profiles\mujoco_vision.yaml --episodes 20
```

The classical estimator uses OpenCV color components plus metric depth back-projection to estimate robot,
resource, and blueprint-cell poses. It does not receive body positions or segmentation IDs. Segmentation
and simulator poses are retained only by the evaluator for recall and planar pose-error metrics. The
capture command writes RGB, depth, segmentation, annotated overlay, and JSON evaluation artifacts.
ConstructionBrain receives detections and confidence, while privileged task state remains the control
baseline until tracking and occlusion recovery are mature.

Tracking and Occlusion Recovery v0 now preserves visual identities across frames with bounded
nearest-assignment tracks. When a resource or blueprint cell is temporarily hidden, the tracker emits a
confidence-decayed prediction with explicit age and missed-frame metadata. Reports keep visible recall
separate from tracked recall. ConstructionBrain uses tracked agent/resource/blueprint availability as a
terminal safety condition, while mission sequencing and grid navigation still use privileged state.

Estimated-State Control v0 now uses those RGB/depth-derived track positions to approve or veto GRAB and
INSTALL. A terminal action proceeds only when both robots are close enough to the selected resource and,
for installation, the resource is close enough to two blueprint cells. The assessment records track IDs,
distances, minimum confidence, and whether prediction-backed tracks were required. Mission sequencing,
resource selection, and grid navigation still use privileged simulator state; this milestone does not
claim end-to-end visual autonomy.

Capture the prediction-backed visual assessment immediately before installation:

```powershell
python scripts\capture_assembly_perception.py --advance-options go_pickup,grab,go_assembly --output-dir artifacts\assembly_perception\estimated_state_pre_install
```

Capture the first installed beam with visible and prediction-backed tracks:

```powershell
python scripts\capture_assembly_perception.py --advance-options go_pickup,grab,go_assembly,install --output-dir artifacts\assembly_perception\tracked_post_install
```

Run the optional CoppeliaSim backend spike after starting CoppeliaSim Edu:

```powershell
python -m pip install -r requirements-sim-coppelia.txt
python scripts\check_coppelia_runtime.py
python scripts\run_coppelia_assembly.py
```

The CoppeliaSim adapter creates the construction scene from the same YAML task, switches CoppeliaSim to
its MuJoCo physics engine, advances the remote simulation deterministically, captures a top-down image,
and saves a reusable `.ttt` scene. The v0 adapter synchronizes kinematic poses from the logical task; it
does not yet claim wheel, gripper, contact-force, or perception parity with the direct MuJoCo backend.
Generated evidence is written under `artifacts/coppelia/` and `logs/coppelia/`.

Build and record the ten-component Modular Room v0 milestone:

```powershell
python scripts\run_modular_construction.py `
  --blueprint configs\blueprints\modular_room_v0.yaml `
  --runtime-profile configs\assembly_profiles\coppelia_local.yaml `
  --brain precedence `
  --record
```

Preview every catalog mesh, scale, and orientation before running the room:

```powershell
python scripts\preview_construction_assets.py
```

The typed compiler validates materials, component IDs, metric poses, team sizes, and an acyclic
dependency graph before converting the architectural blueprint into the existing task contract. The
precedence brain then installs four columns, four walls, and two roof modules with both robots. Runs
write episode, compiled-blueprint, and lab-report files under `logs/construction_runs/<run_id>/`; replay,
final camera images, and the reusable scene go under `artifacts/construction_runs/<run_id>/`.

When available, the adapter loads two KUKA YouBot mobile manipulators from the installed CoppeliaSim Edu
model library. The robot model is referenced at runtime and is not redistributed by this repository. A
small CC0 subset of Kenney's Modular Buildings pack is stored under `assets/third_party/` and mapped in
`configs/construction_asset_catalog.yaml`; see [asset provenance](assets/ASSETS.md). These meshes provide
wall, corner, door, window, and roof visuals for the completed Modular Room v0 blueprint-to-scene
milestone. See [research landscape](docs/research-landscape.md) for the external systems that informed
the graph, benchmark, learning, and decentralized-swarm design choices.

Each run writes a JSON experiment under `logs/construction_brain/` containing typed observations,
resource assignments, option decisions with rationales, execution results, construction metrics, and
backend diagnostics. The brain is deterministic and local; OpenAI API access is not required.

Recommended profiles:

- `configs/assembly_profiles/local_dev.yaml`: CPU regression baseline
- `configs/assembly_profiles/local_gpu.yaml`: local RTX 4060 torch/CUDA validation
- `configs/assembly_profiles/mujoco_local.yaml`: Windows-friendly MuJoCo 3D visual simulation
- `configs/assembly_profiles/mujoco_sensing.yaml`: noisy/dropout-aware MuJoCo Physical AI experiments
- `configs/assembly_profiles/mujoco_vision.yaml`: RGB/depth perception plus noisy physical sensing
- `configs/assembly_profiles/coppelia_local.yaml`: optional CoppeliaSim ZeroMQ backend spike
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
- `logs/copilot/` contains generated copilot reports and the local SQLite run registry
- `.env` and `.env.*` are ignored so local API keys and secrets are not committed
- the intended default runtime profile is `configs/assembly_profiles/local_dev.yaml`
- the intended local GPU validation profile is `configs/assembly_profiles/local_gpu.yaml`
- the intended local 3D simulation profile is `configs/assembly_profiles/mujoco_local.yaml`
- the optional interactive robotics profile is `configs/assembly_profiles/coppelia_local.yaml`
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

- stabilize the current research workbench and copilot workflow
- extend the local assembly sandbox toward resource-based construction scenarios
- add costs, obstacles, idle-time metrics, and recovery diagnostics
- improve visual artifacts for college research and portfolio review
- port the stable task contract into Isaac Lab only after local parity is protected

See [roadmap.md](docs/roadmap.md) for milestone checkpoints and done criteria.
