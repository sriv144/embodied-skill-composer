# Roadmap: From Two-Robot Assembly To Physical AI Construction Swarms

## Construction Intelligence v1: Learned Swarm Brain

Status: active. The implementation checkpoint is complete; long research training and live dynamic
Coppelia acceptance remain intentionally open.

Completed in this checkpoint:

- deterministic 16-32-module train, validation, and held-out cottage families,
- `construction_coordination_v1`, where work consumes simulated time and failures trigger reassignment,
- optional w9 CBS routing behind a deterministic reservation-table A* fallback,
- CP-SAT demonstrations, behavior cloning, MAPPO/IPPO losses, checkpoint and ONNX export,
- a real 64-transition training plumbing run and held-out non-learned controller smoke evaluation,
- five-seed evaluation statistics with standard deviation and bootstrap 95% confidence intervals,
- MuJoCo campaign-to-`SkillProfile` calibration,
- `dynamic_base_logical_payload` Coppelia execution with 20 Hz wheel commands and measured telemetry,
- persistent lab APIs, approval-gated training, run history, and WebSocket progress,
- a routed browser workbench using the real named-node cottage GLB and an original robot GLB,
- a read-only GitHub Pages bundle, CI, secret scanning, licensing, and dependency attribution.

Current local readiness:

- PyTorch `2.11.0+cu130` sees the RTX 4060 and passes CUDA tensor allocation.
- `w9-pathfinding==0.1.3` is not installed because MSVC and the Windows SDK are not present; the tested
  deterministic fallback is active.
- A live Coppelia assignment passed with 2,274 physics steps, 4,540 wheel commands, 4,536 measured
  pose samples, one installed module, and zero post-start robot pose writes. Full-cottage completion and
  live failure recovery remain required.
- MAPPO/IPPO research checkpoints do not exist yet. The UI labels them `not trained`.

Acceptance gates still open:

1. Run MAPPO and IPPO research seeds `7,8,9,10,11` and publish the held-out confidence intervals.
2. Verify the stated completion, CP-SAT gap, failure recovery, and ablation thresholds from actual data.
3. Extend the passed one-module live Coppelia gate to a complete cottage with zero post-initialization
   robot pose writes.
4. Record planned-versus-measured replay and one genuine obstacle or unavailable-robot recovery.

No dual boot, ROS 2, Isaac Sim, humanoid embodiment, or lunar scene is required for these gates.

## Construction v2 Product Track

Status: the first portfolio-grade workbench slice is operational.

Completed:

- typed architectural, module, robot, schedule, brain-event, and execution-trace contracts,
- an approved single-story cottage compiling into 24 modules and four mixed-role robots,
- sequential, greedy, and real OR-Tools CP-SAT controllers,
- 196-second optimized makespan versus 551 seconds sequential on the fixture,
- API-backed obstacle, robot-health, and dropped-resource recovery traces,
- deterministic OpenCV floor-plan footprint intake with mandatory review state,
- a headless Blender 5.1 generator producing `.blend`, named-node `.glb`, and preview renders,
- a React/Three.js workbench for Design, Modules, Plan, Simulate, and Results,
- dependency graph, robot-lane Gantt, brain rationale, replay timeline, metrics, and report export,
- desktop/mobile WebGL verification and focused Construction v2 regression coverage,
- a trace-driven four-robot CoppeliaSim adapter with reusable scene output,
- a PettingZoo Parallel API environment for high-level bidding and cooperative team formation,
- a deterministic MARL-interface oracle that completes all 24 modules in nine decisions.

Next product milestones:

1. Add direct wall/opening editing and a small golden dataset for uploaded-plan parsing.
2. Add conflict-aware multi-agent routing beyond the current deterministic orthogonal routes.
3. Run and evaluate the implemented MAPPO/IPPO bidder against scripted, greedy, auction, and CP-SAT.
4. Validate dynamic Coppelia execution and measured recovery on the live simulator.
5. Add optional IFC export and richer CC0 materials after geometry semantics remain stable.

The browser digital twin is now the primary product experience. Blender is the geometry compiler;
CoppeliaSim provides optional kinematic robotics evidence; Isaac Lab remains a later parity target.
The current PettingZoo environment is a tested coordination interface, not a trained-policy claim.

## Current State

The project has a working local research core:

- local collaborative assembly sandbox,
- two-robot two-beam task,
- hierarchical team-options learner,
- retained low-level MARL baseline,
- scripted option oracle,
- MuJoCo visual backend for the same task semantics,
- Isaac Lab backend stub,
- copilot CLI for local experiment recording and readiness checks.

The verified flagship result is that hierarchical learned options solve the default `2/2 beams` task, while the low-level learned MARL baseline still stalls at `1/2 beams`.

Known limits:

- OpenAI copilot `ask` mode requires API quota.
- CUDA is visible to the active `.venv` torch runtime; the RTX 4060 tensor-allocation check passes.
- Isaac Lab is a contract-preserving stub, not a real simulator backend yet.
- Blender is optional and should not be required by tests or runtime commands.

## Phase 1: Stabilize The Research Workbench

Goal: make the current foundation reproducible before adding simulator complexity.

Done means:

- README links to the vision, roadmap, current results, Isaac prep, and copilot workflow.
- `docs/vision.md` states the Physical AI construction-swarm direction.
- `docs/roadmap.md` tracks milestones and simulator strategy.
- full local tests pass.
- generated outputs remain under `logs/` and `artifacts/`.
- `.env` stays ignored and secrets are never committed.

Primary commands:

```powershell
python -m pytest -q --basetemp .pytest_tmp
python scripts\run_copilot.py benchmark --episodes 1
python scripts\run_copilot.py eval-options --policy scripted --episodes 1
python scripts\run_copilot.py nvidia-check
```

## Phase 2: Resource Inventory And Blueprint Slots

Goal: move from fixed beams toward explicit construction semantics.

Status: complete for the first backward-compatible construction-semantics slice.

Done means:

- scenarios can describe resource inventory and target blueprint slots,
- default beam-only configs still validate and behave the same,
- each current beam can be derived as a default resource-to-blueprint placement,
- metrics include structure completion, resource delivery accuracy, energy cost, idle time, wasted steps, collisions, and coordination efficiency,
- scripted oracle solves the resource/blueprint scenario,
- tests cover resource accounting and blueprint completion.

This phase keeps the local Python grid as the research truth source.

## Phase 3: ConstructionBrain v0

Goal: introduce a clear AI brain interface before depending on larger models or heavier simulators.

Status: complete for v0. The typed observation/decision contract, scripted brain, heuristic allocator,
episode runner, and CLI artifact path are implemented and regression-tested. Physical backends can add
optional typed manipulation feedback without changing local construction semantics; MuJoCo now exposes
alignment error, finger state, measured grip force, and attachment state to the brain.

Done means:

- a `ConstructionBrain` interface can observe resources, blueprint slots, robot state, and progress metrics,
- `ScriptedConstructionBrain` reproduces the current oracle behavior,
- a heuristic allocator can assign resources/slots without changing low-level option execution,
- physical grasp/install rationales can cite simulator feedback while local observations remain compatible,
- learned hierarchical options remain the main policy track,
- low-level MARL remains a comparison baseline.

The brain is layered:

```text
mission planner -> task allocator -> coordinator -> option policy -> low-level controller
```

OpenAI can later assist with mission planning and lab reporting. It should not be required for deterministic tests.

## Phase 4: Simulator Backend Spike

Goal: evaluate one physical simulator bridge without replacing the local grid.

Status: active. The ConstructionBrain runs through the real MuJoCo 3.10 backend with 32 generalized
positions and 28 velocity DoF.
Robots and beams now move through physics-stepped mocap-weld pose tracking, match local task metrics,
and produce a decodable H.264 physical trajectory. Explicit obstacle cells, conflict-aware independent
routes, rigid carried-beam routes, and matching collidable MuJoCo obstacle geometry are implemented.
High-level grasp/install decisions still use shared logical semantics, but MuJoCo now requires dual
gripper contacts from four position-controlled slide fingers, activates a runtime site-based weld while
carrying, opens the fingers, and releases the weld for installation. Manipulation failures, brain retries,
recovery metrics, and physical-alignment gates are implemented. Contact normal-force sensing and a
calibrated `25 N` per-robot minimum-force gate are implemented; wrist torque sensing and learned
low-level manipulation remain future work. Physical Sensing v0 adds seeded alignment/force/joint noise,
whole-sample dropout, exponential filtering, freshness/age metadata, and typed sensor/alignment safety
holds before terminal actions. Physical Sensing v0 alone does not remove privileged task state.
Visual Perception v0 adds a calibrated top-down RGB/depth camera, OpenCV component
detection, metric pose back-projection, brain-visible confidence, and oracle-only recall/error scoring.
It detects agents, resources, and blueprint cells without feeding segmentation IDs or body poses into
the estimator. Tracking and Occlusion Recovery v0 adds persistent IDs, bounded missed-frame prediction,
confidence decay, separate visible/tracked recall, and terminal-action holds when tracked visual targets
are unavailable. Estimated-State Control v0 now evaluates robot-to-resource and
resource-to-blueprint geometry from RGB/depth tracks before GRAB and INSTALL. A 50-episode MuJoCo
campaign completed all 50 episodes: all 300 terminal decisions were visually ready, mean pose error was
`0.0235 m`, final visible blueprint recall was `0.5`, and tracked recall was `1.0`. The controller still
uses privileged mission sequencing, resource selection, and grid navigation. Active viewpoint selection,
appearance embeddings, uncertainty-aware data association, and replacing that remaining privileged state
are future work.

CoppeliaSim Backend Spike v0 is also operational through the ZeroMQ Remote API. The default construction
scenario is generated programmatically, runs the scripted policy to `2/2` beams, advances CoppeliaSim in
deterministic `0.05 s` steps, captures a top-down camera image, and saves a reusable `.ttt` scene. The
validated run used CoppeliaSim's MuJoCo engine and matched the local reward (`10.52`) and completion
metrics. This is currently kinematic pose synchronization, not dynamic wheel/gripper/contact parity.

Modular Room v0 is complete as the first blueprint-to-construction execution milestone. A typed YAML
blueprint compiles into ten legacy-compatible tasks with a stable topological order: four columns, four
walls, and two roof modules. The deterministic precedence brain installs all `10/10` components locally
and through CoppeliaSim, with matching completion and delivery metrics. The Coppelia path imports the CC0
Kenney meshes, loads two installed KUKA YouBots without redistribution, records a historical replay from
per-frame completion state, captures overview/top-down images, and saves a reusable terminal `.ttt`
scene. The asset preview command validates all catalog imports before scenario execution.

Candidate backends:

- **MuJoCo** for clean local physics/control experiments.
- **CoppeliaSim** for robotics prototyping, remote API control, and multi-robot scenes.

Done means:

- one backend spike runs the same simple construction semantics,
- scripted behavior works before learned behavior,
- artifacts and metrics match the local sandbox shape,
- failures are documented without blocking local research progress.

The initial CoppeliaSim spike and Modular Room v0 meet these criteria. The next controller milestone adds
arbitrary unlocked-component selection, role-specific jobs, and concurrent robot activity. That becomes
the environment for graph-based hierarchical RL/MARL and a TERMES-inspired decentralized baseline. The
later dynamic embodiment gate covers wheeled rover control, cooperative gripper attachment, force
sensing, and camera-stream integration.

Do not adopt Unity in this phase. Do not require ROS 2 yet.

## Phase 5: Visual And Presentation Layer

Goal: make the work understandable to professors, peers, and reviewers.

Done means:

- scripted success, learned success, low-level failure, and recovery behavior have saved visual artifacts,
- MuJoCo playback remains aligned with local sandbox metrics,
- Blender assets or renders may be used for optional visuals,
- a short research narrative exists: problem, baseline, hierarchical solution, results, next backend step.

Blender stays optional. If Blender is not installed or not on `PATH`, tests and core commands must still pass.

## Phase 6: NVIDIA Isaac Lab Parity

Goal: move into the NVIDIA Physical AI path only after local semantics are stable.

Done means:

- Linux NVIDIA environment is ready,
- Isaac Sim and Isaac Lab are installed,
- the same two-beam/resource task exists in Isaac with privileged state,
- scripted options solve it first,
- artifacts and metrics match the local sandbox shape,
- learned hierarchical options are evaluated only after scripted parity is proven.

Out of scope for the first Isaac pass:

- end-to-end visual MARL,
- humanoids,
- realistic planetary terrain,
- sim-to-real deployment.

## Long-Term Destination

The long-term destination is not just a robot demo. It is a reproducible Physical AI research system for studying how robot teams can plan, coordinate, learn, and recover while assembling structures from resources.
