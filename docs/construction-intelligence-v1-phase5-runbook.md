# Construction Intelligence v1 — Phase 5 Coppelia Runbook

Phase 5 requires two real, manually approval-gated CoppeliaSim runs. The automated
test harness proves orchestration and evidence-schema behavior, but it is explicitly
recorded as `offline_harness` and can never satisfy the live gate.

## Simulator boundary

The live executor commands KUKA YouBot wheel joints in deterministic stepping mode
and samples robot poses and velocities from CoppeliaSim. Direct robot pose writes are
allowed only while constructing the initial scene, before simulation starts.

Payload transport remains logical and overhead. A module is parented to a carrier
dummy synchronized to the measured robot-base centroid while preserving its
attested full three-dimensional pickup offset. A deterministic transport center
height is bounded by the yard configuration and derived from conservative full-RPY
module AABBs. Finite installed modules, staged modules, and site obstacles may be
overflown only when their independently recomputed vertical separation passes.
Idle and disabled robots remain infinite-height XY exclusion columns: payloads may
not overfly them at any logical height.

The physical yard stages every module at its complete target RPY, so transport never
claims an in-transit payload rotation. After the measured team is proven stopped,
the executor records one logical lift with the staging pose, lifted pose, carrier
pose, measured centroid, full centroid offset, bounded height, zero wheel commands,
and measured-settle proof. Phase 5 does **not** claim physical lift or descent, arm
motion, gripper actuation, grasp contact, cooperative contact dynamics, or physical
payload dynamics.

The planner computes one carrier route and derives both base routes from fixed
measured formation offsets. Carry routes are sampled at no more than 0.1 m
intervals. Approach and return path pairs are scheduled by a bounded deterministic
product-state search that inserts explicit waits; every independent-progress pair
of swept base segments must preserve the configured pair separation before the
executor receives it. Continuous analytic segment checks cover approach, carry,
and return against installed/staged structure, obstacles, idle robots, disabled
robots, and the active payload. Teams return to deterministic dispatch bays after
each installation.

Payload contact with structure is forbidden throughout logical transport. The only
exception is exactly one recorded `logical_installation_snap` at the frozen full
target pose, with the measured team again proven stopped. Its complete target pose,
digest, timestamp, logical transport height, and narrowly derived contact-module
list are independently verified.

## Required external resource

1. Start CoppeliaSim manually.
2. Ensure the ZeroMQ remote API is reachable on `127.0.0.1:23000`.
3. Ensure the configured KUKA YouBot model exists. The default is:
   `C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/mobile/KUKA YouBot.ttm`.
4. Commit the Phase 5 implementation and begin from a clean Git worktree; unverifiable,
   abbreviated, or dirty source provenance is rejected before connecting.
5. Keep the application open until each run exits and its scene is saved.

The runner fails closed when the server or model is unavailable. It writes only a
`gate_failure.json` record with `live_evidence: false`; it never substitutes fake
telemetry or claims that the live gate passed.

## Run the two approval-gated scenarios

Use the same held-out generated cottage seed for both runs:

```powershell
.\.venv\Scripts\python.exe scripts\run_construction_phase5_coppelia.py `
  --scenario nominal --seed 900 --yes

.\.venv\Scripts\python.exe scripts\run_construction_phase5_coppelia.py `
  --scenario unavailable-robot-recovery --seed 900 --yes
```

The deterministic planner installs one dependency-valid module at a time. In the
recovery run it selects a robot with remaining planned work only after at least 25%
of modules are installed, issues exactly one zero-velocity recovery stop, permanently
rejects later commands to that robot, and recomputes every subsequent team from the
remaining fleet.

## Acceptance gates

Both runs must show:

- every generated module installed;
- zero direct robot pose writes after simulation start;
- exhaustive discovery and source-role classification of bundled YouBot scripts;
  the wheel-command and arm/gripper writers are disabled with read-back
  verification, while allowlisted passive omni-wheel geometry-maintenance scripts
  remain enabled, leaving Python as the exclusive wheel-command owner;
- non-zero wheel commands, measured telemetry, and measured command response for
  every active robot;
- physical collision/contact queries for every required robot/robot,
  robot/module, and robot/obstacle pair on every physics step;
- every planned site obstacle instantiated as a respondable simulator object;
- per-robot pickup formation, team-spacing, and install errors inside configured
  tolerances;
- equal-horizon base routes rigidly derived from one carrier route, with synchronized
  measured formation error inside tolerance;
- continuous planned and measured base-to-payload clearance of at least the robot
  footprint radius plus route clearance at every carry sample;
- analytic continuous planned and measured world clearance for every approach,
  carry, and return segment, including between recorded waypoints;
- conservative full-RPY payload envelopes, a finite recorded logical transport
  height, and vertical separation from finite-height structure and obstacles;
- no base or payload overflight of idle or disabled robot columns;
- exactly one attested logical lift and one final-target-only logical installation
  snap per module, both bound to zero-command and measured-settle proofs;
- a recorded maximum formation offset that stays inside the configured finite
  expansion bound;
- one complete planned-versus-measured replay entry for every module;
- a reusable saved `.ttt` scene and a hash-verifiable artifact bundle.

The nominal run must additionally show zero collision stops.

The recovery run must additionally show:

- robot disable at completion fraction `>= 0.25`;
- zero commands to that robot after its recorded stop-command cutoff;
- measured settling after disable, including the required consecutive
  low-velocity samples and displacement record;
- at least one remaining planned job reassigned away from it;
- planned and measured post-disable payload clearance to the settled disabled robot;
- full-cottage completion.

The process exits successfully only when every scenario-specific acceptance gate
passes and the evidence kind is `live_coppelia`.

## Evidence bundle schema

Each successful or diagnostically useful connected run produces:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Provenance, source/configuration/plan digests, limitations, and hashes |
| `scenario.json` | Exact generated cottage and fleet |
| `planned_jobs.json` | Deterministic job IDs, team lineage, approach/carry/return routes, transport heights, clearances, and final snaps |
| `planned_vs_measured_replay.json` | Job timestamps, route-phase start snapshots, typed clearance minima, planned routes, and measured pose samples |
| `wheel_commands.jsonl` | Every wheel command in simulator-time order |
| `measured_telemetry.jsonl` | Measured base poses and velocities |
| `trace.json` | Scene/pose-write inventory, every collision-query round, jobs, logical attachment, installation, and recovery events |
| `metrics.json` | Acceptance gates and run diagnostics |
| `report.md` | Human-readable result and limitations |
| `construction_intelligence.ttt` | Reusable saved live scene |

After both native directories pass verification, run
`scripts/package_construction_coppelia_evidence.py` to create the canonical simulator descriptor
and `evidence_replay.mp4` for each scenario. The renderer uses only measured robot-base telemetry,
the generated scenario, and the typed planned-versus-measured replay. Its overlay states that
payload motion is logical and makes no arm, gripper, grasp-contact, or payload-dynamics claim.

The manifest schema is
`construction_intelligence.coppelia_bundle.v1`. Every artifact except the manifest
itself has a SHA-256 digest and byte count. The run result schema is
`construction_intelligence.coppelia_evidence.v1`.

Before publishing evidence, run `verify_phase5_artifact_bundle()` and retain both
nominal and recovery directories as versioned release assets. Verification checks
the canonical inventory and hashes, parses every typed artifact, validates the
native Coppelia scene signature, reconstructs measured replay and recovery
invariants, and independently recomputes planned/measured segment clearance,
full-RPY transport height, disabled-robot exclusion, logical lift provenance,
final-snap provenance, and zero-motion transition proofs. Merely changing manifest
booleans or rehashing arbitrary files cannot create passing evidence.
