# Construction Intelligence v1 — Phase 5 Coppelia Runbook

Phase 5 requires two real, manually approval-gated CoppeliaSim runs. The automated
test harness proves orchestration and evidence-schema behavior, but it is explicitly
recorded as `offline_harness` and can never satisfy the live gate.

## Simulator boundary

The live executor commands KUKA YouBot wheel joints in deterministic stepping mode
and samples robot poses and velocities from CoppeliaSim. Direct robot pose writes are
allowed only while constructing the initial scene, before simulation starts.

Payload transport remains logical. A module is parented to a carrier dummy that is
synchronized to the measured robot-base centroid while preserving its attested
pickup offset. Phase 5 does **not**
claim arm motion, gripper actuation, grasp contact, cooperative contact dynamics, or
physical payload dynamics.

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
- one complete planned-versus-measured replay entry for every module;
- a reusable saved `.ttt` scene and a hash-verifiable artifact bundle.

The nominal run must additionally show zero collision stops.

The recovery run must additionally show:

- robot disable at completion fraction `>= 0.25`;
- zero commands to that robot after its recorded stop-command cutoff;
- measured settling after disable, including the required consecutive
  low-velocity samples and displacement record;
- at least one remaining planned job reassigned away from it;
- full-cottage completion.

The process exits successfully only when every scenario-specific acceptance gate
passes and the evidence kind is `live_coppelia`.

## Evidence bundle schema

Each successful or diagnostically useful connected run produces:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Provenance, source/configuration/plan digests, limitations, and hashes |
| `scenario.json` | Exact generated cottage and fleet |
| `planned_jobs.json` | Deterministic job IDs, team lineage, and routes |
| `planned_vs_measured_replay.json` | Job timestamps, planned routes, and measured pose samples |
| `wheel_commands.jsonl` | Every wheel command in simulator-time order |
| `measured_telemetry.jsonl` | Measured base poses and velocities |
| `trace.json` | Scene/pose-write inventory, every collision-query round, jobs, logical attachment, installation, and recovery events |
| `metrics.json` | Acceptance gates and run diagnostics |
| `report.md` | Human-readable result and limitations |
| `construction_intelligence.ttt` | Reusable saved live scene |

The manifest schema is
`construction_intelligence.coppelia_bundle.v1`. Every artifact except the manifest
itself has a SHA-256 digest and byte count. The run result schema is
`construction_intelligence.coppelia_evidence.v1`.

Before publishing evidence, run `verify_phase5_artifact_bundle()` and retain both
nominal and recovery directories as versioned release assets. Verification checks
the canonical inventory and hashes, parses every typed artifact, validates the
native Coppelia scene signature, reconstructs measured replay and recovery
invariants, and independently recomputes the live gates; merely changing manifest
booleans or rehashing arbitrary files cannot create passing evidence.
