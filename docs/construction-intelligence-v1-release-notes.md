# Construction Intelligence v1.0.0

Construction Intelligence v1 is the first reproducible public release of the
full-cottage coordination system. It combines deterministic construction planning,
durable multi-agent policy training, measured CoppeliaSim execution, and a local or
read-only browser workbench behind hash-verified evidence artifacts.

## What is included

- A deterministic cottage compiler, four-robot fleet planner, routing, scheduling,
  recovery, trace, replay, and reporting pipeline.
- A frozen five-seed MAPPO/IPPO research protocol with validation-only policy
  selection, held-out evaluation, hierarchical confidence intervals, baseline
  comparisons, and the complete behavior-cloning and failure-curriculum ablations.
- Wheel-driven CoppeliaSim nominal and unavailable-robot recovery runs with measured
  base telemetry, reusable scenes, planned-versus-measured replays, and deterministic
  measured-replay videos.
- A React/Three.js workbench covering floor-plan editing, explicit approval,
  compilation, planning, simulation replay, durable experiments, results, and
  evidence inspection at desktop and mobile widths.
- Selected PyTorch checkpoints and deterministic ONNX actor exports for all 20
  validation-selected policies.

## Release assets

The release contains exactly five top-level assets:

1. `construction-intelligence-v1-public-demo.zip` — the hash-pinned static workbench
   data and canonical public evidence surface.
2. `construction-intelligence-v1-research-evidence.zip` — frozen protocol, run,
   selection, evaluation, aggregation, threshold, and ablation artifacts.
3. `construction-intelligence-v1-coppelia-evidence.zip` — independently verified
   native nominal/recovery bundles plus measured-replay videos.
4. `construction-intelligence-v1-selected-policies.zip` — selected checkpoints,
   manifests, ONNX actors, and any required external-data sidecars.
5. `construction-intelligence-v1-release-assets.json` — source/tag identity, exact
   asset inventory, byte counts, and SHA-256 digests.

Every archive is deterministic, path-bounded, size-bounded, CRC-checked, and bound to
the protected source commit and `v1.0.0` tag. The publication workflow verifies the
already-uploaded draft assets before making the release public; it does not rebuild or
repair evidence in CI.

## Reproduce and inspect

- [Setup and documentation index](https://github.com/sriv144/embodied-skill-composer/blob/v1.0.0/docs/construction-intelligence-v1.md)
- [Research protocol](https://github.com/sriv144/embodied-skill-composer/blob/v1.0.0/docs/construction-intelligence-v1-protocol.md)
- [Architecture](https://github.com/sriv144/embodied-skill-composer/blob/v1.0.0/docs/construction-intelligence-v1-architecture.md)
- [Release asset procedure](https://github.com/sriv144/embodied-skill-composer/blob/v1.0.0/docs/construction-intelligence-v1-release.md)
- [Limitations](https://github.com/sriv144/embodied-skill-composer/blob/v1.0.0/docs/construction-intelligence-v1-limitations.md)
- [Public workbench](https://sriv144.github.io/embodied-skill-composer/)

## Simulator boundary

CoppeliaSim evidence drives mobile bases with wheel commands and closes the loop from
measured poses. Payload attachment, vertical lift, transport height, descent, and
final installation snap are explicitly logical. This release does **not** claim
physical arm/gripper manipulation, cooperative contact dynamics, ROS 2 integration,
IFC export, Isaac Lab parity, or sim-to-real transfer.

See the versioned limitations and the native evidence manifests before interpreting
the results outside this boundary.
