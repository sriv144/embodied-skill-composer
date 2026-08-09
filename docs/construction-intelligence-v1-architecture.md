# Construction Intelligence v1 Architecture

Construction Intelligence v1 is a local, single-user research workbench for compiling a reviewed
cottage design into modular construction work, coordinating four mobile robots, and packaging the
resulting deterministic, learned-policy, and live-simulator evidence. The browser is the product
surface; Python owns authoritative design validation, compilation, experiments, and evidence
verification.

## End-to-end data flow

```text
HouseDesign
  -> shared design validation and explicit approval
  -> deterministic cottage compiler
  -> modules + dependencies + resource constraints
  -> deterministic or learned coordination policy
  -> event trace + metrics + report
  -> browser digital-twin replay

Frozen experiment protocol
  -> durable single-GPU run queue
  -> resumable checkpoints and validation-only policy selection
  -> deterministic held-out evaluation
  -> confidence intervals + acceptance + ablations

Deterministic full-cottage plan
  -> live Coppelia wheel commands
  -> measured robot telemetry and collision/contact observations
  -> nominal and unavailable-robot evidence bundles

Canonical research + simulator evidence
  -> fail-closed public-demo exporter
  -> provenance manifest with hashes and source commits
  -> read-only static workbench and versioned release assets
```

## Design and compilation

`HouseDesign` is the wire contract shared by the editor, API, compiler, and fixture exporter. The
orthogonal editor creates and changes axis-aligned walls and wall-bound doors/windows on a metric
grid. Any edit clears approval. The API and compiler call the same validation implementation, so a
client cannot bypass bounds, dimensions, intersections, or opening-placement rules by posting a
handwritten payload.

The compiler deterministically emits transportable foundation, wall, floor, and roof modules plus
their dependency graph and installation poses. Multiple openings retain their exact wall, offset,
width, height, and sill semantics through compilation and Blender geometry generation.

## Planning and coordination

The deterministic foundation provides sequential, greedy, auction, and CP-SAT controllers,
reservation-table routing, synchronized multi-robot carries, recovery, and trace/report generation.
The optional `w9-pathfinding` integration is an optimization; the tested deterministic fallback is
part of the supported v1 contract.

`construction_coordination_v1` is a PettingZoo temporal environment. Travel, pickup, carrying,
installation, battery use, obstacles, drops, and robot failures consume simulated time. Policies use
masked parameter-shared actors. MAPPO uses a centralized critic; IPPO uses independent critics.
CP-SAT demonstrations may warm-start policies through behavior cloning.

The frozen `construction_intelligence_v1` protocol defines the only release research matrix,
training/validation/held-out splits, checkpoint fractions, policy selection order, deterministic
evaluation suite, confidence method, and acceptance thresholds. Held-out seeds are never model
selection inputs.

## Durable experiment lab

The interactive API and CLI route training through a SQLite-backed subprocess queue with one active
GPU training job. Runs retain configuration and source fingerprints, attempts, PID identity,
heartbeats, progress, cancellation state, checkpoints, artifact locations, and typed append-only
events mirrored to JSONL.

Full checkpoints contain actor, critic, optimizers, counters, learning curves, RNG state,
configuration/design digests, environment schema, commit/tree identity, and lineage. Resume rejects
incompatible or missing declared checkpoints. A failure before the first checkpoint restarts the
same immutable configuration from transition zero and records that provenance explicitly.

## Live Coppelia boundary

The Phase 5 controller drives KUKA YouBot bases using wheel commands and closes the loop from measured
poses. Its verifier requires a complete generated cottage, physical collision/contact queries,
per-robot command response, formation tolerances, planned-versus-measured route/timestamp matching,
zero robot pose writes after initialization, and exact scene/telemetry/trace/report artifacts.

The recovery gate disables one robot after at least 25% completion, removes it from subsequent
command targets, observes it settling, reallocates its remaining work, and still completes the
cottage.

Payload transport is deliberately logical. V1 does not claim arm motion, gripper control,
grasp/contact dynamics, learned low-level manipulation, or physical payload attachment.

## Workbench runtime modes

Local mode connects to the loopback-only API for design approval, compilation, planning, simulation,
training matrices, WebSocket events, cancellation/resume, evaluations, and verified artifact links.
An API failure is an error and never silently changes the data source.

Static mode reads only the canonical public-demo bundle. It is visibly read-only and exposes the
fixture, research, and simulator evidence status recorded by the provenance manifest. It does not
pretend to launch local work or infer live readiness from a completed run label.

## Evidence and release boundary

The public-demo exporter copies an allowlisted set of concrete artifacts, hashes every file, records
source commits/configuration digests, and verifies regeneration. Preview exports label fixtures and
absent evidence. Release exports fail closed unless the complete 20-run research matrix, selected
policies, held-out reports, primary acceptance gates, ablation decisions, reproducibility audit, and
both independently verified live Coppelia bundles are present.

Large checkpoints, ONNX models, evaluation data, reports, scenes, telemetry, and videos are release
assets rather than source-tree run directories. The deployed Pages workbench contains the compact
read-only evidence bundle.

## V1 boundaries

The release excludes Isaac Lab execution, ROS 2, physical grippers, learned low-level manipulation,
IFC export, sim-to-real claims, richer material authoring, and complete removal of privileged
simulator state. The local API remains loopback-only and single-user. These boundaries are product
claims, not a backlog hidden behind the demo.
