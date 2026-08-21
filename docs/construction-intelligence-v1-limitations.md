# Construction Intelligence v1 Limitations

This document defines what Construction Intelligence v1 does not claim. These boundaries apply to
the local workbench, the Pages preview, research reports, simulator evidence, and eventual release
assets.

## Evidence status

- The five-seed MAPPO/IPPO and ablation matrix is not release evidence until every frozen Phase 4
  run, validation-only selection, held-out evaluation, and aggregate report completes.
- Unit, smoke, fixture, and partial research runs demonstrate infrastructure only.
- Full-cottage nominal and unavailable-robot Coppelia acceptance remains open until the two live
  bundles pass independent verification.
- Local Phase 6 checks do not close the workbench phase until protected remote checks pass and its
  review is merged.
- The deployed Pages surface remains a read-only preview until all release gates pass.

The [completion roadmap](roadmap.md) is the authoritative source for gate status.

## Robot and simulator boundary

- Coppelia execution controls KUKA YouBot bases with wheel commands and measures base telemetry.
- Payload transport is logical. V1 does not claim arm trajectories, gripper actuation, physical
  grasping, contact-based cooperative transport, or payload dynamics.
- Learned coordination operates at the high-level scheduling and team-formation layer. V1 does not
  include learned low-level locomotion or manipulation.
- Deterministic simulator and training environments retain privileged task state. V1 does not claim
  complete perception-only autonomy.
- CoppeliaSim must be started for live gates and its ZeroMQ API must be reachable on loopback.
- Release MP4s are deterministic top-down visualizations reconstructed from attested measured base
  telemetry. They are not simulator-camera footage and do not add physical payload/contact claims.

## Design and construction scope

- The editor supports a single orthogonal top-down footprint with axis-aligned walls and wall-bound
  doors/windows. It is not a general CAD or BIM editor.
- V1 does not provide IFC export, structural engineering analysis, building-code validation, richer
  material authoring, multi-story design, terrain modeling, or construction cost estimation.
- Compilation and routing are research abstractions. They are not safety-certified plans for
  physical construction.

## Product and deployment boundary

- The interactive backend is loopback-only, single-user, and intended for a trusted local machine.
  It is not an authenticated multi-user service.
- Static mode is read-only and cannot launch jobs, control CoppeliaSim, or infer live readiness.
- Local mode never silently falls back to preview fixtures when the API fails.
- GitHub Pages does not provide repository-configurable response headers. The preview therefore
  applies an early meta Content Security Policy and a no-referrer policy in its document, but meta
  policies cannot enforce `frame-ancestors` or `X-Content-Type-Options: nosniff`. If the workbench
  moves to another host, those controls must be configured as HTTP response headers at the edge.
- The production dependency audit (`npm audit --omit=dev`) is clean. The full development audit
  currently reports GHSA-mh99-v99m-4gvg through the upstream-only
  `eslint`/`eslint-plugin-jsx-a11y -> minimatch@3.1.5 -> brace-expansion@1.1.16` toolchain.
  No patched 1.x release exists; forcing `brace-expansion@5.0.8` breaks minimatch 3's callable
  CommonJS contract. This dev-only advisory remains explicitly tracked rather than hidden by an
  incompatible override or vulnerable downgrade.
- Large checkpoints, ONNX models, evaluation datasets, reports, scenes, telemetry, and videos belong
  in versioned release assets rather than the source repository.

## Deferred systems

V1 excludes Isaac Lab execution, ROS 2 integration, physical robots, sim-to-real validation,
humanoid embodiments, lunar or planetary environments, and removal of all privileged state.
`w9-pathfinding` remains optional; the deterministic reservation-table fallback is supported.

See the [architecture](construction-intelligence-v1-architecture.md) for component boundaries, the
[research protocol](construction-intelligence-v1-protocol.md) for evidence rules, and the
[Phase 5 runbook](construction-intelligence-v1-phase5-runbook.md) for the live-simulator gate.
