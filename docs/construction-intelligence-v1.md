# Construction Intelligence v1 Setup and Documentation Index

Construction Intelligence v1 is the current flagship of Embodied Skill Composer. It turns an
approved orthogonal cottage design into modular work, coordinates a four-robot fleet, runs
reproducible learned-policy experiments, and packages deterministic and simulator evidence for the
browser workbench.

The deployed Pages workbench remains a **read-only preview**. Phase 4 research, Phase 5 live
Coppelia evidence, Phase 6 protected review, and the final v1.0.0 release gate remain open; consult
the [completion roadmap](roadmap.md) for the authoritative status.

## Quick start

Create the complete editable Python environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[construction,construction-rl,dev,rl,sim-coppelia,sim-mujoco]"
.\.venv\Scripts\python.exe -m pip check
```

Install and launch the local workbench:

```powershell
cd workbench
npm ci
cd ..
powershell -ExecutionPolicy Bypass -File scripts\start_construction_workbench.ps1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The primary journey is
**Design &rarr; Brain &rarr; Simulate &rarr; Experiments &rarr; Results**. Local mode uses the loopback API for
editing, compilation, simulation, experiment control, and verified artifact access.

To build the read-only preview instead:

```powershell
.\.venv\Scripts\python.exe scripts\export_construction_public_demo.py --channel preview
cd workbench
npm run build:static
```

Preview export labels absent or fixture evidence honestly. It does not satisfy the research,
live-simulator, or release gates.

## Runtime modes

| Mode | Data source | Mutating actions | Intended use |
| --- | --- | --- | --- |
| Local | Loopback-only Python API | Design, compile, simulate, queue, resume, and cancel | Single-user research workbench |
| Static preview | Hash-recorded public-demo files | None | Pages review while release gates remain open |
| Static release | Canonical release bundle | None | Available only after every v1 gate passes |

A local API failure remains an error; the application never silently switches to static fixture
data.

## Documentation map

| Document | Purpose |
| --- | --- |
| [Completion roadmap](roadmap.md) | Authoritative phase status, evidence links, and remaining gates |
| [Architecture](construction-intelligence-v1-architecture.md) | Design-to-evidence data flow, component ownership, and runtime boundaries |
| [Research protocol](construction-intelligence-v1-protocol.md) | Frozen matrix, seed isolation, checkpoint selection, evaluation, and acceptance rules |
| [Reproducibility](construction-intelligence-v1-reproducibility.md) | Environments, quality gates, fingerprints, durable jobs, and strict resume behavior |
| [Public-demo provenance](construction-intelligence-public-demo-provenance.md) | Preview/release channels, input descriptors, hashes, and fail-closed packaging |
| [Release assets](construction-intelligence-v1-release.md) | Deterministic ZIP staging, draft verification, and fail-closed GitHub publication |
| [Phase 3 evidence](construction-intelligence-v1-phase3-evidence.md) | Protocol smoke matrix and protected-check evidence |
| [Phase 5 runbook](construction-intelligence-v1-phase5-runbook.md) | Required live Coppelia nominal and unavailable-robot runs |
| [Phase 6 evidence](construction-intelligence-v1-phase6-evidence.md) | Local editor, workbench, accessibility, responsive, and browser acceptance evidence |
| [V1 limitations](construction-intelligence-v1-limitations.md) | Explicit research, simulator, product, and deployment boundaries |
| [Dependencies and assets](dependencies-and-assets.md) | Package, license, model, and visual-asset provenance |

Platform-specific legacy and future-backend setup notes remain in
[Windows/VS Code setup](setup/windows-vscode.md) and
[Linux/NVIDIA/Isaac preparation](setup/linux-nvidia-isaac.md). Isaac Lab is not part of v1.

## Evidence interpretation

- Unit and smoke profiles validate plumbing; they are not research evidence.
- Validation seeds select checkpoints. Held-out seeds never influence selection or tuning.
- A completed registry row is not by itself proof of learned-policy or live-simulator acceptance.
- Phase 5 requires independently verified nominal and unavailable-robot bundles from CoppeliaSim.
- Release packaging fails closed when a canonical artifact, hash, source identity, or acceptance gate
  is missing.

Read the [limitations](construction-intelligence-v1-limitations.md) alongside every result or demo.
