# Construction Intelligence v1 Completion Roadmap

## Release checklist

- [x] Foundation implementation
- [x] Phase 1 — publish and protect the checkpoint
- [x] Phase 2 — quality, reproducibility, and durable jobs
- [x] Phase 3 — freeze the research protocol
- [ ] Phase 4 — run the learned-policy evidence
- [ ] Phase 5 — complete dynamic Coppelia evidence
- [ ] Phase 6 — complete the product workbench
- [ ] Phase 7 — publish Construction Intelligence v1.0.0

Phase evidence:

| Phase | Acceptance evidence |
| --- | --- |
| 1 | PR #19 merged as `8d88fa8`; protected `main`; Python, workbench, and secrets checks green; native secret scanning and push protection enabled; Pages preview returned HTTP 200. |
| 2 | PR #20 merged as `a23aadb` after duplicate protected-branch and pull-request matrices passed: Python, workbench, and secret checks all green. The current-commit Pages build/deploy passed and the public workbench returned HTTP 200. On clean commit `5d30174`, 213 tests passed and 2 live-Coppelia tests skipped with 82.19% combined statement/branch coverage; Ruff, full-source mypy (111 files), `pip check`, compile, secret scan, lock regeneration, and static build passed. Production subprocess run `20260716T080821Z-training-83d0f663` completed with checkpoint, ONNX, manifest, JSONL events, and clean source/environment fingerprints. Research-profile run `20260716T080907Z-training-6717722c` was terminated after its checkpoint, reconciled to `interrupted`, resumed as attempt 2 from the same zero-transition state (`training_resumed`), then cancelled cleanly. See [reproducibility](construction-intelligence-v1-reproducibility.md). |
| 3 | PR [#21](https://github.com/sriv144/embodied-skill-composer/pull/21) passed the protected Python, workbench, secret-scan, and Gitleaks gates. Its clean-commit production subprocess smoke completed all 20 runs, a real interruption/reconciliation/resume, 100 checkpoint exports, 1,000 validation episodes, and the complete 240-episode held-out matrix. See the [Phase 3 evidence record](construction-intelligence-v1-phase3-evidence.md) and [research protocol](construction-intelligence-v1-protocol.md). |
| 4 | Pending the complete 20-run research matrix, validation-only selections, held-out evaluation, threshold results, and ablation report. |
| 5 | Pending live full-cottage nominal and unavailable-robot evidence. |
| 6 | Local editor, lab UX, accessibility, responsive, and browser gates pass; protected review is pending. See the [Phase 6 evidence record](construction-intelligence-v1-phase6-evidence.md). |
| 7 | Pending canonical artifact publication, tag, and release. |

The checklist advances only after a phase's branch is reviewed, required checks are green, and the
linked evidence exists. Product work may proceed while the serial GPU queue runs.

## Open release gates

### Phase 4 - learned-policy evidence

The frozen research queue is active and is producing checkpoints and run artifacts. Those partial
artifacts are not release evidence. Phase 4 remains open until all 20 research runs complete, each
policy is selected only with validation seeds `800-804`, every selected policy is evaluated
deterministically on held-out seeds `900-904`, the three primary thresholds pass, and the full
ablation matrix has a reproducible report.

The source of truth for matrix membership, checkpoint fractions, split isolation, policy selection,
hierarchical bootstrap intervals, thresholds, and ablation interpretations is the
[versioned research protocol](construction-intelligence-v1-protocol.md). Environment fingerprints,
strict resume semantics, and quality gates are defined in
[reproducibility](construction-intelligence-v1-reproducibility.md).

### Phase 5 - live Coppelia evidence

Phase 5 requires two real full-cottage runs: a nominal deterministic schedule and a recovery run
that disables one robot after at least 25% completion. Both must preserve zero post-initialization
robot pose writes and publish reusable scenes, telemetry, metrics, traces, and reports. Logical
payload transport remains an explicit limitation.

Follow the [Phase 5 Coppelia runbook](construction-intelligence-v1-phase5-runbook.md). Offline and
fixture evidence cannot satisfy this gate.

### Phase 6 - product workbench

The local editor, runtime modes, experiment controls, evidence views, responsive flows, and
accessibility checks are implemented and locally verified. The phase remains open until its branch
passes protected review and all required remote checks. See the
[Phase 6 evidence record](construction-intelligence-v1-phase6-evidence.md).

The primary product journey is
**Design &rarr; Brain &rarr; Simulate &rarr; Experiments &rarr; Results**. Static
mode remains read-only; a local API failure must remain visible and must never silently substitute
fixture data.

### Phase 7 - public v1.0.0 release

Release packaging remains blocked on canonical Phase 4 and Phase 5 evidence and a completed Phase 6
review. The final gate requires a hash-pinned public demo, consolidated documentation, versioned
release assets, the complete quality matrix, a successful Pages smoke test, package version
`1.0.0`, tag `v1.0.0`, and a public GitHub release.

The deployed Pages workbench remains a **preview** until this gate passes. See the
[public-demo provenance contract](construction-intelligence-public-demo-provenance.md) and
[v1 limitations](construction-intelligence-v1-limitations.md).

## Current implementation reference

The implementation foundation already provides:

- deterministic cottage generation, design validation, modular compilation, routing, planning,
  recovery, traces, reports, and browser replay;
- a temporal PettingZoo coordination environment, CP-SAT demonstrations, MAPPO/IPPO training,
  durable single-GPU jobs, strict resumable checkpoints, ONNX export, and experiment APIs;
- dynamic wheel-driven Coppelia execution with measured telemetry and explicitly logical payload
  transport;
- a local React/Three.js workbench and a read-only static preview;
- fail-closed evidence verification and public-demo provenance packaging.

Start with the [v1 setup and documentation index](construction-intelligence-v1.md), then use the
[architecture](construction-intelligence-v1-architecture.md) and
[limitations](construction-intelligence-v1-limitations.md) to interpret the system's scope.

## Historical roadmap appendix - superseded

This appendix records how the project reached the current v1 foundation. Its milestone names and
statuses are **historical** and must not be used as current release status; only the checklist above
controls the v1 gates.

| Historical track | Retained outcome | Current relationship |
| --- | --- | --- |
| Tabletop and warehouse baselines | Perception, planning, retry, benchmark, and lightweight RL scaffolding | Regression and comparison surfaces, not the v1 flagship |
| Two-robot assembly | Scripted options, learned hierarchical options, low-level MARL comparison, and playback artifacts | Research baseline retained alongside the four-robot cottage system |
| ConstructionBrain v0 | Typed observation/decision boundary, scripted and heuristic coordinators, and experiment records | Predecessor to the current coordination and lab contracts |
| MuJoCo backend spike | Physics-stepped assembly, sensing/noise studies, RGB/depth tracking, and saved videos | Optional earlier physical-AI evidence; not a v1 release gate |
| Coppelia backend and Modular Room spikes | Deterministic remote-API scenes and reusable `.ttt` outputs | Kinematic predecessor to the Phase 5 wheel-driven full-cottage gate |
| Construction v2 product track | Approved cottage, 24-module deterministic fixture, four-robot schedules, Blender assets, and browser digital twin | Deterministic foundation of Construction Intelligence v1 |
| Isaac Lab parity proposal | Linux/NVIDIA preparation and backend-contract notes | Deferred beyond v1; no Isaac execution claim |

The historical north star remains available in [vision.md](vision.md). Retained assembly results are
documented in [assembly hierarchical options](results/assembly-hierarchical-options.md), and future
NVIDIA preparation is explicitly separated in [Isaac preparation](isaac-prep.md).
