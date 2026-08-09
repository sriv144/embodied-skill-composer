# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## Prior runs (on unmerged `claude/sweet-clarke-*` branches)

- **2026-04-24** added `.github/workflows/tests.yml` (pytest on push/PR,
  Python 3.11, headless matplotlib) and an MIT `LICENSE`. That work is not
  merged to `main`; this run does not duplicate it.

## 2026-05-28 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~80 / 100 — top-tier tech (robotics
+ hierarchical RL + multi-agent + MuJoCo), a documented flagship result, and
strong existing docs.

**Branch:** `claude/sweet-clarke-0QUHr` (from `main`).

### Implemented

- **`docs/architecture.md`** — a Mermaid diagram of the
  perception → planning → option-execution → RL → backend flow, a component
  table mapped to the `src/` packages, the "why hybrid" rationale, and the
  runtime-profile matrix. (Previously only setup/results docs existed.)
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant v2.1, rounding out the OSS
  template set alongside the existing `CONTRIBUTING.md`.
- **`.github/dependabot.yml`** — weekly grouped updates for pip and GitHub
  Actions.
- **Seeded this `RESEARCH_LOG.md`.**

### Why this was prioritized

The codebase and README are already strong, so the highest-leverage *safe*
work is showcase/maintenance polish: a visual architecture doc makes the hybrid
design legible to a reviewer in seconds, and the CoC + Dependabot signal a
maintained, professional repo. All four files are additive — zero source code
touched, so there is no way to break the existing test suite or demos.

### Evaluated and skipped

- **Ruff / lint CI job.** No style config committed; adding one would surface a
  wall of pre-existing findings as spurious red. Needs a formatting-baseline
  commit first.
- **MuJoCo / PyBullet / GPU CI.** Need system packages or a GPU; brittle on
  hosted runners.
- **Isaac Lab backend stub.** Genuine feature work already on the roadmap, not
  an incremental polish.
- **Cross-platform (bash) README quickstart.** The README commands are all
  PowerShell; adding POSIX equivalents is worthwhile but is a focused docs PR.

### Next-run candidates

1. Add POSIX/bash equivalents to the README run matrix (currently PowerShell
   only).
2. Apply a `ruff format` baseline commit, then wire a lint CI job.
3. Weekly benchmark workflow that runs `scripts/benchmark_assembly_policies.py`
   and uploads the diagnostics JSON as an artifact.
4. Merge the unmerged `tests.yml` + `LICENSE` to `main`.
