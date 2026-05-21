# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## 2026-05-21 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~87 / 100 — the strongest
project on the account (top of 6).

**Branch:** `claude/sweet-clarke-InUHR` (from `main`).

### Implemented

- **`.github/workflows/tests.yml`** — runs `pytest -q` on every push and PR
  to `main`, Python 3.11, after installing `requirements.txt` +
  `requirements-rl.txt`. Headless via `MPLBACKEND=Agg`, pip-cached on both
  requirements files, cancel-in-progress concurrency, 20-minute timeout.
  `main` previously had no CI signal at all.
- **MIT `LICENSE`** — the repo was previously unlicensed.
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant 2.1, rounding out the OSS
  template set alongside the existing `CONTRIBUTING.md`.
- **README status badges** + a Continuous Integration section.
- **Seeded this `RESEARCH_LOG.md`.**

### Why this was prioritized

The codebase already has a flagship result, 11 pytest files, multi-profile
requirements, and a CONTRIBUTING guide. The single missing load-bearing
piece on `main` was a verifiable CI signal — it converts the green-badge
story from "claimed" to "verified by Actions" and makes every future
regression visible. All changes are purely additive: zero risk to existing
functionality.

### Evaluated and skipped

- **Ruff / black lint job.** `ruff` is pinned in `requirements.txt` but no
  style config exists; a lint job would surface a wall of pre-existing
  findings as spurious CI red. Defer to a dedicated lint-baseline run.
- **MuJoCo / PyBullet / CUDA CI jobs.** Hosted runners lack the system
  packages and GPU; keep CI on the pure-Python regression path.
- **Isaac Lab backend stub.** Genuine feature work already on the roadmap,
  not an incremental polish.

### Next-run candidates

1. Add a `ruff` lint job with the formatting baseline applied in the same PR.
2. Weekly `benchmark` job running `scripts/benchmark_assembly_policies.py`,
   uploading the diagnostics JSON as an artifact.
3. `docs/architecture.md` with a planner / executor / perception / backend
   diagram.
4. Dockerize the CPU regression profile for reproducible local evaluation.
