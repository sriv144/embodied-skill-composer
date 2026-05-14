# Research Log

This file tracks autonomous research and improvement runs against this repository.
Each entry summarizes the date, what was reviewed, what was implemented, and
what was deferred.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100 — top 1 of 6 across the portfolio.

**Branch:** `claude/sweet-clarke-snPHW`.

### Implemented

- Added `.github/workflows/ci.yml` running pytest on Python 3.11 and 3.12
  against `requirements.txt` + `requirements-rl.txt`. Surfaces regressions in
  the assembly env, hierarchical options trainer, planner, executor, and
  warehouse benchmark on every push.
- Added an MIT `LICENSE` so the repo is unambiguously open and reusable for
  portfolio reviewers.
- Seeded this `RESEARCH_LOG.md`.

### Next-run candidates

- Add a `ruff` and `mypy` CI job once their config is checked into
  `pyproject.toml`.
- Add a short demo GIF or `docs/results/` screenshot referenced from the
  README "Flagship Result" section.
- Stub the Isaac Lab backend so the existing assembly task contract can be
  loaded against `IsaacAssemblyAdapter` and benchmarked.

## 2026-05-14 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100 — still top 1 of 6 across the
portfolio.

**Branch:** `claude/sweet-clarke-5vVWf`.

### Implemented

No code changes this run. This commit only updates the research log so future
autonomous runs do not duplicate work that is already sitting on an unmerged
`claude/*` branch.

### Why no implementation this run

The two safe, ready-to-ship items from 2026-04-27 — CI workflow and MIT
LICENSE — already exist on the prior `claude/sweet-clarke-snPHW` branch and
are waiting for the maintainer to merge into `main`. Re-implementing them on
a new branch would create two parallel CI files and two parallel LICENSE
files on different claude branches with conflicting SHAs, which is worse than
leaving the merge to the maintainer.

The remaining 2026-04-27 next-run candidates all have hard prerequisites that
a one-shot autonomous run cannot satisfy safely:

- **ruff/mypy CI:** requires a pinned `pyproject.toml` config first; adding
  strict gates without it would either go red on day one or silently no-op.
- **Demo GIF / screenshot:** needs an actual artifact captured from a real
  MuJoCo or PyBullet run. Cannot be generated from inside this agent.
- **Isaac Lab backend stub:** scoped well beyond a single auto-research pass;
  correct path forward is a dedicated multi-day implementation.

### Next-run candidates

After `claude/sweet-clarke-snPHW` merges to `main`:

1. Land `pyproject.toml` with shared `ruff` + `mypy` config, then add a
   `lint` job to the existing CI workflow that gates on `ruff check` and
   `mypy backend`.
2. Capture and commit a short demo GIF or screenshot of the flagship
   `2/2 beams` assembly result under `docs/results/`, referenced from the
   README.
3. Begin the Isaac Lab adapter stub: a thin `IsaacAssemblyAdapter` that
   implements the same env contract as the MuJoCo adapter, gated behind an
   optional `[isaac]` extra in `requirements.txt`.
4. Add a CONTRIBUTING-friendly `make smoke` target that runs the
   2-beams benchmark end-to-end on CPU in under 60s.
