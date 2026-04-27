# Research Log

This file tracks autonomous research and improvement runs against this repository.
Each entry summarizes the date, what was reviewed, what was implemented, and
what was deferred.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100 — top 1 of 6 across the portfolio.

**Branch:** `claude/sweet-clarke-snPHW`.

### Implemented

- Added `.github/workflows/ci.yml` running pytest on Python 3.11 and 3.12 against
  `requirements.txt` + `requirements-rl.txt`. Surfaces regressions in the
  assembly env, hierarchical options trainer, planner, executor, and warehouse
  benchmark on every push.
- Added an MIT `LICENSE` so the repo is unambiguously open and reusable for
  portfolio reviewers.
- Seeded this `RESEARCH_LOG.md`.

### Why this was prioritized

This repo is the highest-scoring on the portfolio dashboard — clear
hierarchical-options story with a verified `2/2 beams` flagship result, MuJoCo
backend, and a structured roadmap to Isaac Lab. It already has
`CONTRIBUTING.md` and a strong README, but was missing two basic engineering
signals: a CI badge and a license. Both are the cheapest wins that materially
raise interview-grade polish without touching code paths.

### Evaluated and skipped

- **README rewrite:** the README is already strong and accurately reflects the
  flagship results. Touching it risked drift.
- **Strict ruff/mypy CI jobs:** ruff and mypy are listed in `requirements.txt`
  but no shared config is checked in. A strict job would likely create
  immediate CI red. Deferred until config is pinned in `pyproject.toml`.
- **Isaac Lab backend stub:** scoped well beyond a single auto-research pass;
  correct path forward is a dedicated multi-day implementation.

### Next-run candidates

- Add a `ruff` and `mypy` CI job once their config is checked into
  `pyproject.toml`.
- Add a short demo GIF or `docs/results/` screenshot referenced from the
  README "Flagship Result" section.
- Stub the Isaac Lab backend so the existing assembly task contract can be
  loaded against `IsaacAssemblyAdapter` and benchmarked.
