# Research Log

Long-running notebook for the auto-researcher: what was evaluated, what was
shipped, and what is queued for next runs.

## 2026-06-04 - Auto-Researcher v4

**Resume score going in:** 85 / 100

The README is genuinely excellent: it documents the hierarchical-options
result, lists ten distinct entry-point scripts, and explains *why* the
project is hybrid (perception + explicit planning + selective RL). The
biggest unforced error against that quality bar is the absence of any CI
signal - there is no `.github/` folder at all, so `pytest` only runs on the
author's laptop.

### Implemented (branch: claude/sweet-clarke-ahxvL)

- **feat: `.github/workflows/ci.yml`** - matrix CI on Python 3.11 and
  3.12, installs `requirements.txt` plus `requirements-rl.txt` if it
  exists, runs `ruff check` non-blocking, then `pytest -q`. The RL extras
  install is wrapped in an `if [ -f ... ]` so a future split of optional
  deps does not break the workflow.
- **chore: seed RESEARCH_LOG.md** with this entry.

### Why this was prioritized

- The repo already documents `python -m pytest -q --basetemp .pytest_tmp`
  as the canonical local check, and the `tests/` directory contains 11
  test modules covering assembly env / benchmark / tooling / training,
  the warehouse collection planner / perception / integration, and the
  tabletop executor. Putting all of that on every push is high-value.
- `requirements.txt` is intentionally lightweight (numpy / matplotlib /
  pydantic / networkx / pyyaml / pytest / ruff / mypy) so the workflow is
  fast and unlikely to flake.

### Evaluated and skipped

- **MuJoCo / Isaac job.** Both backends are documented in the README as
  optional Windows / Linux-NVIDIA local profiles. Running them headless
  in CI is fragile (MuJoCo GL contexts) and unnecessary for a correctness
  signal. Out of scope.
- **mypy gate.** `mypy>=1.10` is already in `requirements.txt`. A typing
  gate would be valuable but requires an audit of the current type
  coverage before turning it strict. Deferred.
- **Codecov / coverage badge.** Nice-to-have, not load-bearing for a
  research repo. Skipped.
- **Touching the assembly code or MARL training pipeline.** The current
  results table in the README (`scripted=1.000, hierarchical=1.000,
  low-level MARL=0.000`) is the headline result. Touching any of the
  underlying code in an auto-researcher run is too risky relative to the
  marginal gain.

### Next-run candidates

1. Add a `mypy` job once `src/` type coverage is audited.
2. Add a job that renders the assembly playback visualizer in headless
   mode and uploads the summary PNG as an artifact - that gives reviewers
   an inline preview of the flagship result on every push.
3. Add a CONTRIBUTING.md badge to README pointing at the new CI.
4. Audit whether the RL training scripts can be exercised in a tiny
   `episodes=2` smoke run inside CI.
