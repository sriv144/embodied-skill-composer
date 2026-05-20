# Research Log

This file tracks autonomous codebase-improvement runs. Each entry records
what was implemented, why it was prioritized, and what was deferred.

## 2026-05-20 — Auto-Researcher v4

**Resume-worthiness score at start of run: 86 / 100**
(High tech-stack prestige: hierarchical RL, multi-agent coordination,
MuJoCo/PyBullet simulation. Recent commits, strong README, real test suite.)

### Implemented (branch `claude/sweet-clarke-sFMHv`)
- **Continuous Integration** — added `.github/workflows/ci.yml` running ruff
  linting and the full pytest regression suite on every push and pull
  request. The repository ships a substantial test suite (11 test modules
  covering planner, executor, perception, benchmarks, and the assembly
  hierarchical-options stack) plus ruff/mypy/pytest config in
  `pyproject.toml`, but had no CI to guard regressions. The workflow mirrors
  the documented minimal local setup (`requirements.txt` +
  `requirements-rl.txt`).

### Why this was prioritized
The project is feature-rich and well documented but had no automated
verification. A green CI badge materially improves how the repo reads to
recruiters and protects the flagship hierarchical-options result from silent
regressions. It is additive and cannot break existing functionality.

### Evaluated and skipped
- **MuJoCo / PyBullet integration tests in CI** — skipped; these need heavy
  optional simulation dependencies and GPU/headless rendering setup that is
  out of scope for a fast, reliable CI lane.
- **mypy gate in CI** — skipped this run; strict typing config exists but a
  type-clean baseline was not verified, so adding a blocking gate risked a
  red build. Candidate for a future run once the baseline is confirmed.
- **README rewrite** — skipped; the README is already comprehensive
  (architecture, 10 documented run modes, roadmap).

### Next-run candidates
- Add a non-blocking mypy job, then promote it to blocking once clean.
- Add an Isaac Lab backend stub behind the existing task contract.
- Cache the pip environment more aggressively and add a coverage report.
