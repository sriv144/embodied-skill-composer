# Research Log

This file tracks Auto-Researcher passes against this repository: what was
implemented, what was evaluated and skipped, and what is queued for next run.

## 2026-06-05 — Auto-Researcher v4

**Resume score at start of run:** 68 / 100

embodied-skill-composer is the most technically prestigious repo of the
six — a real PPO + MuJoCo / PyBullet robotics RL stack with split
requirements files for CPU / CUDA / mujoco / pybullet, a CONTRIBUTING.md,
and a docs/ tree. The two visible gaps were a missing LICENSE and no CI.
Both were added safely without touching training code.

### Implemented (branch `claude/sweet-clarke-eZfYm`)

- `LICENSE` (MIT) so the repo is legally usable / forkable.
- `.github/workflows/ci.yml`:
    - **ruff** `E9,F63,F7,F82` over `src/` and `tests/` for real syntax
      and undefined-name bugs.
    - `python -m compileall` belt-and-braces import-time smoke check.

### Why this was prioritized

LICENSE + CI signal are the two universal trust markers reviewers scan
for first. Zero behavior changes, zero new runtime dependencies, no
impact on the GPU / mujoco / pybullet install paths. The training
pipeline depends on heavy native deps (torch + cu130 wheels, mujoco,
pybullet) that would push CI runtime past 10 min and risk flaky failures
— so this run intentionally limits CI to static checks that pass
deterministically on a stock ubuntu runner.

### Evaluated and skipped

- **Run pytest in CI.** The test tree imports from `src/` which pulls in
  torch + gym + mujoco; cold-installing those on every PR is expensive and
  flaky. Deferred until a `tests/conftest.py` adds `pytest.importorskip`
  shims so the unit tests can run without the simulator stack.
- **Add an Isaac Lab smoke job.** The recent commits build toward an
  Isaac backend; a CI smoke test would be valuable but Isaac is not
  installable on a stock GitHub runner. Deferred.
- **Polish the README with a screenshot / GIF of a trained policy.**
  Real showcase win, but requires generating media that should be
  reviewed before publishing. Deferred.

### Candidates for next run

1. Add a `tests/conftest.py` with `pytest.importorskip("mujoco")` /
   `"torch"` guards so the unit tests can run hermetically in CI.
2. Wire a separate GPU-required workflow gated by `workflow_dispatch`
   for the full training smoke test.
3. Add a screenshot or short GIF of a trained PPO policy in the README
   hero section.
4. Add a Mermaid diagram to `docs/` showing the
   `env -> wrappers -> PPO -> rollout buffer -> update` loop.
