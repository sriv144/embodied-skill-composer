# Research Log

This log tracks autonomous-research / auto-improvement passes over the
repository. Each entry records what was scored, what was implemented,
and what was deliberately skipped, so future runs can avoid re-doing
work that has already shipped.

## 2026-06-02 — Auto-Researcher v4

### Resume-worthiness score at start of run

`88 / 100` — ranked #1 of 6.

Breakdown:

- Tech stack prestige: 25 / 25 — Multi-agent RL, hierarchical options,
  PPO, MuJoCo, planned Isaac Lab. Top-tier robotics stack.
- Commit recency: 22 / 25.
- Feature completeness: 19 / 20 — Flagship hierarchical-options policy
  solves the 2/2 beams assembly task with a published benchmark vs.
  scripted and low-level-MARL baselines.
- Stars / visibility: 8 / 15.
- README quality: 14 / 15 — Excellent, with verified results table,
  10 run-modes documented, and architecture notes.

### Implemented on branch `claude/sweet-clarke-MN9a4`

- **ci: add `.github/workflows/ci.yml`.** Runs the existing pytest
  suite (12 modules covering tabletop, warehouse perception / planner,
  collaborative assembly env / tooling / training / benchmark, and
  integration) on Python 3.11 and 3.12 for every push and PR. The
  workflow installs the base `requirements.txt` and, if present, the
  optional `requirements-rl.txt`. This was the highest-leverage,
  lowest-risk change available because the test suite already exists
  and is fully self-contained — no API keys, no external services.
- **docs: seed this `RESEARCH_LOG.md`.**

### Evaluated and skipped

- Adding a `ruff` lint job. Skipped this run because `ruff check` over
  a research codebase usually produces a flurry of low-signal warnings
  on first contact, and fixing them all is a separate, larger PR.
  Queued as a follow-up: introduce ruff with an explicit ignore list
  in `pyproject.toml`.
- Adding `mypy` strict checking. Same reason — the repo already lists
  `mypy` in `requirements.txt`, but no `mypy.ini` / `pyproject` config
  is in place. Queued.
- Pushing the Isaac Lab backend stub. Skipped — the README roadmap
  flags this as the next research milestone and it is too large for
  an auto-researcher pass.
- Touching any `src/` code. Skipped to keep this commit purely
  additive and risk-free; the flagship policy already works.

### Candidates for next run

1. Introduce `pyproject.toml` with a curated `ruff` config and wire it
   into CI so style is enforced going forward.
2. Add a `mypy` job (advisory at first, then required) targeting
   `src/embodied_skill_composer/assembly/`.
3. Cache the MuJoCo wheel so a future `requirements-sim-mujoco.txt`
   install doesn't dominate CI runtime.
4. Stand up the Isaac Lab backend stub described in the README
   roadmap.
5. Publish the verified benchmark numbers as a versioned JSON artifact
   in CI so regressions in `train_assembly_options.py` are visible on
   every push.
