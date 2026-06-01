# Research Log

This log tracks autonomous-research improvements applied to this repository.
Each run records what was implemented, what was considered and skipped, and
what the next candidate improvements are. Do not delete entries; append new ones.

## 2026-06-01 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~78 / 100

The project is already strong: hybrid robotics + hierarchical team-options +
MARL baseline + MuJoCo 3D + classical-CV perception, with a detailed README,
tests, a `pyproject.toml` configured for ruff / mypy / pytest, and multiple
requirements files for CPU / RL / MuJoCo / CUDA. The largest visible gap was
the absence of any GitHub Actions — the typecheck/test config exists locally
but nothing enforces it.

### What was implemented

Branch: `claude/sweet-clarke-nPZT4`

- **`.github/workflows/ci.yml`** — a lint + typecheck + test workflow that:
  - runs on push/PR to `main`
  - cancels superseded runs via the standard `concurrency` group
  - sets up Python 3.11 with pip caching keyed off the requirements files
  - installs `requirements.txt` + `requirements-rl.txt` (CPU-safe variants)
  - installs the project itself in editable mode so `src/` packages resolve
  - runs `ruff check src tests` against the existing `[tool.ruff]` config
  - runs `mypy src` against the existing `[tool.mypy]` config with
    `continue-on-error: true` so the build does not break while typing is
    tightened incrementally
  - runs `pytest -q`, picking up `pythonpath` / `testpaths` from `pyproject.toml`
- **`RESEARCH_LOG.md`** — seeded (this file).

### Why this was prioritized

A passing CI badge on the front of a robotics / RL repo is a strong
signal for recruiters and collaborators. Everything CI needs was already
in `pyproject.toml`; wiring the workflow is a low-risk way to convert latent
quality into a visible signal. Lint/test on every PR also catches regressions
in the existing assembly / warehouse / MARL pipelines.

### What was evaluated and skipped

- **CUDA / MuJoCo job on GHA.** Skipped — GitHub-hosted runners don't have
  GPUs, and the MuJoCo viewer needs a display. Limited the workflow to the
  CPU regression baseline that already works locally.
- **`.env.example`.** Skipped — inspection of the source did not reveal any
  required environment variables (config is via YAML profiles in `configs/`).
  Adding an empty `.env.example` would just be noise.
- **Ruff / mypy strictness ratchet.** Out of scope for an atomic CI commit;
  better to land CI first and tighten the rules in follow-up PRs once we see
  the actual error counts on `main`.
- **Workflow that runs on the PyBullet or MuJoCo extras.** Skipped — those
  installs are heavy and slow on GHA. Can be added as a manual `workflow_dispatch`
  job later if the maintainers want a deeper smoke test.

### Next-run candidates

1. Tighten the ruff ruleset and remove `continue-on-error` from the mypy step
   once the existing error count is paid down.
2. Add a `workflow_dispatch` job that installs `requirements-pybullet.txt` /
   `requirements-sim-mujoco.txt` and runs the corresponding simulator smoke
   tests headlessly (xvfb).
3. Add coverage reporting (`pytest-cov` + Codecov upload).
4. Add a README badge linking to the CI status once `main` has a green run.
5. Camera-based evaluation for the assembly task (already on the roadmap in
   `README.md` — not a CI concern, but the obvious next feature ratchet).
