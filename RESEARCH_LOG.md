# Research Log

Automated improvement history for this repository, maintained by the
Auto-Researcher agent. Each entry records what was changed, why it was
prioritized, and what was deliberately left for a later run.

## 2026-05-29 — Auto-Researcher v4

**Resume-worthiness score at start of run: 82 / 100**
(tech stack 25 · commit recency 25 · feature completeness 17 · stars 3 · README 12)

### Implemented (branch `claude/sweet-clarke-facaX`)
- Added a GitHub Actions CI workflow (`.github/workflows/ci.yml`) that runs on
  every push and pull request to `main`, plus manual dispatch:
  - installs the package and base runtime dependencies on Python 3.11 and 3.12,
  - runs `ruff` as an advisory lint step,
  - runs the full `pytest` suite.
- The torch-dependent MARL / hierarchical-option training smoke tests already
  guard themselves with `importlib.util.find_spec("torch")` +
  `pytest.mark.skipif`, and import the heavy trainers lazily inside the test
  bodies. They therefore skip cleanly on the lightweight runner while the
  planner, executor, integration, assembly-environment, and assembly-backend
  tests execute on base dependencies only — yielding a fast, deterministic,
  green pipeline that cannot affect application behavior.

### Why this was prioritized
This was the highest-scoring target and its single most glaring gap was the
absence of continuous integration. The project already ships a real test suite
and `ruff` / `mypy` / `pytest` configuration in `pyproject.toml`, so wiring up
CI is high resume value (demonstrates CI/CD discipline on an RL + robotics
codebase), clean to implement, and zero-risk to the application itself.

### Evaluated and skipped
- **`.env.example`** — skipped; this is a local simulation sandbox with no API
  keys or external services, so an env template would add noise.
- **README rewrite** — the README is already strong (architecture, verified
  results table, per-mode run commands). Left untouched to avoid any risk of
  corrupting good documentation.
- **Installing torch in CI** — skipped this run to keep the pipeline fast and
  guaranteed-green; the seeded smoke tests are tiny, so a future job that
  installs `requirements-rl.txt` (CPU torch) is a safe follow-up.

### Next-run candidates
- Add an optional CI job that installs `requirements-rl.txt` (CPU torch) and
  runs the MARL / option-training smoke tests.
- Add a CI status badge to the top of `README.md`.
- Add `mypy` as an advisory CI step (configuration already present in
  `pyproject.toml`).
