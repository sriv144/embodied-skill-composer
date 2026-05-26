# Research Log

This log tracks autonomous-agent improvements (auto-researcher runs).
Each entry records what was implemented, why, what was skipped, and
candidates for future runs.

## 2026-05-26 — Auto-Researcher v4

**Resume score at run start:** 78/100
(robotics + multi-agent RL stack, recent activity, strong README,
full test suite, no CI badge, 1 GitHub star)

### Implemented (branch: `claude/sweet-clarke-DU4LA`)
- **CI workflow** (`.github/workflows/ci.yml`): runs ruff lint +
  the existing pytest suite under Python 3.11 on push/PR to `main`.
  Installs `requirements.txt` and `requirements-rl.txt` and defers
  to the existing pyproject configuration for pytest discovery.
  Adds a visible build-passing signal without changing project code.

### Why prioritised
- Project already has 11 test files passing locally per README, but
  nothing enforces them on push.
- A CI workflow is additive (cannot break existing functionality)
  and high-value: a green badge on a robotics RL repo signals
  engineering hygiene without requiring code changes.

### Evaluated and skipped
- **Adding `.env.example`** — project is local-only sim with no env
  vars surfaced in README; skipped to avoid noise.
- **Replacing `run_project.ps1` with a Makefile** — PowerShell is
  the documented setup path; diverging now would invalidate the
  README.
- **Adding mypy to CI** — mypy is configured in pyproject but type
  coverage is not yet enforced; running it now might block PRs on
  pre-existing typing gaps. Deferred until a separate "strict types"
  pass.

### Next-run candidates
- Add a README CI status badge once the workflow has a run history.
- Matrix test on Python 3.11 + 3.12 once compatibility is known.
- Optional `requirements-dev.txt` split so production installs stay
  lean (currently pytest/ruff/mypy ship in `requirements.txt`).
- A separate, longer-timeout MuJoCo job once `requirements-sim-mujoco`
  is known stable in headless mode.
