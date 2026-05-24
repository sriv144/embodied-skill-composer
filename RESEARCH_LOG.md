# Research Log

A running log of automated research + improvement passes on this repo.
Each entry records what was implemented, why, what was evaluated but
skipped, and candidate work for the next pass.

## 2026-05-24 — Auto-Researcher v4

**Resume score at start of run:** 78 / 100
*(robotics + RL + multi-agent + MuJoCo, with a thorough README, a
flagship results table, hierarchical-options PPO baseline, and 11+
test files — the project is mature, presentation is already strong.)*

**Branch:** `claude/sweet-clarke-lJSR8`

**What was implemented**

- `.github/workflows/ci.yml` with two jobs:
  1. `lint-and-compile` (required): `ruff check src tests scripts`
     plus `python -m compileall -q src tests scripts`. These are
     deterministic and platform-agnostic, so the badge stays honest.
  2. `tests` (informational, `continue-on-error: true`): installs
     `requirements.txt`, does `pip install -e .`, runs
     `pytest -q -k "not training and not mujoco"`. Training and
     MuJoCo tests are deselected because they need GPU / native
     viewer dependencies that the Ubuntu runner doesn't have.
     Marked informational so initial parity gaps between the
     project's primary Windows dev environment and Linux CI don't
     produce a false-red signal until the author confirms.

No source code was touched. Only CI configuration and this log were
added.

**Why this was prioritised**

The project is feature-complete with a clear flagship result
(hierarchical learned options solve 2/2 beams vs. low-level MARL at
0/2). The README, results doc, and CONTRIBUTING.md are all in good
shape. The single visible gap was the absence of any CI signal on
pushes — the pinned dev tooling (ruff, mypy, pytest) in
`requirements.txt` plus the existing `[tool.ruff]` and
`[tool.pytest.ini_options]` configuration in `pyproject.toml` made
adding a workflow almost mechanical, with very low breakage risk.

**Evaluated and skipped**

- Adding mypy to CI: `pyproject.toml` already has a strict mypy
  config (`check_untyped_defs`, `warn_unused_ignores`,
  `warn_redundant_casts`, `warn_return_any`, `strict_optional`).
  Running it under those settings without first verifying clean
  state would likely produce a long error list and burn CI minutes.
  Worth doing as a focused follow-up that fixes errors in the same
  PR.
- Adding an Isaac backend stub: the README's roadmap calls for
  porting the assembly task contract into Isaac. Out of scope for
  an automated improvement pass — needs design judgement.
- Touching MuJoCo / training tests: deliberately left untouched.
  Either would need a GPU runner or a careful mock layer.
- Replacing the bombastic results phrasing: README is fine; no
  edit warranted.

**Next-run candidates**

- Add a separate `mypy` CI job once a quick pass confirms the
  existing code is type-clean under the project's strict mypy config.
- Coverage reporting via `pytest --cov` and a Codecov upload.
- Pre-commit hook config (`.pre-commit-config.yaml`) wiring ruff
  and mypy so contributors get the same checks locally.
- Dependabot config under `.github/dependabot.yml` for pip and
  GitHub Actions.
- Add a GIF / screenshot of the assembly playback (script already
  exists in `scripts/visualize_assembly_episode.py`) to README to
  drive showcase value further.
