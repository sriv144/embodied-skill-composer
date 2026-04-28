# Research Log

A running log of automated improvement runs against this repo.

## 2026-04-28 — Auto-Researcher v4

**Resume score (start of run):** 89 / 100

- Tech stack prestige: 25 (hierarchical PPO + MARL, MuJoCo, Isaac-prep, multi-agent options learning)
- Commit recency: 25 (pushed 2026-04-19)
- Feature completeness: 18 (working hierarchical-options policy, scripted/learned/MARL baselines, MuJoCo backend)
- Stars / visibility: 7 (1 star)
- README quality: 14 (already excellent: flagship result table, modes, profiles, architecture)

### Implemented on `claude/sweet-clarke-jnSqo`

1. **`.github/workflows/tests.yml`** — a CI workflow that:
   - Runs on push and PR to `main`, plus manual dispatch.
   - Installs the project in editable mode plus `requirements.txt` (which already pins `pytest`, `ruff`, `mypy`).
   - Runs `ruff check src tests` followed by `pytest -q`.
   - Deliberately skips the optional heavy backends (PyBullet, MuJoCo, CUDA torch). Those require local hardware and are not needed for the regression suite.
   - Uses a concurrency group so superseded pushes auto-cancel.
2. **This `RESEARCH_LOG.md`**.

### Why these were prioritized

- The repo had `pyproject.toml` configured for ruff + pytest + mypy and a `requirements.txt` that already declares all three, but no GitHub Actions workflow to run them. Adding CI is a one-file change with high recruiter signal.
- The base requirements deliberately exclude the heavy simulation backends, so a CI environment can install and run tests without GPU / MuJoCo. The workflow follows that same convention.
- Pure additive change — no existing files modified, no behaviour change locally.

### Evaluated and skipped this run

- **MuJoCo / PyBullet smoke tests in CI.** Out of scope without specialised runners.
- **mypy step.** Considered, but full strict-mode mypy across the whole `src/` tree is likely to fail on first run (the project is research code with many third-party shims). Would need a curated `mypy --check` invocation — logged for next run.
- **README badges.** Will land naturally once the workflow has produced a stable run.
- **CONTRIBUTING.md improvements.** Already exists and is reasonable. No urgent gap.
- **Code-level changes.** The repo's RL pipeline is intricate; any change without local execution is too risky and explicitly ruled out by the run constraints.

### Next-run candidates

- Add a CI status badge to README once `tests.yml` has produced its first green run.
- Add a curated `mypy` invocation (start with `--ignore-missing-imports` over `src/embodied_skill_composer/core/` and grow coverage incrementally).
- Add a docs build / link-check step (the docs/ tree is rich and worth keeping internally consistent).
- Add a tiny pre-commit config (ruff, end-of-file-fixer, trailing-whitespace) to keep PRs clean.
- Generate small WebP/MP4 result clips from the existing benchmark outputs and embed them in the README to make the flagship result more visible.
