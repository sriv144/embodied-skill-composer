# Research Log

This file tracks autonomous improvements made by the Auto-Researcher agent.
Each entry records what was implemented, what was evaluated and skipped,
and candidates queued for the next run, so we never repeat work.

## 2026-06-09 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 81/100

**Branch:** `claude/sweet-clarke-rj6hm9`

### Implemented this run
- `.github/workflows/ci.yml` — first CI workflow for this repo. Sets up Python 3.11, installs `requirements.txt` plus the optional RL stack, then runs `pytest -q --basetemp .pytest_tmp` exactly as the README documents. Adds a green-badge signal that hierarchical-options / MARL regressions stay healthy.
- `RESEARCH_LOG.md` — seeded so future agent runs have memory of what's been tried.

### Why these were prioritized
- Repo is feature-dense (two-robot collaborative assembly, hierarchical options, PPO fine-tuning, MuJoCo backend) with a real test suite, but had no CI. CI is the highest-leverage polish step: it surfaces regressions, gives the README a verified-tests badge, and adds resume signal that the project is reproducibly buildable.
- Risk is low: the workflow only runs the existing pytest entry point that the README already promises to work.

### Evaluated and skipped
- README rewrite — already strong (flagship-result table, mode-by-mode runbook, roadmap). Not worth the diff risk.
- Ruff / black formatting pass — would touch many files; needs an opinionated style decision from the maintainer before being safe.
- Dockerizing the local sandbox — high resume value but non-trivial because of optional MuJoCo / CUDA wheels. Defer to a focused run.
- Isaac Lab backend stub — listed in the roadmap; needs a real Isaac install path to validate, out of scope for an autonomous run.

### Next-run candidates
1. Add a `ruff` + `mypy` lint job to CI once a `pyproject.toml` lint section is agreed.
2. Publish a short results artifact (`docs/results/`) from the CI run so the flagship-result table can be auto-refreshed.
3. Add a GitHub release / changelog stub once the Isaac Lab milestone lands.
4. Add a `requirements-dev.txt` so contributors get pytest / ruff in one command.
