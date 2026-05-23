# Research Log — Embodied Skill Composer

Durable memory for the auto-researcher agent. Each run appends an entry
documenting what was implemented, what was deliberately skipped, and the
next viable improvement. Do not delete prior entries.

---

## 2026-05-23 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100

**Implemented on branch `claude/sweet-clarke-dmZJS`:**
- Added `.github/workflows/ci.yml` — the repository's first CI workflow.
  Installs CPU torch via the official PyTorch wheel index, the base
  `requirements.txt`, and the editable package, then runs `ruff check`
  and the full `pytest` suite (`tests/test_assembly_*`,
  `tests/test_collection_*`, `tests/test_planner.py`,
  `tests/test_executor.py`, etc.). Triggers on `main`, every
  `claude/**` branch, and PRs.
- Seeded this `RESEARCH_LOG.md`.

**Why prioritized:** This is a strong robotics + hierarchical RL repo
with an unusually deep test suite (11 test files spanning assembly env,
benchmark, tooling, training, planner, executor, and collection
integration) — but zero CI. Wiring up CI is the single highest-leverage
resume signal here: it converts the existing pytest coverage into
visible green badges on every PR with no risk to the research code.

**Evaluated and skipped this run:**
- *Adding MuJoCo simulation tests to CI.* Skipped: `requirements-sim-mujoco.txt`
  pulls native GL/EGL deps that don't run cleanly on `ubuntu-latest`
  without extra setup, and the existing tests already cover the task
  contract without MuJoCo.
- *README polish.* The README is already excellent (10 numbered run
  modes, flagship-result table, architecture summary, roadmap). No
  clear win without screenshots, which require running locally.
- *Adding mypy to CI.* Skipped: the project has strict mypy config in
  `pyproject.toml` but the codebase hasn't been audited for mypy clean.
  Running it now would fail loudly and gate every PR.
- *Adding Claude-backed planner option* alongside the current
  scripted+RL planners. Skipped: real feature work, deserves its own
  scoped PR.

**Next-run candidates (in priority order):**
1. Add an Isaac Lab backend stub matching the existing assembly task
   contract (already on the README roadmap).
2. Add mypy to CI as a non-blocking advisory step, then triage and fix
   findings in a follow-up commit.
3. Add a short `docs/screenshots/` set rendered from
   `scripts/visualize_assembly_episode.py` and embed in the README
   beside the flagship-result table.
4. Wire `scripts/check_gpu_runtime.py` into a separate self-hosted-GPU
   workflow so the CUDA path is regularly validated.
