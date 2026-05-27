# Research Log

This log tracks autonomous research-and-development passes over the
embodied-skill-composer repository. Each run records the resume-impact
score, what was implemented (and why), what was evaluated and skipped,
and candidates for the next pass.

---

## 2026-05-27 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 72 / 100

**Branch:** `claude/sweet-clarke-HM9gk`

### What was implemented

- **CI workflow** (`.github/workflows/ci.yml`) — GitHub Actions pipeline
  that runs `ruff` lint and the pytest suite against Python 3.11 and
  3.12 on every push and pull request to `main`. The repository already
  ships a strong test suite under `tests/` (assembly env, benchmark,
  planner, executor, collection perception, integration coverage) but
  had no CI before this run, so regressions on `main` were invisible.

### Why this was prioritized

A recruiter or interviewer skimming the GitHub page sees the
CI badge first — "green build" is a near-zero-cost signal that the
robotics + RL stack is actually maintained, not a one-shot dump. The
repo already passes its own tests locally, so wiring CI is
low-risk / high-visibility.

### Evaluated and skipped

- **README overhaul** — The current README is already detailed,
  including a flagship-result table, hierarchical-options write-up,
  MuJoCo backend, GPU runtime check, and a roadmap. Skipped to avoid
  churning prose for the sake of churn.
- **Isaac Lab backend** — Mentioned in the roadmap and would be
  resume-gold, but requires Linux + NVIDIA hardware and a multi-day
  port. Out of scope for a single autonomous pass.
- **MARL low-level policy retraining** — Current low-level baseline
  stalls at 1/2 beams; improving it is a research project, not a
  drive-by edit.
- **Pre-commit hooks** — Worth adding but redundant with CI for now.

### Next-run candidates

1. Add a results badge / status block to the top of the README once CI
   has run at least once on main.
2. Wire a coverage report (`pytest --cov`) and post it as a CI summary.
3. Stand up the Isaac Lab backend stub from the roadmap behind a
   feature flag in `configs/assembly_profiles/isaac_gpu.yaml`.
4. Add a small `examples/` directory with a one-command demo gif for
   the hierarchical-options policy — strongest possible README polish.
