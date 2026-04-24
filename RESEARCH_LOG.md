# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## 2026-04-24 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~85 / 100

The strongest project on the account. Flagship result already documented in
the README (hierarchical team options beating low-level MARL 2/2 vs 1/2 beams),
CONTRIBUTING guide in place, multiple requirements files for CPU/GPU/MuJoCo
profiles, and a `tests/` directory with real coverage. The obvious missing
loadbearing piece was a CI signal proving that the test suite still passes.

### Implemented on branch `claude/sweet-clarke-wCHAT`

- **`.github/workflows/tests.yml`.** Runs `pytest -q` on every push and PR
  to `main`, on Python 3.11, after installing both `requirements.txt` and
  `requirements-rl.txt`. Uses the repo's existing test layout, so no code
  changes are required to wire it up.
  - Pip caching keyed on both requirements files.
  - `MPLBACKEND=Agg` so any matplotlib calls stay headless in CI.
  - Cancel-in-progress concurrency so PR pushes don't queue duplicate jobs.
  - 20-minute timeout guard.
- **MIT LICENSE** added — the repo was previously unlicensed.
- **Seeded this `RESEARCH_LOG.md`.**

### Why prioritized over alternatives

CI is the single highest-leverage addition: it turns an already-strong
repo's green-badge story from "claimed" to "verified by Actions." It also
unlocks every future change (refactors, new backends) by making regressions
visible. Zero risk to existing functionality — pure additive wiring.

### Evaluated and skipped

- **Ruff / black lint job.** No existing style config in the repo; adding one
  would surface a wall of pre-existing findings that look like "I broke
  everything." Defer to a dedicated lint-normalisation run with a clean
  baseline commit.
- **MuJoCo / PyBullet job.** MuJoCo and PyBullet both need system packages
  and are brittle on hosted Ubuntu runners. Keep CI to the pure-Python
  regression path for now.
- **GPU / CUDA smoke test.** Hosted GitHub runners have no GPU; this has to
  be a self-hosted runner and is out of scope for a zero-infra run.
- **Isaac Lab backend stub.** Already on the roadmap; genuine feature work,
  not an auto-researcher-style incremental polish.

### Next-run candidates

1. Add a `ruff` (or `black + isort`) lint job alongside tests, with the
   formatting baseline applied in the same PR.
2. Add a `benchmark` job that runs `scripts/benchmark_assembly_policies.py`
   weekly and uploads the diagnostics JSON as an artifact.
3. Add a `docs/architecture.md` with a diagram of the planner / executor /
   perception / backend stack.
4. Dockerize the CPU regression profile for reproducible local evaluation.
5. Add a `CODE_OF_CONDUCT.md` to round out the OSS template set.
