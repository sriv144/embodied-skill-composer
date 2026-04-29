# Research Log

A running log of autonomous research and improvement work performed against
this repository. Each entry records what changed, why, and what was
intentionally deferred.

## 2026-04-29 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100 (top of the 6-repo cohort).

**Branch:** `claude/sweet-clarke-dI8Uk`

### What was implemented
- Added a GitHub Actions CI workflow at `.github/workflows/ci.yml` that runs
  the tabletop and warehouse pytest suite on Python 3.10 and 3.11. The
  optional MuJoCo and PyBullet simulator suites are excluded because they
  require GPU/native binaries unavailable on stock CI runners.
- Added an MIT `LICENSE` so the repository is unambiguously open-source and
  acceptable as portfolio material.
- Seeded this `RESEARCH_LOG.md` to anchor future autonomous runs.

### Why this was prioritized
The README and architecture were already strong (hierarchical team-options
MARL with PPO, a MuJoCo backend, scripted/learned/baseline benchmarking),
so the marginal value of more code was low and the breakage risk was high.
The missing pieces that genuinely hurt resume credibility were:
1. No CI badge / no green-tick proof that `pytest -q` actually passes.
2. No license, which makes external reviewers skip the repo entirely.
3. No persistent research memory for the auto-researcher itself.

All three are reversible, additive, and cannot break existing functionality.

### Evaluated and skipped
- **README rewrite:** the existing README is already detailed and benchmarked.
  Touching it risks regression of accurate result tables.
- **Isaac Lab backend stub:** mentioned in the README roadmap. Too large for a
  single autonomous run, and would need real Isaac Lab installed to validate.
- **Visual MARL evaluation:** roadmap item, depends on Isaac Lab work first.
- **Adding `.env.example`:** the project does not currently read any secrets
  at import time, so a stub `.env.example` would be misleading.

### Next-run candidates
- Add a coverage badge once CI has produced a baseline report.
- Auto-generate a results gallery PNG from `artifacts/assembly_playback/`.
- Port the assembly task contract to an Isaac Lab backend stub (interface
  only, no training) so the roadmap item is unblocked.
- Wire `ruff` and `mypy --strict` into the CI workflow once a baseline of
  warnings has been triaged.
