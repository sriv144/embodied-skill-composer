# Research Log

A running log of autonomous research-and-development cycles on Embodied Skill Composer.
Each entry summarizes the resume-impact score at the start of the run, what was shipped
on the listed branch, what was evaluated and skipped, and ideas left for the next pass.

---

## 2026-05-18 — Auto-Researcher v4

**Resume score at start of run:** 75/100
**Branch:** `claude/sweet-clarke-9dHiZ`

### Implemented
- **docs: add MIT `LICENSE`** so the public repo can legally be reused.
- **docs: add `CITATION.cff`** so the project surfaces GitHub's "Cite this repository"
  button and renders cleanly in Zenodo, Hugging Face, and other academic tooling.
- **ci: add `.github/workflows/ci.yml`** that installs `requirements.txt` and
  `requirements-rl.txt`, then runs the existing pytest suite on every push and PR
  to `main`. This makes the green test signal visible from the repo homepage and
  catches regressions automatically.

### Why this was prioritized
- The repo was already well-documented, had a flagship benchmark table, and had a
  working pytest suite — but had **no LICENSE, no CITATION, and no visible CI badge**.
  These three gaps are the classic "great project with no stars" pattern. They are
  zero-risk additive files that immediately raise resume polish.
- Risk is minimal: no source file or backend behavior changes; the CI workflow only
  runs what `README.md` already documents as the local test command.

### Evaluated and skipped (with reasons)
- **Adding ruff + pre-commit config** — the project's coding-style baseline is
  unclear and forcing a formatter could touch hundreds of lines. Deferred to a
  follow-up cycle where ruff config is added separately from any reformatting commit.
- **Recording an MP4 demo of the MuJoCo scripted policy as a README hero asset** —
  requires running the simulation in CI or committing a large binary; the existing
  benchmark table already conveys the flagship result.
- **Isaac Lab backend port** — large multi-week feature already on the roadmap;
  not a single-cycle change.
- **Adding learned RGB-D perception** — research-scoped, multi-week effort.

### Next-run candidates
1. Add `ruff` config + `.pre-commit-config.yaml` (separate from any reformat).
2. Generate a README hero animation (`artifacts/assembly_playback/summary.png`)
   from a deterministic seed and check it in.
3. Add a smoke test that imports the top-level package without optional sim deps.
4. Wire a `coverage` artifact upload to the new CI workflow.
5. Add a `Makefile` mirroring the PowerShell helper for Linux contributors.
