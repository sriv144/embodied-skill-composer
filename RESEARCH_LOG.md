# Auto-Researcher Research Log — embodied-skill-composer

This file is the persistent memory for the auto-researcher agent. Each run
appends a new section describing what was evaluated, what was implemented
(and on which branch), and what is on deck for the next run.

---

## 2026-06-06 — Auto-Researcher v4

**Resume score at start of run:** 83 / 100

- Tech stack prestige: 25/25 (robotics + hierarchical RL/MARL + MuJoCo +
  team-options + simulator-agnostic backend design)
- Commit recency: 22/25 (updated 2026-05-11, ~26 days before this run)
- Feature completeness: 18/20 (flagship result table, tabletop + warehouse +
  assembly modes, benchmark suite, MuJoCo backend, GPU runtime check)
- Stars + visibility: 4/15 (1 star)
- README quality: 14/15 (excellent — results table up top, clear modes,
  setup notes, roadmap)

### Implemented this run

Branch: `claude/sweet-clarke-TGFOp`

- `.github/workflows/ci.yml` — pytest + ruff CI on PRs and pushes to `main`.
  - lint job: ruff check, non-blocking, GitHub-annotated output
  - test job: installs the package in editable mode plus `requirements.txt`
    and runs the full `pytest` suite (the repo's `pyproject.toml` already
    declares `pythonpath = ["src"]` and `testpaths = ["tests"]`)

### Why this was prioritized

The repo already pins `pytest>=8.2`, `ruff>=0.5`, and `mypy>=1.10` in
`requirements.txt`, has 19 test files, and the README explicitly tells
contributors to run pytest before pushing — but `.github/workflows/` did
not exist, so none of this was enforced on incoming changes. Adding CI is
pure additive plumbing: no source code is touched, so existing
functionality cannot break, and a green CI badge immediately strengthens
the repo's resume signal as a "production-discipline" robotics project.

### Evaluated and skipped this run

- **`.env.example`.** Skimmed the repo — runtime profiles live in
  `configs/assembly_profiles/*.yaml`, not env vars, and the README does
  not reference any. Adding an empty `.env.example` would be cargo culting.
- **README badges.** The README is already long and well-structured;
  adding badges that point to a brand-new (and likely red-on-first-run)
  workflow is premature. Add once CI is stable.
- **mypy in CI.** `mypy` is pinned but strict-mode runs would flag a
  large amount of untyped code on first attempt. Defer until a typed
  baseline exists.

### Next-run candidates

1. Add a `pre-commit` config (ruff + black or ruff-format) so contributors
   catch lint locally.
2. Add mypy to CI in non-strict mode once the test job is consistently
   green for a couple of runs.
3. Add an `isaac-prep` placeholder CI matrix entry (skipped by default)
   that documents the planned Isaac Lab profile gate.
4. Add a CI smoke step that runs one short headless episode of the
   scripted assembly policy and uploads the playback PNG summary as an
   artifact — visible "the robot actually does something" proof per PR.
