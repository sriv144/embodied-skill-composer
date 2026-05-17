# Research Log

This file tracks autonomous research and improvement runs against this
repository. Each entry summarizes the date, what was reviewed, what was
implemented, and what was deferred.

## 2026-05-17 — Auto-Researcher v4

**Resume-worthiness score at start of run: 86 / 100** (rank 2 of 6).

| Signal | Score |
| --- | --- |
| Tech stack prestige (hierarchical MARL + MuJoCo + Isaac plan) | 25 / 25 |
| Commit recency (updated 2026-05-11) | 22 / 25 |
| Feature completeness (verified flagship row, scripted + learned + MARL baselines) | 19 / 20 |
| Stars + visibility | 5 / 15 |
| README quality (strong, results table, multi-backend command surface) | 15 / 15 |

### Implemented this run (branch: `claude/sweet-clarke-Quaj7`)

- **docs(readme): hierarchical-options flow diagram.** Inserted a
  `## Hierarchical Options Flow` section between the Flagship Result table
  and `## What It Does`, with a Mermaid `flowchart TD` covering:
  perception → team-option observation → high-level policy (scripted oracle
  or PPO) → deterministic option executor → low-level primitives →
  `AssemblyTaskBackend.step` → reward shaping + metrics → PPO replay loop
  → benchmark suite. A `Backends` subgraph pins the three backend choices
  (`local_sandbox`, `mujoco_local`, `isaac_lab` stub) against the same step
  contract so portability is visible.
  Renders inline on GitHub. Rest of the README is preserved verbatim.

### Why this was prioritized

The README is already this repo's strongest asset (clear flagship row,
multi-backend command surface, explicit roadmap). The single missing
resume-grade signal called out in the previous run was a visual. A Mermaid
diagram is zero-risk (no code paths touched, no CI to break) and directly
answers the "what is hierarchical-options actually doing?" question that a
portfolio reviewer would ask before diving into `options_trainer.py`.

No source files were touched, so the previous CI workflow (on
`claude/sweet-clarke-snPHW`) is unaffected if both branches are eventually
merged.

### Evaluated and skipped

- **Isaac Lab backend stub.** Listed as a next-run candidate but already
  shipped on `main` — `IsaacLabAssemblyBackend` in
  `src/embodied_skill_composer/assembly/backends.py` is a contract-preserving
  stub. Nothing to add this run; the diagram surfaces its presence instead.
- **Strict `ruff` / `mypy` CI job.** `[tool.ruff]` and `[tool.mypy]` are in
  `pyproject.toml`, but the mypy config is strict (`check_untyped_defs`,
  `warn_return_any`, `strict_optional`). Adding a CI job blind would likely
  go red on the first commit. Defer until the codebase has been swept once.
- **Demo GIF.** Higher impact but requires producing media; left for a run
  with a local visualizer loop.
- **Real-cluster screenshot of the playback visualizer output.** Same
  constraint — needs a local matplotlib run.

### Next-run candidates

1. Sweep the codebase with `ruff check --fix` locally, then add a CI lint
   job once the diff is clean.
2. Generate and embed `artifacts/assembly_playback/summary.png` from the
   scripted-options run under the new Flagship Result section.
3. A short docs page under `docs/results/` capturing a learned-vs-scripted
   episode side-by-side as inline PNGs.
4. Begin replacing the stub IsaacLab backend with a real Isaac Sim driver
   behind the existing `AssemblyTaskBackend` protocol.

### Prior research-log context

Previous runs (most recent first, none merged to `main`):

- `claude/sweet-clarke-snPHW` (2026-04-27) — pytest CI on Python 3.11 +
  3.12, MIT LICENSE, seeded research log.
- `claude/sweet-clarke-wCHAT` (2026-04-24) — pytest workflow, LICENSE,
  earlier research log.
