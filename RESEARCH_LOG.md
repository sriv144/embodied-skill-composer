# Research Log

Maintained by the Auto-Researcher passes. Each entry records what was looked
at, what was implemented, and what was deferred so future runs do not redo
the same work.

## 2026-06-08 — Auto-Researcher v4

**Resume-worthiness score (start of run):** ~72 / 100
- Tech stack prestige (25): 25 — hierarchical RL + MARL + MuJoCo +
  perception is exactly the robotics-research signal recruiters look for.
- Commit recency (25): 22 — last push 2026-05-11, just inside the
  30-day window.
- Feature completeness (20): 17 — verified flagship result (2/2 beams,
  scripted + learned options) plus a retained MARL baseline.
- Stars + visibility (15): 3 — 1 star; this is the main bottleneck.
- README quality (15): 13 — already strong: flagship result table, run
  commands per mode, architecture by directory, roadmap. Mostly text-only.

**Implemented on branch `claude/sweet-clarke-fX5uP`:**
- `.github/workflows/ci.yml` — ruff (advisory) + py_compile of tracked
  `.py` files. Avoids installing the heavy CUDA / MuJoCo wheels in CI by
  intent; the goal is a fast lint signal, not a full integration run.
- `SECURITY.md` — vulnerability reporting policy with simulator /
  hardware safety notes specific to learned MARL policies.
- `docs/architecture-overview.md` — Mermaid diagrams of the hierarchical
  control stack and the scripted-vs-learned policy ladder, plus a code
  pointers table.
- `RESEARCH_LOG.md` — this file.

**Why prioritized:** the README is already excellent prose, but it leans on
the reader to assemble the picture mentally. A Mermaid diagram of the
hierarchical stack and the policy ladder is high-leverage visual signal for
a recruiter or reviewer who skims for ten seconds. Adding CI gives the
repo a green badge without touching the simulator pipeline.

**Evaluated and skipped this run:**
- Wiring `pytest` into CI. The default test suite leans on the `local_dev`
  profile and the tabletop adapters; running it in a clean GitHub runner
  needs a deps-install step that this run intentionally avoids until the
  matrix has been verified. Queued for the next pass.
- Promoting the diagram into the top of `README.md`. The current README is
  long; in-place edits are higher risk and would shift the existing flagship
  table out of the first viewport. Doing the diagram as `docs/...` keeps the
  edit additive.
- Adding screenshots / GIFs of the MuJoCo rollout. High resume signal but
  requires running the simulator locally; not in scope for a doc-only pass.

**Next-run candidates:**
- Add a `## Architecture` section to `README.md` linking to
  `docs/architecture-overview.md`.
- Commit a short GIF of the MuJoCo 3D rollout (`artifacts/mujoco_*.mp4`
  exists as a CLI target).
- Add a `make benchmark` / `tox`-style wrapper so the benchmark suite has a
  single canonical invocation visible from CI.
- Smoke-test the `scripts/check_gpu_runtime.py` path in CI on a CPU runner
  (it should fall back gracefully).
