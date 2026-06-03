# Research Log

This file tracks autonomous research-agent activity on
embodied-skill-composer. Each entry records (a) what was implemented,
(b) why it was prioritized, (c) what was evaluated but skipped, and
(d) candidates for next runs.

Do not delete prior entries — they exist so future runs can avoid
re-doing the same work.

---

## 2026-06-03 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 79 / 100

- Stack prestige (25): 25 — hierarchical multi-agent RL, PPO,
  PyBullet + MuJoCo backends, imitation warm-start, planned Isaac
  Lab port. Top-of-stack robotics.
- Commit recency (25): 18 — 23 days since last push.
- Feature completeness (20): 18 — flagship two-robot assembly
  task with verified `2/2 beams` result table, scripted /
  learned-hierarchical / low-level MARL baselines, MuJoCo 3D
  visualization, GPU runtime check, runtime profiles.
- Stars / visibility (15): 4 — 1 star but very specific topic tags.
- README (15): 14 — results table up front, command examples for
  every workflow, architecture summary, roadmap.

### Implemented on `claude/sweet-clarke-oCTQH`

1. **Added `.github/workflows/ci.yml`** — a Python 3.11 + 3.12
   matrix that runs `ruff check`, a non-blocking `mypy` pass,
   and `pytest -q`. CPU-only torch wheel via the official PyTorch
   index URL keeps the install step under five minutes. The
   project's `pyproject.toml` already configures ruff / mypy /
   pytest, so the workflow simply invokes them.
2. **Seeded `RESEARCH_LOG.md`** so subsequent auto-researcher
   runs see prior decisions.

### Why these were prioritized

The project's flagship result — `Learned hierarchical options:
100% success on 2/2 beams` vs. `Low-level MARL: 0% on 1/2 beams`
— is genuinely impressive, but with no CI badge a reviewer can't
tell the table reflects current `main`. A green CI status check is
the single highest-leverage change for credibility.

Intentionally left `.env.example` out: the project's local
sandbox does not require any API keys, and inventing env vars
would add noise without function.

### Evaluated and skipped

- **Cross-platform README commands.** README is currently
  PowerShell-only. Adding bash equivalents is useful but not
  resume-critical. Defer.
- **MuJoCo headless rendering in CI.** Would require system
  packages (libosmesa, libglfw); the CPU-only smoke test
  is enough to prove the planner and learners build.
- **Type-strictness ratchet.** mypy is currently non-blocking. Once
  the codebase passes cleanly, flip it to fail-on-error.
- **Isaac Lab backend stub.** Listed in the roadmap; needs design
  work, not a single commit.

### Next-run candidates

1. Add a results plot / GIF (rendered from `scripts/visualize_assembly_episode.py`)
   embedded in the README so the `2/2 vs 1/2` story is visual.
2. Cache the CPU torch wheel across CI runs to shrink build time.
3. Make `mypy` blocking and fix the first wave of issues.
4. Add a `scripts/quick_demo.py` that runs the shortest possible
   end-to-end episode for new contributors.
5. Build the Isaac Lab task-contract stub described in the roadmap.
