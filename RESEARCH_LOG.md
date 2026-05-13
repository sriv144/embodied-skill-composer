# Research Log

This log tracks automated research and improvement runs by the
auto-researcher agent. Each entry captures the project's resume-worthiness
score at run start, what was implemented (with branch), why it was
prioritized, what was evaluated and skipped, and the next-run candidates.

---

## 2026-05-13 — Auto-Researcher v4

**Resume score at start of run:** 88 / 100
- Tech stack prestige: 24/25 (robotics + RL + multi-agent, MuJoCo / PyBullet)
- Commit recency: 25/25 (active within the last 48 hours)
- Feature completeness: 18/20 (flagship 2-robot assembly task working end-to-end)
- Stars + visibility: 6/15
- README quality: 15/15 (architecture, runnable commands, roadmap)

**Implemented (branch: `claude/sweet-clarke-HHSYf`):**
- `.github/workflows/ci.yml` — GitHub Actions workflow that runs `ruff check`
  and the existing `pytest` suite on Python 3.11 for every push / PR to `main`.
  The repo already has 11 test files under `tests/` but no CI; this closes
  the credibility gap when reviewers browse the Actions tab.
- `RESEARCH_LOG.md` — this file (seeded for future runs).

**Why this was prioritized:**
- Highest resume score in the portfolio + zero CI signal was the single
  largest credibility gap. A green CI badge on a robotics + RL repo is the
  cheapest, highest-trust improvement available.
- Tests already exist and are CPU-friendly, so CI is low-risk to add.
- The CI matches the project's declared `requires-python = ">=3.11"` and
  installs both `requirements.txt` and `requirements-rl.txt`, so the
  assembly / MARL tests can import `stable-baselines3` and friends.

**Evaluated and skipped this run:**
- README badge insertion — deferred to avoid a large README diff in the
  same commit as CI bring-up. Will land next run once CI is green.
- `.env.example` — the project does not yet consume any environment
  variables (no API keys, no broker creds). Adding an empty stub would be
  noise; defer until the Isaac Lab / ROS 2 integrations land.
- MuJoCo headless rendering in CI — too heavy for the free runner; the
  README's GUI commands stay developer-local.
- Refactor of `run_project.ps1` into a cross-platform Makefile — useful
  but not resume-bearing; leave for a maintenance pass.

**Next-run candidates (ranked):**
1. Add a README hero image / animated GIF of the MuJoCo 3D assembly
   episode — highest visual return on a recruiter skim.
2. Add a `coverage` step to CI and a coverage badge.
3. Wire a `release.yml` workflow that builds the package on tags.
4. Add a Linux-only headless MuJoCo smoke test job (guarded by
   `if: matrix.os == 'ubuntu-latest'`).
5. Publish the assembly hierarchical-options result as a short technical
   write-up (`docs/results/`) with a benchmark plot.
