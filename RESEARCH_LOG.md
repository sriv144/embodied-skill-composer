# Embodied Skill Composer — Auto-Researcher Log

A cumulative record of automated research + implementation passes on this
repository. Each entry captures what was evaluated, what shipped, and what
was deferred so that future runs avoid duplicating work.

## 2026-06-10 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 91 / 100
(tech 25, recency 22, completeness 19, stars 11, README 14)

**Branch:** `claude/sweet-clarke-0tcsmd`

### Implemented

- `.github/workflows/ci.yml` — first GitHub Actions pipeline for the
  repo.
  - `lint` job runs `ruff check src tests` in report-only mode
    (`continue-on-error: true`) to surface style issues without blocking
    the first PR cycle while existing code is migrated.
  - `test` job installs the package in editable mode plus
    `requirements.txt` and `requirements-rl.txt`, then runs the
    tabletop / warehouse / hierarchical-options pytest suite under
    `tests/` with the README's recommended `--basetemp=.pytest_tmp`.
  - MuJoCo / PyBullet / Isaac Lab extras are intentionally **not**
    installed in CI; those backends are validated locally via the
    runtime profiles documented in `README § GPU runtime check`.
  - Triggered on push to `main`, PRs targeting `main`, and
    `workflow_dispatch`. Concurrency cancels stale runs per ref.

### Why prioritized

This repo already has the strongest README of the six (results table,
flagship hierarchical-options policy, MuJoCo + Isaac roadmap). The single
most visible gap was the absence of any automated tests in CI — the
`tests/` directory ships solid coverage but interviewers cannot tell
without cloning. Adding a CI badge converts existing test discipline
into a visible signal at near-zero implementation risk. The workflow
does not touch any source code or runtime profile, so the
flagship 2/2-beam assembly result is unaffected.

### Evaluated and skipped

- **Adding a CI badge to the top of `README.md`.** Deferred until the
  workflow has produced its first successful run, so the badge URL is
  green from the moment it's added.
- **Wiring `pytest-cov` to upload a coverage artifact.** Sensible next
  step but would expand the CI surface beyond a narrow first commit;
  scheduled for a follow-up.
- **Adding `requirements-sim-mujoco.txt` to the CI matrix.** MuJoCo
  needs OpenGL headers and large wheels; running it in GitHub-hosted
  runners is fragile and not worth the green-badge risk this pass.
- **Documentation pass.** README is already strong; no edits this run.

### Next-run candidates

1. Add a CI status badge to `README.md` after the first successful run.
2. Extend the `test` job with `pytest-cov` and upload `coverage.xml`
   as a workflow artifact.
3. Add a separate `mujoco` workflow (manual `workflow_dispatch` only)
   that installs `requirements-sim-mujoco.txt` for smoke-running the
   scripted 3D assembly episode.
4. Audit `src/embodied_skill_composer/assembly/learners/` for an
   Anthropic Claude planner option (currently the policy is PPO-only).
