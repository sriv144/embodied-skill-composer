# Research Log

This file tracks autonomous research and improvement runs against this repo.
Each run lists what was implemented, what was evaluated and skipped, and the
next-run candidate list.

## 2026-05-15 — Auto-Researcher v4

**Resume-worthiness score at start of run: 91 / 100**

Signal breakdown:
- Tech stack prestige: 25/25 (multi-agent RL, MARL, MuJoCo, Isaac-ready)
- Commit recency: 24/25 (last push 2026-05-11)
- Feature completeness: 19/20 (scripted + learned hierarchical options ship a 100% success-rate flagship result)
- Stars + visibility: 8/15
- README quality: 15/15 (architecture, results table, runbook, roadmap)

### Implemented this run

Branch: `claude/sweet-clarke-dYRjh`

- `feat(ci)`: added `.github/workflows/ci.yml` running ruff (hard gate), mypy and pytest (non-blocking) on every push and PR to `main`. The repo already configures `ruff`, `mypy`, and `pytest` in `pyproject.toml` but has no CI wiring — this closes the gap with zero runtime impact. Torch-dependent training tests are already `skipif`-guarded, so they cleanly skip on GHA runners without RL extras installed.

### Why this was prioritized

The repo's results table (`scripted: 1.000`, `learned hierarchical: 1.000`, `low-level MARL: 0.000`) is the kind of artifact that pulls weight in an interview. CI that publicly proves the lint + import surface is healthy raises the floor on that credibility without touching the RL training paths. It is the highest-impact, lowest-risk change available on a single run.

### Evaluated and skipped this run

- Refactor the MARL trainer to share replay buffers across agents — skipped: high blast radius on the RL learning curves; needs a dedicated benchmark run before/after to prove it does not regress the flagship result.
- Add a Dockerfile + devcontainer — skipped: project already documents Windows + Linux/NVIDIA setup and a Docker image would have to bundle MuJoCo licensing logic; out of scope for an unattended run.
- Add a curriculum-stage replay video to the README — skipped: requires running the visualizer, which is not safe to do unattended.

### Next-run candidates

1. Promote mypy from non-blocking to hard-gate after a one-shot pass over outstanding type errors.
2. Add a coverage step (`pytest --cov=embodied_skill_composer`) and upload to Codecov / artifact.
3. Auto-render the assembly playback summary into `docs/results/` on every push and embed it in the README.
4. Add a smaller CPU-only training smoke job (1 iteration, 2 episodes) to guard against regressions in the hierarchical options trainer.
