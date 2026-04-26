# Research Log

A running ledger of autonomous-improvement passes against this repository.
Each entry records the resume-worthiness score at the start of the run, what
was implemented, what was evaluated and skipped, and what the next pass should
look at.

## 2026-04-26 — Auto-Researcher v4

- Branch: `claude/sweet-clarke-T7V0c`
- Resume score at start of run: **87 / 100**
  - Tech stack prestige: 25 (hierarchical MARL + PPO + MuJoCo + planned Isaac)
  - Commit recency: 25 (last meaningful commit 2026-04-16)
  - Feature completeness: 18 (flagship 2/2 beams result is reproducible)
  - Stars + visibility: 5
  - README quality: 14 (architecture, results table, full setup matrix)

### Implemented this run
- `CITATION.cff` so the project is machine-citable from research papers and
  GitHub's "Cite this repository" UI.
- `SECURITY.md` with a coordinated-disclosure path for sim/exec issues.
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md` and
  `PULL_REQUEST_TEMPLATE.md` tailored to this repo's subsystems and the
  flagship `2/2 beams` benchmark.
- This `RESEARCH_LOG.md` seed file.

### Why this was prioritized
A prior `claude/sweet-clarke-wCHAT` pass already added LICENSE, a CI workflow,
and a research log seed. Re-doing those would duplicate work. The remaining
high-leverage repository-quality gaps were academic citability, a security
contact path, and contributor-facing templates — all of which materially
improve the GitHub landing page that recruiters and collaborators see first,
without touching any RL code or risking the flagship benchmark.

### Evaluated and skipped
- **README badges row** — README is already comprehensive (9.7 KB) and the
  natural place for badges depends on the CI workflow that lives on a sibling
  branch. Will revisit once the CI workflow lands on `main`.
- **`pre-commit` config (ruff/black)** — `pyproject.toml` does not currently
  declare lint config; adding hooks without aligning them risks breaking the
  contributor flow described in `CONTRIBUTING.md`. Defer to a focused pass.
- **Removing PowerShell-only quick-start commands from README** — Linux/Isaac
  setup notes already exist under `docs/setup/`; rewriting the front-page
  examples is a larger doc rework, not a one-shot win.

### Next-run candidates
1. Add a `Makefile` (or cross-platform `tasks.py`) that wraps the most-used
   `python scripts/...` invocations so the README quick-start is one line.
2. Land a CHANGELOG.md and start tagging releases (`v0.1.0`) so the flagship
   `2/2 beams` result is anchored to a citable artifact.
3. Add a `pre-commit` config and matching `ruff` + `black` settings in
   `pyproject.toml`, then a CI lint job to enforce them.
4. Publish a short blog-post-style writeup under `docs/results/` with frames
   from `scripts/visualize_assembly_episode.py`, and link it from the README
   so the headline result is visually obvious on the landing page.
