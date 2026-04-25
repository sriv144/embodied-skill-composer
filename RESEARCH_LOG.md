# Research Log

Persistent memory used by the auto-researcher agent. Each run appends a dated
section so future runs can see what was evaluated, what shipped, and what was
deliberately skipped.

## 2026-04-23 - Auto-Researcher v4

**Resume-worthiness score at start of run: 85 / 100**
- Tech stack prestige: 24/25 (hierarchical MARL + team-options PPO + MuJoCo 3D + Isaac-prep)
- Commit recency: 25/25 (last push 2026-04-19)
- Feature completeness: 17/20 (tabletop baselines, warehouse flagship, collaborative assembly, MuJoCo backend, benchmark suite, visualizer)
- Stars / visibility: 5/15 (1 star)
- README quality: 14/15 (flagship result table, 10 numbered run-modes, architecture map, roadmap)

### Branch
`claude/sweet-clarke-ONkrp`

### Implemented
- `.editorconfig` - unified line-endings and indent rules across Windows
  (PowerShell) and Linux (Isaac / MuJoCo). Repo officially supports both
  platforms so editor drift is a real risk.
- `.github/dependabot.yml` - weekly pip updates grouped by ecosystem
  (core, rl-stack, sim-backends) plus github-actions updates. CUDA torch
  majors are pinned intentionally and ignored, matching the behavior of
  `scripts/setup_cuda_torch_windows.ps1`.

### Why this was prioritized
Both additions are orthogonal to the two unmerged claude/* branches on
this repo (see below), zero-risk, and strengthen the "runs on Windows,
Linux, Isaac, MuJoCo, CUDA" story the project tells. They reinforce
contributor quality signals without touching runtime code.

### Prior claude/* branches observed (unmerged on main)
- `claude/sweet-clarke-QFlkH` - added a cross-platform Makefile + an
  earlier RESEARCH_LOG seed. This ONkrp branch intentionally does NOT
  touch Makefile. RESEARCH_LOG.md will trivially conflict on merge and
  the resolution is to concatenate entries.
- `claude/awesome-knuth-TXG01` - added `.github/workflows/ci.yml`. This
  ONkrp branch does not touch `.github/workflows/*` so a three-way
  merge is conflict-free on CI.

### Evaluated and skipped
- **New CI workflow**: already owned by `claude/awesome-knuth-TXG01`.
- **New Makefile or task runner**: already owned by
  `claude/sweet-clarke-QFlkH`.
- **Dockerfile**: the project is a hybrid sim stack spanning PyBullet,
  MuJoCo, CUDA-specific torch wheels, and a planned Isaac Lab profile.
  A generic Dockerfile would either duplicate the requirements-*.txt
  matrix or miss GPU passthrough. Needs real design work, not a
  drive-by add.
- **LICENSE file**: the repo is already public with a stated authoring
  pattern but no LICENSE is visible at root. Defer to an explicit
  authorial choice rather than auto-picking one.
- **Dedicated tests for the MuJoCo backend**: would require importing
  `mujoco` in CI, which is heavier than a safe baseline pass.

### Next-run candidates
1. After QFlkH + awesome-knuth-TXG01 merge, add a `pre-commit-config.yaml`
   (ruff + trailing-whitespace) consistent with how AegisQuant was
   hardened in its own QFlkH branch.
2. Add a short `docs/results/mujoco-vs-sandbox.md` once the MuJoCo
   learned policy has a verified success rate, mirroring the existing
   `assembly-hierarchical-options.md` results write-up.
3. Add a `LICENSE` file (most likely MIT given the research/learning
   framing) - requires the repo owner's explicit choice.
4. Record a short asciicast / gif of `visualize_assembly_episode.py`
   output and embed at the top of the README under the flagship table.
5. Add a single-command `python -m embodied_skill_composer.demo` entry
   point so reviewers can see the flagship result without reading 10
   separate PowerShell snippets.

## 2026-04-25 - Auto-Researcher v4

**Resume-worthiness score at start of run: 86 / 100** (unchanged tier;
no new commits to main since 2026-04-19, but the cumulative claude/*
backlog has grown).

### Branch
`claude/sweet-clarke-lOIj4`

### Implemented
- `.pre-commit-config.yaml` - ruff (lint + format) plus standard
  whitespace hooks (trailing-whitespace, end-of-file-fixer, check-yaml,
  check-toml, check-merge-conflict, check-added-large-files,
  mixed-line-ending). Hooks are configured non-aggressively: no auto-fix
  on commit, format runs in `--check` mode, and `mixed-line-ending`
  excludes `*.ps1` so PowerShell scripts keep CRLF as expected on
  Windows. Hooks only fire when a contributor opts in via
  `pre-commit install`, so this is zero-risk for current workflows.

### Why this was prioritized
The 2026-04-23 entry above explicitly listed pre-commit as the next
clean delta. It is orthogonal to every existing claude/* branch on
this repo:
- `awesome-knuth-TXG01` adds `.github/workflows/ci.yml` (no overlap).
- `sweet-clarke-QFlkH` adds `Makefile` (no overlap).
- `sweet-clarke-ONkrp` adds `.editorconfig` + `.github/dependabot.yml`
  (no overlap; editorconfig informs ruff but does not duplicate config).

A conflict-free, zero-runtime-risk commit was the right safe lever
given that the codebase ships hybrid sim backends (PyBullet, MuJoCo,
pending Isaac) where any code-touching change requires real backend
validation.

### Prior claude/* branches observed (unmerged on main)
- `claude/sweet-clarke-ONkrp` (2026-04-23) - editorconfig + dependabot
- `claude/sweet-clarke-QFlkH` - Makefile
- `claude/awesome-knuth-TXG01` - CI workflow

Merge order suggestion: awesome-knuth-TXG01 (CI) -> sweet-clarke-QFlkH
(Makefile) -> sweet-clarke-ONkrp (editorconfig+dependabot) ->
sweet-clarke-lOIj4 (pre-commit). Each pair only conflicts on
RESEARCH_LOG.md which resolves by concatenating dated sections.

### Evaluated and skipped
- **Code-touching changes (e.g. demo entry point, MuJoCo-vs-sandbox
  results doc)**: still require running the actual sim stack to be
  safe. Out of scope for a single safe-by-default run.
- **LICENSE**: still owner's choice.
- **Asciicast / gif of visualizer**: requires a recorded run.
- **Update CI to also run pre-commit**: depends on awesome-knuth-TXG01
  merging first.

### Next-run candidates
1. Once awesome-knuth-TXG01 + lOIj4 both merge, extend `ci.yml` to add
   a `pre-commit run --all-files` step so the hooks are enforced in CI
   not just local opt-in.
2. Inventory existing ruff failures across `src/` and `scripts/` and
   open a tracking issue with a one-shot `ruff check --fix` plan; do
   not auto-apply without review since this codebase has algorithmic
   constants that linters love to misclassify.
3. Add `docs/results/mujoco-vs-sandbox.md` once the MuJoCo learned
   policy has a verified success rate.
4. Add `LICENSE` once owner picks a license.
5. Record an asciicast / gif of `visualize_assembly_episode.py` and
   embed at the top of the README.
