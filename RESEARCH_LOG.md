# Research Log

A running ledger of autonomous-improvement passes against this repository.
Each entry records the resume-worthiness score at the start of the run,
what was implemented, what was evaluated and skipped, and what the next
pass should look at.

## 2026-05-19 — Auto-Researcher v4

**Resume-worthiness score at start of run: 87 / 100** (rank 1 of 6).

| Signal | Score |
| --- | --- |
| Tech stack prestige (hierarchical MARL + PPO + MuJoCo + planned Isaac) | 25 / 25 |
| Commit recency (updated 2026-05-11) | 22 / 25 |
| Feature completeness (flagship 2/2 beams result reproducible, 10 numbered run modes) | 18 / 20 |
| Stars + visibility (1 star) | 5 / 15 |
| README quality (architecture, results table, full setup matrix) | 14 / 15 |

### Implemented this run (branch: `claude/sweet-clarke-lpd1a`)

No code or config changes. This commit only seeds `RESEARCH_LOG.md` on
the pre-assigned branch so the next run has continuity.

### Why no implementation this run

embodied-skill-composer is the strongest repo in the portfolio and its
open-source quality surface is already covered by an unusually deep
backlog of unmerged `claude/sweet-clarke-*` and `claude/awesome-knuth-*`
branches:

- `awesome-knuth-TXG01` — CI workflow (`.github/workflows/ci.yml`).
- `sweet-clarke-QFlkH` — cross-platform Makefile.
- `sweet-clarke-ONkrp` — `.editorconfig` + `.github/dependabot.yml`.
- `sweet-clarke-lOIj4` — `.pre-commit-config.yaml`.
- `sweet-clarke-wCHAT` — `.github/workflows/tests.yml` + MIT LICENSE.
- `sweet-clarke-T7V0c` — CITATION.cff + SECURITY.md + ISSUE / PR
  templates.

Every low-risk additive doc / template / config is already on at least
one of those branches. Re-shipping any of them here would violate the
no-duplicate-work guardrail. The remaining genuinely-new candidates
(CHANGELOG.md tied to a real v0.1.0 tag, asciicast / gif of
`visualize_assembly_episode.py`, MuJoCo-vs-sandbox results write-up,
an opinionated `docs/architecture.md`) all require either an owner
decision (versioning, license choice) or a recorded sim run, neither
of which fits a safe-by-default autonomous pass.

Token budget this run was instead spent on the three repos with
clear, unblocked next-run candidates:

- `FinLens` — Anthropic Claude provider option behind `LLM_PROVIDER`
  (`claude/admiring-davinci-lpd1a`).
- `Autonomous-SRE-Agent` — helm lint + docker compose validation
  workflow (`claude/fervent-edison-lpd1a`).
- `salesnuero` — first CI workflow on the repo, backend compile +
  frontend build (`claude/compassionate-keller-lpd1a`).

### Evaluated and skipped

- **CHANGELOG.md anchored to `v0.1.0`.** Cleanest one-shot doc available,
  but the owner has not picked a tag scheme and seeding one would
  commit them to a versioning style they may not want.
- **`docs/results/mujoco-vs-sandbox.md`.** Genuinely high-impact but
  requires a real recorded MuJoCo success rate to be honest about, and
  fabricating numbers into a results doc would be worse than not shipping.
- **Asciicast / short gif from `scripts/visualize_assembly_episode.py`.**
  Highest visual lift on the README, but needs a real local sim run to
  produce — outside the scope of an unattended cloud session.
- **Single-command `python -m embodied_skill_composer.demo` entry.**
  Tempting, but "demo" semantics depend on which backend the user wants
  (tabletop, warehouse, assembly, MuJoCo). Owner choice.

### Next-run candidates (priority order)

1. After `awesome-knuth-TXG01` (CI) merges to `main`, extend it with a
   `pre-commit run --all-files` step so the hooks added on `lOIj4` are
   enforced in CI, not just on local opt-in.
2. Add `docs/architecture.md` with a diagram of the
   planner / executor / perception / backend stack — mirrors the
   AegisQuant `docs/ARCHITECTURE.md` pattern from `lucid-darwin-ONkrp`.
3. Inventory ruff failures across `src/` and `scripts/` and open a
   tracking issue with a one-shot `ruff check --fix` plan; do NOT
   auto-apply (the codebase has algorithmic constants that linters
   misclassify).
4. Record an asciicast / gif of `visualize_assembly_episode.py` and
   embed at the top of the README, under the flagship results table.
5. Once the owner picks a release tag scheme, seed CHANGELOG.md against
   the agreed `v0.1.0` and start tagging releases so the flagship
   `2/2 beams` result becomes a citable artifact.

### Prior research-log context

Previous runs on unmerged `claude/sweet-clarke-*` /
`claude/awesome-knuth-*` branches (most recent first, none merged to
`main`):

- `T7V0c` (2026-04-26) — CITATION + SECURITY + ISSUE/PR templates.
- `lOIj4` (2026-04-25) — `.pre-commit-config.yaml` (ruff + whitespace).
- `ONkrp` (2026-04-23) — `.editorconfig` + `.github/dependabot.yml`.
- `wCHAT` (2026-04-24) — `.github/workflows/tests.yml` + MIT LICENSE.
- `QFlkH` — cross-platform Makefile.
- `awesome-knuth-TXG01` — first `.github/workflows/ci.yml`.
