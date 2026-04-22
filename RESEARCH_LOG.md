# Research Log

Automated improvement log maintained by Auto-Researcher.
Each run appends a dated entry describing what was implemented, what was skipped, and why.

---

## 2026-04-22 — Auto-Researcher v4

**Resume score at the start of this run:** 87/100 (top-1 this run — hierarchical MARL + PPO + MuJoCo + planned Isaac Lab backend, verified benchmark table in README showing learned hierarchical options solving 2/2 beams while low-level MARL stalls at 1/2).

**Prior open auto-researcher branch on this repo:** `claude/awesome-knuth-TXG01` (2026-04-21) added `.github/workflows/ci.yml` and a RESEARCH_LOG on that branch. It has not merged yet, so this run deliberately avoids `.github/` to keep both branches independently mergeable.

**Implemented (branch `claude/sweet-clarke-QFlkH`):**
- `Makefile`: cross-platform developer entrypoints mirroring the README's PowerShell snippets (install, install-rl, install-mujoco, install-pybullet, install-dev, test, lint, format, typecheck, bench-scripted, bench-rl, assembly-scripted, assembly-learned, assembly-benchmark, train-assembly-options, train-assembly-marl, mujoco-scripted, mujoco-learned, check-gpu, clean). Overrides via `PYTHON=...` and `RUNTIME_PROFILE=...`; `typecheck` is advisory (leading `-`) since the repo's mypy config is strict and not all source is clean yet.
- Seeded this `RESEARCH_LOG.md` on main.

**Why this was prioritized:**
The README documents the entire dev + RL + MuJoCo workflow in PowerShell only, which is a visible friction point for any Linux / macOS collaborator — and a headline project like this is exactly what a recruiter would clone on a Mac. A Makefile is the standard, zero-runtime-risk way to close that gap. It does not change Python imports, test behavior, or simulator config; it just exposes what already works.

**Evaluated and skipped this run:**
- Adding a second CI workflow or extending `.github/workflows/ci.yml`: the prior `claude/awesome-knuth-TXG01` branch owns that file and queued mypy-advisory / MuJoCo-smoke / nightly-RL jobs. Touching `.github/` from this branch would create a merge conflict. Re-queued to land once that branch merges.
- `docker-compose.yml` / `Dockerfile`: MuJoCo + PyTorch + PyBullet images are heavy and Linux/NVIDIA-setup-sensitive; the repo's `docs/setup/linux-nvidia-isaac.md` already walks through the native path. Image work is better scoped to a dedicated run.
- README results-table regeneration (queued on the prior branch): the README already surfaces a results table. Re-generating it from `docs/results/assembly-hierarchical-options.md` is a nice-to-have, not a next-highest-priority.
- `.env.example`: the repo is offline-first (simulator + local RL); there are no API keys to template.

**Next-run candidates (updated from prior branch's queue where still relevant):**
- Once `claude/awesome-knuth-TXG01` lands: add mypy as an advisory non-failing CI job, then a MuJoCo headless smoke job, then a nightly scheduled short RL training sanity run.
- Surface results tables from `docs/results/assembly-hierarchical-options.md` in the README as a generated block and link the Makefile `assembly-benchmark` target so reproducing the table is `make assembly-benchmark`.
- Add a `devcontainer.json` that preloads Python 3.11 + the CPU baseline so the VS Code quick-start matches `make install`.
- Add a short `docs/results/README.md` index listing every file under `docs/results/` with a one-line description and the command used to regenerate.
