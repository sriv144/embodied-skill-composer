# RESEARCH_LOG.md

Persistent memory for the auto-researcher agent. Read top-to-bottom before deciding what to ship on the next pass.

---

## 2026-06-12 — Auto-Researcher v4

**Resume score at start of run:** 82 / 100 (top-2 of the 6-repo portfolio)

**Score breakdown:**
- Tech stack prestige: 24/25 — robotics + multi-agent RL + hierarchical options + MuJoCo backend + planned Isaac Lab is an elite robotics-RL stack.
- Commit recency: 22/25 — updated 2026-05-11.
- Feature completeness: 18/20 — flagship hierarchical-options result hits 1.000 success on the 2/2-beams assembly task; MARL baseline + MuJoCo 3D + playback visualizer all ship.
- Stars / visibility: 4/15 — 1 star, low discovery.
- README quality: 14/15 — flagship result table, runtime-profile matrix, install + commands per surface. Excellent.

### What was implemented this pass (branch `claude/sweet-clarke-qiarsa`)

Pure additive scaffolding — zero source-code, CI, runtime profile, or training script touched:

- `.github/ISSUE_TEMPLATE/bug_report.yml` with runtime-profile dropdown (local_dev / local_gpu / mujoco_local / isaac_gpu) and surface dropdown matching the README's 10 mode breakdown.
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/PULL_REQUEST_TEMPLATE.md` with a **result-drift check** that pins the headline `scripted=1.000 / learned-hierarchical=1.000 / low-level-MARL=0.000` baseline — protects the README's flagship table from silent regression.
- `CHANGELOG.md`.

### Why these and not something bigger?

Open-PR inventory (PRs #1–#17) already covers CI (ruff + pytest + mypy on 3.11/3.12), MIT LICENSE, CITATION.cff, Mermaid architecture docs, CODE_OF_CONDUCT, dependabot, Linux quickstart script. Stacking more CI on top is duplicative. Issue templates with a runtime-profile dropdown are uniquely valuable for a multi-backend robotics repo — the README itself hammers on "which runtime profile?" being the first triage question. The result-drift check in the PR template is novel and protects the most important README signal.

### Evaluated and skipped

- **Wiring up a results-CSV emitter that dumps `eval_assembly_options.py` numbers to a versioned `docs/results/*.csv`** — high value but needs to read the actual eval script to avoid contract drift. Queued.
- **MuJoCo playback GIF in README** — needs the MuJoCo backend run; out of scope here.
- **Promoting an existing CI PR to ready-for-review** — maintainer decision.
- **`scripts/quickstart.sh` (already in PR #7)** — duplicate.

### Next-run candidates (priority order)

1. **Result-tracking CSV**: `scripts/dump_assembly_metrics.py --policy {scripted,learned,marl} --runtime-profile X` -> `docs/results/assembly_metrics_YYYY-MM-DD.csv`. Lets the PR template's drift check be machine-checkable.
2. **MuJoCo demo GIF** (3-5s) in README, generated from `run_mujoco_assembly.py --policy scripted --gui --record`.
3. **Merge the lowest-risk of the open scaffolding PRs** (issue templates + LICENSE + CITATION.cff) into one clean PR.
4. **`docs/results/comparison.md`**: an auto-generated comparison table backed by the CSV emitter in (1).
5. **Isaac Lab backend stub** that satisfies the task contract — README already names this as the next backend milestone.
