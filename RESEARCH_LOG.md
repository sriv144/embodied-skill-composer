# Research Log

Automated improvement log maintained by Auto-Researcher.
Each run appends a dated entry describing what was implemented, what was skipped, and why.

---

## 2026-04-21 — Auto-Researcher v4

**Resume score at the start of this run:** 84/100 (top-ranked this run — MARL + PPO + MuJoCo + Isaac Lab plan, detailed README with benchmark tables).

**Implemented (branch `claude/awesome-knuth-TXG01`):**
- Added `.github/workflows/ci.yml`: runs `ruff check src tests` and `pytest -q` on Python 3.11 for pushes and PRs against `main`. Core `requirements.txt` only — RL/MuJoCo/PyBullet extras are excluded so the matrix stays fast and doesn't need GPU wheels.
- Seeded this `RESEARCH_LOG.md`.

**Why this was prioritized:**
Repo had zero `.github/` directory. Adding CI is a pure upside for a research-grade robotics project: it proves the test suite runs cleanly, protects against regressions in the tabletop + warehouse + assembly baselines, and sends a strong resume signal ("this author ships with CI"). Low risk — the workflow only exercises the CPU regression path the repo already documents as the default dev loop.

**Evaluated and skipped this run:**
- Multi-Python matrix (3.11, 3.12): `pyproject.toml` pins `>=3.11` but downstream sims (MuJoCo, Isaac Lab) are Python-version sensitive. Keeping a single 3.11 job until extras are exercised in CI.
- RL training smoke job (`train_assembly_options.py`): too slow for PR CI; belongs in a separate scheduled nightly workflow.
- MuJoCo / PyBullet jobs: require headless GL + apt deps; will add a separate `sim.yml` if demanded.
- `mypy` gate: repo has `[tool.mypy]` config but many source files likely aren't strict-clean yet — turning it into a hard gate without a triage pass would block PRs.

**Next-run candidates:**
- Add a nightly `scheduled-rl.yml` that trains a tiny assembly-options policy and asserts success rate ≥ 0.5 on the scripted oracle.
- Add a headless MuJoCo smoke job (`scripts/run_mujoco_assembly.py --policy scripted --no-gui`).
- Turn `mypy` into an advisory job (non-failing) first, then gate.
- Publish results tables from `docs/results/assembly-hierarchical-options.md` into the README as a generated block.
