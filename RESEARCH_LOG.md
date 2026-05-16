# Research Log

A running log of automated research-and-development passes against this repository.

## 2026-05-16 — Auto-Researcher v4

**Resume-worthiness score at start of run: 84 / 100**

| Signal | Score |
| --- | --- |
| Tech stack prestige (RL + multi-agent + robotics + MuJoCo) | 25 / 25 |
| Commit recency (updated 2026-05-11) | 22 / 25 |
| Feature completeness (10 documented modes, hierarchical options, MuJoCo backend) | 20 / 20 |
| Stars + visibility (1 star) | 3 / 15 |
| README quality (strong architecture, setup, results table) | 14 / 15 |

### Implemented this run (branch: `claude/sweet-clarke-Ow84F`)

- **ci: add GitHub Actions workflow for syntax + tests.** Added `.github/workflows/ci.yml` that compiles `src/` with `compileall` and runs the existing `pytest` regression suite on every push and PR against `main`. The repository already ships a test suite under `tests/` plus a README invoking `pytest -q --basetemp .pytest_tmp`, but nothing currently exercises it in CI.

### Why this was prioritized

This is the strongest of the 6 target repos on raw tech and feature completeness, but it had zero CI signal. For a research-flavored RL/robotics project a green CI badge is a cheap, high-trust marker. Wiring up the existing `pytest` suite is the safest possible first CI iteration — it requires no new dependencies and no new test code.

### Evaluated and skipped

- **MuJoCo / CUDA jobs in CI.** Heavy GPU and sim dependencies would balloon CI time and likely fail on default runners. The regression value comes from CPU-only logic tests that already exist in `tests/`.
- **README screenshots / GIFs of MuJoCo runs.** High value, but requires running the simulator and recording media locally — not feasible from an autonomous PR pass.
- **Lint / type-check addition (ruff, mypy).** Promising but risks quietly failing the new CI on its first run. Deferred to a dedicated pass once the test job is proven green.

### Next-run candidates

1. Add a `ruff` lint job alongside the existing test job once CI is stable.
2. Add an Anthropic / Claude-backed planner option on top of the existing hierarchical option layer.
3. Commit a small `docs/results/assembly-hierarchical-options.png` summarising the success-rate table from the README so GitHub renders the flagship result visually.
4. Add a top-level Mermaid architecture diagram inline in the README.
