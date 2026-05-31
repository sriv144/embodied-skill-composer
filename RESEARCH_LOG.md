# Research Log

Automated improvement log maintained by the auto-researcher agent.

---

## 2026-05-31 — Auto-Researcher v4

**Resume score at start of run:** 77 / 100 (2nd of 6 target repos)

### Implemented (branch: `claude/sweet-clarke-D2Gmr`)

- **`.github/workflows/ci.yml`** — first CI workflow for the repo. Runs
  `ruff check`, `mypy`, and `pytest` on Python 3.11 and 3.12 for every push
  and PR. The repo already configures ruff and mypy in `pyproject.toml`, so
  the workflow just wires them into CI. `mypy` is set to
  `continue-on-error: true` for now so type-only regressions don't immediately
  red-light the suite.
- **`LICENSE`** — MIT, dated 2026. Previously the repo had no license file,
  which blocked external reuse.
- **`scripts/quickstart.sh`** — Linux/macOS equivalent of the existing
  `run_project.ps1`. The README's setup snippets are all PowerShell, which
  was a real onboarding obstacle for non-Windows users. Supports
  `--with-mujoco` and `--skip-tests` flags.

### Why this was prioritized

- The repo is technically strong (multi-agent RL, hierarchical options,
  MuJoCo backend) but read as "Windows-only research code" to a Linux-first
  reviewer because every README snippet uses PowerShell paths.
- No CI signal means a reviewer can't tell at a glance whether the test
  suite passes — a quick win for credibility.
- No LICENSE means the repo is technically "all rights reserved."

### Evaluated and skipped

- **Adding a CONTRIBUTING.md.** The repo already has one (`CONTRIBUTING.md`
  is present at the root), so skipped.
- **Adding an Isaac Lab backend stub.** The roadmap mentions it but the
  surface area is large (sim adapter + task contract bridge + GPU CI) —
  multi-PR work, not appropriate for a single auto-researcher commit.
- **Recording a short GIF of the MuJoCo assembly task for the README.**
  Requires a GPU runner and `scripts/run_mujoco_assembly.py --record`. Worth
  doing in a future run with proper artifact handling.
- **Adding Cohere or local SentenceTransformer fallback.** Not relevant to
  this repo — no embedding model used here.

### Next-run candidates

1. Add a results badge to the README pointing at the new CI workflow.
2. Add a `docs/` MkDocs site that surfaces the existing `docs/results/` and
   `docs/setup/` markdown.
3. Record a short MuJoCo GIF and embed it under the "Flagship Result"
   section of the README.
4. Drop `continue-on-error` from the `mypy` step once the codebase is
   strict-clean.
5. Add a `docker/Dockerfile` so the assembly stack can be reproduced on any
   host without a Python install.
