# Research Log

A running record of auto-researcher passes against this repo.

## 2026-05-25 -- Auto-Researcher v4

**Resume score at start:** 83/100. Most polished repo in the set:
hierarchical-options results table in the README, dedicated `docs/`
with setup notes for Windows/VSCode and Linux/NVIDIA, MuJoCo backend,
GPU sanity-check script, `CONTRIBUTING.md`, multiple requirements
variants per backend, and a benchmark suite.

**Implemented on branch `claude/sweet-clarke-FVAfQ`:**

- MIT `LICENSE`.
- Seeded this `RESEARCH_LOG.md`.

**Why nothing larger this pass:** The highest-value improvements all
require changing RL training/eval behavior:
- Add Isaac Lab backend stub (mentioned in the roadmap) -- needs the
  task contract migration, can't be a one-shot diff.
- Add a CI workflow that runs `pytest -q` -- low risk in isolation, but
  the test suite pulls torch + MuJoCo on import paths; the install
  surface in CI is non-trivial and worth its own pass.
- Add training-curves Markdown reports auto-generated after each PPO
  run -- needs to wire into `scripts/train_assembly_options.py` and
  introduce a new artifacts contract.

None of those are safe drive-by additions; documenting them here keeps
the ratchet visible for the next pass.

**Evaluated and skipped:**

- Replacing the Windows-flavored README commands with cross-platform
  equivalents: would help linux/mac users, but the README explicitly
  targets Windows-first development today.
- Adding pre-commit hooks: low signal-to-noise on a repo with a known
  active developer and few external contributors.

**Next-run candidates:**

- CI workflow restricted to the lightweight subset of tests (no
  MuJoCo / no torch GPU paths).
- Isaac Lab backend stub that satisfies the task contract API and
  raises NotImplementedError on the simulator hooks.
- An auto-generated training-curves Markdown summary committed under
  `docs/results/`.
- Promote `docs/results/assembly-hierarchical-options.md` into the
  README via a hero figure.
