# Construction Intelligence v1 Phase 3 Evidence

## Scope

This record covers the frozen research protocol and its non-synthetic unit-profile end-to-end gate.
The gate uses the production SQLite-backed subprocess queue, checkpoint/resume implementation,
validation selector, held-out evaluator, hierarchical aggregation, report generator, and artifact
auditor. It does not substitute the 1.5M-transition research runs required by Phase 4.

## Provenance

| Field | Value |
| --- | --- |
| Source commit | `d864d3087b484d29d7145e4464117be4f9dc6ee1` |
| Worktree-change digest | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Protocol digest | `79e5444bb108573a9ce1597493e4556cf4fd7ba59607f81e7fe2826b0e74c707` |
| Execution profile | `unit` |
| Matrix ID | `construction_intelligence_v1-unit-e2e-79e5444bb108-20260726T091518Z-57f53e67` |
| Verification schema | `construction-intelligence-phase3-e2e-v1` |
| Verification SHA-256 | `350e57d7873b2c367ebc9607646615d8c0d22039128a6f13eab12b125ff3541c` |
| Run date | 2026-07-26 |

The verifier started and finished against the same clean committed source fingerprint. Generated
run data remains under the ignored `logs/construction_intelligence/phase3_e2e/d864d30` directory;
the compact hashes below make the canonical outputs independently checkable without committing the
189 MB working run.

## Acceptance results

| Gate | Result |
| --- | ---: |
| Production subprocess training runs completed | 20 / 20 |
| Fractional policies exported at 10%, 25%, 50%, 75%, and 100% | 100 / 100 |
| Validation candidates evaluated | 100 / 100 |
| Validation episodes on seeds 800–804 | 1,000 / 1,000 |
| Policies selected using validation only | 20 / 20 |
| Held-out episodes on seeds 900–904 | 240 / 240 |
| Learned-policy held-out episodes | 200 |
| Baseline held-out episodes | 40 |

The durable-run proof forcibly stopped
`construction_intelligence_v1-unit-ippo_full-seed-10` after a positive checkpoint at transition 8.
The harness stopped the dispatcher and then invoked the production stale-run reconciler, which
fenced the exact owned worker and persisted the run as interrupted. Attempt 2 resumed from transition
8 and completed. This exercises real persisted state and process ownership rather than a mocked
status transition.

The short unit policies passed the MAPPO no-failure completion, IPPO no-failure completion, and
MAPPO failure-completion evaluators. They did not pass the research-only MAPPO-to-CP-SAT makespan
threshold (`1.5077×` observed versus `1.15×` required). That outcome is expected for a plumbing
profile and does not satisfy or fail Phase 4; the frozen 1.5M-transition matrix must pass every
primary threshold before release.

## Canonical artifact hashes

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `matrix_selections.json` | 2,601,759 | `6239bef9014bfc7725d8607d7dd0a929416c86a8d70256764b9a75b6fa78227f` |
| `evaluation.json` | 778,614 | `f6f5fcc8be7a2c0d972cc132944247e75e2b5eb599637bcfcb73079323b843d1` |
| `episodes.csv` | 250,408 | `95735a398135ed6738a2dbe647b047432a2b78c5772f306c3702b0b538acf616` |
| `report.md` | 8,761 | `b534450fb7f5deb89aeaae070340a6a77b02bbbba2f12b4ee79429ae26f8c883` |

## Reproduction

From a clean committed checkout with the Construction Intelligence dependencies installed:

```powershell
.\.venv\Scripts\python.exe scripts\verify_construction_phase3_e2e.py `
  --output-root "$PWD\logs\construction_intelligence\phase3_e2e\$(git rev-parse --short HEAD)-$(Get-Date -Format yyyyMMddTHHmmss)"
```

The output directory must not exist before launch. A successful run writes `verification.json` only
after the full matrix, forced interruption/resume, validation selection, held-out evaluation, report
audit, and final source-fingerprint check have passed.

## Quality and review status

The local gate refresh on 2026-07-26 passed:

- Ruff across `src`, `scripts`, and `tests`;
- mypy across 118 files in `src` and `scripts`;
- `pip check` and byte compilation;
- 277 tests passed, 3 opt-in/live tests skipped, and combined statement/branch coverage was 82.32%
  against the required 82.19% floor;
- the repository secret scanner;
- a clean `npm ci` with zero reported vulnerabilities; and
- the static TypeScript/Vite production build.

The existing Vite chunk-size warning remains explicitly assigned to Phase 6. The hosted Gitleaks
action and the rest of the protected remote matrix remain pending until the review branch is
published; their result will be appended before Phase 3 is checked on the project roadmap.
