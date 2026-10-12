# Construction Intelligence v1 — completion status

Updated October 12, 2026. This is an acceptance-gate report, not a claim that v1 is released.

## Verified complete

Three of the seven numbered roadmap phases have passed their recorded gates:

- **Phase 1:** published checkpoint, protected `main`, required remote checks, and Pages preview.
- **Phase 2:** dependency/reproducibility foundations, typing cleanup, durable subprocess jobs,
  resumable checkpoints, cancellation, and restart reconciliation.
- **Phase 3:** the original versioned protocol and complete unit/smoke matrix, selection isolation,
  aggregation, reporting, and real interruption/resume evidence.

The deterministic compiler, fleet planning, recovery, traces, and workbench foundation are also
implemented. Counting phases is not a weighted percentage of engineering work or a time estimate.

## Implemented, awaiting final acceptance

- **Phase 6 workbench:** orthogonal editor, approval reset, shared validation, 12 parser goldens,
  local/static modes, experiment controls, results, responsive flows, and accessibility checks.
  Final hardening adds stale-replay protection, visible async errors, standalone diagnostic
  evaluation registration/display, and a real CPU resume → evaluation → downloadable results
  browser proof. Compilation notices were collapsed by the desktop grid, and mobile CSS hid the
  accessible progressbar. The layout fix passed all eight local/static desktop/mobile browser
  flows, including the real CPU resume/evaluation/download journey; fresh remote checks remain.
- **Phase 7 release infrastructure:** provenance-bound demo export, selected policy/ONNX and
  native simulator asset packaging, release checks, and deployment verification. Final hardening
  adds all 20 per-update training CSVs with a hash/provenance index, and keeps validation completion
  separate from rollout metrics. Further release-gate hardening independently recomputes research
  and ablation claims from downloaded archives, regenerates the canonical demo twice during staging,
  and makes draft publication wait for authorized deployment plus the exact-hash Pages smoke.
  These additions have focused tests; their complete remote checks remain pending.
  Canonical release artifacts cannot be generated from failed or
  incomplete research/simulator evidence.
- **Quality hardening:** an independent 87.85% line-coverage gate, separate from the retained 82.19%
  combined statement/branch gate, and retained XML evidence for future baseline audits. The new line
  floor is rounded upward from the verified current remote baseline, 16,033 / 18,252 lines.

Focused hardening verification passed: 54 frontend unit tests, 19 backend lab-path tests, 44
training-curve/research-packager tests, and 59 line-gate tests. Ruff, full-source mypy on 138 files,
byte compilation, dependency integrity, and the repository secret scan also passed. The completed
browser journey now passes locally on the Phase 6 fix. Complete coverage, protected remote checks,
and simulator-backed canonical round-trip remain unverified on the latest combined hardening
checkpoint; no narrow check is presented as their substitute.

PR [#25](https://github.com/sriv144/embodied-skill-composer/pull/25), checkpoint
`bd06db9a0443d4719be8d4aca34db882128d617e`, passed both remote Python, workbench, and secret checks.
The [pull-request run](https://github.com/sriv144/embodied-skill-composer/actions/runs/38111292762)
recorded 1,222 Python tests passed, three approval-gated tests skipped, 85.25% combined coverage,
full-source mypy on 136 files, and eight Playwright tests. That pass does **not** certify the later
hardening changes above; they require their own checks.

The subsequent release-fidelity audit passed 97 focused research/provenance/curve tests, 29
mocked release-asset tests, and 21 workflow/notes/channel tests. It found and closed archive-claim
verification, canonical regeneration, draft-lookup/permission, duplicate diagnostics, and
publication-before-deployment gaps. The Pages environment now permits `main` and the exact
`v1.0.0` tag; the publication reviewer is unchanged. These are implementation/verification results,
not a completed live or research release gate.

## Remaining release-critical work

1. **Phase 4 — corrected research evidence.** The original 20-run / 30M-transition matrix and
   240 held-out episodes completed, but MAPPO median makespan was 1.310722× CP-SAT, above the
   unchanged 1.15× limit. Completion thresholds passed. Preserve those historical failed results.
   All four validation-only pilots completed training and validation, but neither shared PPO-epoch
   setting qualified at the final 150K checkpoint. MAPPO nominal makespan ratios were 2.098546×
   (six epochs) and 2.374003× (two epochs); one algorithm also missed nominal completion for each
   setting. There is **no eligible pilot winner and no authorized corrected full-matrix launch**.
   See the [final pilot decision](evidence/construction-intelligence-validation-pilot-20261012-decision.json)
   and earlier scoped [training-artifact audit](evidence/construction-intelligence-validation-pilot-20261011-training-audit.json).
   Validation-only traces identify deterministic all-WAIT collapse and loss of parallel assignments;
   the current reward can prefer lower travel despite substantially worse makespan. A separately
   fingerprinted training-reward candidate must be tested before another research commitment.
   Freeze a corrected protocol only after pilot
   eligibility and complete smoke/resume gates, then run all 20 fresh 1.5M-transition experiments,
   validation-only selection, and held-out evaluation. All primary gates must pass.
2. **Phase 5 — live simulator evidence.** Complete and independently verify a full nominal cottage,
   then unavailable-robot recovery on the same source. Preserve zero post-start robot pose writes,
   real wheel commands/measured telemetry, safety tolerances, reusable scenes, and planned/measured
   replay. An earlier native crash and an interrupted GUI-suppressed attempt are not passing
   evidence. The headless retry ended at 9/26 modules with a route-control timeout, despite zero
   recorded collision stops and post-start pose writes. All nine retained native artifact hashes
   match their attestation; see the [failed-attempt record](evidence/construction-intelligence-phase5-20261012-route-timeout.json).
   Correct and test the native heading/ground-plane projection before a fresh nominal run; recovery
   may start only after nominal verification. Logical payload transport remains an explicit limitation.
3. **Phase 6 — final acceptance and merge.** Pass the new completed-job browser journey plus all
   existing local/static, mobile, keyboard, accessibility, unit, lint, build, and protected checks.
   Update its dedicated PR/evidence and merge only when its gates are proven.
4. **Phase 7 — publish.** Regenerate the public demo from passing canonical research/simulator
   artifacts, verify every asset/hash/source identity, finish consolidated documentation, pass
   release/deployed-Pages checks, merge the reviewed stack, and publish tag/release `v1.0.0`.

## Recoverability and boundaries

All four completed pilots and their final validation decision are retained. After earlier process handles disappeared, the dedicated durable
pilot queue was reattached to reconcile/resume compatible work; its immutable source was not edited.
The exact termination cause is unverified. Simulator attempts use separate output roots and are
never substituted for one another. No held-out data selects pilots or policies.

See the [roadmap](roadmap.md), [candidate protocol amendment](construction-intelligence-v1-protocol-amendment-1.1.0.md),
[historical research results](construction-intelligence-v1-research-1.0.0-results.md), and
[simulator runbook](construction-intelligence-v1-phase5-runbook.md) for the authoritative gates.
