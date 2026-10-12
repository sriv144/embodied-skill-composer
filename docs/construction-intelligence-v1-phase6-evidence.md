# Construction Intelligence v1 — Phase 6 Evidence

Status: the latest Phase 6 hardening passed all eight local browser flows on 2026-10-12.
Fresh protected remote acceptance, branch review, and merge remain pending; an earlier
implementation checkpoint passed remote checks.

## Product journey

The workbench now exercises the same `HouseDesign` contract from design through evidence:

1. edit a grid-snapped orthogonal floor plan,
2. validate dimensions, bounds, intersections, and openings,
3. explicitly approve the current revision,
4. compile the approved design,
5. inspect the task graph and fleet allocation,
6. replay deterministic construction,
7. launch or resume durable training work in local mode, and
8. inspect learned-policy, confidence-interval, ablation, and Coppelia evidence.

Every geometry edit revokes approval. Invalid or unapproved designs cannot compile.
The public static bundle is explicitly read-only. A local API failure renders an error
and never switches to fixture data.

## Acceptance-audit hardening — 2026-10-11

Replay refresh now clears the previous design/controller trace, ignores out-of-order responses,
and surfaces failures visibly. Compilation success is announced from the parent so replay loading
cannot silently remove its status. Recovery operations announce success/failure accessibly.

Completed standalone evaluations are registered with `matrix_id=null` and the explicit
`diagnostic` split, and Results exposes their real summaries and bounded downloadable artifacts.
These records cannot substitute for frozen research acceptance. Validation checkpoint completion
uses its own typed metric/chart, separate from training rollout endings and installed-module
fractions; chart headings and accessible section IDs are unique.

The new local browser flow uses real full CPU resume checkpoints interrupted at 32 of 64
transitions, resumes through the workbench, verifies `training_resumed` and completion events,
policy/ONNX artifacts and provenance, evaluates validation seed 800, then downloads the actual
report from its Results record. It runs for desktop and mobile; it does not train a research
matrix or use held-out data for selection. This stronger browser proof passed local browser
execution on 2026-10-12, as recorded below; fresh protected remote acceptance remains pending.

Focused verification in the isolated hardening checkout: ESLint and TypeScript passed; 54
Vitest tests across ten files passed; 19 backend lab-path tests passed; owned Python Ruff/mypy
and diff checks passed. These are scoped checks, not a replacement for full remote CI,
production builds, browser/accessibility acceptance, or protected review.

## Browser CI regression repair — 2026-10-12

The failed [PR #23 acceptance run](https://github.com/sriv144/embodied-skill-composer/actions/runs/38140060855)
and [PR #25 acceptance run](https://github.com/sriv144/embodied-skill-composer/actions/runs/38142195468)
exposed two real layout defects. The parent compilation notice was a third child in a two-row
desktop grid and became visually collapsed; the desktop failure reproduced locally before
repair. Mobile CSS also hid the named progress bars with `display: none`, removing them from
both the display and accessibility tree even when the resumed job had completed correctly.

The workbench now reserves a dedicated automatic-height live-status row between its header and
view, and mobile run rows show their progress bar across a second row. The browser test retains
the actual `100` percent accessible-value check and additionally requires the bar to be visible;
checkpoint, event, manifest, ONNX, evaluation, and artifact assertions were not weakened.

Verification scoped to `codex/phase-6-workbench` on Windows with local Chrome:

- All 49 Vitest tests across 10 files passed, including a new status-row regression.
- ESLint passed with zero warnings; strict TypeScript and `git diff --check` passed.
- Local and static production builds passed; the largest minified chunk remained 308.14 kB
  with no Vite chunk-size warning.
- All eight browser flows passed in separate local and static invocations: six local desktop/mobile
  tests (2.6 minutes) and two read-only static desktop/mobile journeys (1.5 minutes).
- Both local journeys resumed a real interrupted CPU unit checkpoint from 32 to 64 transitions,
  checked resume/completion events and policy/ONNX provenance, evaluated diagnostic validation
  seed 800, downloaded the real Results report, and exercised matrix WebSocket/cancel controls.
- The journeys' responsive, focus/occlusion, CSP, and serious/critical accessibility gates passed.

These are Phase 6 checkout counts, distinct from the earlier 54-test isolated hardening checkout.
They do not certify the complete Python coverage matrix, Linux CI, the research acceptance
matrix, live Coppelia gates, or a public release. Phase 6 remains unchecked until its fresh
required remote checks pass and the pull request is reviewed and merged.

## Acceptance coverage

| Gate | Evidence |
| --- | --- |
| Orthogonal editor | Wall create/move/resize/delete; door/window create/move/resize/delete; 0.25 m snapping; keyboard-selectable SVG geometry and handles. |
| Shared validation | Typed Python validation gates both `POST /api/design/validate` and rebuild/compile. The client projects the same issue codes for immediate feedback. Rebuild is atomic. |
| Multiple openings | The compiler preserves every opening’s wall-relative offset, width, height, and sill and emits deterministic wall panels around each aperture. |
| Parser goldens | A schema-versioned manifest defines exactly 12 deterministic cases: clean landscape/portrait, low contrast, inverted contrast, speckle noise, one/two/three openings, competing footprints, non-orthogonal ambiguity, too-small input, and undecodable bytes. Ambiguous inputs fail closed and every parsed plan remains unapproved. |
| Experiment workbench | Single and matrix launch, complete durable states, cancel/resume, typed reconnecting WebSocket events, validation selection, held-out launch, learning curves, per-seed tables, confidence intervals, primary acceptance, ablations, and simulator evidence boundaries. |
| Evidence integrity | Static mode consumes versioned `research-summary.json` and `coppelia-evidence.json`. Local results refresh every five seconds. Coppelia readiness requires independently verified nominal and unavailable-robot manifests; API reachability or a completed registry row is not evidence. |
| Artifact safety | Results expose concrete downloadable files through bounded, traversal-safe URLs. Raw absolute paths and artifact directories are never presented as links. |
| Runtime truthfulness | Immutable `local`/`static` contract. Static mode performs no `/api` requests. Local fetch failures remain visible errors. |
| Accessibility | Skip link, visible focus, accessible names, semantic progress, live status/error announcements, keyboard editor interaction, reduced-motion handling, and automated axe scans. |
| Responsive behavior | The complete Design → Brain → Simulate → Experiments → Results journey is exercised at desktop and Pixel 7 widths with horizontal-overflow assertions. Mobile uses one document scroll owner, a sticky project header, unobscured fixed navigation, and an actual Results bottom-scroll/navigation-reset proof. |
| Bundle size | Native SVG research charts replace the full ECharts bundle. Three.js is split by dependency depth; the largest minified production chunk is 308.14 kB and Vite emits no chunk-size warning. |
| Frontend quality | ESLint, strict TypeScript, Vitest/Testing Library with coverage, Playwright local/static desktop/mobile flows, and serious/critical axe gates are required in CI. |

## Local acceptance baseline

- ESLint: passed with zero warnings.
- TypeScript: passed.
- Vitest: 37 passed across 8 files.
- Vitest coverage: 59.89% lines overall; editor logic 91.70% lines; run-event hook 81.96% lines; results evidence view 86.25% lines.
- Local and static production builds: passed; no chunk exceeded 500 kB.
- Playwright: 6 passed across local/static and desktop/mobile projects after the independent scroll/occlusion review.
- The mobile regression proves the Results document is taller than the viewport, reaches its bottom, then resets to scroll position zero with the project header visible after navigation.
- A headed Playwright CLI review of the rebuilt static artifact rendered the editor, simulator, and results surfaces without CSP errors.
- Accessibility: zero serious or critical axe violations in the tested journeys.
- Python regression suite: 391 passed and 3 approval-gated tests skipped.
- Ruff, full-source mypy, compilation, `pip check`, and the repository secret scanner: passed.

Phase 6 remains unchecked in the roadmap until its pull request passes all protected
remote gates and is merged.

## Final hardening — 2026-10-10

The dependency lock was updated with compatible versions only. A fresh `npm ci` and
`npm audit` reported zero vulnerabilities; ESLint, TypeScript, all 37 frontend unit
tests, and both production builds passed. Unit line coverage was 60.00%; the largest
production chunk remained 308.14 kB with no chunk-size warning.

The separate public-provenance fixture now uses a 0.05 m tracking tolerance within
the unchanged 0.12 m physical-yard reserve. Three focused tracking-margin tests passed.
This fixture correction does not change robot geometry or any safety threshold.

The API launcher accepts `--coppelia-evidence-root` alongside `--registry-path`, so
verified evidence in independent execution workspaces need not be copied into the
default log directory. Eight isolated CLI tests passed, including loopback binding,
default behavior, custom paths, invalid roots, and incompatible reload rejection.

Final full Python coverage, browser/accessibility revalidation, canonical research
result integration, and protected remote checks remain pending. Earlier baseline
passes do not substitute for those final gates.

The published checkpoint `05bec64` subsequently passed both protected Linux CI
matrices for [PR #23](https://github.com/sriv144/embodied-skill-composer/pull/23):
580 Python tests passed, three approval-gated tests were skipped, combined coverage
was 82.78%, full-source mypy checked 129 files, and six local/static browser journeys
passed. Workbench lint, unit coverage, builds, lock integrity, and secret checks
also passed. See the [push CI run](https://github.com/sriv144/embodied-skill-composer/actions/runs/38070145418)
and [pull-request CI run](https://github.com/sriv144/embodied-skill-composer/actions/runs/38070148849).
The later external-evidence launcher and CLI-origin result verifier still require
their own final full-source checks; this checkpoint does not certify those additions.
