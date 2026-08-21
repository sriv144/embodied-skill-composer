# Construction Intelligence Public Demo Provenance

The public-demo exporter is an evidence packager, not an experiment runner. It copies only
hash-pinned inputs, validates the claims they can support, generates stable read-only indexes, and
writes `provenance.json`. It never promotes unit, smoke, fixture, or incomplete evidence into a
release claim.

## Channels

`preview` is the default channel. With no input descriptor, the command builds the reviewed cottage
fixture and visibly records that learned-policy and live Coppelia evidence are absent:

```powershell
.\.venv\Scripts\python.exe scripts\export_construction_public_demo.py `
  --channel preview `
  --output workbench\public\demo
```

`release` requires three explicit input descriptors:

```powershell
.\.venv\Scripts\python.exe scripts\export_construction_public_demo.py `
  --channel release `
  --deterministic-bundle release-inputs\deterministic\deterministic-bundle.json `
  --research-bundle release-inputs\research-bundle.json `
  --simulator-bundle release-inputs\simulator-bundle.json `
  --source-commit <40-character-commit> `
  --source-clean `
  --source-tree-digest <64-character-clean-tree-digest> `
  --output workbench\public\demo
```

Create that canonical deterministic descriptor from the clean release commit with
`scripts/export_construction_public_demo.py --deterministic-input-only --output
release-inputs/deterministic`. The default exporter remains a truthful fixture preview and never
silently promotes itself to canonical evidence.

The final packaging command must run from a clean, explicitly recorded commit. Each evidence
descriptor also records its own clean source identity. Those input commits may differ when, for
example, a long Phase 4 matrix was pinned before a later workbench-only commit; the exporter
preserves every identity instead of rewriting history. It rejects mismatches *within* a bundle,
such as a run, selection, or simulator summary that disagrees with its descriptor's source commit.

## Input descriptor schema

Every descriptor uses schema `construction-intelligence-public-demo-input-v1`:

```json
{
  "schema_version": "construction-intelligence-public-demo-input-v1",
  "kind": "research",
  "evidence_status": "canonical",
  "source": {
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "dirty": false,
    "tree_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "created_at": "2026-07-26T00:00:00Z",
  "configuration_digests": ["<one digest for every distinct frozen run configuration>"],
  "protocol_digest": "<construction_intelligence_v1 protocol digest>",
  "profile": "research",
  "matrix_id": "<canonical matrix id>",
  "artifacts": [
    {
      "role": "matrix",
      "path": "matrix.json",
      "target": "evidence/research/matrix.json",
      "sha256": "<sha256 of matrix.json>",
      "media_type": "application/json"
    }
  ]
}
```

`path` is resolved relative to the descriptor unless it is absolute. `target` is always a safe,
POSIX-style path relative to the public-demo root. Research targets must live below
`evidence/research/`, and simulator targets below `evidence/coppelia/`. Every declared input is
hashed before any output directory is replaced.

### Deterministic bundle

A canonical release bundle contains exactly these roles:

- `project`, `scenarios`, `policies`, `runs`, and `report`;
- `house` and `robot`;
- `trace_sequential`, `trace_greedy`, `trace_optimized`, and `trace_recovery`.

A preview fixture may contain a smaller reviewed set, but its `evidence_status` must be `fixture`.

### Research bundle

The canonical Phase 4 descriptor contains exactly:

- `matrix`;
- `selections`;
- `evaluation`, `episodes`, and `report`;
- `acceptance` and `ablations`;
- `reproducibility_audit` and `release_completeness`.

The exporter verifies the research profile, all 20 completed matrix runs, all 20 immutable
validation-only selections, seeds 800–804, the five checkpoint fractions, matching source and
configuration digests, and the complete 240-episode held-out grid on seeds 900–904. The held-out
grid must contain 200 learned-policy and 40 baseline episodes with hierarchical confidence
interval summaries and per-training-seed and per-scenario-seed tables. All four primary acceptance
checks must pass. Both pre-registered ablation interpretations must be present; an honestly
unsupported hypothesis is allowed.

Phase 4 can create this descriptor after its durable executor writes the final matrix detail,
`matrix_selections.json`, `evaluation.json`, `episodes.csv`, `report.md`, `acceptance.json`,
`ablations.json`, `reproducibility_audit.json`, and `release_completeness.json`. Both final audits
must be complete and blocker-free. The descriptor points at those files and records their exact
SHA-256 values; the exporter does not need the live SQLite database.

### Simulator bundle

The canonical Phase 5 descriptor points directly at the two native bundles produced by
`run_construction_phase5_coppelia.py`; no conversion step or hand-authored summary is needed. For
both the `nominal` and `recovery` prefixes it contains roles for `manifest`, `scenario`,
`planned_jobs`, `replay`, `wheel_commands`, `telemetry`, `trace`, `metrics`, `report`, and `scene`.
The recovery roles point at the runner's `unavailable_robot_recovery` scenario.

The exporter independently invokes `verify_phase5_artifact_bundle()` for each directory. It requires
the native `construction_intelligence.coppelia_bundle.v1` manifest to attest a clean,
approval-confirmed, completed `live_coppelia` run with `live_gate_passed: true`, and verifies every
file against the native byte counts and hashes. Nominal metrics must show full installation, zero
post-start pose writes, zero collision stops, real wheel commands, measured telemetry, and both
formation and installation tolerances. Recovery metrics must additionally show disablement at or
after 25% completion, no later commands to that robot, reassignment of remaining work, and full
completion. Both native manifests must state `"payload_transport_model": "logical_carrier"`. The
public output explicitly says this does not demonstrate arm, gripper, grasp-contact, or payload
dynamics.

## Workbench evidence indexes

Every export writes `research-summary.json` and `coppelia-evidence.json`. A preview without
canonical inputs writes explicit `status: "absent"` payloads with empty evidence collections; it
does not omit the files or make the workbench infer a result from a failed request.

A canonical research summary contains the held-out hierarchical confidence intervals,
per-training-seed and per-scenario-seed tables, checkpoint-validation learning curves, primary
acceptance decisions, both pre-registered ablation interpretations, and concrete links to
individual hash-pinned files. A canonical simulator summary contains independently verified
nominal and unavailable-robot recovery manifests, metrics, measured-telemetry replay videos, and
individual file links. It sets `ready: true` only when both native live gates pass and both MP4
files decode successfully.

Every public link is a safe path below the exported bundle. Research and simulator descriptors
must match the exact v1 role-to-target and media-type contracts: JSON, JSONL, CSV, Markdown, MP4,
and the binary Coppelia scene are allowed, while HTML, SVG, XML, JavaScript, CSS, and other
same-origin active content cannot be published under an evidence link. MP4 files are generated
from attested base telemetry and explicitly label payload transport as logical; they are not
simulator-camera recordings or evidence of arm, gripper, or contact physics. The local API uses an equivalent
file-level contract under `/api/lab/.../artifacts/<file>` and rejects traversal, executable web
content, unregistered directories, and raw host paths.

## Output provenance schema

`provenance.json` uses schema `construction-intelligence-public-demo-provenance-v1` and records:

- release channel and exact exporter source identity;
- each input descriptor's SHA-256, evidence status, source identity, protocol digest,
  configuration digests, profile, and matrix ID;
- every emitted artifact's relative path, byte size, SHA-256, source kind, and role;
- the deterministic timestamp and ordering strategies;
- a canonical payload hash for the provenance manifest itself.

No wall-clock time is embedded. `generated_at` is the fixed Unix epoch, JSON keys and artifact paths
are sorted, and evidence timestamps are copied only from hash-pinned inputs. Repeating an export
with identical inputs produces byte-identical output.

The manifest cannot contain the hash of its own final bytes without self-reference, so
`provenance.json` is the sole coverage exclusion. Its `manifest_payload_sha256` hashes the canonical
manifest with that field set to `null`. Every other file is covered. Verify a generated directory
with:

```powershell
.\.venv\Scripts\python.exe scripts\export_construction_public_demo.py `
  --verify-only `
  --output workbench\public\demo
```

Verification fails for a missing, extra, changed, or size-mismatched artifact, or for a changed
manifest payload.

## Versioned release assets

The verified public-demo directory is one input to the final GitHub release packager; it is not
uploaded as an unstructured directory. `scripts/package_construction_release.py` packages it with
the canonical research evidence, Coppelia evidence, and all 20 validation-selected checkpoint and
ONNX exports. The packager independently verifies the release-channel public demo before creating
any ZIP, includes every ONNX external-data sidecar, uses sorted fixed-metadata ZIP entries, and
writes the five-file checksums/identity contract consumed by the manual draft-release workflow.

See [Construction Intelligence v1 release assets](construction-intelligence-v1-release.md) for the
exact local staging, draft creation, verification, and publication sequence.
