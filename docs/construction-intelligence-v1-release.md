# Construction Intelligence v1 Release Assets

Construction Intelligence v1 is published from five pre-staged files. The publication workflow
does not build, repair, replace, or infer evidence:

- `construction-intelligence-v1-public-demo.zip`;
- `construction-intelligence-v1-research-evidence.zip`;
- `construction-intelligence-v1-coppelia-evidence.zip`;
- `construction-intelligence-v1-selected-policies.zip`;
- `construction-intelligence-v1-release-assets.json`.

The JSON identity manifest binds the repository version and tag, the clean 40-character source
commit, a SHA-256 digest of the committed Git tree listing, every ZIP's byte size and SHA-256, and
the four required asset roles. The identity manifest is the fifth required file and is validated
semantically rather than attempting an impossible self-hash.

## Stage locally

Run staging only from the clean commit that will receive `v1.0.0`. All repository version
declarations must already agree; the packager never changes versions.

First stage the canonical deterministic cottage baseline. Unlike the default preview exporter,
this mode requires a clean Git worktree and writes the complete canonical descriptor consumed by
release packaging:

```powershell
.\.venv\Scripts\python.exe scripts\export_construction_public_demo.py `
  --deterministic-input-only `
  --output release-inputs\deterministic
```

Use `release-inputs\deterministic\deterministic-bundle.json` as the deterministic descriptor
below.

First package the two independently verified native Phase 5 directories. This renders the public
nominal and recovery MP4 files deterministically from measured base telemetry and the typed replay;
it does not alter either native bundle or present logical payload motion as contact dynamics.

```powershell
.\.venv\Scripts\python.exe scripts\package_construction_coppelia_evidence.py `
  --nominal-bundle release-inputs\phase5-nominal `
  --recovery-bundle release-inputs\phase5-recovery `
  --output release-inputs\coppelia
```

Use the emitted `release-inputs\coppelia\simulator-bundle.json` as the simulator descriptor below.

```powershell
.\.venv\Scripts\python.exe scripts\package_construction_release.py `
  --output output\construction-intelligence-v1-release `
  --deterministic-bundle release-inputs\deterministic\deterministic-bundle.json `
  --research-bundle release-inputs\research-bundle.json `
  --simulator-bundle release-inputs\coppelia\simulator-bundle.json `
  --release-version 1.0.0 `
  --release-tag v1.0.0
```

Staging first invokes the canonical public-demo release exporter and verifier with all three
descriptor files. It then:

1. creates the complete release-channel public demo;
2. copies the exact research and Coppelia subtrees, including both measured replay videos, into
   separate evidence archives;
3. resolves all 20 validation-selected checkpoints from the canonical selection evidence;
4. verifies each checkpoint's SHA-256, checkpoint fraction, transition count, lineage,
   configuration digest, source commit, experiment identity, training seed, and resume provenance;
5. loads that exact checkpoint and exports its actor to ONNX;
6. includes `actor.onnx` and every file written beside it, including ONNX external-data sidecars;
7. writes sorted ZIP entries with a fixed 1980 timestamp, regular-file permissions, no comments or
   extra fields, and deterministic level-9 deflate;
8. independently verifies the complete five-file staging directory before replacing the output.

Repeat the command into a second output directory and compare it when performing the final
regeneration audit. Identical inputs must produce byte-identical files.

## Verify locally

Before creating the draft:

```powershell
.\.venv\Scripts\python.exe scripts\package_construction_release.py `
  --verify-only `
  --output output\construction-intelligence-v1-release `
  --release-version 1.0.0 `
  --expected-tag v1.0.0
```

Verification rejects a missing or extra top-level asset, a changed size or hash, duplicate or
unsorted ZIP entries, non-canonical ZIP metadata, path traversal, corrupt data, a research or
Coppelia archive that differs from the public-demo subtree, a non-release public demo, a version,
tag, or clean-source mismatch, an incomplete policy matrix, or any policy/checkpoint mismatch.
Before Python's ZIP parser, CRC scanning, or extraction it parses a bounded, non-ZIP64 central
directory and enforces role-specific compressed, expanded, member-count, per-member, path-length,
and compression-ratio budgets. The maximum verification footprint is below 5 GiB, including all
four downloaded archives and the three trees extracted concurrently.

After creating the tag, add `--require-tag-at-head` to require that `refs/tags/v1.0.0` resolves to
the checked-out commit.

## Create the draft

Create and push the tag only after the release commit and staging output are final. Then create an
unpublished GitHub draft with exactly the five verified files:

```powershell
git tag v1.0.0
git push origin v1.0.0

gh release create v1.0.0 --draft --verify-tag `
  --title "Construction Intelligence v1.0.0" `
  --notes-file docs\construction-intelligence-v1-release-notes.md `
  output\construction-intelligence-v1-release\construction-intelligence-v1-public-demo.zip `
  output\construction-intelligence-v1-release\construction-intelligence-v1-research-evidence.zip `
  output\construction-intelligence-v1-release\construction-intelligence-v1-coppelia-evidence.zip `
  output\construction-intelligence-v1-release\construction-intelligence-v1-selected-policies.zip `
  output\construction-intelligence-v1-release\construction-intelligence-v1-release-assets.json
```

Do not edit an uploaded asset in place. If any byte changes, restage all assets, replace the draft's
five files, and rerun local verification.

## Publish the verified v1 draft

The manual workflow is intentionally hard-coded to the one v1 release:

```powershell
gh workflow run release.yml
```

`.github/workflows/release.yml` requires `v1.0.0` to resolve to the current protected `main`
commit. Its read-only verification job installs dependencies from the hash-locked Linux
constraints, downloads the existing draft, requires the exact five-file inventory, and verifies
every identity, hash, archive, public-demo claim, selected checkpoint, and tag-to-HEAD relation.
It passes only a SHA-256 fingerprint of that complete asset set to a fresh publication runner. The
write-token job rechecks the tag, `main`, draft state, and downloaded asset-set fingerprint before
running `gh release edit v1.0.0 --draft=false`. Both jobs require a non-prerelease draft and bind
its exact title and normalized body to the committed
`construction-intelligence-v1-release-notes.md` file. Configure the
`construction-intelligence-v1-release` GitHub environment with required reviewers and protect the
`v1.0.0` tag. A failed, cancelled, changed, or unapproved run leaves the release unpublished.
