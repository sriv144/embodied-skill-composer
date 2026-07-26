# Construction Intelligence v1 Research Protocol

The versioned protocol at
[`configs/construction/experiments/construction_intelligence_v1.yaml`](../configs/construction/experiments/construction_intelligence_v1.yaml)
is the source of truth for learned-coordination evidence. The loader rejects unknown fields,
changed seed partitions, incomplete checkpoint grids, unregistered variants, and split overlap.
Every queued run stores both the protocol digest and the canonical runtime-aware training
configuration digest.

## Frozen Matrix

Each profile expands to the same 20-run shape:

| Variant | Algorithm | Role | Seeds | Research transitions |
| --- | --- | --- | --- | ---: |
| `mappo_full` | MAPPO | primary | 7–11 | 1,500,000 |
| `ippo_full` | IPPO | primary | 7–11 | 1,500,000 |
| `mappo_no_bc` | MAPPO | behavior-cloning ablation | 7–11 | 1,500,000 |
| `mappo_no_failure_curriculum` | MAPPO | failure-curriculum ablation | 7–11 | 1,500,000 |

The no-behavior-cloning variant sets both expert episodes and cloning epochs to zero. The
no-failure-curriculum variant leaves all other training settings unchanged and disables sampled
training failures.

Every run saves resumable and evaluation-ready checkpoints at 10%, 25%, 50%, 75%, and 100%.
Research targets are exactly 150,000, 375,000, 750,000, 1,125,000, and 1,500,000 transitions.
Each resumable snapshot contains actor, critic, both optimizers, counters, learning curves, Python,
NumPy and Torch RNG state, configuration and design digests, source identity, and checkpoint
lineage.

## Split Isolation And Selection

- Training scenarios: seeds 0–799.
- Validation and checkpoint selection: seeds 800–804.
- Final held-out evaluation: seeds 900–904.

All five checkpoints are evaluated deterministically on every validation seed, both without and
with failures. A run is selectable only when the exact five-checkpoint grid is complete and every
candidate shares the same configuration digest and source commit. The preregistered ordering is:

1. higher mean completion;
2. lower mean makespan;
3. fewer transitions;
4. lexicographically smaller checkpoint ID.

The frozen selection record preserves the checkpoint path and SHA-256, transition fraction,
candidate ranking, lineage, configuration digest, source commit, and resume provenance. Selection
records are immutable. The API never accepts a client-authored checkpoint, metric, or ranking: a
confirmed selection request makes the server evaluate all five persisted checkpoints and freeze the
pre-registered ordering itself. The held-out API and executor refuse to start until all 20 selections
and their complete server-generated evidence files exist.

## Final Evaluation And Statistics

The selected 20 policies are evaluated on seeds 900–904, both without and with failures. Sequential,
greedy, auction, and CP-SAT baselines are evaluated on the same grid. This yields 200 learned-policy
episodes and 40 baseline episodes.

Reports include per-training-seed and per-scenario-seed tables. Confidence intervals use a
deterministic crossed hierarchical bootstrap: training seeds and the shared scenario-seed grid are
resampled independently, and each replicate evaluates their Cartesian product. This preserves the
paired scenario difficulty shared by every trained policy.

Primary release thresholds are:

- MAPPO and IPPO mean no-failure completion at least 95%;
- MAPPO median makespan no more than 1.15 times CP-SAT;
- MAPPO mean failure-suite completion at least 85%.

Behavior cloning is supported when final completion improves by at least 5 percentage points or
transitions-to-95% fall by at least 20%. Transitions-to-95% is the earliest frozen checkpoint whose
mean no-failure completion across all five training seeds and all five validation scenarios reaches
95%; if the ablation never reaches 95% while the full variant does, the reduction is treated as
100%. Failure curriculum is supported when failure completion improves by at least 10 points
without reducing no-failure completion by more than 2 points. Unsupported hypotheses are reported
and do not block release; incomplete evidence does.

## Execution

The durable executor can launch or reattach to the canonical matrix:

```powershell
python scripts\run_construction_experiment.py execute --profile research
```

It retains the one-GPU FIFO guarantee, automatically resumes interrupted runs with compatible
checkpoints up to the configured attempt limit, evaluates validation checkpoints, freezes
selections, runs the held-out matrix, and writes the structured acceptance artifact. A research
run is not marked evidence-complete by assertion: the executor hashes and validates every training
configuration, learning curve, policy manifest, ONNX export, resumable/fractional checkpoint,
selection record, held-out table, report, acceptance result, and ablation result. Read-only status
and explicit selection/held-out stages are also available:

```powershell
python scripts\run_construction_experiment.py status
python scripts\run_construction_experiment.py select <matrix-id>
python scripts\run_construction_experiment.py heldout <matrix-id>
```

The corresponding loopback API provides matrix launch/list/status, immutable selection, held-out
launch, individual run control, and typed persisted event streams. The existing single-run REST and
CLI behavior remains supported.

## Phase 3 End-to-End Gate

The non-synthetic unit-profile gate runs all 20 policies through the production SQLite/subprocess
queue, terminates one owned worker after a positive checkpoint, reconciles and resumes it, evaluates
all 100 fractional policies on 1,000 validation episodes, and produces the complete 240-episode
held-out report:

```powershell
.\.venv\Scripts\python.exe scripts\verify_construction_phase3_e2e.py `
  --output-root "$PWD\logs\construction_intelligence\phase3_e2e\$(git rev-parse --short HEAD)-$(Get-Date -Format yyyyMMddTHHmmss)"
```

The command fails fast unless the source worktree is clean and committed, the output path does not
already exist, and the source fingerprint remains unchanged for the full run. Its final
`verification.json` records the source commit, protocol digest, interruption/resume lineage, exact
episode counts, and SHA-256 hashes for the canonical selection and held-out artifacts. The matching
pytest gate is opt-in with `RUN_CONSTRUCTION_PHASE3_E2E=1`.

The accepted clean-commit run is recorded in the
[Phase 3 evidence record](construction-intelligence-v1-phase3-evidence.md).
