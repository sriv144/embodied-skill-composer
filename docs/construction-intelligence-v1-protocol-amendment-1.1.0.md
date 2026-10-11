# Protocol 1.1.0 amendment — candidate, not release evidence

The original protocol remains archived byte-for-byte at
[`construction_intelligence_v1-1.0.0.yaml`](../configs/construction/experiments/archive/construction_intelligence_v1-1.0.0.yaml).
Its validated fingerprint is
`79e5444bb108573a9ce1597493e4556cf4fd7ba59607f81e7fe2826b0e74c707` and its
research source is `5166577d9f1d4f1211bfb35727e84b3588c03bcb`. Original runs,
selections, and results must not be overwritten or relabelled as this revision.

## Evidence and reasons

The [validation-only audit](evidence/construction-intelligence-v1-protocol-1.0.0-validation-audit.json)
contains the source, selection-file, learning-curve, and immutable-selection hashes used for this
amendment. It contains no held-out inputs.

Validation seed 801 places an obstacle at grid cell `(3, 3)`. The old sampler excluded the
staging centre `(3, 4)` but not `foundation_0_0`'s required two-base approach formation cell
`(3, 3)`. That foundation and seven dependent modules cannot be installed: the scenario ceiling
is `10/18 = 55.556%`, and the five-scenario nominal validation ceiling is `91.111%`. All ten
selected primary policies' incomplete validation episodes occur on this seed. No seed is
removed, skipped, or relabelled by the correction.

There is a separate policy-stability problem. On feasible validation scenarios 800 and 802–804,
the mean nominal completion of IPPO's 10% candidates is 100%, falling to 76.606% at the final
checkpoint; MAPPO falls from 99.615% to 88.662%. These later failures have zero invalid bids
and reach the 256-decision limit. This motivates a controlled PPO-update pilot, not a claim that
a proposed hyperparameter change has already solved the problem.

Two diagnostic/terminology defects were also found. The old `rollout_terminal_fraction` includes
both successful termination and time-limit truncation; it is not structure completion. The old
failure setting samples a constant 35% failure mixture rather than a progressive curriculum.

## Preregistered correction

1. Reserve robot start cells, module centres, and both staging and target formation cells when
   sampling nominal and injected obstacle failures. Reject obstacles that disconnect required
   interaction cells. Preserve the initial candidate shuffle and failure-kind/timing stream;
   fail explicitly if a requested obstacle count cannot be placed. Routing does not silently
   bypass blocked endpoints.
2. Record actual installed-module fraction and successful completion separately from
   termination, truncation, and collector-budget cutoffs. Preserve the legacy terminal metric's
   terminal-or-truncated meaning; do not invent missing historical metrics when resuming a curve.
3. Make the revised matrix's failure curriculum explicit: failure probability increases linearly
   from 0 to 0.35 over the first 50% of training transitions, then remains 0.35. The no-curriculum
   ablation disables failures throughout. Use the persisted transition counter for resumed runs.
   Existing single-run configurations retain their constant-probability default.
4. Before freezing the revised research matrix, compare PPO epochs 6 (control) and 2 (candidate)
   from scratch for MAPPO and IPPO, training seed 7, 150,000 transitions each. Keep the research
   architecture, 128 expert episodes, 40 cloning epochs, and all other training settings equal.
   These four runs are explicitly validation pilots, never release runs. Set the recorded
   curriculum reference budget to 1.5M transitions: the pilot is the first 10% of the research
   curriculum, not an accelerated ramp. Evaluate the five fractional checkpoints on seeds
   800–804, both failure modes, and compare validation makespans against CP-SAT on that same
   validation grid. Fractional pilot checkpoints describe trajectories only; hyperparameter
   comparison uses each run's **final 150K** checkpoint, the first actual research milestone.
   Choose further changes only using training and validation evidence; no held-out result may
   select a pilot or checkpoint.

Both algorithms share one final PPO epoch count. A pilot candidate is eligible only if its final
validation results meet the unchanged primary thresholds applied to validation: nominal mean
completion ≥95% for both algorithms, MAPPO failure mean completion ≥85%, and MAPPO median
nominal makespan / CP-SAT median nominal makespan ≤1.15. Rank eligible epoch counts by:

1. higher minimum nominal mean completion across MAPPO and IPPO;
2. higher mean nominal completion across both algorithms;
3. higher MAPPO failure mean completion;
4. lower MAPPO / CP-SAT median nominal makespan ratio;
5. lower pooled nominal mean makespan;
6. fewer epochs (an exact tie prefers 2).

If no candidate is eligible, record no winner and continue training/validation investigation;
do not automatically launch a full research matrix. Eligibility is only a validation proxy, not
a held-out acceptance result or a guarantee that final release gates will pass.

The complete revised 20-run matrix must be frozen on a clean source commit after these pilots.
All twenty runs start from scratch; old resume state is incompatible with the changed source,
scenario generator, and configuration fingerprints. Both baselines and learned policies must be
reevaluated under the corrected generator. The transition budget remains 1.5M per research run.

## Unchanged release requirements

Training seeds 7–11, scenario splits 0–799 / 800–804 / 900–904, five checkpoint fractions,
validation-only checkpoint ranking, deterministic held-out evaluation, 2,000-sample hierarchical
bootstrap, all primary thresholds, and both ablation interpretation boundaries are unchanged.
Neither the defect nor failed historical results authorize lowering those thresholds.

The no-failure-training ablation tests the complete failure-exposure intervention. It does not
isolate the causal benefit of a progressive ramp compared with a constant-probability mixture.

Protocol 1.1.0 remains a candidate until its unit/smoke/resume gates and controlled pilots are
recorded. Phase 4 stays unchecked until the revised complete research matrix passes.

## Durable pilot execution

On the clean committed correction checkout, inspect the frozen pilot before dispatch:

```powershell
python scripts/run_construction_validation_pilot.py plan --device cuda
python scripts/run_construction_validation_pilot.py execute --device cuda
```

The executor uses a dedicated `validation-pilot.sqlite` registry, a kernel-owned executor lock,
and the existing serial subprocess training queue. Rerun the same command on the same source
commit to reconcile or resume compatible jobs. Cancellation is terminal; missing or incompatible
checkpoints fail closed. Interrupting the executor detaches without cancelling its durable worker.
All five policy counters must match 15K, 37.5K, 75K, 112.5K, and 150K transitions before evaluation.
Source, configuration, design, split, and per-episode provenance are checked before results are
registered. The summary records no winner if neither epoch budget meets the validation proxy;
it cannot launch a research matrix or become release evidence.
