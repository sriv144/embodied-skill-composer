# Validation-only reward-time candidate — October 12, 2026

This is a preregistered diagnostic pilot, not a frozen full research protocol or a release result.
The completed [PPO-epoch pilot](evidence/construction-intelligence-validation-pilot-20261012-decision.json)
has no eligible winner. Its evidence and the failed original research matrix remain unchanged.

## Observed failure mechanism

Eight CPU-only deterministic traces reproduced the saved selected/final validation-seed-800
completion, makespan, travel, and idle metrics to `1e-9` on immutable source
`2eba09c3fdff9babc15c7615ca6a437f8a9704cf`.

- Final MAPPO/six-epoch and IPPO/two-epoch policies entered all-WAIT fixed points with ready work,
  idle robots, no active jobs, and no pending failures. Actor inputs stayed byte-identical through
  the remaining decisions; neither invalid bids nor collisions explained the noncompletion.
- MAPPO/two-epoch selected → final became 274.8995 s slower while travelling 35.5 m less;
  undiscounted reward increased from 4.8050077 to 4.8530994.
- IPPO/six-epoch selected → final became 186.0228 s slower while travelling 26.5 m less;
  reward increased from 4.8135838 to 4.8510819.

The training reward's time term is `-0.15 * elapsed_s / 1800`, while travel costs `0.002 / m`.
One metre saved therefore offsets 24 seconds of makespan. This is evidence of a reward/objective
mismatch and lost parallelism, not proof that the proposed intervention will solve either defect.
The diagnostic summary SHA-256 is
`ea96bf682f83aa7bb5e90780d9d86621a739792a51867fb0a3346bf716c478fe`.
The local detailed traces remain outside Git with the completed pilot artifacts.

## Intervention and decision rule

Compare `reward_time_coefficient=0.15` (control) against `1.5` (treatment) for MAPPO and IPPO:
four serial runs, training seed 7, 150K transitions each, six PPO epochs, otherwise identical
research-profile fields. Retain 128 expert episodes, 40 behavior-cloning epochs, 2,048 rollout
decisions, minibatches of 512, hidden size 128, and checkpoints at 10/25/50/75/100%.
The failure curriculum retains the first 150K transitions of its 1.5M-transition reference schedule.

All four trainings finish before deterministic validation on seeds 800–804, with and without
failures, against the same CP-SAT validation baseline. Evaluate all five checkpoints for learning
diagnostics, but compare intervention eligibility using **only each final 150K checkpoint**:
both algorithms' nominal completion ≥95%, MAPPO failure completion ≥85%, and MAPPO nominal
median makespan / CP-SAT ≤1.15. Pick one shared eligible coefficient with the existing completion,
failure, and makespan lexicographic order; break exact ties toward the smaller coefficient.
An eligible pilot does not itself authorize automatic full-matrix dispatch or establish release acceptance.

The opt-in launcher is:

```powershell
python scripts/run_construction_validation_pilot.py plan --comparison reward-time --device cuda
python scripts/run_construction_validation_pilot.py execute --comparison reward-time --device cuda `
  --registry logs/construction_intelligence/reward-time-pilot.sqlite `
  --evidence-root logs/construction_intelligence/reward_time_pilot
```

Run only from a clean, immutable checkout. The generated manifest binds source/configuration/design
digests, run membership, intervention, curriculum, and selection rules. Nondefault reward settings
enter the resumable configuration digest; incompatible resumes must fail. The historical default
retains its original digest for backwards-compatible resume.

No action masks, WAIT behavior, physics, controller evaluation, held-out selection, acceptance
thresholds, or ablation definitions change. If the candidate passes, separately version the full
protocol, pass unit/smoke/resume gates, and rerun the complete affected five-seed matrices and
ablations. If it fails, retain and report the failure before choosing another training/validation-only
intervention. Never tune against held-out results.
