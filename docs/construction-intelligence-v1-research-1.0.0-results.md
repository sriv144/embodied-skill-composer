# Protocol 1.0.0 research results — complete, release gate failed

The original frozen research matrix completed on 2026-10-11. All 20 runs reached exactly
1,500,000 transitions (30M total), all 20 immutable policies were selected on validation only,
and the complete 240-episode held-out matrix was generated. Independently rehashing all 326
reproducibility files (883,151,472 bytes) found no missing or mismatched files.

This is a completed experiment, **not a passing v1 release**.

| Primary gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| MAPPO nominal mean module completion | 99.8519% | ≥95% | Pass |
| IPPO nominal mean module completion | 100% | ≥95% | Pass |
| MAPPO median nominal makespan / CP-SAT | 1.310722× | ≤1.15× | Fail |
| MAPPO failure mean module completion | 99.8667% | ≥85% | Pass |

Behavior cloning was unsupported by the preregistered interpretation: final completion gain
was 0.385185 percentage points, below five points; transitions-to-95% could not be established
on the original validation suite. The historical failure-training intervention was also
unsupported: failure completion changed by −0.133333 points, while nominal completion changed
by +3.555556 points. The original implementation used a constant failure mixture, not a
progressive curriculum. Unsupported hypotheses do not themselves block release; the failed
primary makespan gate does.

## Frozen identity and artifact ledger

- Source: `5166577d9f1d4f1211bfb35727e84b3588c03bcb`.
- Protocol: `1.0.0`, SHA-256 `79e5444bb108573a9ce1597493e4556cf4fd7ba59607f81e7fe2826b0e74c707`.
- Matrix: `construction_intelligence_v1-research-79e5444bb108`.
- Evaluation: `20261011T003001Z-construction_intelligence_v1-heldout`.
- Held-out grid: seeds 900–904 × both failure modes; 40 baseline and 200 learned-policy episodes.
- Historical output root: `logs/construction_intelligence/experiments/heldout/20261011T003001Z-construction_intelligence_v1-heldout` in the frozen execution workspace.

| File | SHA-256 |
| --- | --- |
| `acceptance.json` | `2f82855d0f67ba8a9c09f1383f69cb901fe367f76865991d87676a2a17f8b83e` |
| `ablations.json` | `779c2c122c5c55d125e486f2aee5048e048837aa924a88add711bae0c210c563` |
| `evaluation.json` | `86d883a122ce4ecdaa2fb2082917c4588ec5a21a211bd0b624fec4d9ef5b1555` |
| `episodes.csv` | `079493d5ead6498add24b20e5231f8192df8b15d2e6e86aa82cb11a3381715c1` |
| `report.md` | `52a26a6292a94130d69353eb70152673fce3de3da382d7ee98c15f2d1c61a3c0` |
| `reproducibility_audit.json` | `f7888bc148d3ab4be797850abec4b2f8a40cab1d9647c820fdd2de56e8321b7e` |
| `release_completeness.json` | `8cdbcf5a0ce584fb1fc178dfcf06a1b1f5ea68042dd7ccf400deac475f3b153f` |

The full run directories and model weights are not committed. Their local existence and hashes
are not a claim that release assets have already been published.

## What happens next

A separately recorded [validation-only audit](evidence/construction-intelligence-v1-protocol-1.0.0-validation-audit.json)
identified an endpoint-feasibility defect and training instability before these terminal results
were used for reporting. The [candidate protocol amendment](construction-intelligence-v1-protocol-amendment-1.1.0.md)
preserves the failed historical evidence, all seed partitions, and every acceptance threshold.
Corrective model selection and tuning use training and validation only. The complete affected
20-run matrix and baselines must be rerun on the new frozen source; old checkpoints cannot be
resumed into the corrected experiment.
