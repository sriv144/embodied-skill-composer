## Summary

<!-- One or two sentences on what changed and why. -->

## Scope

- [ ] Tabletop / warehouse task contract
- [ ] Collaborative assembly task contract
- [ ] Hierarchical team options (scripted or learned)
- [ ] Low-level MARL baseline
- [ ] MuJoCo backend / runtime profile
- [ ] Perception / planner / executor
- [ ] Docs / CI / scaffolding only

## Test plan

- [ ] `pytest -q --basetemp .pytest_tmp` is green locally
- [ ] `eval_assembly_options.py --policy scripted` still hits the documented success-rate baseline
- [ ] `eval_assembly_options.py --policy learned` still hits the documented success-rate baseline (if policy / env touched)
- [ ] MuJoCo 3D smoke (`run_mujoco_assembly.py --policy scripted`) still completes (if MuJoCo backend touched)

## Result drift check (skip if N/A)

The README headline result is **scripted=1.000, learned-hierarchical=1.000, low-level-MARL=0.000** on `2/2 beams`. If your change moves these numbers, update the README result table in the same PR.
