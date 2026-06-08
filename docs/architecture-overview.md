# Architecture Overview — Embodied Skill Composer

This page complements the README with a visual map of how perception,
planning, option execution, and learning fit together for the two-robot
collaborative-assembly flagship.

## Hierarchical control stack

```mermaid
flowchart TB
    subgraph Perception
        P1["Sensors / oracle state"] --> P2["Classical CV builder"]
        P2 --> P3["WorldState"]
    end

    subgraph High_Level["High-level policy"]
        H1["Team options head\nPPO over option indices"] --> H2["Selected team option\n(e.g. fetch_beam_1, install_beam_2)"]
    end

    subgraph Option_Routing["Deterministic option router"]
        O1["Per-agent sub-policies\nscripted + recoverable"] --> O2["Low-level actions"]
    end

    subgraph Sim_Backend["Backend (pluggable)"]
        S1["Local 2D sandbox"]
        S2["MuJoCo 3D"]
        S3["Isaac Lab (planned)"]
    end

    P3 --> H1
    H2 --> O1
    O2 --> S1
    O2 --> S2
    O2 --> S3
    S1 --> P1
    S2 --> P1
    S3 --> P1
```

## Policy ladder for benchmarking

```mermaid
flowchart LR
    A["Scripted options\n(oracle coordination)"] --> R["Common task contract"]
    B["Learned hierarchical options\n(PPO + imitation warm-start)"] --> R
    C["Low-level learned MARL\n(retained baseline)"] --> R
    R --> M["Benchmark metrics\nsuccess rate, return, beams installed"]
```

## Why hierarchical options

The retained low-level MARL baseline stalls at 1/2 beams in the default task,
while the hierarchical options policy clears 2/2 beams reliably. Promoting
the team-option abstraction out of scripted code and into the learned head
is what closes that gap. The pluggable backend layer keeps the same task
contract whether the rollout runs in the local 2D sandbox, MuJoCo, or a
future Isaac Lab build.

## Where to look in the code

| Path | Responsibility |
| --- | --- |
| `src/embodied_skill_composer/assembly/` | Task contract, backend selection, scripted oracle, learners |
| `src/embodied_skill_composer/perception/` | Oracle and classical-CV world-state builders |
| `src/embodied_skill_composer/sim/` | Tabletop and warehouse adapters |
| `src/embodied_skill_composer/rl/` | Pickup-policy training scaffolding |
| `scripts/train_assembly_options.py` | Hierarchical options training entrypoint |
| `scripts/benchmark_assembly_policies.py` | Cross-policy benchmark runner |
| `configs/assembly_profiles/` | Runtime profiles (CPU, local GPU, MuJoCo, Isaac stub) |
