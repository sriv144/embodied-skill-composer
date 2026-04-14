# Linux + NVIDIA + Isaac Bring-Up

## Target Platform

The current `isaac_gpu` runtime profile is a contract-preserving stub. The real Isaac milestone assumes:

- Linux host
- NVIDIA GPU
- CUDA-capable driver stack
- Isaac Lab / Isaac Sim compatible environment

## Bring-Up Goal

The first objective is **not** large-scale training. It is to make the Isaac backend match the current local assembly task contract:

- `reset(seed)`
- `get_team_option_observation()`
- `get_team_option_mask()`
- `execute_team_option(option)`
- `build_artifact()`
- `get_option_episode_diagnostics()`

## Recommended Order

1. Validate Python, CUDA, and Isaac installation on Linux.
2. Bring up an `isaac_lab` backend that can instantiate the two-robot assembly task.
3. Match scripted option execution first.
4. Match artifact and benchmark output shape.
5. Reuse the hierarchical options learner only after backend parity is stable.

## Out Of Scope For First Isaac Pass

- camera perception for assembly
- end-to-end visual MARL
- large clutter/randomization expansion
- sim-to-real work
