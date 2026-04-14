# Isaac Prep Notes

## Goal

The next milestone is **not** to train a full Isaac policy immediately. The goal is to port the current assembly task contract into an Isaac-compatible backend while preserving the successful hierarchical options control loop.

## Current Assumptions

- Current development machine: Windows local prototyping environment
- Future Isaac target: Linux workstation or cloud machine
- GPU expectation: NVIDIA GPU with CUDA support
- Simulator target: Isaac Lab / Isaac Sim stack

## What Must Stay Stable

The Isaac port should preserve the current task-facing interface:

- `reset(seed)`
- `get_team_option_observation()`
- `get_team_option_mask()`
- `execute_team_option(option)`
- `build_artifact()`
- `get_option_episode_diagnostics()`

This keeps the local sandbox as the regression environment while Isaac becomes a higher-fidelity backend.

## First Isaac Backend Milestone

The first backend slice should focus on:

- matching two-robot assembly task semantics,
- matching episode success/failure metrics,
- matching option-level diagnostics,
- matching training/eval entrypoints as closely as practical.

Do **not** make visual perception or end-to-end visual MARL part of the first Isaac milestone.

## Recommended Bring-Up Order

1. Implement an `isaac_lab` backend stub behind the current backend factory.
2. Recreate the two-beam assembly task in Isaac with privileged state only.
3. Match artifact structure and benchmark output against the local sandbox.
4. Verify scripted options first.
5. Reuse the hierarchical options learner after backend parity is demonstrated.
