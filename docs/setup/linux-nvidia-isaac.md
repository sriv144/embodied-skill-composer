# Linux + NVIDIA + Isaac Bring-Up

## Why Linux Is The Real Isaac Target

Keep Windows for current local development, benchmark regression, and VS Code workflows.

Move to Linux before serious Isaac backend implementation or training. The current laptop is:

- `Windows 11 Home`
- `NVIDIA GeForce RTX 4060 Laptop GPU`
- about `8 GB` VRAM
- about `15.7 GB` system RAM

That is enough for local torch checks and smaller runs, but it is likely tight for heavier Isaac workflows.

## Recommended Environment

Preferred order:

1. Separate Linux machine or cloud GPU box
2. Dual-boot Linux
3. WSL only as a temporary exploration path

Recommended Linux target:

- Ubuntu LTS
- recent NVIDIA driver
- CUDA-compatible stack
- isolated conda or micromamba environment dedicated to Isaac

Cloud GPU Ubuntu VMs are usually **not free** beyond trial credits. Budget for hourly GPU runtime plus storage. For early Isaac bring-up, prefer an RTX 4090 / RTX 6000 Ada / L40S class instance with at least `24 GB` VRAM and `32-64 GB` RAM over a tiny free notebook runtime.

## Bring-Up Goal

The first Isaac milestone is **backend parity**, not scale.

Match the current local assembly task contract:

- `reset(seed)`
- `get_team_option_observation()`
- `get_team_option_mask()`
- `execute_team_option(option)`
- `build_artifact()`
- `get_option_episode_diagnostics()`

The first success condition is:

- same two-beam assembly task
- scripted options work first
- artifact and metric fields match the local sandbox
- hierarchical option loop stays backend-agnostic

## Recommended Order

1. Validate Python, torch, CUDA, and NVIDIA visibility on Linux.
2. Install Isaac Sim / Isaac Lab in an isolated environment.
3. Bring up an `isaac_lab` backend that instantiates the same assembly task semantics.
4. Match scripted option execution and diagnostics.
5. Match benchmark output shape.
6. Reuse the hierarchical learner only after parity is stable.

## Out Of Scope For First Isaac Pass

- camera perception for assembly
- end-to-end visual MARL
- larger clutter/randomization expansions
- sim-to-real work
