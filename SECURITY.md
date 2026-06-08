# Security Policy

## Supported Versions

Only the latest commit on `main` is actively maintained. Older branches and
legacy demo modes (tabletop baselines, single-agent warehouse) are kept for
regression coverage but are not security-patched.

## Reporting a Vulnerability

If you discover a security issue, please do not open a public GitHub issue.
Contact the maintainer directly so the fix can be coordinated before
disclosure.

When reporting, please include:

- A description of the issue and its impact
- A minimal reproduction (config profile, command, or training script)
- Any logs, stack traces, or affected file paths
- Your suggested fix or mitigation, if you have one

## Simulator and Hardware Safety

This repository targets simulated robots first (PyBullet, MuJoCo, and the
planned Isaac Lab backend). Policies trained here are **not** validated for
real-hardware execution. If you wire a learned policy through ROS or any
physical actuator bridge:

- Always run with hardware E-stop and software action clipping in the loop.
- Re-validate the policy on the target backend before any physical trial.
- Treat MARL coordination policies as research artifacts — they are tuned
  for the local sandbox, not for safety-critical multi-agent execution.

## Secrets Hygiene

This project does not require API keys today. If you extend it with a learned
perception model that calls an LLM (for example for high-level planning over
options), keep credentials in `.env`, never in committed configs.

## Dependencies

Dependencies are split across `requirements.txt`, `requirements-rl.txt`,
`requirements-sim-mujoco.txt`, `requirements-pybullet.txt`, and the CUDA
wheel files. Keep simulator wheels out of the default install so contributors
on non-GPU machines are not forced into CUDA-only torch builds.
