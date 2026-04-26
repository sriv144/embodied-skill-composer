# Security Policy

## Supported Versions

This is a research project. Only the `main` branch receives security fixes.

## Reporting a Vulnerability

If you discover a security issue (e.g. a malicious YAML task file that escapes
the sandbox runtime, an unsafe pickle path in a checkpoint loader, or a
command-injection in any of the `scripts/` entry points), please **do not**
open a public GitHub issue.

Instead, email the maintainer privately with:

- A clear description of the issue and the affected file/path.
- A minimal reproduction (config snippet, command, expected vs. actual).
- Your assessment of severity and any suggested mitigation.

You can expect an acknowledgement within 7 days. Coordinated disclosure is
preferred for any RCE-class issue in the simulator backends or RL training
entry points.

## Out of scope

- Issues in third-party simulators (PyBullet, MuJoCo, Isaac Lab).
- Issues in pinned ML dependencies — please report those upstream.
- Performance / resource-exhaustion issues that require an attacker to already
  control training configs on a trusted host.
