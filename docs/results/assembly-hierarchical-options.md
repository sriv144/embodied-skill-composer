# Assembly Hierarchical Options Results

## Task

The current flagship benchmark is a two-robot collaborative assembly task in a local grid sandbox. Two agents must:

1. reach the current beam pickup cells,
2. coordinate a grab,
3. transport the beam to the assembly cells,
4. install it,
5. recover and repeat for the second beam.

## Verified Comparison

Using the default `assembly_env.yaml` and `assembly_training.yaml` configuration:

- The retained **low-level learned MARL** baseline still fails the full task.
- The **hierarchical team-options learner** solves the full `2/2 beams` task consistently.

Most recent verified local numbers:

| Policy | Success Rate | Mean Return | Mean Beams Installed |
| --- | ---: | ---: | ---: |
| Scripted options | 1.000 | 10.52 | 2.00 |
| Learned hierarchical options | 1.000 | 10.52 | 2.00 |
| Low-level learned MARL | 0.000 | -0.02 | 1.00 |

## Why The Hierarchical Policy Works Better

The low-level learner was forced to discover long-horizon sequencing, transport coordination, and post-install recovery through primitive joint actions. It repeatedly solved the first beam and then deadlocked before the second pickup.

The hierarchical policy fixes that by learning over **team options** instead of raw actions:

- `go_pickup`
- `grab`
- `go_assembly`
- `install`
- `reset_to_pickup_route`
- `reposition_after_install`
- `wait`
- `align_for_terminal_action`

Travel and recovery are executed deterministically, while PPO only learns the higher-level task sequence. That shrinks the search space and makes the beam-1 to beam-2 transition explicit.

## Why This Matters For Isaac

This result is useful because it shows the learning interface we should preserve when moving to Isaac Lab:

- keep a shared task contract,
- keep option-level observations and masks,
- preserve artifact and metrics formats,
- port only the backend dynamics first.
