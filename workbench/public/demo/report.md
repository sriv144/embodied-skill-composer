# Construction Research Report: Cedar Ridge Modular Cottage

## Architectural Compilation

- Design: `cottage_v1`
- Build modules: `24`
- Robot fleet: `4`
- Footprint: `8.0 x 6.0 m`
- Roof: `gable` at `28 deg`

## Controller Comparison

| Controller | Makespan | Travel | Energy | Idle robot time |
| --- | ---: | ---: | ---: | ---: |
| sequential | 551 s | 200.2 m | 74.1 Wh | 1433 s |
| greedy | 204 s | 200.2 m | 74.8 Wh | 35 s |
| optimized | 196 s | 200.2 m | 74.6 Wh | 6 s |

The optimized schedule reduces fixture makespan by **64.4%** relative to deterministic sequential construction.

## Optimized Critical Path

`foundation_0_0` -> `north_00` -> `roof_0_0`

## AI Brain Decisions

- `000s` assign `foundation_0_1` to robot_2, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `001s` assign `foundation_1_0` to robot_1, robot_3: Selected because prerequisites and robot capacity are satisfied.
- `028s` assign `foundation_0_0` to robot_2, robot_3: Selected from the precedence-ready set to protect the critical path.
- `028s` assign `foundation_1_1` to robot_1, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `053s` assign `east_01` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `053s` assign `interior_panel_1` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `057s` assign `interior_panel_0` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `057s` assign `west_02` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `072s` assign `south_01` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `075s` assign `north_01` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `077s` assign `south_00` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `078s` assign `east_00` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `091s` assign `north_00` to robot_3: Selected from the precedence-ready set to protect the critical path.
- `098s` assign `east_02` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `098s` assign `south_03` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `104s` assign `west_00` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `113s` assign `north_03` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `116s` assign `north_02` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `121s` assign `west_01` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `122s` assign `south_02` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `139s` assign `roof_0_1` to robot_1, robot_2: Selected because prerequisites and robot capacity are satisfied.
- `139s` assign `roof_1_1` to robot_3, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `169s` assign `roof_1_0` to robot_3, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `170s` assign `roof_0_0` to robot_1, robot_2: Selected from the precedence-ready set to protect the critical path.

## Interpretation

The CP-SAT controller is an explainable scheduling oracle, not a learned policy. It demonstrates where multi-robot coordination creates measurable value before MARL is introduced. Browser and simulator playback consume the same execution trace.

## Limitations

- Geometry is modular and architectural, not structurally certified.
- Transport uses metric task-level motion rather than wheel/gripper dynamics.
- Floor-plan interpretation requires human approval before compilation.
- CoppeliaSim and OpenAI services are optional validation and assistance layers.

## Public Demo Provenance

Exported deterministically from the reviewed `cottage_v1` fixture. No trained MAPPO or IPPO result is included in this curated bundle.
