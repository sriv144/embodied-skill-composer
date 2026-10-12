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
| sequential | 551 s | 200.0 m | 74.1 Wh | 1433 s |
| greedy | 203 s | 200.0 m | 74.8 Wh | 30 s |
| optimized | 197 s | 200.0 m | 74.7 Wh | 8 s |

The optimized schedule reduces fixture makespan by **64.2%** relative to deterministic sequential construction.

## Optimized Critical Path

`foundation_0_0` -> `north_00` -> `roof_0_0`

## AI Brain Decisions

- `000s` assign `foundation_0_1` to robot_2, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `000s` assign `foundation_1_0` to robot_1, robot_3: Selected because prerequisites and robot capacity are satisfied.
- `027s` assign `foundation_0_0` to robot_1, robot_3: Selected from the precedence-ready set to protect the critical path.
- `028s` assign `interior_panel_1` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `028s` assign `south_03` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `047s` assign `foundation_1_1` to robot_2, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `053s` assign `north_00` to robot_1: Selected from the precedence-ready set to protect the critical path.
- `053s` assign `north_01` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `076s` assign `south_00` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `076s` assign `south_02` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `076s` assign `west_01` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `076s` assign `west_02` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `094s` assign `east_01` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `094s` assign `south_01` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `096s` assign `west_00` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `098s` assign `east_02` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `113s` assign `east_00` to robot_1: Selected because prerequisites and robot capacity are satisfied.
- `114s` assign `north_03` to robot_3: Selected because prerequisites and robot capacity are satisfied.
- `118s` assign `north_02` to robot_4: Selected because prerequisites and robot capacity are satisfied.
- `122s` assign `interior_panel_0` to robot_2: Selected because prerequisites and robot capacity are satisfied.
- `140s` assign `roof_0_1` to robot_3, robot_4: Selected because prerequisites and robot capacity are satisfied.
- `140s` assign `roof_1_1` to robot_1, robot_2: Selected because prerequisites and robot capacity are satisfied.
- `170s` assign `roof_0_0` to robot_1, robot_2: Selected from the precedence-ready set to protect the critical path.
- `170s` assign `roof_1_0` to robot_3, robot_4: Selected because prerequisites and robot capacity are satisfied.

## Interpretation

The CP-SAT controller is an explainable scheduling oracle, not a learned policy. It demonstrates where multi-robot coordination creates measurable value before MARL is introduced. Browser and simulator playback consume the same execution trace.

## Limitations

- Geometry is modular and architectural, not structurally certified.
- Transport uses metric task-level motion rather than wheel/gripper dynamics.
- Floor-plan interpretation requires human approval before compilation.
- CoppeliaSim and OpenAI services are optional validation and assistance layers.

## Public Demo Provenance

This is a reviewed deterministic fixture preview. It contains no canonical MAPPO/IPPO research result and no live Coppelia evidence. The generated `release-status.json` and `provenance.json` make those boundaries machine-readable.
