# Research Landscape And Design Borrowing

Embodied Skill Composer remains its own repository. The projects below are references for interfaces,
evaluation, and research questions; their code is not copied wholesale into this system.

## Fabrica

[Fabrica](https://github.com/yunshengtian/Fabrica) treats general multi-part assembly as an integrated
planning-and-learning problem for two coordinated arms. The useful lesson here is architectural: keep
geometry-aware planning, execution, and learned policies as separable layers with inspectable handoffs.

Adopted now:

- typed part/material identity and metric target poses,
- an explicit assembly dependency graph,
- deterministic planning as an oracle above future learned control.

Deferred:

- contact-rich dual-arm manipulation,
- learned grasp and insertion policies,
- Fabrica's embodiment-specific planning stack.

## WorkBenchMark

[WorkBenchMark](https://workbenchmark.github.io/) represents target assemblies as graphs and evaluates
both task reasoning and physical execution. Its separation of pick and assembly areas closely matches
this project's material-yard/work-zone setup.

Adopted now:

- structured YAML task definitions,
- graph dependencies and exact relative/metric placement intent,
- reproducible metrics and tier-friendly scenario design,
- separate material staging and construction zones.

Deferred:

- RGB-D-only autonomous execution,
- tight-tolerance insertion and stability scoring,
- real-robot benchmark parity.

## Learn2Assemble

[Learn2Assemble](https://niklasfunk.com/publication/learn2assemble/) studies learned assembly sequencing
with structured representations. It supports the central research hypothesis here: learning should
choose useful high-level construction actions over a structured state, rather than rediscovering every
primitive motion from scratch.

Adopted now:

- a stable structured observation/action contract,
- scripted and precedence oracles for future policy comparison,
- component-level sequencing as the next hierarchical RL target.

Deferred:

- new RL training on the ten-component room,
- generalization to unseen blueprints,
- graph neural policy representations.

## TERMES

[TERMES](https://people.seas.harvard.edu/~jkwerfel/) demonstrates construction without centralized
roles by using local rules, environmental state, and precedence constraints. It is the right contrast
to the current centralized precedence brain.

Adopted now:

- dependencies as explicit construction constraints,
- progress encoded in the evolving structure,
- a planned decentralized baseline rather than treating central coordination as the only answer.

Deferred:

- local-only robot observations and stigmergic coordination,
- dynamic robot populations,
- simultaneous independent installation jobs.

## Resulting Project Position

Modular Room v0 deliberately combines the least risky ideas first: WorkBenchMark-style structured tasks,
Fabrica-style layered planning/execution, Learn2Assemble-style future high-level learning, and TERMES as
the future decentralized comparator. The current Coppelia controller remains
`kinematic_cooperative_pose_sync`; it is a reproducible execution visualization, not yet a claim of
dynamic mobile manipulation.
