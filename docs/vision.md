# Vision: Physical AI Construction Swarm Simulator

## Product Definition

The concrete product is:

> An AI system that converts architectural intent into a buildable modular representation, then plans, coordinates, and evaluates a multi-robot construction process.

Its first honest scope is a reviewed, single-story orthogonal floor plan rather than arbitrary image-to-engineering reconstruction. A façade image may guide appearance, but metric walls, openings, modules, dependencies, payload constraints, and target poses remain explicit and reviewable. The AI contribution is visible in interpretation confidence, dependency reasoning, robot assignment, makespan optimization, disturbance recovery, and later learned coordination.

The final research experience connects five activities: review the design, inspect its modules, understand the plan, watch embodied execution, and compare measurable outcomes. This keeps the project grounded in architecture, 3D geometry, robotics, and multi-agent intelligence instead of becoming only a house renderer or only a robot animation.

## North Star

Embodied Skill Composer is growing into a **hybrid Physical AI construction-swarm simulator**. In plain terms, it is an **AI construction swarm simulator**: a research workbench where multiple autonomous robots coordinate to collect resources, move materials, and assemble structures under simulated physical constraints.

The motivating long-term picture is remote and planetary construction. Imagine robot teams working in lunar, Martian, disaster-zone, offshore, or hazardous industrial environments where humans cannot work safely or continuously. The robots receive a target structure, inspect available resources, divide the work, act in a simulated world, recover from failures, and improve through repeated experiments.

The realistic solo-project version is smaller and stronger:

> Build a reproducible simulator where simple robot teams plan, coordinate, and learn how to assemble modular structures from limited resources.

## Why This Is Physical AI

This project is related to Physical AI because the agent is not only producing text or classifying data. It must make decisions that become actions in a world with geometry, constraints, collisions, time cost, resource limits, and eventually physics.

The core loop is:

```text
resources + blueprint + robot state
        -> AI brain chooses coordinated actions
        -> simulator applies movement, contact, costs, and failures
        -> metrics score construction progress
        -> policy/planner improves
```

The current local sandbox is abstract, but it already contains the right shape: robot positions, beams, pickup targets, assembly targets, actions, options, rewards, collisions, recovery, benchmarks, and diagnostics. The next job is to make those construction concepts explicit before increasing simulator realism.

## The AI Brain

The AI brain should be layered instead of one giant model:

- **Mission planner**: interprets the requested structure or experiment goal.
- **Task allocator**: assigns resources and blueprint slots to robot teams.
- **Coordinator**: prevents robots from blocking each other and handles recovery.
- **Option policy**: chooses high-level skills such as pickup, grab, transport, install, wait, or recover.
- **Low-level controller**: executes movement, grasping, placement, and later robot-specific control.

OpenAI-style models are useful for mission planning, lab assistance, explanations, and experiment design. Reinforcement learning is useful for option policies and coordination. Robotics simulators such as MuJoCo, CoppeliaSim, and Isaac Lab are useful when the physical embodiment matters.

## Research Insight

The project is not only about moving blocks. The core research question is:

> How much structure should we give robot teams so learning becomes reliable instead of brittle?

The current result already points in a useful direction. A low-level learned MARL baseline stalls after the first beam, while a hierarchical team-options learner solves the two-beam task. That suggests the right path is not "RL for everything." The right path is a hybrid system:

- explicit structure for long-horizon construction,
- deterministic or scripted skills where reliability matters,
- learned policies where adaptation and coordination matter,
- simulator-agnostic metrics so local experiments can move into richer backends later.

## Realistic Solo Scope

Start with:

- two or three simple robots,
- beams, blocks, or panels as resources,
- target blueprint slots,
- collision, time, idle, and energy costs,
- scripted oracle and learned-policy comparisons,
- reproducible benchmarks and reports.

Do not start with:

- humanoids,
- full planetary terrain,
- raw-material processing,
- end-to-end visual MARL,
- realistic construction tools,
- sim-to-real deployment.

Those are later research tracks after the core coordination and construction semantics are stable.

## Simulator Strategy

Use simulators in layers, not all at once:

1. **Local Python grid**
   - Research truth source for task semantics, resources, blueprints, metrics, and tests.
   - Fast enough for CI, regression, and experiment sweeps.

2. **MuJoCo**
   - Clean physics/control path for local 3D episodes.
   - Useful for making movement and embodiment more realistic while keeping experiments manageable.

3. **CoppeliaSim**
   - Worth evaluating as a robotics prototyping backend for multi-robot scenes, remote APIs, and educational robotics workflows.
   - Future spike only; not the immediate source of truth.

4. **Blender**
   - Optional assets, mockups, renders, and presentation visuals.
   - Not a runtime simulator dependency.

5. **NVIDIA Isaac Lab**
   - Long-term Physical AI backend for higher-fidelity robotics work.
   - Use only after local task semantics and benchmark contracts are stable.

6. **ROS 2**
   - Deferred until backend/control architecture matters.
   - Treat as a future bridge toward real-robot style systems, not an immediate requirement.

Unity is deferred. It may help with demos later, but it should not distract from the research core now.

## Working Principle

Move slowly enough that every milestone leaves behind:

- a runnable command,
- a saved artifact,
- a benchmark number,
- a short explanation,
- and a test that protects the result.

That is how this project can grow from a college research prototype into a serious Physical AI simulation workbench without losing its center.
