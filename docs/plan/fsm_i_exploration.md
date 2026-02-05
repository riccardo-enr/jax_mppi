# FSMI Exploration Plan

## Goal

Implement the real FSMI (Fast Shannon Mutual Information) exploration logic to bias exploration toward unexplored areas, running at a lower rate than the MPPI control loop.

Look at the docs/plan/i_mppi.qmd for the theory behind FSMI.

The FSMI module should run at 5 Hz while the MPPI control loop runs at 50 Hz.

## Scope

- Add FSMI state computation and scoring for exploration targets.
- Integrate FSMI output as a bias/goal for MPPI, without changing MPPI core control rate.
- Basically the FSMI returns a score for each area to explore correlated with the centroid position of the area to explore.
- Ensure FSMI runs at a lower frequency (e.g., every N MPPI ticks or at a fixed Hz).

## Assumptions / Open Questions

- Clarify which map representation is used for "explored vs unexplored" (occupancy grid, TSDF, voxel grid, etc.).
- Define the FSMI target abstraction (frontiers, waypoints, information gain hotspots).
- Define how FSMI influences MPPI (goal override, cost shaping, trajectory prior).

## Plan

- [ ] Identify and document the current map/coverage representation and available signals for "explored."
- [ ] Define the FSMI state structure and scoring function for "information gain" or "unexploredness."
- [ ] Implement FSMI target selection (frontiers or hotspots) and a target cache.
- [ ] Add a scheduler to run FSMI at lower frequency than MPPI (e.g., every N control steps).
- [ ] Integrate FSMI output into MPPI (cost shaping or dynamic goal update).
- [ ] Add configuration knobs (FSMI rate, scoring weights, frontier thresholds).
- [ ] Add minimal tests or logging to verify FSMI target updates and MPPI biasing.

## Completion Criteria

- FSMI updates at the configured lower rate and selects targets in unexplored regions.
- MPPI behavior demonstrably biases toward FSMI targets without destabilizing control.
- Configuration documented and example run validates exploration bias.
