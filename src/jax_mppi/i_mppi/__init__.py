"""I-MPPI building blocks.

This module implements the two-layer I-MPPI architecture:

Layer 2 (FSMI Analyzer, ~5-10 Hz):
    - FSMIModule: Full FSMI with O(n²) computation
    - FSMITrajectoryGenerator: Generates reference trajectories maximizing information

Layer 3 (I-MPPI Controller, ~50 Hz):
    - UniformFSMI: O(n) fast FSMI for local reactivity
    - biased_*_command: MPPI variants with mixture sampling

Cost function structure:
    J = Dynamics + Tracking(Layer2_Ref) - λ * UniformFSMI(Local)
"""

from .environment import (
    GOAL_POS,
    INFO_ZONES,
    WALLS,
    augmented_dynamics,
    informative_running_cost,
    running_cost,
)
from .fsmi import (
    FSMIConfig,
    FSMIModule,
    FSMITrajectoryGenerator,
    UniformFSMI,
    UniformFSMIConfig,
)
from .planner import (
    biased_kmppi_command,
    biased_mppi_command,
    biased_smppi_command,
)

__all__ = [
    # Layer 2: Full FSMI for reference trajectory
    "FSMIConfig",
    "FSMIModule",
    "FSMITrajectoryGenerator",
    # Layer 3: Fast Uniform-FSMI for local reactivity
    "UniformFSMI",
    "UniformFSMIConfig",
    # Environment
    "GOAL_POS",
    "INFO_ZONES",
    "WALLS",
    "augmented_dynamics",
    "informative_running_cost",
    "running_cost",
    # Biased MPPI commands
    "biased_mppi_command",
    "biased_smppi_command",
    "biased_kmppi_command",
]
