from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@dataclass
class FSMIConfig:
    info_threshold: float = 20.0
    goal_pos: jax.Array = field(
        default_factory=lambda: jnp.array([9.0, 5.0, -2.0])
    )


class FSMITrajectoryGenerator:
    """Layer 2: FSMI-driven trajectory generator."""

    def __init__(self, config: FSMIConfig, info_zones: jax.Array):
        self.config = config
        self.info_zones = info_zones

    def get_target(self, info_levels: jax.Array) -> tuple[jax.Array, str]:
        """Determine target based on info levels (simple state machine)."""
        target_pos = self.config.goal_pos
        target_mode = "GOAL"

        # Simple logic: Visit Info 1, then Info 2, then Goal
        # In a real FSMI, this would optimize viewpoints based on the map.
        if info_levels[0] > self.config.info_threshold:
            target_pos = jnp.array([
                self.info_zones[0, 0],
                self.info_zones[0, 1],
                -2.0,
            ])
            target_mode = "INFO 1"
        elif info_levels[1] > self.config.info_threshold:
            target_pos = jnp.array([
                self.info_zones[1, 0],
                self.info_zones[1, 1],
                -2.0,
            ])
            target_mode = "INFO 2"

        return target_pos, target_mode
