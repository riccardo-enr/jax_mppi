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
        # Logic must be JAX-traceable (no python control flow on data).
        # We use jax.lax.cond or jnp.where.

        # Defaults
        goal_pos = self.config.goal_pos
        info1_pos = jnp.array([
            self.info_zones[0, 0],
            self.info_zones[0, 1],
            -2.0,
        ])
        info2_pos = jnp.array([
            self.info_zones[1, 0],
            self.info_zones[1, 1],
            -2.0,
        ])

        # Logic:
        # if info[0] > thresh: target = info1
        # elif info[1] > thresh: target = info2
        # else: target = goal

        # Note: We can't return strings (target_mode) in JIT.
        # We'll return an integer: 0=GOAL, 1=INFO1, 2=INFO2

        pred1 = info_levels[0] > self.config.info_threshold
        pred2 = info_levels[1] > self.config.info_threshold

        target_pos = jax.lax.select(
            pred1,
            info1_pos,
            jax.lax.select(pred2, info2_pos, goal_pos),
        )

        # For visualization purposes only (outside JIT), we might map this back.
        # Inside JIT, we return the mode as an integer.
        target_mode = jax.lax.select(
            pred1,
            1,
            jax.lax.select(pred2, 2, 0),
        )

        return target_pos, target_mode
