from dataclasses import dataclass
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class GridMap:
    grid: chex.Array  # (H, W) float32 in [0, 1]
    origin: chex.Array  # (2,) [x, y] of bottom-left corner
    resolution: float  # meters per cell
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def tree_flatten(self):
        # We must cast resolution to array if we want it to be differentiable,
        # but typically it's static parameter for map.
        # However, for JIT compatibility of passing the *whole* object,
        # scalar float usually goes into aux_data unless it's a tracer.
        # But resolution is float. Let's put it in aux_data?
        # Or better: treat it as child if it might be traced?
        # Usually resolution is constant.
        # Let's put width, height in aux. origin and grid in children.
        # resolution can be aux.
        return (self.grid, self.origin), (
            self.width,
            self.height,
            self.resolution,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        grid, origin = children
        width, height, resolution = aux
        return cls(
            grid=grid,
            origin=origin,
            resolution=resolution,
            width=width,
            height=height,
        )


def create_grid_map(
    origin: jax.Array, width: int, height: int, resolution: float
) -> GridMap:
    grid = jnp.zeros((height, width))
    return GridMap(
        grid=grid,
        origin=origin,
        resolution=resolution,
        width=width,
        height=height,
    )


def world_to_grid(
    pos: jax.Array, origin: jax.Array, resolution: float
) -> jax.Array:
    """Convert world coordinates to grid indices (float)."""
    return (pos - origin) / resolution


def grid_to_world(
    indices: jax.Array, origin: jax.Array, resolution: float
) -> jax.Array:
    """Convert grid indices to world coordinates (center of cell)."""
    return origin + (indices + 0.5) * resolution


@partial(jax.jit, static_argnames=["resolution", "width", "height"])
def rasterize_environment(
    walls: jax.Array,
    info_zones: jax.Array,
    origin: jax.Array,
    width: int,
    height: int,
    resolution: float,
) -> GridMap:
    """
    Rasterize walls and info zones into a GridMap.
    Walls -> 1.0 (Occupied)
    Info Zones -> 0.5 (Unknown/High Entropy)
    Free Space -> 0.0 (Free)
    """
    # Create coordinate grid
    y_range = jnp.arange(height)
    x_range = jnp.arange(width)
    Y, X = jnp.meshgrid(y_range, x_range, indexing="ij")

    # World coordinates of each cell center
    world_X = origin[0] + (X + 0.5) * resolution
    world_Y = origin[1] + (Y + 0.5) * resolution

    # Rasterize Walls
    # Walls are line segments [x1, y1, x2, y2]
    # We treat them as thick lines or just check distance to segment
    def dist_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = jnp.dot(ap, ab) / jnp.dot(ab, ab)
        t = jnp.clip(t, 0.0, 1.0)
        closest = a + t * ab
        return jnp.linalg.norm(p - closest)

    def is_wall(x, y):
        p = jnp.array([x, y])

        def check_wall(w):
            start = w[:2]
            end = w[2:]
            d = dist_segment(p, start, end)
            # Wall thickness/radius
            return d < 0.2  # 20cm radius

        # Vectorize over walls
        in_walls = jax.vmap(check_wall)(walls)
        return jnp.any(in_walls)

    # Rasterize Info Zones
    # Info zones are rectangles [cx, cy, w, h, val]
    def is_info_zone(x, y):
        p = jnp.array([x, y])

        def check_zone(z):
            center = z[:2]
            size = z[2:4]
            half_size = size / 2.0
            d = jnp.abs(p - center) - half_size
            return jnp.all(d <= 0.0)

        in_zones = jax.vmap(check_zone)(info_zones)
        return jnp.any(in_zones)

    # Vectorize over grid
    # We use vmap over flattened grid for simplicity then reshape

    # Flatten coords
    coords = jnp.stack([world_X.flatten(), world_Y.flatten()], axis=1)

    is_wall_mask = jax.vmap(lambda p: is_wall(p[0], p[1]))(coords)
    is_zone_mask = jax.vmap(lambda p: is_info_zone(p[0], p[1]))(coords)

    # Priority: Wall > Zone > Free
    # Wall = 1.0
    # Zone = 0.5
    # Free = 0.0

    vals = jnp.zeros_like(is_wall_mask, dtype=jnp.float32)
    vals = jnp.where(is_zone_mask, 0.5, vals)
    vals = jnp.where(is_wall_mask, 1.0, vals)

    grid = vals.reshape(height, width)

    return GridMap(
        grid=grid,
        origin=origin,
        resolution=resolution,
        width=width,
        height=height,
    )
