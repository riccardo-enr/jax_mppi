"""Environment setup for the I-MPPI interactive simulation.

Creates an office-like occupancy grid with rooms, corridors, and
information zones for the quadrotor exploration scenario.
"""

import jax.numpy as jnp

from jax_mppi.i_mppi.map import GridMap


def create_occupancy_grid():
    """Create an office-like occupancy grid with rooms and corridors.

    Returns:
        Tuple of (grid, map_origin, resolution, width, height).
    """
    world_width = 14.0
    world_height = 12.0
    resolution = 0.1

    width = int(world_width / resolution)
    height = int(world_height / resolution)

    grid = 0.5 * jnp.ones((height, width))

    # Known free space (central corridor)
    grid = grid.at[35:85, 5:135].set(0.2)

    # Outer walls
    grid = grid.at[0:5, :].set(0.9)
    grid = grid.at[115:120, :].set(0.9)
    grid = grid.at[:, 0:5].set(0.9)
    grid = grid.at[:, 135:140].set(0.9)

    # Room 1: Bottom-left office
    grid = grid.at[85:115, 5:45].set(0.9)
    grid = grid.at[35:115, 5:10].set(0.9)
    grid = grid.at[35:85, 40:45].set(0.9)
    grid = grid.at[35:45, 40:45].set(0.2)  # doorway
    grid = grid.at[40:80, 10:40].set(0.5)  # unknown interior
    grid = grid.at[40:50, 30:40].set(0.35)

    # Room 2: Top-left office
    grid = grid.at[5:35, 5:45].set(0.9)
    grid = grid.at[5:35, 40:45].set(0.9)
    grid = grid.at[28:36, 40:45].set(0.2)  # doorway
    grid = grid.at[10:30, 10:40].set(0.5)

    # Room 3: Bottom-right office
    grid = grid.at[85:115, 95:135].set(0.9)
    grid = grid.at[85:115, 130:135].set(0.9)
    grid = grid.at[35:85, 95:100].set(0.9)
    grid = grid.at[40:50, 95:100].set(0.2)  # doorway
    grid = grid.at[40:80, 100:130].set(0.5)
    grid = grid.at[50:60, 105:115].set(0.8)
    grid = grid.at[65:75, 120:125].set(0.8)

    # Room 4: Top-right office
    grid = grid.at[5:35, 95:135].set(0.9)
    grid = grid.at[5:35, 95:100].set(0.9)
    grid = grid.at[28:36, 95:100].set(0.2)  # doorway
    grid = grid.at[10:30, 100:130].set(0.5)
    grid = grid.at[25:32, 100:110].set(0.35)

    # Central obstacles
    grid = grid.at[45:55, 50:60].set(0.85)
    grid = grid.at[65:75, 70:80].set(0.85)
    grid = grid.at[40:45, 85:90].set(0.8)
    grid = grid.at[75:80, 20:25].set(0.8)

    # Narrow passages
    grid = grid.at[35:85, 45:52].set(0.2)
    grid = grid.at[55:65, 60:70].set(0.2)

    # Info zones (high uncertainty)
    grid = grid.at[50:75, 12:35].set(0.5)
    grid = grid.at[55:70, 15:30].set(0.55)
    grid = grid.at[12:28, 102:128].set(0.5)
    grid = grid.at[15:25, 105:125].set(0.55)

    # Additional complexity
    grid = grid.at[70:82, 48:52].set(0.9)
    grid = grid.at[72:80, 48:52].set(0.2)
    grid = grid.at[72:80, 45:48].set(0.52)
    grid = grid.at[55:70, 90:95].set(0.9)
    grid = grid.at[35:36, 40:45].set(0.75)
    grid = grid.at[84:85, 95:100].set(0.75)

    map_origin = jnp.array([0.0, 0.0])
    return grid, map_origin, resolution, width, height


def create_grid_map():
    """Create the full grid map object for the simulation.

    Returns:
        Tuple of (grid_map_obj, grid_array, map_origin, resolution).
    """
    grid_array, map_origin, resolution, width, height = create_occupancy_grid()
    grid_map_obj = GridMap(
        grid=grid_array,
        origin=map_origin,
        resolution=resolution,
        width=width,
        height=height,
    )
    return grid_map_obj, grid_array, map_origin, resolution
