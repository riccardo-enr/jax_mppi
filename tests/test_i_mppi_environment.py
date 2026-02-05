import jax.numpy as jnp
from jax_mppi.i_mppi.environment import dist_rect

def test_dist_rect():
    center = jnp.array([0.0, 0.0])
    size = jnp.array([2.0, 2.0]) # Half size 1.0, 1.0 (width, height)

    # Inside
    assert dist_rect(jnp.array([0.0, 0.0]), center, size) == 0.0
    assert dist_rect(jnp.array([0.5, 0.5]), center, size) == 0.0
    assert dist_rect(jnp.array([0.9, 0.9]), center, size) == 0.0

    # Boundary (approx)
    assert jnp.isclose(dist_rect(jnp.array([1.0, 0.0]), center, size), 0.0, atol=1e-6)

    # Outside
    # x=2.0, center=0.0, half_w=1.0 -> d_x = |2-0| - 1 = 1.0
    # y=0.0, center=0.0, half_h=1.0 -> d_y = |0-0| - 1 = -1.0
    # max(d, 0) = [1.0, 0.0] -> norm = 1.0
    assert jnp.isclose(dist_rect(jnp.array([2.0, 0.0]), center, size), 1.0)

    assert jnp.isclose(dist_rect(jnp.array([0.0, 2.0]), center, size), 1.0)

    # Corner
    # x=2, y=2 -> d=[1,1] -> norm=sqrt(2)
    assert jnp.isclose(dist_rect(jnp.array([2.0, 2.0]), center, size), jnp.sqrt(2.0))
