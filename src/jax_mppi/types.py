from typing import Callable
import jax

# Dynamics: (state, action) -> next_state  or  (state, action, t) -> next_state
DynamicsFn = Callable[..., jax.Array]
# Cost: (state, action) -> scalar_cost  or  (state, action, t) -> scalar_cost
RunningCostFn = Callable[..., jax.Array]
# Terminal: (states, actions) -> scalar_cost
TerminalCostFn = Callable[[jax.Array, jax.Array], jax.Array]
