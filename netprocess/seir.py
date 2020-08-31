import jax
from jax import lax
from jax import numpy as jnp
from . import networks, primitives


"""
Node state structure: (state_no, ticks_in_state, node_feature_vector)
Edges: no state
"""


def build_sir_update_function(edge_beta: float, gamma: float):
    """

    Returns a function `(rng_key, edges, states) -> new_states`
    """

    def trans_from_s(_time, adj_counts):
        num_i = adj_counts[1]
        # Probability of being nfected by at least one neighbor
        p = 1.0 - (1.0 - edge_beta) ** num_i
        return jnp.array([0.0, 1.0, 0.0]) * p + jnp.array([1.0, 0.0, 0.0]) * (1.0 - p)

    upfun = primitives.build_update_function(
        [
            trans_from_s,
            jnp.array([0.0, 1.0 - gamma, gamma]),
            jnp.array([0.0, 0.0, 1.0]),
        ]
    )

    def update_state(rng_key, edges, states):
        adjacent_states = networks.count_inneighbor_states(edges, states, 3)
        # Note: since we do not use t2 at all, JAX JIT should omit its computation
        s2, t2 = upfun(
            rng_key,
            states,
            jnp.zeros(states.shape[0], dtype=jnp.int32),
            adjacent_states,
        )
        return s2

    return update_state
