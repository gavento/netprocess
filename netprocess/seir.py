import jax
from jax import lax
from jax import numpy as jnp

from . import networks, primitives, messages
from .utils import ArrayDict, cond


def build_sir_update(
    edge_beta: float,
    gamma: float,
    count_time=False,
    count_infected=False,
    jit=True,
):
    """
    Returns a SIR-update function with optional stats keeping.

    States: 0=S, 1=I, 2=R
    Returns: `update_SIR(rng_key, nodes_dict, edges_from_to) -> node_dict`.
    """

    def sir_ef(r, ed, fnd, tnd):
        infection = cond(
            jnp.logical_and(fnd["state"] == 1, tnd["state"] == 0),
            lambda: jnp.int32(jax.random.bernoulli(r, edge_beta)),
            0,
        )
        return {}, {"infection": infection}, {"infection": infection}

    def sir_nf(r, nd, ied, oed):
        state = primitives.switch(
            nd["state"],
            [
                # S -> {S, I}:
                lambda: cond(ied["infection"] > 0, 1, 0),
                # I -> {I, R}:
                lambda: cond(jax.random.bernoulli(r, gamma), 2, 1),
                # R -> {R}:
                2,
            ],
        )
        d = {"state": state}

        if count_time:
            d["time"] = cond(state == nd["state"], lambda: nd["time"] + 1, 0)
        if count_infected:
            d["infected"] = nd["infected"] + oed["infection"]
        return d

    f = messages.build_step_edge_node_messages(sir_ef, sir_nf, jit=jit)

    def update_SIR(rng_key, nodes_dict, edges_from_to):
        return f(rng_key, nodes_dict, {}, edges_from_to)[0]

    return update_SIR


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
