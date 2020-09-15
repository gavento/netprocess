import jax
import jax.numpy as jnp

from ..jax_utils import cond, switch
from ..network_process import OperationBase


class SIRUpdateOperation(OperationBase):
    def __init__(self, state_key="compartment", prefix=""):
        self.prefix = prefix
        self.state_key = state_key

    def prepare_state_pytrees(self, state):
        state.nodes_pytree.setdefault(
            "compartment", jnp.zeros(state.n, dtype=jnp.int32)
        )
        state.nodes_pytree.setdefault("infected", jnp.zeros(state.n, dtype=jnp.int32))
        state.params_pytree.setdefault("edge_beta", 0.05)
        state.params_pytree.setdefault("gamma", 0.1)

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        infection = cond(
            jnp.logical_and(from_node["compartment"] == 1, to_node["compartment"] == 0),
            lambda: jnp.int32(jax.random.bernoulli(rng_key, params["edge_beta"])),
            0,
        )
        return {}, {"infection": infection}, {"infection": infection}

    def update_node(self, rng_key, params, node, in_edges, out_edges):
        compartment = switch(
            [
                # S -> {S, I}:
                lambda: cond(in_edges["sum"]["infection"] > 0, 1, 0),
                # I -> {I, R}:
                lambda: cond(jax.random.bernoulli(rng_key, params["gamma"]), 2, 1),
                # R -> {R}:
                2,
            ],
            node["compartment"],
        )
        return {
            "compartment": compartment,
            "infected": node["infected"] + out_edges["sum"]["infection"],
        }


class ComputeImmediateROperation:
    def update_params_and_record(self, rng_key, state, new_nodes, new_edges):
        # TODO
        return {}, {"immediate_R_mean": mean, "immediate_R_samples": count}
