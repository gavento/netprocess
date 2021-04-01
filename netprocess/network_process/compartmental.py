import copy

import attr
import jax
import jax.numpy as jnp

from ..jax_utils import cond, switch
from ..network_process import OperationBase


@attr.s
class Transition:
    """
    Definition of one unary or binary transition for GenericCompartmentalUpdateOp.

    All those transitions are modelled as with exponential delay.

    Unary example:
        Transition("I", "R", param_key="immunity_loss_rate", default_val=0.01)
    Binary example:
        Transition("I", "R", param_key="immunity_loss_rate", default_val=0.01)
    """

    # Compartment names: from, to, [source_state]
    f = attr.ib()
    t = attr.ib()
    s = attr.ib(default=None)
    # Indices of the above compartments
    fi = attr.ib(init=False, default=None)
    ti = attr.ib(init=False, default=None)
    si = attr.ib(init=False, default=None)
    # Name of the probability parameter and of the same adjusted for delta_t
    param_key = attr.ib(kw_only=True)
    adjusted_param_key = attr.ib(kw_only=True, default=None)
    default_val = attr.ib(kw_only=True, default=None)
    activated_key = attr.ib(kw_only=True, default=None)
    binary = attr.ib(type=bool, default=False, kw_only=True)


class GenericCompartmentalUpdateOp(OperationBase):
    def __init__(
        self,
        compartments=("S",),
        transitions=(),
        state_key="compartment",
        delta_t_key="delta_t",
        aux_prefix="",
    ):
        self.compartments = compartments
        self.aux_prefix = aux_prefix
        self.state_key = state_key
        self.delta_t_key = delta_t_key

        self.transitions = copy.deepcopy(transitions)
        for i, t in enumerate(self.transitions):
            t.fi = self.compartments.index(t.f)
            t.ti = self.compartments.index(t.t)
            assert (t.s != None) == (t.binary)
            if t.adjusted_param_key is None:
                t.adjusted_param_key = f"_{self.aux_prefix}adjusted_{t.param_key}"
            if t.binary:
                t.si = self.compartments.index(t.s)
                if t.activated_key is None:
                    t.activated_key = (
                        f"_{self.aux_prefix}activated_{t.f}{t.t}_{t.s}_{i}"
                    )

    def prepare_state_pytrees(self, state):
        """
        Prepare the state for running the process.

        Ensures the state has all state variables, missing parameters get defaults (and raise errors if none),
        probabilities are adjusted to delta_t, jax arrays are ensured for all pytree vals.
        """
        if self.state_key not in state.nodes_pytree:
            state.nodes_pytree[self.state_key] = jnp.zeros(state.n, dtype=jnp.int32)
        state.params_pytree.setdefault(self.delta_t_key, 1.0)
        assert state.params_pytree[self.delta_t_key] < 1.000001

        for t in self.transitions:
            if t.default_val is not None:
                state.params_pytree.setdefault(t.param_key, t.default_val)
            assert t.param_key in state.params_pytree
            state.params_pytree[t.adjusted_param_key] = (
                state.params_pytree[t.param_key] * state.params_pytree[self.delta_t_key]
            )

        state._ensure_ndarrays()

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        """
        Run all binary transitions on the directed edge, recording each activation separately.
        """
        ret = {}
        for t in self.transitions:
            if t.binary:
                ret[t.activated_key] = cond(
                    jnp.logical_and(
                        from_node[self.state_key] == t.si,
                        to_node[self.state_key] == t.fi,
                    ),
                    lambda: jnp.int32(
                        jax.random.bernoulli(rng_key, params[t.adjusted_param_key])
                    ),
                    jnp.int32(0),
                )
        return ret

    # TODO: HERE
    # TODO: figure out transitions with the same target state (issue of switch?)
    # TODO: figure out transitions with the same source state (issue of under-definedness? parallel poisson processes)
    def update_node(self, rng_key, params, node, in_edges, out_edges):
        compartment = switch(
            [
                # S -> {S, I}:
                lambda: cond(in_edges["sum"][self.INFECTION] > 0, 1, 0),
                # I -> {I, R}:
                lambda: cond(jax.random.bernoulli(rng_key, params["gamma"]), 2, 1),
                # R -> {R}:
                2,
            ],
            node[self.STATE],
        )
        return {
            self.state_key: compartment,
            self.INFECTED: node[self.INFECTED] + out_edges["sum"][self.INFECTION],
        }
