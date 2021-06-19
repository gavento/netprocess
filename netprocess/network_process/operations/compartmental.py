import copy

import attr
import jax
import jax.numpy as jnp

from ...jax_utils import cond, switch
from ...utils import PytreeDict, PRNGKey
from .base import OperationBase
from typing import Tuple


@attr.s
class PoissonTransition:
    """
    Definition of a poisson transition for GenericCompartmentalUpdateOp.

    States:
        `f` - transition from state name or number
        `t` - transition to state name or number
    """

    # Compartment names: from, to
    f = attr.ib()
    t = attr.ib()
    # Indices of the above compartments
    fi = attr.ib(init=False, default=None)
    ti = attr.ib(init=False, default=None)
    # Name of the key recording activation of this in the node (default: None)
    rate_key = attr.ib(kw_only=True, default=None)
    default_rate = attr.ib(kw_only=True, default=None)

    def _init_for_operation(self, _index: int, op: "PoissonCompartmentalUpdateOp"):
        self.f, self.fi = op._lookup_compartment(self.f)
        self.t, self.ti = op._lookup_compartment(self.t)

    def sample_transition_time(
        self, rng_key: PRNGKey, params: PytreeDict, _node, _in_edges, _out_edges
    ) -> jnp.float32:
        """Sample the transition delay of the transition"""
        return jax.random.exponential(rng_key) / params[self.rate_key]

    def update_edge(
        self, _op, _rng_key, _params, _edge, _from_node, _to_node
    ) -> PytreeDict:
        """Add any temporary attributes to an edge (for binary transitions)"""
        return {}


@attr.s
class BinaryPoissonTransition(PoissonTransition):
    # Compartment names: source_state
    s = attr.ib()
    # Indices of the above compartments, filled by the Operation
    si = attr.ib(init=False, default=None)
    edge_activated_time_key = attr.ib(kw_only=True, default=None)

    def _init_for_operation(self, index: int, op: "PoissonCompartmentalUpdateOp"):
        super()._init_for_operation(op)
        self.s, self.si = op._lookup_compartment(self.s)
        if self.edge_activated_time_key is None:
            self.edge_activated_time_key = (
                f"_{op.aux_prefix}activated_{self.f}{self.t}_{self.s}_{index}"
            )

    def update_edge(self, op, rng_key, params, edge, from_node, to_node) -> PytreeDict:
        """Sample the transition delay of the transition on a single edge"""
        time = cond(
            jnp.logical_and(
                from_node[op.state_key] == self.si,
                to_node[op.state_key] == self.fi,
            ),
            lambda: jax.random.exponential(rng_key) / params[self.rate_key],
            jnp.float32(jnp.inf),
        )

        return {self.edge_activated_time_key: time}

    def sample_transition_time(
        self, _rng_key, _params, _node, in_edges: PytreeDict, _out_edges
    ) -> jnp.float32:
        """Sample the transition delay of the transition"""
        return in_edges["min"][self.edge_activated_time_key]


class PoissonCompartmentalUpdateOp(OperationBase):
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
            t._init_for_operation(i, self)

        self.from_states = jnp.array([t.fi for t in self.transitions], dtype=jnp.int32)
        self.to_states = jnp.array([t.ti for t in self.transitions], dtype=jnp.int32)

    def _lookup_compartment(self, state_name_or_no) -> Tuple[str, int]:
        if isinstance(state_name_or_no, str):
            return state_name_or_no, self.compartments.index(state_name_or_no)
        elif isinstance(state_name_or_no, int):
            assert state_name_or_no < len(self.compartments)
            return str(state_name_or_no), state_name_or_no
        else:
            raise TypeError

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
            if t.param_key is not None:
                if t.default_rate is not None:
                    state.params_pytree.setdefault(t.param_key, t.default_rate)
                assert t.param_key in state.params_pytree

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        """
        Run all binary transitions on the directed edge, recording each activation separately.
        """
        ret = {}
        rngs = jax.random.split(rng_key, len(self.transitions))
        for i, t in enumerate(self.transitions):
            ret.update(t.update_edge(self, rngs[i], params, edge, from_node, to_node))
        return ret

    def update_node(self, rng_key, params, node, in_edges, out_edges):
        if not self.transitions:
            return {self.state_key: node[self.state_key]}
        rngs = jax.random.split(rng_key, len(self.transitions))

        def trans_time(i: int, t: PoissonTransition):
            return cond(
                node[self.state_key] == t.fi,
                lambda: t.sample_transition_time(
                    rngs[i], params, node, in_edges, out_edges
                ),
                jnp.inf,
            )

        times = jnp.array([trans_time(i, t) for i, t in enumerate(self.transitions)])
        return {
            self.state_key: cond(
                jnp.min(times) < params[self.delta_t_key],
                self.to_states[jnp.argmin(times)],
                node[self.state_key],
            )
        }
