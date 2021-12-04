import copy
from typing import Tuple

import attr
import jax
import jax.numpy as jnp

from ..operations.base import (
    OperationBase,
    EdgeUpdateData,
    NodeUpdateData,
    ParamUpdateData,
)
from ..utils import PRNGKey, PytreeDict, KeyOrValueT, KeyOrValue
from ..utils.jax_utils import cond, switch


class PoissonTransition:
    """
    Definition of a poisson transition for GenericCompartmentalUpdateOp.

    States:
        `f` - transition from state name or number
        `t` - transition to state name or number
    """

    def __init__(self, f, t, rate: KeyOrValueT):
        self.f = f
        self.t = t
        # Indices of the above compartments
        self.fi = None
        self.ti = None

        self.rate = KeyOrValue(rate)

    def _init_for_operation(self, _index: int, op: "PoissonCompartmentalUpdateOp"):
        self.f, self.fi = op._lookup_compartment(self.f)
        self.t, self.ti = op._lookup_compartment(self.t)

    def sample_transition_time(self, data: NodeUpdateData) -> jnp.float32:
        """Sample the transition delay of this transition"""
        return jax.random.exponential(data.rng_key) / self.rate.get_from(data.params)

    def update_edge(self, op, data: EdgeUpdateData) -> PytreeDict:
        """Add any temporary attributes to an edge (for binary transitions)"""
        return {}


class BinaryPoissonTransition(PoissonTransition):
    def __init__(self, f, t, s, rate: KeyOrValueT):
        super().__init__(f, t, rate)
        # Compartment names: source_state
        self.s = s
        # Indices of the above compartments
        self.si = None
        self.time_edge_activated = None

    def _init_for_operation(self, index: int, op: "PoissonCompartmentalUpdateOp"):
        super()._init_for_operation(index, op)
        self.s, self.si = op._lookup_compartment(self.s)
        self.time_edge_activated = KeyOrValue(
            f"_{op.aux_prefix}activated_{self.f}{self.t}_{self.s}_{index}"
        )

    def update_edge(self, op, data: EdgeUpdateData) -> PytreeDict:
        """Sample the transition delay of the transition on a single edge"""
        time = cond(
            jnp.logical_and(
                op.state.get_from(data.from_node) == self.si,
                op.state.get_from(data.to_node) == self.fi,
            ),
            lambda: jax.random.exponential(data.rng_key)
            / self.rate.get_from(data.params),
            jnp.float32(jnp.inf),
        )
        return {self.time_edge_activated.key: time}

    def sample_transition_time(self, data: NodeUpdateData) -> jnp.float32:
        """Sample the transition delay of the transition"""
        return data.in_edges["min"][self.time_edge_activated.key]


class PoissonCompartmentalUpdateOp(OperationBase):
    def __init__(
        self,
        compartments=("S",),
        transitions=(),
        state_key: str = "compartment",
        delta_t: KeyOrValueT = 1.0,
        aux_prefix="",
    ):
        self.compartments = compartments
        self.aux_prefix = aux_prefix
        self.state = KeyOrValue(state_key, default=0)
        self.delta_t = KeyOrValue(delta_t)

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
        self.state.ensure_in(state.node_props, repeat_times=state.n)
        self.delta_t.ensure_in(state.params)

        for t in self.transitions:
            t.rate.ensure_in(state.params)

    def update_edge(self, data: EdgeUpdateData) -> PytreeDict:
        """
        Run all binary transitions on the directed edge, recording each activation separately.
        """
        ret = {}
        rngs = jax.random.split(data.rng_key, len(self.transitions))
        for i, t in enumerate(self.transitions):
            ret.update(t.update_edge(self, data._replace(rng_key=rngs[i])))
        return ret

    def update_node(self, data: NodeUpdateData) -> PytreeDict:
        if not self.transitions:
            return {self.state.key: self.state.get_from(data.node)}
        rngs = jax.random.split(data.rng_key, len(self.transitions))

        def trans_time(i: int, t: PoissonTransition):
            return cond(
                self.state.get_from(data.node) == t.fi,
                lambda: t.sample_transition_time(data._replace(rng_key=rngs[i])),
                jnp.inf,
            )

        times = jnp.array([trans_time(i, t) for i, t in enumerate(self.transitions)])
        return {
            self.state.key: cond(
                jnp.min(times) < self.delta_t.get_from(data.params),
                self.to_states[jnp.argmin(times)],
                self.state.get_from(data.node),
            )
        }
