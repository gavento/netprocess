from typing import List, Union

import jax
import jax.numpy as jnp

from ..utils import PRNGKey, PytreeDict
from .base import OperationBase
from ..process.state import ProcessState, ProcessStateData


class AdvanceTimeOp(OperationBase):
    """
    Operation that advances time param by the current delta_t.

    NB: You probably want to use this last in the operation order.
    """

    def __init__(
        self,
        t_key="t",
        delta_t_key="delta_t",
    ):
        self.t_key = t_key
        self.delta_t_key = delta_t_key

    def prepare_state_pytrees(self, state):
        state.params_pytree.setdefault(self.delta_t_key, 1.0)
        state.params_pytree.setdefault(self.t_key, 0.0)

    def update_params(self, _rng_key, state, _orig_state) -> PytreeDict:
        params2 = jax.tree_map(lambda x: x, state.params_pytree)
        params2[self.t_key] = (
            state.params_pytree[self.t_key] + state.params_pytree[self.delta_t_key]
        )
        return params2


class CountNodeStatesOp(OperationBase):
    def __init__(
        self, states: Union[int, List[str]], key: str = "state", dest: str = None
    ):
        if isinstance(states, int):
            self.state_names = [f"S{i}" for i in range(states)]
        else:
            self.state_names = states
        self.states = len(self.state_names)
        self.key = key
        self.dest = dest if dest is not None else f"{self.key}_count"

    def create_record(
        self,
        rng_key: PRNGKey,
        state: ProcessStateData,
        orig_state: ProcessStateData,
    ) -> PytreeDict:
        counts = jnp.sum(
            jax.nn.one_hot(state.nodes_pytree[self.key], self.states), axis=0
        )
        return {self.dest: counts}

    def get_traces(self, state: ProcessState):
        """
        Return a dict `{trace_name: y_array}` for plotting.

        Includes `"x": 0..s-1`.
        """
        data = state.all_records()[self.dest]
        d = {s: data[:, i] for s, i in enumerate(self.state_names)}
        d.update(x=jnp.arange(data.shape[0]))
        return d


class CountNodeTransitionsOp(OperationBase):
    def __init__(
        self, states: Union[int, List[str]], key: str = "state", dest: str = None
    ):
        if isinstance(states, int):
            self.state_names = [f"S{i}" for i in range(states)]
        else:
            self.state_names = states
        self.states = len(self.state_names)
        self.key = key
        self.dest = dest if dest is not None else f"{self.key}_transitions"

    def create_record(
        self,
        rng_key: PRNGKey,
        state: ProcessStateData,
        orig_state: ProcessStateData,
    ) -> PytreeDict:
        transitions = (
            self.states * state.nodes_pytree[self.key]
            + orig_state.nodes_pytree[self.key]
        )
        counts = jnp.sum(jax.nn.one_hot(transitions, self.states * self.states), axis=0)
        counts_from_to = jnp.reshape(counts, (self.states, self.states))
        return {self.dest: counts_from_to}

    def get_traces(
        self, state: ProcessState, diagonal=False, zeros=False, fraction=False
    ):
        """
        Return a dict `{trace_name: y_array}` for plotting.

        Includes `"x": 0..s-1`.
        Optionally contains diagonal traces (state to itself), transitions that never
        happened, and/or contains fraction from source state rather than counts.
        """
        data = state.all_records()[self.dest]
        if fraction:
            data = data / state.n
        d = {}
        for s0, i0 in enumerate(self.state_names):
            for s1, i1 in enumerate(self.state_names):
                r = data[:, i0, i1]
                if (diagonal or i0 != i1) and (zeros or jnp.sum(r) > 0):
                    d[f"{s0} -> {s1}"] = r
        d.update(x=jnp.arange(data.shape[0]))
        return d
