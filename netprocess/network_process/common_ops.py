import jax
import jax.numpy as jnp

from ..utils import PRNGKey, PytreeDict
from .state import ProcessStateData


class CountNodeStatesOp:
    def __init__(self, states: int, key: str = "state", dest: str = None):
        self.states = states
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
        return {}, {self.dest: counts}


class CountNodeTransitionsOp:
    def __init__(self, states: int, key: str = "state", dest: str = None):
        self.states = states
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
