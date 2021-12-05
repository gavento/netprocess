from ..process.state import ProcessState
from ..utils import KeyOrValue, KeyOrValueT
from .base import EdgeUpdateData, NodeUpdateData, OperationBase, ParamUpdateData


class Fun(OperationBase):
    """
    Wrapper for functionally-defined operations.

    Each of `edge_f`, `node_f`, and `params_f` takes the respective `EdgeUpdateData` etc. and
    returns a dict of updated attributes.
    """

    def __init__(self, edge_f=None, node_f=None, params_f=None):
        self.edge_f = edge_f
        self.node_f = node_f
        self.params_f = params_f

    def update_edges(self, data: EdgeUpdateData) -> dict:
        if self.edge_f is None:
            return {}
        return self.edge_f(data)

    def update_nodes(self, data: NodeUpdateData) -> dict:
        if self.node_f is None:
            return {}
        return self.node_f(data)

    def update_params(self, data: ParamUpdateData) -> dict:
        if self.params_f is None:
            return {}
        return self.params_f(data)


class IncrementParam(OperationBase):
    """
    Operation that increments one param by another param or value.

    Useful e.g. for incrementing counters or time passed.
    """

    def __init__(
        self,
        value_key: str,
        increment: KeyOrValueT = 1,
        default=None,
        dtype=None,
    ):
        assert isinstance(value_key, str)
        self.value = KeyOrValue(value_key, default=default, dtype=dtype)
        self.increment = KeyOrValue(increment, dtype=dtype)

    def prepare_state_data(self, data: ProcessState):
        self.value.ensure_in(data)
        self.increment.ensure_in(data)

    def update_params(self, data: ParamUpdateData) -> dict:
        return {
            self.value.key: self.value.get_from(data.prev_state)
            + self.increment.get_from(data.prev_state)
        }


# class CountNodeStatesOp(OperationBase):
#     def __init__(
#         self, states: Union[int, List[str]], key: str = "state", dest: str = None
#     ):
#         if isinstance(states, int):
#             self.state_names = [f"S{i}" for i in range(states)]
#         else:
#             self.state_names = states
#         self.states = len(self.state_names)
#         self.key = key
#         self.dest = dest if dest is not None else f"{self.key}_count"

#     def create_record(
#         self,
#         rng_key: PRNGKey,
#         state: ProcessStateData,
#         orig_state: ProcessStateData,
#     ) -> PytreeDict:
#         counts = jnp.sum(
#             jax.nn.one_hot(state.node_props[self.key], self.states), axis=0
#         )
#         return {self.dest: counts}

#     def get_traces(self, state: ProcessState):
#         """
#         Return a dict `{trace_name: y_array}` for plotting.

#         Includes `"x": 0..s-1`.
#         """
#         data = state.all_records()[self.dest]
#         d = {s: data[:, i] for s, i in enumerate(self.state_names)}
#         d.update(x=jnp.arange(data.shape[0]))
#         return d


# class CountNodeTransitionsOp(OperationBase):
#     def __init__(
#         self, states: Union[int, List[str]], key: str = "state", dest: str = None
#     ):
#         if isinstance(states, int):
#             self.state_names = [f"S{i}" for i in range(states)]
#         else:
#             self.state_names = states
#         self.states = len(self.state_names)
#         self.key = key
#         self.dest = dest if dest is not None else f"{self.key}_transitions"

#     def create_record(
#         self,
#         rng_key: PRNGKey,
#         state: ProcessStateData,
#         orig_state: ProcessStateData,
#     ) -> PytreeDict:
#         transitions = (
#             self.states * state.node_props[self.key] + orig_state.node_props[self.key]
#         )
#         counts = jnp.sum(jax.nn.one_hot(transitions, self.states * self.states), axis=0)
#         counts_from_to = jnp.reshape(counts, (self.states, self.states))
#         return {self.dest: counts_from_to}

#     def get_traces(
#         self, state: ProcessState, diagonal=False, zeros=False, fraction=False
#     ):
#         """
#         Return a dict `{trace_name: y_array}` for plotting.

#         Includes `"x": 0..s-1`.
#         Optionally contains diagonal traces (state to itself), transitions that never
#         happened, and/or contains fraction from source state rather than counts.
#         """
#         data = state.all_records()[self.dest]
#         if fraction:
#             data = data / state.n
#         d = {}
#         for s0, i0 in enumerate(self.state_names):
#             for s1, i1 in enumerate(self.state_names):
#                 r = data[:, i0, i1]
#                 if (diagonal or i0 != i1) and (zeros or jnp.sum(r) > 0):
#                     d[f"{s0} -> {s1}"] = r
#         d.update(x=jnp.arange(data.shape[0]))
#         return d
