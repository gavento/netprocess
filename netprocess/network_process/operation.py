import jax
import jax.numpy as jnp

from ..utils import PytreeDict, PRNGKey
from .state import ProcessState, ProcessStateData


class OperationBase:
    def prepare_state_pytrees(self, state: ProcessState):
        """
        Prepare the (freshly created) state pytrees to be ready for this op.

        In particular, add all updated and required arrays to
        state.params_pytree, state.nodes_pytree and state.edges_pytree,
        check their shape and type if they exist.
        """
        pass

    def update_edge(
        self,
        rng_key: PRNGKey,
        params: PytreeDict,
        edge: PytreeDict,
        from_node: PytreeDict,
        to_node: PytreeDict,
    ) -> PytreeDict:
        """
        Compute and return the edge updates and messages to from_node and to_node.

        Returns a single pytree dict. Underscored items are temporary, non-nderscored
        items are edge updates. All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.
        Must always return the same key sets! Must be JITtable.

        `params`, `edge`, `from_node` and `to_node` are all dict pytrees.

        Must be JIT-able.
        """
        return {}

    def update_node(
        self,
        rng_key: PRNGKey,
        params: PytreeDict,
        node: PytreeDict,
        in_edges: PytreeDict,
        out_edges: PytreeDict,
    ) -> PytreeDict:
        """
        Compute and return the node update items.

        Must always return the same key set! Must be JITtable.
        All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.

        `params` and `node` are dict pytrees. `in_edges` and `out_edges` are nested
        aggregates of update_edge pytrees. Currently, "sum", "prod", "min", and "max"
        are passed. Note that unused aggregates are dropped during JIT.
        Also note that for bool dtypes, "sum" and "prod" is of dtype `int32`.

        Must be JIT-able.
        """
        return {}

    def update_params(
        self,
        rng_key: PRNGKey,
        state: ProcessStateData,
        orig_state: ProcessStateData,
    ) -> PytreeDict:
        """
        Compute and return the param updates.

        Return a pytree dictionary.
        Must always return the same key sets. Must be JIT-able.
        `state` includes temporary (underscored) properties.
        """
        return {}

    def create_record(
        self,
        rng_key: PRNGKey,
        state: ProcessStateData,
        orig_state: ProcessStateData,
    ) -> PytreeDict:
        return {}
        """
        Compute and return the param updates and any records.

        Return `record_pytree`.
        Must always return the same key sets. Must be JIT-able.
        `state` includes temporary (underscored) properties.
        """

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
