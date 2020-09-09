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
    ) -> (PytreeDict, PytreeDict, PytreeDict):
        """
        Compute and return the edge update keys and messages to from_node and to_node.

        Return `(edge_update_dict, from_update_dict, to_update_dict)`.
        Must always return the same key sets!
        Underscore prefixed items are not kept in state and only passed to create_record
        and used for debugging.

        `params`, `edge`, `from_node` and `to_node` are all dict pytrees.

        Must be JIT-able.
        """
        return {}, {}, {}

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

        Must always return the same key set!
        Underscore prefixed items are not kept in state and only passed to create_record
        and used for debugging.

        `params` and `node` are dict pytrees. `in_edges` and `out_edges` are nested
        aggregates of update_edge pytrees. Currently, "sum", "prod", "min", and "max"
        are passed. Note that unused aggregates are dropped during JIT.
        Also note that for bool dtypes, "sum" and "prod" is of dtype `int32`.

        Must be JIT-able.
        """
        return {}

    def update_params_and_record(
        self,
        rng_key: PRNGKey,
        old_state: ProcessStateData,
        updated_nodes: PytreeDict,
        updated_edges: PytreeDict,
    ) -> (PytreeDict, PytreeDict):
        """
        Compute and return the param updates and any records.

        Return `(param_updates_pytree, record_pytree)`.
        Must always return the same key sets. Must be JIT-able.
        `updated_nodes' and `updated_edges` include teporary underscored items.
        """
        return {}, {}
