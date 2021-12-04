from dataclasses import dataclass, replace

from netprocess.utils.prop_tree import PropTree

from ..utils import PRNGKey


class EdgeUpdateData(PropTree):
    """
    Structure holding data for node update operations.

    Has props:
    * `state` - entire ProcessStateData, use `state.node` and `state.edge` with caution, as well as `state.prng_key`
    * `prng_key` - edge-local PRNG key
    * `edge` - props of the edge to be updated
    * `src_node`, `tgt_node` - props of the endpoint nodes of the edge
    """

    _FROZEN = True
    _OTHER_PROPS = False

    state: PropTree
    prng_key: PRNGKey
    edge: PropTree
    src_node: PropTree
    tgt_node: PropTree


class NodeUpdateData(PropTree):
    """
    Structure holding data for node update operations.

    Has props:
    * `state` - entire ProcessStateData, use `state.node` and `state.edge` with caution, as well as `state.prng_key`
    * `prng_key` - node-local PRNG key
    * `node` - props of the node to be updated
    * `src_node`, `tgt_node` - props of the endpoint nodes of the edge
    * `in_edges`, `out_edges` - nested aggregates of edge properties and the dicts returned by update_edge
    * `edges` - aggregated over the union of `in_edges` and `out_edges`

    Aggregates: Currently, "sum", "prod", "min", and "max" are passed. Note that for bool dtypes, "sum" and "prod" is of dtype `int32`.
    Use e.g. as: `data.in_edges["max"]["some_property"]`
    """

    _FROZEN = True
    _OTHER_PROPS = False

    state: PropTree
    prng_key: PRNGKey
    node: PropTree
    in_edges: PropTree
    out_edges: PropTree
    edges: PropTree


class ParamUpdateData(PropTree):
    """
    Structure holding data for global parameter update operations.
    """

    _FROZEN = True
    _OTHER_PROPS = False

    state: PropTree
    prev_state: PropTree
    prng_key: PRNGKey


class OperationBase:
    """
    Base class for an operation in the network process.
    """

    def prepare_state_data(self, state: "netprocess.process.ProcessStateData"):
        """
        Prepare the (freshly created) `ProcessStateData` PropTree to be ready for this operation.

        In particular, add all updated and required ndarrays to
        state.params, state.node_props and state.edge_props,
        optionally check their shape and type if they exist.
        """
        pass

    def update_edge(self, data: EdgeUpdateData) -> PropTree:
        """
        Compute and return the edge updates and messages to from_node and to_node.

        Returns a single pytree dict. Underscored items are temporary, non-nderscored
        items are edge updates. All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.
        Must always return the same key sets! Must be JITtable.

        Must be JIT-able.
        """
        return {}

    def update_node(self, data: NodeUpdateData) -> PropTree:
        """
        Compute and return the node update items.

        Must always return the same key set! Must be JITtable.
        All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.

        Must be JIT-able.
        """
        return {}

    def update_params(self, data: ParamUpdateData) -> PropTree:
        """
        Compute and return the param updates.

        Return a pytree dictionary.
        Must always return the same key sets. Must be JIT-able.
        Both `data.new_state` and `data_prev_state` include temporary (underscored) properties.
        """
        return {}

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
