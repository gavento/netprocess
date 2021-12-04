from ..utils import PytreeDict, PRNGKey
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EdgeUpdateData:
    """
    Structure holding data for node update operations.

    `from_node` and `to_node` are the property pytrees of source and tagret nodes.
    In undirected graphs, the order of from/to is arbitrary.
    """

    rng_key: PRNGKey
    params: PytreeDict
    edge: PytreeDict
    from_node: PytreeDict
    to_node: PytreeDict

    def _replace(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class NodeUpdateData:
    """
    Structure holding data for node update operations.

    `in_edges`, `out_edges` and `edges` are nested aggregates of edge properties and the dicts returned by update_edge.
    `edges` is aggregated over the union of `in_edges` and `out_edges`.
    Currently, "sum", "prod", "min", and "max" are passed. Note that for bool dtypes, "sum" and "prod" is of dtype `int32`.

    Use as: `data.in_edges["max"]["some_property"]`
    """

    rng_key: PRNGKey
    params: PytreeDict
    node: PytreeDict
    in_edges: PytreeDict
    out_edges: PytreeDict
    edges: PytreeDict

    def _replace(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class ParamUpdateData:
    """
    Structure holding data for global parameter update operations.
    """

    rng_key: PRNGKey
    state: "netprocess.process.ProcessStateData"
    prev_state: "netprocess.process.ProcessStateData"

    def _replace(self, **kw):
        return replace(self, **kw)


class OperationBase:
    def prepare_state_pytrees(self, state: "netprocess.process.ProcessState"):
        """
        Prepare the (freshly created) state pytrees to be ready for this op.

        In particular, add all updated and required ndarrays to
        state.params, state.node_props and state.edge_props,
        optionally check their shape and type if they exist.
        """
        pass

    def update_edge(self, data: EdgeUpdateData) -> PytreeDict:
        """
        Compute and return the edge updates and messages to from_node and to_node.

        Returns a single pytree dict. Underscored items are temporary, non-nderscored
        items are edge updates. All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.
        Must always return the same key sets! Must be JITtable.

        Must be JIT-able.
        """
        return {}

    def update_node(self, data: NodeUpdateData) -> PytreeDict:
        """
        Compute and return the node update items.

        Must always return the same key set! Must be JITtable.
        All items are seen by all the later update functions
        in the same step, underscored items are not persistet to next step.

        Must be JIT-able.
        """
        return {}

    def update_params(self, data: ParamUpdateData) -> PytreeDict:
        """
        Compute and return the param updates.

        Return a pytree dictionary.
        Must always return the same key sets. Must be JIT-able.
        Both `data.new_state` and `data_prev_state` include temporary (underscored) properties.
        """
        return {}

    def create_record(self, data: ParamUpdateData) -> PytreeDict:
        """
        Compute and return the param updates and any records.

        Return `record_pytree`.
        Must always return the same key sets. Must be JIT-able.
        Both `data.new_state` and `data_prev_state` include temporary (underscored) properties.
        """
        return {}

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
