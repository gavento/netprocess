class OperationBase:
    def prepare_state_pytrees(self, state):
        pass

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        """
        Compute and return the edge update keys and messages to from_node and to_node.

        Must always return the same key sets!
        Underscore prefixed items are not kept in state and only passed to create_record
        and used for debugging.

        `params`, `edge`, `from_node` and `to_node` are all dict pytrees.
        """
        return {}, {}, {}

    def update_node(self, rng_key, params, node, in_edges, out_edges):
        """
        Compute and return the node update items.

        Must always return the same key set!
        Underscore prefixed items are not kept in state and only passed to create_record
        and used for debugging.

        `params` and `node` are dict pytrees. `in_edges` and `out_edges` are nested
        aggregates of update_edge pytrees. Currently, "sum", "mul", "min", and "max"
        are passed. Note that unused operations are dropped during JIT.
        """
        return {}

    def update_params_and_record(
        self, rng_key, params, old_nodes, new_nodes, old_edges, new_edges
    ):
        """
        Compute and return the param updates and any records.

        Must always return the same key sets!

        `params` and `node` are dict pytrees. `in_edges` and `out_edges` are nested
        aggregates of update_edge pytrees. Currently, "sum", "mul", "min", and "max"
        are passed. Note that unused operations are dropped during JIT.
        """
        return {}, {}
