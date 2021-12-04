import logging
import random

import jax
import jax.numpy as jnp
import numpy as np

from netprocess.operations.base import EdgeUpdateData, NodeUpdateData, ParamUpdateData

from ..network import Network
from ..utils import PRNGKey, PytreeDict
from .state import ProcessState, ProcessStateData, _filter_check_merge
from .tracing import Tracer
from ..operations import OperationBase

log = logging.getLogger(__name__)


class NetworkProcess:
    def __init__(self, operations, record=()):
        self.operations = tuple(operations)
        assert all(isinstance(op, OperationBase) for op in self.operations)
        self._run_jit = jax.jit(self._run, static_argnames=["tracing", "jit"])
        self._traced = 0
        self._tr = Tracer(tracing=False)
        self._tr.log_line("<Never traced>")
        self.record = record

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.operations}>"

    def run(self, state: ProcessState, steps=1, jit=True) -> ProcessState:
        step0 = state.params["step"]
        steps_array = jnp.arange(step0, step0 + steps, dtype=jnp.int32)
        state_data = state.as_pytree()
        if jit:
            state_update, records = self._run_jit(
                state_data, steps_array, tracing=True, jit=True
            )
        else:
            state_update, records = self._run(
                state_data, steps_array, tracing=False, jit=False
            )
        return state.copy_updated(state_update, [records])

    def trace_log(self):
        return f"Traced {self._traced} times, last log:\n{self._tr.get_log()}"

    def warmup_jit(self, state=None, n=None, m=None, steps=1, block=True):
        """Force the compilation of the JITted run function and wait for it (if `block`)."""
        if state is None:
            assert n >= 2 and m >= 1
            state = self.new_state([[0, 1]] * m, n=n, seed=42)
        else:
            assert n is None and m is None
        # Run the jitted function
        new_state = self.run(state, steps=steps)
        # Wait for all computations
        if block:
            new_state.block_on_all()

    def _run(
        self,
        state: ProcessStateData,
        steps_array: jnp.DeviceArray,
        tracing: bool,
        jit: bool,
    ):
        """Returns (new_state, all_records_pytree). JIT-able when jit=True."""
        if tracing:
            msg = f"Tracing {self} with n={state.n}, m={state.m}, steps={steps_array.shape[0]}"
            self._tr = Tracer(tracing=True)
            self._traced += 1
            log.debug(msg)
        if jit:
            return jax.lax.scan(
                lambda s, i: self._run_step(s, i),
                state,
                steps_array,
            )
        else:
            rs = []
            for si in steps_array:
                state, r = self._run_step(state, si)
                rs.append(r)
            return state, jax.tree_multimap(lambda *r: jnp.stack(list(r)), *rs)

    def _run_step(self, state: ProcessStateData, step: jnp.int32):
        """Returns (new_state, record_pytree). JIT-able."""

        # Original state, never updated
        orig_state = state

        # Set step number
        state = state.copy()
        # NB: this shuld be a noop with correct external step numbering
        state.params["step"] = step

        # Run all the update steps, updating the staself._run_step(s, si)te
        state = self._run_update_edges(state)
        state = self._run_update_nodes(state)
        state = self._run_update_params(state, orig_state)

        records = {rn: state.params[rn] for rn in self.record}

        # Finally, increment the step number
        state.params["step"] = step + 1

        # Create the new state, filtering underlines and checking
        # that we only update existing keys
        state = state._replace(
            rng_key=jax.random.fold_in(state.rng_key, 0),
            params=_filter_check_merge(orig_state.params, state.params, "params"),
            node_props=_filter_check_merge(
                orig_state.node_props, state.node_props, "node"
            ),
            edge_props=_filter_check_merge(
                orig_state.edge_props, state.edge_props, "edge"
            ),
        )
        return state, records

    def _run_update_edges(self, state: ProcessStateData) -> ProcessStateData:
        """
        Compute edge operations and update the state (incl. the rng_key).
        """

        def edge_f(shared_rng_key, params, e_pt, from_pt, to_pt):
            "Apply all operations on each edge"
            e_pt = e_pt.copy()
            rng_key = jax.random.fold_in(shared_rng_key, e_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(rng_key, i)
                updates = op.update_edge(
                    EdgeUpdateData(
                        r,
                        self._tr.TD(params, "P"),
                        self._tr.TD(e_pt, "E"),
                        self._tr.TD(from_pt, "fromN"),
                        self._tr.TD(to_pt, "toN"),
                    )
                )
                self._tr.log_operation_io(op, updates)
                e_pt.update(updates)
            return e_pt

        # Extract node values for edge endpoints
        n2e_from_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 0]], state.node_props
        )
        n2e_to_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 1]], state.node_props
        )

        self._tr.log_line("Edge updates:")
        new_edges_pytree = jax.vmap(edge_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(state.rng_key, 0),
            state.params,
            state.edge_props,
            n2e_from_pytree,
            n2e_to_pytree,
        )
        return state._replace(
            edge_props=new_edges_pytree, rng_key=jax.random.fold_in(state.rng_key, 1)
        )

    def _run_update_nodes(self, state: ProcessStateData) -> ProcessStateData:
        """
        Apply all operation node updates on state.
        """

        def node_f(
            shared_rng_key, params, n_pt, in_edges_agg, out_edges_agg, both_edges_agg
        ):
            "Apply all operations on each node"
            n_pt = n_pt.copy()
            rng_key = jax.random.fold_in(shared_rng_key, n_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(rng_key, i)
                updates = op.update_node(
                    NodeUpdateData(
                        r,
                        self._tr.TD(params, "P"),
                        self._tr.TD(n_pt, "N"),
                        self._tr.TD(in_edges_agg, "inE", depth=2),
                        self._tr.TD(out_edges_agg, "outE", depth=2),
                        self._tr.TD(both_edges_agg, "E", depth=2),
                    )
                )
                self._tr.log_operation_io(op, updates)
                n_pt.update(updates)
            return n_pt

        def scatter_op(agg_op, zval, e_vals, e_endpoints):
            "Compute given aggregate of e_vals grouped by e_endpoints"
            # Get min/max operation-neutral value for integers and infinities
            # NB: this is numpy - should be static vals in the computation graph
            if np.isneginf(zval) and np.issubdtype(e_vals.dtype, np.integer):
                zval = np.iinfo(e_vals.dtype).min
            elif np.isinf(zval) and np.issubdtype(e_vals.dtype, np.integer):
                zval = np.iinfo(e_vals.dtype).max
            # Array of neutral values
            z = jnp.full(
                state.node_props["i"].shape + e_vals.shape[1:],
                zval,
                dtype=e_vals.dtype,
            )
            e_endpoints_exp = jnp.expand_dims(e_endpoints, 1)
            dims = jax.lax.ScatterDimensionNumbers(
                tuple(range(1, len(e_vals.shape))), (0,), (0,)
            )
            return agg_op(z, e_endpoints_exp, e_vals, dims)

        # Compute edge-to-node value aggregates
        in_edges_agg, out_edges_agg, both_edges_agg = {}, {}, {}
        for agg_op, agg_name, zval, comb2 in (
            (jax.lax.scatter_add, "sum", 0, lambda x1, x2: x1 + x2),
            (jax.lax.scatter_mul, "prod", 1, lambda x1, x2: x1 * x2),
            (jax.lax.scatter_min, "min", np.inf, jnp.minimum),
            (jax.lax.scatter_max, "max", -np.inf, jnp.maximum),
        ):
            in_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, zval, a, state.edges[:, 1]),
                state.edge_props,
            )
            out_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, zval, a, state.edges[:, 0]),
                state.edge_props,
            )
            both_edges_agg[agg_name] = jax.tree_util.tree_multimap(
                comb2, in_edges_agg[agg_name], out_edges_agg[agg_name]
            )

        self._tr.log_line("Node updates:")
        new_nodes_pytree = jax.vmap(node_f, in_axes=(None, None, 0, 0, 0, 0))(
            jax.random.fold_in(state.rng_key, 0),
            state.params,
            state.node_props,
            in_edges_agg,
            out_edges_agg,
            both_edges_agg,
        )
        return state._replace(
            node_props=new_nodes_pytree,
            rng_key=jax.random.fold_in(state.rng_key, 1),
        )

    def _run_update_params(
        self, state: ProcessStateData, orig_state: ProcessStateData
    ) -> ProcessStateData:
        """
        Compute parameter updates.
        """
        self._tr.log_line("Param updates:")
        new_params_pytree = state.params.copy()
        for i, op in enumerate(self.operations):
            updates = op.update_params(
                ParamUpdateData(
                    jax.random.fold_in(state.rng_key, i + 1),
                    self._tr.TS(state),
                    self._tr.TS(orig_state, "orig"),
                )
            )
            self._tr.log_operation_io(op, updates)
            new_params_pytree.update(updates)
        return state._replace(
            params=new_params_pytree,
            rng_key=jax.random.fold_in(state.rng_key, 0),
        )

    def new_state(
        self,
        network: Network,
        *,
        seed=None,
        params: PytreeDict = {},
        node_props: PytreeDict = {},
        edge_props: PytreeDict = {},
    ):
        """
        Create a new ProcessState with initial pytree elements for all operations.

        `seed` may be jax PRNGKey, a 64 bit number (used as a seed) or None (randomize).
        The given param/sprops are used as overrides over the Network pytrees (those are untouched).
        """
        if seed is None:
            seed = random.randint(0, 1 << 64 - 1)
        if isinstance(seed, (jnp.ndarray, np.ndarray)):
            assert seed.shape == (2,)
            assert seed.dtype == jnp.int32
            rng_key = jnp.array(seed)
        else:
            rng_key = jax.random.PRNGKey(seed)

        # Note: all pytree elements are converted to jax arrays later in the state constructor
        # Note: all pytrees are properly copied later in the state constructor
        state = ProcessState(
            rng_key,
            network=network,
            params=dict(network.h5_as_pytree("/params"), **params),
            node_props=dict(network.h5_as_pytree("/node_props"), **node_props),
            edge_props=dict(network.h5_as_pytree("/edge_props"), **edge_props),
            process=self,
        )
        # Prepare state for operations
        for op in self.operations:
            op.prepare_state_pytrees(state)
        # Ensure the operaions added only ndarrays
        state._ensure_ndarrays()
        return state
