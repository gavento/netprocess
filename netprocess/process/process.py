import logging
import random
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from ..network import Network
from ..operations import EdgeUpdateData, NodeUpdateData, OperationBase, ParamUpdateData
from ..utils import PropTree
from .state import ProcessState, ProcessState
from .tracing import Tracer

log = logging.getLogger(__name__)


class NetworkProcess:
    """
    A network process is an immutable sequence of operations on `ProcessState`.

    The state is created for a network via `state0 = process.new_state(net, ...)` and then
    eveolved with e.g. `state1 = process.run(state0, steps=10)`.
    State also accumulates any gathered recorded properties on every step.
    """

    def __init__(self, operations: Iterable[OperationBase], record_keys=()):
        self.operations = tuple(operations)
        assert all(isinstance(op, OperationBase) for op in self.operations)
        self._run_jit = jax.jit(self._run, static_argnames=["tracing", "jit"])
        self._traced = 0
        self._tr = Tracer(tracing=False)
        self._tr.log_line("<Never traced>")
        self.record_keys = record_keys

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.operations}>"

    def run(self, state: ProcessState, steps=1, jit=True) -> ProcessState:
        """
        Run `steps` steps of the process on the given `state`, returning a new state.

        By default, JIT-compiles the operations for GPU or CPU.
        Recorded keys are automatically added to the new state `records`.
        """
        state = state.copy()
        net, rec = state._network, state._records
        state._network, state._records = None, None

        steps_array = jnp.arange(state.step, state.step + steps, dtype=jnp.int32)
        if jit:
            state, records = self._run_jit(state, steps_array, tracing=True, jit=True)
        else:
            state, records = self._run(state, steps_array, tracing=False, jit=False)

        state._network, state._records = net, rec
        if len(records) > 0:
            state.records.add_record(records)
        return state

    def trace_log(self) -> str:
        """
        Return the tracing log as a string.
        """
        return f"Traced {self._traced} times, last log:\n{self._tr.get_log()}"

    def warmup_jit(self, state=None, n=None, m=None, steps=1, block=True):
        """
        Force the compilation of the JITted run() function for given `n`, `m` and `steps`, and wait for it (if `block`).
        """
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
        state: ProcessState,
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

    def _run_step(self, data: ProcessState, step: jnp.int32):
        """Returns (new_state, record_pytree). JIT-able."""

        # Original state, never updated
        prev_data = data.copy()
        data = data.copy()

        # Set step number
        # NB: this shuld be a noop with correct external step numbering
        data["step"] = step

        # Run all the update steps, updating the staself._run_step(s, si)te
        data = self._run_update_edges(data)
        data = self._run_update_nodes(data)
        data = self._run_update_params(data, prev_data)

        records = {rk: data[rk] for rk in self.record_keys}

        # Create the new state, filtering underlines and checking
        # that we only update existing keys
        data = data._filter_underscored()
        for k in data.keys():
            assert k in prev_data

        # Finally, increment the step number and update the PRNG
        data = data._replace(
            prng_key=jax.random.fold_in(data.prng_key, 1), step=step + 1
        )
        return data, records

    def _run_update_edges(self, state: ProcessState) -> ProcessState:
        """
        Apply all edge operations and return updated state (incl. prng_key).
        """

        def edge_f(shared_rng_key, state, e_pt, src_pt, tgt_pt):
            "Apply all operations on each edge"
            e_pt = e_pt.copy()
            prng_key = jax.random.fold_in(shared_rng_key, e_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(prng_key, i)
                updates = op.update_edge(
                    self._tr.wrap(
                        EdgeUpdateData(
                            prng_key=r,
                            state=state,
                            edge=e_pt,
                            src_node=src_pt,
                            tgt_node=tgt_pt,
                        )
                    )
                )
                self._tr.log_operation_io(op, updates)
                e_pt.update(updates)
            return e_pt

        # Extract node values for edge endpoints
        n2e_src_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edge["src"]], state.node
        )
        n2e_tgt_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edge["tgt"]], state.node
        )
        self._tr.log_line("Edge updates:")
        new_edge_data = jax.vmap(edge_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(state.prng_key, 0),
            state,
            state.edge,
            n2e_src_pytree,
            n2e_tgt_pytree,
        )
        return state._replace(
            edge=new_edge_data, prng_key=jax.random.fold_in(state.prng_key, 1)
        )

    def _run_update_nodes(self, state: ProcessState) -> ProcessState:
        """
        Apply all node operations and return updated state (incl. prng_key).
        """

        def node_f(
            shared_rng_key, state, n_pt, in_edges_agg, out_edges_agg, both_edges_agg
        ):
            "Apply all operations on each node"
            n_pt = n_pt.copy()
            prng_key = jax.random.fold_in(shared_rng_key, n_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(prng_key, i)
                updates = op.update_node(
                    self._tr.wrap(
                        NodeUpdateData(
                            prng_key=r,
                            state=state,
                            node=n_pt,
                            in_edges=in_edges_agg,
                            out_edges=out_edges_agg,
                            edges=both_edges_agg,
                        )
                    )
                )
                self._tr.log_operation_io(op, updates)
                n_pt.update(updates)
            return n_pt

        def scatter_op(agg_op, zval, e_vals, e_endpoints):
            "Compute given aggregate of e_vals grouped by e_endpoints"
            # Get min/max operation-neutral value for integers and infinities
            # NB: this is numpy - should be static vals in the computation graph
            if np.issubdtype(e_vals.dtype, np.bool_):
                e_vals = jnp.int32(e_vals)
            if np.isneginf(zval) and np.issubdtype(e_vals.dtype, np.integer):
                zval = np.iinfo(e_vals.dtype).min
            elif np.isinf(zval) and np.issubdtype(e_vals.dtype, np.integer):
                zval = np.iinfo(e_vals.dtype).max
            # Array of neutral values
            z = jnp.full(
                state.node["i"].shape + e_vals.shape[1:],
                zval,
                dtype=e_vals.dtype,
            )
            e_endpoints = jnp.where(state.edge["active"], e_endpoints, state.n + 1)
            e_endpoints_exp = jnp.expand_dims(e_endpoints, 1)
            dims = jax.lax.ScatterDimensionNumbers(
                tuple(range(1, len(e_vals.shape))), (0,), (0,)
            )
            return agg_op(z, e_endpoints_exp, e_vals, dims, mode="drop")

        # Compute edge-to-node value aggregates
        in_edges_agg, out_edges_agg, both_edges_agg = {}, {}, {}
        for agg_op, agg_name, zval, comb2 in (
            (jax.lax.scatter_add, "sum", 0, lambda x1, x2: x1 + x2),
            (jax.lax.scatter_mul, "prod", 1, lambda x1, x2: x1 * x2),
            (jax.lax.scatter_min, "min", np.inf, jnp.minimum),
            (jax.lax.scatter_max, "max", -np.inf, jnp.maximum),
        ):
            in_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, zval, a, state.edge["tgt"]),
                state.edge,
            )
            out_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, zval, a, state.edge["src"]),
                state.edge,
            )
            both_edges_agg[agg_name] = jax.tree_util.tree_multimap(
                comb2, in_edges_agg[agg_name], out_edges_agg[agg_name]
            )

        self._tr.log_line("Node updates:")
        new_node_data = jax.vmap(node_f, in_axes=(None, None, 0, 0, 0, 0))(
            jax.random.fold_in(state.prng_key, 0),
            state,
            state.node,
            in_edges_agg,
            out_edges_agg,
            both_edges_agg,
        )
        return state._replace(
            node=new_node_data,
            prng_key=jax.random.fold_in(state.prng_key, 1),
        )

    def _run_update_params(
        self, state: ProcessState, orig_state: ProcessState
    ) -> ProcessState:
        """
        Apply all (global) state updates and return an updated state (incl. prng_key).
        """
        self._tr.log_line("Param updates:")
        state = state.copy()
        for i, op in enumerate(self.operations):
            updates = op.update_params(
                self._tr.wrap(
                    ParamUpdateData(
                        prng_key=jax.random.fold_in(state.prng_key, i + 1),
                        state=state,
                        prev_state=orig_state,
                    )
                )
            )
            self._tr.log_operation_io(op, updates)
            state.update(updates)
        return state._replace(prng_key=jax.random.fold_in(state.prng_key, 0))

    def new_state(
        self,
        network: Network,
        props: PropTree = {},
        *,
        seed=None,
        record_stride: int = 1,
    ) -> ProcessState:
        """
        Create a new ProcessState, also ensuring the required initial properties for all operations.

        State properties are taken from `props` or using defaults.
        `seed` may be jax PRNGKey, a 64 bit number (used as a seed) or None (randomize).
        The given param/sprops are used as overrides over the Network parameters (if any; those are unmodified).
        """
        if seed is None:
            seed = random.randint(0, 1 << 64 - 1)
        if isinstance(seed, (jnp.ndarray, np.ndarray)):
            assert seed.shape == (2,)
            assert seed.dtype == jnp.int32
            prng_key = jnp.array(seed)
        else:
            prng_key = jax.random.PRNGKey(seed)

        # Note: all pytree elements are converted to jax arrays later in the state constructor
        # Note: all pytrees are properly copied later in the state constructor
        sd = ProcessState.from_network(
            network, prng_key, props=props, record_stride=record_stride
        )
        # Prepare state for operations
        for op in self.operations:
            op.prepare_state_data(sd)
        sd._assert_shapes()
        return sd
