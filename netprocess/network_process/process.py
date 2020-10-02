import logging
import random
import time
import typing

import jax
import jax.numpy as jnp
import networkx as nx

from .. import networks
from ..utils import PRNGKey, PytreeDict
from .tracing import Tracer
from .state import ProcessState, ProcessStateData, _filter_check_merge

log = logging.getLogger(__name__)


class NetworkProcess:
    def __init__(self, operations):
        from .operation import OperationBase

        self.operations = tuple(operations)
        assert all(isinstance(op, OperationBase) for op in self.operations)
        self._run_jit = jax.jit(self._run, static_argnums=[2])
        self._traced = 0
        self._tr = Tracer(tracing=False)
        self._tr.log_line("<Never traced>")

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.operations}>"

    def run(self, state: ProcessState, steps=1, jit=True) -> ProcessState:
        steps_array = jnp.zeros((steps, 1))
        state_data = state.as_pytree()
        if jit:
            state_update, records = self._run_jit(state_data, steps_array, True)
        else:
            state_update, records = self._run(state_data, steps_array, False)
        return state.copy_updated(state_update, [records])

    def trace_log(self):
        return f"Traced {self._traced} times, last log:\n{self._tr.get_log()}"

    def warmup_jit(self, state=None, n=None, m=None, block=True):
        """Force the compilation of the JITted run function and wait for it (if `block`)."""
        if state is None:
            assert n >= 2 and m >= 1
            state = self.new_state([[0, 1]] * m, n=n, seed=42)
        else:
            assert n is None and m is None
        # Run the jitted function
        new_state = self.run(state, steps=1)
        # Wait for all computations
        if block:
            new_state.block_on_all()

    def _run(
        self, state: ProcessStateData, steps_array: jnp.DeviceArray, tracing: bool
    ):
        """Returns (new_state, all_records_pytree). JIT-able."""
        if tracing:
            msg = f"Tracing {self} with n={state.n}, m={state.m}, steps={steps_array.shape[0]}"
            self._tr = Tracer(tracing=True)
            self._traced += 1
            log.debug(msg)
        return jax.lax.scan(
            lambda s, _: self._run_step(s),
            state,
            steps_array,
        )

    def _run_step(self, state: ProcessStateData):
        """Returns (new_state, record_pytree). JIT-able."""

        # Original state, never updated
        orig_state = state

        # Run all the update steps, updating the state
        state = self._run_update_edges(state)
        state = self._run_update_nodes(state)
        state = self._run_update_params(state, orig_state)
        # Note: this folds in values 1..(#ops) to the state rng
        records = self._run_create_record(state, orig_state)

        # Create the new state, filtering underlines and checking
        # that we only update existing keys
        state = state._replace(
            rng_key=jax.random.fold_in(state.rng_key, 0),
            params_pytree=_filter_check_merge(
                orig_state.params_pytree, state.params_pytree, "params"
            ),
            nodes_pytree=_filter_check_merge(
                orig_state.nodes_pytree, state.nodes_pytree, "node"
            ),
            edges_pytree=_filter_check_merge(
                orig_state.edges_pytree, state.edges_pytree, "edge"
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
                    r,
                    self._tr.TD(params, "P"),
                    self._tr.TD(e_pt, "E"),
                    self._tr.TD(from_pt, "fromN"),
                    self._tr.TD(to_pt, "toN"),
                )
                self._tr.log_operation_io(op, updates)
                e_pt.update(updates)
            return e_pt

        # Extract node values for edge endpoints
        n2e_from_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 0]], state.nodes_pytree
        )
        n2e_to_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 1]], state.nodes_pytree
        )

        self._tr.log_line("Edge updates:")
        new_edges_pytree = jax.vmap(edge_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(state.rng_key, 0),
            state.params_pytree,
            state.edges_pytree,
            n2e_from_pytree,
            n2e_to_pytree,
        )
        return state._replace(
            edges_pytree=new_edges_pytree, rng_key=jax.random.fold_in(state.rng_key, 1)
        )

    def _run_update_nodes(self, state: ProcessStateData) -> ProcessStateData:
        """
        Apply all operation node updates on state.
        """

        def node_f(shared_rng_key, params, n_pt, in_edges_agg, out_edges_agg):
            "Apply all operations on each node"
            n_pt = n_pt.copy()
            rng_key = jax.random.fold_in(shared_rng_key, n_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(rng_key, i)
                updates = op.update_node(
                    r,
                    self._tr.TD(params, "P"),
                    self._tr.TD(n_pt, "N"),
                    self._tr.TD(in_edges_agg, "inE", depth=2),
                    self._tr.TD(out_edges_agg, "outE", depth=2),
                )
                self._tr.log_operation_io(op, updates)
                n_pt.update(updates)
            return n_pt

        def scatter_op(agg_op, agg_base, e_vals, e_endpoints):
            "Compute given aggregate of e_vals grouped by e_endpoints"
            # Get neutral value of the op from the op on an empty array
            zval = agg_base(jnp.array((), dtype=e_vals.dtype))
            # Array of neutral values
            z = jnp.full(
                state.nodes_pytree["i"].shape + e_vals.shape[1:], zval, dtype=zval.dtype
            )
            e_endpoints_exp = jnp.expand_dims(e_endpoints, 1)
            dims = jax.lax.ScatterDimensionNumbers(
                tuple(range(1, len(e_vals.shape))), (0,), (0,)
            )
            return agg_op(z, e_endpoints_exp, e_vals, dims)

        # Compute edge-to-node value aggregates
        in_edges_agg, out_edges_agg = {}, {}
        for agg_op, agg_name, agg_base in (
            (jax.lax.scatter_add, "sum", jnp.sum),
            (jax.lax.scatter_mul, "prod", jnp.prod),
            (jax.lax.scatter_min, "min", jnp.min),
            (jax.lax.scatter_max, "max", jnp.max),
        ):
            in_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, agg_base, a, state.edges[:, 1]),
                state.edges_pytree,
            )
            out_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(
                    agg_op,
                    agg_base,
                    a,
                    state.edges[:, 0],
                ),
                state.edges_pytree,
            )

        self._tr.log_line("Node updates:")
        new_nodes_pytree = jax.vmap(node_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(state.rng_key, 0),
            state.params_pytree,
            state.nodes_pytree,
            in_edges_agg,
            out_edges_agg,
        )
        return state._replace(
            nodes_pytree=new_nodes_pytree,
            rng_key=jax.random.fold_in(state.rng_key, 1),
        )

    def _run_update_params(
        self, state: ProcessStateData, orig_state: ProcessStateData
    ) -> ProcessStateData:
        """
        Compute parameter updates.
        """
        self._tr.log_line("Param updates:")
        new_params_pytree = state.params_pytree.copy()
        for i, op in enumerate(self.operations):
            updates = op.update_params(
                jax.random.fold_in(state.rng_key, i + 1),
                self._tr.TS(state),
                self._tr.TS(orig_state, "orig"),
            )
            self._tr.log_operation_io(op, updates)
            new_params_pytree.update(updates)
        return state._replace(
            params_pytree=new_params_pytree,
            rng_key=jax.random.fold_in(state.rng_key, 0),
        )

    def _run_create_record(
        self, state: ProcessStateData, orig_state: ProcessStateData
    ) -> PytreeDict:
        "Create records from all operations for this step."
        self._tr.log_line("Creating records:")
        records = {}
        for i, op in enumerate(self.operations):
            updates = op.create_record(
                jax.random.fold_in(state.rng_key, i + 1),
                self._tr.TS(state),
                self._tr.TS(orig_state, "orig"),
            )
            self._tr.log_operation_io(op, updates)
            records.update(updates)
        return records

    def new_state(
        self,
        edges_or_graph: typing.Union[jnp.DeviceArray, nx.Graph],
        n: int = None,
        *,
        seed=None,
        params_pytree={},
        nodes_pytree={},
        edges_pytree={},
    ):
        """
        Create a new ProcessState with initial pytree elements for all operations.

        `seed` may be jax PRNGKey, a 64 bit number (used as a seed) or None (randomize).
        """
        if seed is None:
            seed = random.randint(0, 1 << 64 - 1)
        if isinstance(seed, jnp.DeviceArray):
            assert seed.shape == (2,)
            assert seed.dtype == jnp.int32
            rng_key = seed
        else:
            rng_key = jax.random.PRNGKey(seed)

        if isinstance(edges_or_graph, nx.Graph):
            edges = networks.nx_graph_to_edges(edges_or_graph)
            if edges_pytree:
                raise ValueError(
                    "non-empty edges_pytree while passing a graph is unstable"
                )
            if n is not None:
                assert n == edges_or_graph.order()
            n = edges_or_graph.order()
        else:
            edges = jnp.array(edges_or_graph, dtype=jnp.int32)
            if n is None:
                raise ValueError("n is needed when only edge-list is given")
            assert (edges < n).all()

        state = ProcessState(
            rng_key,
            n,
            edges,
            params_pytree=params_pytree,
            nodes_pytree=nodes_pytree,
            edges_pytree=edges_pytree,
            process=self,
        )
        # Prepare state for operations
        for op in self.operations:
            op.prepare_state_pytrees(state)
        # Ensure the operaions added only ndarrays
        state._ensure_ndarrays()
        return state
