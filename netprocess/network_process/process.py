import logging
import random
import time
import typing

import jax
import jax.numpy as jnp
import networkx as nx
from jax.random import PRNGKey

from .. import networks, utils
from .state import ProcessState, ProcessStateData, _filter_check_merge

log = logging.getLogger(__name__)


class NetworkProcess:
    def __init__(self, operations):
        from .operation import OperationBase

        self.operations = tuple(operations)
        assert all(isinstance(op, OperationBase) for op in self.operations)
        self._run_jit = jax.jit(self._run, static_argnums=[2])
        self._traced = 0
        self._trace_log = []

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
        return f"Traced {self._traced} times, last log:\n" + (
            "\n".join(self._trace_log)
        )

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

    def _run(self, state: ProcessStateData, steps_array: jnp.int32, tracing: bool):
        """Returns (new_state, all_records_pytree). JIT-able."""
        if tracing:
            self._traced += 1
            self._trace_log = []
        log.debug(
            f"Tracing {self} with n={state.n}, m={state.m}, steps={len(steps_array)}"
        )
        return jax.lax.scan(
            lambda s, _: self._run_step(s, tracing=tracing), state, steps_array
        )

    def _run_step(self, state: ProcessStateData, tracing=False):
        """Returns (new_state, record_pytree). JIT-able."""

        # Randomness - split into new key, forget the old one
        step_rng_key, new_rng_key = jax.random.split(state.rng_key)
        state = state._replace(rng_key=None)
        orig_state = state

        accessed = set()

        def tdict(d, l):
            "Helper to wrap dicts in tracing dict"
            if tracing:
                return utils.TracingDict(d, accessed, f"{l}[", "]")
            else:
                return d

        # Step 1: Extract node values for edge endpoints
        n2e_from_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 0]], state.nodes_pytree
        )
        n2e_to_pytree = jax.tree_util.tree_map(
            lambda a: a[state.edges[:, 1]], state.nodes_pytree
        )

        if tracing:
            self._trace_log.append(
                f"Param keys: {', '.join(state.params_pytree.keys())}"
            )
            self._trace_log.append(
                f"Node keys:  {', '.join(state.nodes_pytree.keys())}"
            )
            self._trace_log.append(
                f"Edge keys:  {', '.join(state.edges_pytree.keys())}"
            )

        # Step 2: Compute edge operations and messages
        def edge_f(shared_rng_key, params, e_pt, from_pt, to_pt):
            "Apply all operations on each edge"
            e_pt = e_pt.copy()
            rng_key = jax.random.fold_in(shared_rng_key, e_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(rng_key, i)
                accessed.clear()
                updates = op.update_edge(
                    r,
                    tdict(params, "P"),
                    tdict(e_pt, "E"),
                    tdict(from_pt, "fromN"),
                    tdict(to_pt, "toN"),
                )
                if updates and tracing:
                    self._trace_log.append(
                        f"  {op}: {', '.join(sorted(accessed))} -> {', '.join(updates.keys())}"
                    )
                e_pt.update(updates)
            return e_pt

        if tracing:
            self._trace_log.append("Edge updates:")
        new_edges_pytree = jax.vmap(edge_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(step_rng_key, 0),
            state.params_pytree,
            state.edges_pytree,
            n2e_from_pytree,
            n2e_to_pytree,
        )
        state = state._replace(edges_pytree=new_edges_pytree)

        # Step 3: Compute edge-to-node value aggregates
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

        in_edges_agg, out_edges_agg = {}, {}
        for agg_op, agg_name, agg_base in (
            (jax.lax.scatter_add, "sum", jnp.sum),
            (jax.lax.scatter_mul, "prod", jnp.prod),
            (jax.lax.scatter_min, "min", jnp.min),
            (jax.lax.scatter_max, "max", jnp.max),
        ):
            in_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(agg_op, agg_base, a, state.edges[:, 1]),
                new_edges_pytree,
            )
            out_edges_agg[agg_name] = jax.tree_util.tree_map(
                lambda a: scatter_op(
                    agg_op,
                    agg_base,
                    a,
                    state.edges[:, 0],
                ),
                new_edges_pytree,
            )

        # Step 4: Compute node operations and updates
        def node_f(shared_rng_key, params, n_pt, in_edges_agg, out_edges_agg):
            "Apply all operations on each node"
            n_pt = n_pt.copy()
            rng_key = jax.random.fold_in(shared_rng_key, n_pt["i"])
            for i, op in enumerate(self.operations):
                r = jax.random.fold_in(rng_key, i)
                accessed.clear()
                in_edges_agg = {
                    k: tdict(v, f"inE[{k}]") for k, v in in_edges_agg.items()
                }
                out_edges_agg = {
                    k: tdict(v, f"outE[{k}]") for k, v in out_edges_agg.items()
                }
                updates = op.update_node(
                    r, tdict(params, "P"), tdict(n_pt, "N"), in_edges_agg, out_edges_agg
                )
                if updates and tracing:
                    self._trace_log.append(
                        f"  {op}: {', '.join(sorted(accessed))} -> {', '.join(updates.keys())}"
                    )
                n_pt.update(updates)
            return n_pt

        if tracing:
            self._trace_log.append("Node updates:")
        new_nodes_pytree = jax.vmap(node_f, in_axes=(None, None, 0, 0, 0))(
            jax.random.fold_in(step_rng_key, 1),
            state.params_pytree,
            state.nodes_pytree,
            in_edges_agg,
            out_edges_agg,
        )
        state = state._replace(nodes_pytree=new_nodes_pytree)

        # Step 5: Compute param updates
        if tracing:
            self._trace_log.append("Param updates:")
        new_params_pytree = state.params_pytree.copy()
        for op in self.operations:
            updates = op.update_params(
                jax.random.fold_in(step_rng_key, 2),
                state,
                orig_state,
            )
            if updates and tracing:
                self._trace_log.append(f"  {op}: ? -> {', '.join(updates.keys())}")
            new_params_pytree.update(updates)
        state = state._replace(params_pytree=new_params_pytree)

        # Step 6: Create any records
        if tracing:
            self._trace_log.append("Creating records:")
        records = {}
        for op in self.operations:
            updates = op.create_record(
                jax.random.fold_in(step_rng_key, 2),
                state,
                orig_state,
            )
            if updates and tracing:
                self._trace_log.append(f"  {op}: ? -> {', '.join(updates.keys())}")
            records.update(updates)

        # Create the new state, filtering underlines and checking
        # that we only update existing keys
        state = state._replace(
            rng_key=new_rng_key,
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
        return state, [records]

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
            rng_key = PRNGKey(seed)

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
