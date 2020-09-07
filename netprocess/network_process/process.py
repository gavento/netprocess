import logging
import random
import time
import typing

import jax
import jax.numpy as jnp
import networkx as nx
from jax.random import PRNGKey

from .. import networks
from .state import ProcessState, ProcessStateData

log = logging.getLogger(__name__)


class NetworkProcess:
    def __init__(self, operations):
        self.operations = tuple(operations)
        self._run_jit = jax.jit(self._run)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.operations}>"

    def run(self, state: ProcessState, steps=1):
        steps = jnp.array(steps, dtype=jnp.int32)
        state_data = state.as_pytree()
        new_state, records = self._run_jit(state_data, steps)
        return state.copy_updated(new_state, [records])

    def warmup(self, n=16, block=True):
        """Force the compilation of the JITted run function and wait for it (if `block`)."""
        assert n >= 2
        state = self.new_state([[0, 1], [1, 0]], n=n, seed=42)
        # Run the jitted function
        t0 = time.time()
        new_state, records = self.run(state, 1)
        # Wait for all computations
        if block:
            for pytree in [
                records,
                new_state.params_pytree,
                new_state.nodes_pytree,
                new_state.edges_pytree,
            ]:
                for a in jax.tree_util.tree_leaves(pytree):
                    a.block_until_ready()
            t1 = time.time()
            log.debug(f"Warmup of {self} (n={n}) took {t1-t0:.2f}s")

    def _run(self, state: ProcessStateData, steps: jnp.int32):
        """Returns (new_state, all_records_pytree)"""
        step_array = jnp.zeros((steps, 1))
        return jax.lax.scan(lambda s, _: self._run_step(s), state, step_array)

    def _run_step(self, state: ProcessStateData):
        """Returns (new_state, record_pytree)"""
        return state, {}  ########### TODO

        # Randomness
        edge_rng_key, node_rng_key = jax.random.split(rng_key)
        edge_rng_keys = jax.random.split(edge_rng_key, num=m)
        node_rng_keys = jax.random.split(node_rng_key, num=n)
        # Extract node values for edges
        n2e_from_dict = {k: nodes_dict[k][e_from] for k in nodes_dict}
        n2e_to_dict = {k: nodes_dict[k][e_to] for k in nodes_dict}
        # Compute edge operations and messages
        edge_update_dict, from_update_dict, to_update_dict = jax.vmap(edge_f)(
            edge_rng_keys, edges_dict, n2e_from_dict, n2e_to_dict
        )
        # Compute node input values
        def collect_sum(e_vals, e_endpoints, k=None):
            z = jnp.zeros((n,) + e_vals.shape[1:], dtype=e_vals.dtype)
            e_endpoints_exp = jnp.expand_dims(e_endpoints, 1)
            dims = lax.ScatterDimensionNumbers(
                tuple(range(1, len(e_vals.shape))), (0,), (0,)
            )
            # print("CSUM", k, e_vals.shape, e_endpoints.shape, z.shape, e_endpoints_exp.shape, dims)
            return lax.scatter_add(z, e_endpoints_exp, e_vals, dims)

        inedge_updates_sum = {
            k: collect_sum(to_update_dict[k], e_to, k) for k in to_update_dict
        }
        outedge_updates_sum = {
            k: collect_sum(from_update_dict[k], e_from, k) for k in from_update_dict
        }
        # Compute edge operations and messages
        node_update_dict = jax.vmap(node_f)(
            node_rng_keys, nodes_dict, inedge_updates_sum, outedge_updates_sum
        )

        # Combine node values
        new_nodes_dict = dict(nodes_dict)
        new_nodes_dict.update(node_update_dict)
        new_edges_dict = dict(edges_dict)
        new_edges_dict.update(edge_update_dict)
        return new_nodes_dict, new_edges_dict

    def new_state(
        self,
        edges_or_graph: typing.Union[jnp.DeviceArray, nx.Graph],
        n: int = None,
        *,
        seed=None,
        params={},
        nodes_pytree={},
        edges_pytree={},
    ):
        """
        Create a new ProcessState with initial pytree elements for all operations.

        `seed` may be jax PRNGKey, a 64 bit number (used as a seed) or None (randomize).
        """
        if seed is None:
            seed = random.randint(0, 1 << 64 - 1)
        if isinstance(seed, PRNGKey):
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
            params_pytree=params,
            nodes_pytree=nodes_pytree,
            edges_pytree=edges_pytree,
            process=self,
        )
        for op in self.operations:
            op.prepare_state_pytrees(state)
        return state
