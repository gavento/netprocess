from typing import Callable

import jax
import jax.numpy as jnp

from ..network import Network
from ..utils import PRNGKey, ArrayTree, jax_utils
from .records import ProcessRecords

NodeFn = Callable[["ProcessState", ArrayTree, ArrayTree], ArrayTree]
EdgeFn = Callable[["ProcessState", ArrayTree, ArrayTree, ArrayTree], ArrayTree]
StateFn = Callable[["ProcessState"], ArrayTree]


@jax.tree_util.register_pytree_node_class
class ProcessState(ArrayTree):
    """
    Structure holding array data for a state to be passed to JIT-able functions.

    A subclass of ArrayTree, additionally holding data on the original network
    and the collected records (those are not accessible in JIT-ted code).

    Attribute properties:

    * always contains: `n`, `m`, `prng_key`, `step`
    * `node` - arrays for each node
        * always contains: `i`, `in_deg`, `out_deg`, `deg`, `weight`, `active`
    * `edge` - arrays for each edge
        * always contains: `i`, `src`, `tgt`, `weight`, `active`

    Note: *Inactive edges* do not pass values to nodes (e.g. are not present in aggregates
        like `in_edges.max.foo`) but still compute their updates (so no compute savings).

        *Inactive nodes* still pass value over any *active edges* and they still compute
        their updates (so no compute savings). Therefore `active` on nodes is mostly just a marker.
        Operations and aggregatins need to ignore inactive edges and nodes as appropritate.
        Their values are also recorded.
    """

    __slots__ = ("_network", "_records", "_record_set")

    @property
    def n(self) -> jnp.ndarray:
        return self["n"]

    @property
    def m(self) -> jnp.ndarray:
        return self["m"]

    @property
    def step(self) -> jnp.ndarray:
        return self["step"]

    @property
    def prng_key(self) -> PRNGKey:
        return self["prng_key"]

    @property
    def edge(self) -> ArrayTree:
        return self["edge"]

    @property
    def node(self) -> ArrayTree:
        return self["node"]

    @property
    def network(self) -> Network:
        if self._network is None:
            raise Exception(f"self.network is not available")
        return self._network

    @property
    def records(self) -> ProcessRecords:
        if self._records is None:
            raise Exception(f"self.records is not available")
        return self._records

    @classmethod
    def from_network(
        cls, net: Network, prng_key: PRNGKey, record_stride: int = 1, props: dict = {}
    ) -> "ProcessState":
        s = cls(**props)
        s["n"] = net.n
        s["m"] = net.m
        s["step"] = 0
        s["prng_key"] = prng_key

        s["edge.i"] = jnp.arange(s.m, dtype=jnp.int32)
        s["edge.src"] = net.edges[:, 0]
        s["edge.tgt"] = net.edges[:, 1]
        s["edge.weight"] = jnp.ones(net.m, dtype=jnp.float32)
        s["edge.active"] = jnp.ones(net.m, dtype=jnp.bool_)

        s["node.i"] = jnp.arange(s.n, dtype=jnp.int32)
        s["node.in_deg"] = jnp.bincount(s.edge["tgt"], length=net.n)
        s["node.out_deg"] = jnp.bincount(s.edge["src"], length=net.n)
        s["node.deg"] = s.node["in_deg"] + s.node["out_deg"]
        s["node.weight"] = jnp.ones(net.n, dtype=jnp.float32)
        s["node.active"] = jnp.ones(net.n, dtype=jnp.bool_)

        s._records = ProcessRecords(stride=record_stride)
        s._network = net
        s._record_set = ArrayTree()
        s._assert_shapes()
        return s

    def next_prng_key(self) -> PRNGKey:
        """Return a new PRNG key, also advancing `state.prng_key`."""
        self["prng_key"], prng2 = jax.random.split(self.prng_key)
        return prng2

    def apply_node_fn(self, node_fn: NodeFn):
        """Apply `node_fn` to every node, updating `self` inplace.

        Updates `self.node` properties and advances `self.prng_key`.
        Call for each node gets its own independent `state.prng_key`.
        JIT-able.

        Args:
            edge_fn (EdgeFn): function applied to every edge. Must be JIT-able.
        """
        edges = self._aggregate_edges().copy(frozen=True)
        self.node.update(
            jax.vmap(
                lambda state, pk, node, edges: ArrayTree(
                    node_fn(state.copy(replacing={"prng_key": pk}), node, edges)
                ),
                in_axes=(None, 0, 0, 0),
            )(
                self._bare_copy(frozen=True),
                jax.random.split(self.next_prng_key(), self["node.i"].shape[0]),
                self.node.copy(frozen=True),
                edges,
            ).copy(
                frozen=False
            )
        )

    def apply_edge_fn(self, edge_fn: EdgeFn):
        """Apply `edge_fn` to every edge, updating `self` inplace.

        Updates `self.edge` properties and advances `self.prng_key`.
        Call for each edge gets its own independent `state.prng_key`.
        JIT-able.

        Args:
            edge_fn (EdgeFn): function applied to every edge.
        """
        assert not self._frozen
        assert not self.edge._frozen
        src_nodes = jax.tree_util.tree_map(
            lambda a: a[self.edge["src"]], self.node
        ).copy(frozen=True)
        tgt_nodes = jax.tree_util.tree_map(
            lambda a: a[self.edge["tgt"]], self.node
        ).copy(frozen=True)
        assert not self._frozen
        assert not self.edge._frozen
        r = jax.vmap(
            lambda state, pk, edge, src, tgt: ArrayTree(
                edge_fn(state.copy(replacing={"prng_key": pk}), edge, src, tgt)
            ),
            in_axes=(None, 0, 0, 0, 0),
        )(
            self._bare_copy(frozen=True),
            jax.random.split(self.next_prng_key(), num=self["edge.i"].shape[0]),
            self.edge.copy(frozen=True),
            # self.edge["i"],
            src_nodes,
            # self.edge["i"],
            tgt_nodes,
        ).copy(
            frozen=False
        )
        assert not self._frozen
        assert not self.edge._frozen
        self.edge.update(r)

    def tree_flatten(self):
        f, a = super().tree_flatten()
        return (f, (a, self._network, self._records, self._record_set))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        a, _network, _records, _record_set = aux_data
        pt = super().tree_unflatten(a, children)
        assert isinstance(pt, ProcessState)
        pt._network, pt._records, pt._record_set = _network, _records, _record_set
        if pt._records is not None:
            pt._records = pt._records.copy()
        return pt

    def record_value(self, key, value=None):
        """Add a record to the current record set of the current step.

        Accepts both a key of an existing value (not of a subtree),
        or the given value for arbitrary key.

        Should be used within operations as the record set is collected
        and actually recorded at the end of every step.
        Does nothing on its own outside of `process.run()`.
        """
        if value is None:
            value = self[key]
        assert isinstance(value, jnp.ndarray)
        assert key not in self._record_set
        self._record_set[key] = value

    def _take_record_set(self) -> ArrayTree:
        r = self._record_set
        self._record_set = ArrayTree()
        return r.copy(frozen=True)

    def block_on_all(self):
        """Block until all arrays are actually computed (e.g. for timing).

        This does not copy the arrays to CPU from GPU.
        """
        self._records.block_on_all()
        for v in self.leaf_values():
            v.block_until_ready()

    def _assert_shapes(self):
        for v in self.node.leaf_values():
            assert v.shape[0] == self.n
        for v in self.edge.leaf_values():
            assert v.shape[0] == self.m

    def _filter_underscored(self) -> "ProcessState":
        """Returns an unfrozen copy with all `_underscored` keys removed (incl. underscored subtrees).

        Also removes any empty subtrees."""
        s2 = self.copy(frozen=False)
        for k in tuple(s2.leaf_keys()):
            if k in s2 and any(kp.startswith("_") for kp in k.split(".")):
                del s2[k]
        # Deeper nodes go before shallower ones in this order
        for k, v in sorted(tuple(s2._iter_subtrees()), reverse=True):
            if k in s2 and len(v) == 0:
                del s2[k]

        return s2

    def _pad_to(self, n: int = None, m: int = None):
        """Return a copy of self padded or shortened to N and/or M (whichever is given)"""
        s = self.copy()
        if n is not None:
            s = s._replace(n=n, node=jax_utils.pad_pytree_to(s.node, self.n, n))
        if m is not None:
            s = s._replace(m=m, node=jax_utils.pad_pytree_to(s.edge, self.m, m))
        return s

    def _aggregate_edges(self) -> ArrayTree:
        """Compute edge-to-node value aggregates"""
        in_edges_agg = jax_utils.create_scatter_aggregates(
            self.node["i"].shape[0],
            self.edge,
            self.edge["tgt"],
            self.edge["active"],
        )
        out_edges_agg = jax_utils.create_scatter_aggregates(
            self.node["i"].shape[0],
            self.edge,
            self.edge["src"],
            self.edge["active"],
        )
        COMB_2 = {
            "min": jnp.minimum,
            "max": jnp.maximum,
            "sum": lambda x, y: x + y,
            "prod": lambda x, y: x * y,
        }
        both_edges_agg = {
            agg_name: jax.tree_util.tree_multimap(
                comb2, in_edges_agg[agg_name], out_edges_agg[agg_name]
            )
            for agg_name, comb2 in COMB_2.items()
        }
        return ArrayTree(
            {"in": in_edges_agg, "out": out_edges_agg, "all": both_edges_agg}
        )

    def _bare_copy(self, frozen=None):
        """Create a copy without `_network` and `_records`"""
        s = self.copy(frozen=frozen)
        s._network, s._records, s._record_set = None, None, None
        return s
