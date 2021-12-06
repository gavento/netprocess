from typing import Any

import jax
import jax.numpy as jnp

from ..network import Network
from ..utils import PRNGKey, PropTree, jax_utils
from .records import ProcessRecords


@jax.tree_util.register_pytree_node_class
class ProcessState(PropTree):
    """
    Structure holding array data for a state to be passed to JIT-able functions. A subclass of PropTree.

    State is almost always meant as **immutable**.
    Prefer to use e.g. `state._replace({"a.b": 42}, c=3.14, d={'e':2.7})`, or copy() it and modify only right after.

    Attribute properties:

    * always contains: `n`, `m`, `prng_key`, `step`
    * `node` - arrays for each node
        * always contains: `i`, `in_deg`, `out_deg`, `deg`, `weight`, `active`
    * `edge` - arrays for each edge
        * always contains: `i`, `src`, `tgt`, `weight`, `active`

    The state property tree may have other top-level properties (e.g. parameters), as well as any nested `PropTree`.
    """

    __slots__ = ("_network", "_records")

    n: jnp.int32
    m: jnp.int32
    step: jnp.int32
    prng_key: PRNGKey
    edge: PropTree
    node: PropTree

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
        s._assert_shapes()
        return s

    @property
    def network(self) -> Network:
        if self._network is None:
            raise Exception(f"self.network is not available in NetworkProcess.run()")
        return self._network

    @property
    def records(self) -> ProcessRecords:
        if self._records is None:
            raise Exception(f"self.records is not available in NetworkProcess.run()")
        return self._records

    def _assert_shapes(self):
        for k, v in self.node.items():
            assert v.shape[0] == self.n
        for k, v in self.edge.items():
            assert v.shape[0] == self.m

    def _filter_underscored(self) -> "ProcessState":
        s2 = self.copy()
        for k in tuple(self.keys()):
            if any(kp.startswith("_") for kp in k.split(".")):
                s2._delitem_f(k)
        return s2

    def _pad_to(self, n: int = None, m: int = None):
        """
        Return a copy of self padded or shortened to
        N and/or M (whichever is given)
        """
        s = self.copy()
        if n is not None:
            s = s._replace(n=n, node=jax_utils.pad_pytree_to(s.node, self.n, n))
        if m is not None:
            s = s._replace(m=m, node=jax_utils.pad_pytree_to(s.edge, self.m, m))
        return s

    def tree_flatten(self):
        assert isinstance(self, ProcessState)
        f, a = super().tree_flatten()
        return (f, (a, self._network, self._records))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        a, _network, _records = aux_data
        pt = super().tree_unflatten(a, children)
        assert isinstance(pt, ProcessState)
        pt._network = _network
        pt._records = _records
        if pt._records is not None:
            pt._records = pt._records.copy()
        return pt

    def block_on_all(self):
        """
        Block until all arrays are actually computed (does not copy them to CPU).
        """
        self._records.block_on_all()
        for v in jax.tree_util.tree_leaves(self):
            v.block_until_ready()
