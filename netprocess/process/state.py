import dataclasses

import jax
import jax.numpy as jnp
from numpy import float32
from netprocess.utils.prop_tree import PropTree

from netprocess.utils.types import PRNGKey

from ..network import Network
from ..utils import Pytree, PytreeDict, jax_utils, PRNGKey
import typing
from ..utils.prop_tree import PropTree


@jax.tree_util.register_pytree_node_class
class ProcessStateData(PropTree):
    """
    Structure holding array data for a state to be passed to JIT-able functions.

    Properties:
    * n, m, prng_key, step
    * node
      * i, in_deg, out_deg, deg
    * edge
      * src, dst
    """

    n: jnp.int32
    m: jnp.int32
    step: jnp.int32
    prng_key: PRNGKey
    edge: PropTree
    node: PropTree

    @classmethod
    def from_network(
        cls, net: Network, prng_key: PRNGKey, **props
    ) -> "ProcessStateData":
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

        s._assert_shapes()
        return s

    def _assert_shapes(self):
        for k, v in self.node.items():
            assert v.shape[0] == self.n
        for k, v in self.edge.items():
            assert v.shape[0] == self.m

    def _filter_underscored(self) -> "ProcessStateData":
        d2 = {}
        for k in self.keys():
            if not any(kp.startswith("_") for kp in k.split(".")):
                d2[k] = self[k]
        return self.__class__(d2)

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


class ProcessState:
    """
    A high-level process state wrapping `ProcessStateData`.
    """

    def __init__(
        self,
        data: ProcessStateData,
        network: Network,
        process: "NetworkProcess" = None,
        record_chunks=(),
    ):
        """
        Internal constructor, use NetworkProcess.new_state() or self.copy_updated to create.
        """
        self.data = data
        self._network = network
        self._process = process
        self._record_chunks = list(record_chunks)

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError:
            return super().__getattribute__(key)

    def _updated(self, new_data: ProcessStateData, new_records=()):
        """
        Return a copy of self updated with given ProcessStateData and single step records.

        Filters out underscored pytree entries of the update and checks
        that the updated pytree elements (top-level ones) exist in old state.

        `new_records` is an iterable of `record_pytree`s.
        Note that state `m`, `n` and `edges` must stay the same.
        """
        assert isinstance(new_data, ProcessStateData)
        assert isinstance(self.data, ProcessStateData)
        assert self.n == new_data.n
        assert self.m == new_data.m

        return ProcessState(
            data=new_data,
            network=self._network,
            process=self._process,
            record_chunks=list(self._record_chunks) + list(new_records),
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} N={self.n} M={self.m} records={self.count_records()} process={self._process}>"

    def count_records(self):
        if len(self._record_chunks) == 0:
            return 0
        if len(self._record_chunks[0]) == 0:
            return None  # No records - unable to determine
        return int(
            sum(jax.tree_util.tree_leaves(pt)[0].shape[0] for pt in self._record_chunks)
        )

    def last_record(self):
        """
        Return pytree of the last record (as 1x... array)
        """
        if len(self._record_chunks) == 0:
            raise ValueError(f"No records in {self!r}")
        return jax.tree_util.tree_map(lambda a: a[[-1]], self._record_chunks[-1])

    def all_records(self):
        """
        Return the concatenated records pytree.

        Concatenates the records if chunked, replaces the chunks with the single
        resulting chunk.
        """
        if len(self._record_chunks) == 0:
            raise ValueError(f"No records in {self!r}")
        if len(self._record_chunks) > 1:
            self._record_chunks = [jax_utils.concatenate_pytrees(self._record_chunks)]
        return self._record_chunks[0]

    def block_on_all(self):
        """
        Block until all arrays are actually computed (does not copy them to CPU).
        """
        for v in jax.tree_util.tree_leaves(self.data):
            v.block_until_ready()
        if len(self._record_chunks) > 0:
            for v in jax.tree_util.tree_leaves(self._record_chunks[-1]):
                v.block_until_ready()
