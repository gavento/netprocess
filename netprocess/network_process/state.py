import collections

import jax
import jax.numpy as jnp

from .. import jax_utils


ProcessStateData = collections.namedtuple(
    "ProcessStateData",
    ["rng_key", "edges", "params_pytree", "nodes_pytree", "edges_pytree", "n", "m"],
)


class ProcessState:
    """

    * `params_pytree` always contains "n" (# of nodes) and "m" (# of edges)
    * `nodes_pytree` always contains "i" (node index)
    * `edges_pytree` always contains "i" (edge index)
    """

    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        n: int,
        edges: jnp.DeviceArray,
        params_pytree={},
        nodes_pytree={},
        edges_pytree={},
        _record_chunks=(),
        process=None,
    ):
        self.rng_key = rng_key
        # Data
        self.edges = jax_utils.ensure_array(edges)
        self.params_pytree = jax_utils.ensure_pytree(params_pytree)
        self.nodes_pytree = jax_utils.ensure_pytree(nodes_pytree)
        self.edges_pytree = jax_utils.ensure_pytree(edges_pytree)
        # Data sizes
        self.n = jnp.array(n, dtype=jnp.int32)
        self.params_pytree.setdefault("n", self.n)
        self.m = jnp.array(self.edges.shape[0], dtype=jnp.int32)
        self.params_pytree.setdefault("m", self.m)
        # Ensure nodes and edges have numbers
        if "i" not in self.edges_pytree:
            self.edges_pytree["i"] = jnp.arange(self.m, dtype=jnp.int32)
        if "i" not in self.nodes_pytree:
            self.nodes_pytree["i"] = jnp.arange(self.n, dtype=jnp.int32)
        # Optional Process reference
        self.process = process
        # Chunked stats records
        self._record_chunks = list(_record_chunks)
        self._check_data()

    def copy_updated(self, sd: ProcessStateData, new_records=()):
        """
        Return a copy of self updated with ProcessStateData.

        `new_records` is an iterable of `record_pytree`s.
        Note that m, n and edges should stay the same.
        """
        assert isinstance(sd, ProcessStateData)
        assert self.__class__ == ProcessState
        assert self.n == sd.n
        assert self.m == sd.m
        return ProcessState(
            rng_key=sd.rng_key,
            n=sd.n,
            edges=sd.edges,
            params_pytree=jax.tree_util.tree_map(lambda x: x, sd.params_pytree),
            nodes_pytree=jax.tree_util.tree_map(lambda x: x, sd.nodes_pytree),
            edges_pytree=jax.tree_util.tree_map(lambda x: x, sd.edges_pytree),
            _record_chunks=list(self._record_chunks) + list(new_records),
            process=self.process,
        )

    def _check_data(self):
        assert self.edges.shape == (self.m, 2)
        for a in jax.tree_util.tree_leaves(self.params_pytree):
            assert isinstance(a, jnp.DeviceArray)
        for a in jax.tree_util.tree_leaves(self.nodes_pytree):
            assert isinstance(a, jnp.DeviceArray)
            assert a.shape[0] == self.n
        for a in jax.tree_util.tree_leaves(self.edges_pytree):
            assert isinstance(a, jnp.DeviceArray)
            assert a.shape[0] == self.m

    def __repr__(self):
        return f"<{self.__class__.__name__} N={self.n} M={self.m} records={self.count_records()} process={self.process}>"

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

    def as_pytree(self):
        """
        Return the state data as a named tuple "view" - to use as pytree.

        Does not contain the history.
        """
        return ProcessStateData(
            rng_key=self.rng_key,
            edges=self.edges,
            params_pytree=self.params_pytree,
            nodes_pytree=self.nodes_pytree,
            edges_pytree=self.edges_pytree,
            n=self.n,
            m=self.m,
        )
