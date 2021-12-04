import dataclasses

import jax
import jax.numpy as jnp

from netprocess.utils.types import PRNGKey

from ..network import Network
from ..utils import Pytree, PytreeDict, jax_utils, PRNGKey
import typing


@dataclasses.dataclass(frozen=True)
class ProcessStateData:
    """
    Structure holding array data for a state to be passed to JIT-able functions.
    """

    rng_key: PRNGKey
    edges: jnp.ndarray
    params: PytreeDict
    node_props: PytreeDict
    edge_props: PytreeDict
    n: jnp.int32
    m: jnp.int32

    def _replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def copy(self):
        """Copy the state, reusing all the ndarrays and immutable objects."""
        return dataclasses.replace(
            self,
            params=jax.tree_map(lambda x: x, self.params),
            node_props=jax.tree_map(lambda x: x, self.node_props),
            edge_props=jax.tree_map(lambda x: x, self.edge_props),
        )

    @classmethod
    def _pad_pytree_to(_cls, pt: Pytree, old_n, n=None):
        """Internal, pad or shrink first dim of all leaves if `n` is not None.
        Return new first dim size and new pytree."""
        if n is None:
            return old_n, pt
        elif old_n < n:
            return (
                n,
                jax.tree_map(
                    lambda a: jnp.pad(
                        a, [(0, n - old_n)] + ([(0, 0)] * (len(a.shape) - 1))
                    ),
                    pt,
                ),
            )
        else:
            return n, jax.tree_map(lambda a: a[:n], pt)

    def _pad_to(self, n=None, m=None):
        """
        Return a copy of self padded or shortened to
        N and/or M (whichever is given)
        """
        np = jax_utils.pad_pytree_to(self.node_props, self.n, n)
        ep = jax_utils.pad_pytree_to(self.edge_props, self.m, m)
        return self._replace(node_props=np, n=n, edge_props=ep, m=m)


jax.tree_util.register_pytree_node(
    ProcessStateData,
    lambda p: (dataclasses.astuple(p), None),
    lambda _a, d: ProcessStateData(*d),
)


class ProcessState:
    """

    * `params` always contains "n" (# of nodes) and "m" (# of edges)
    * `node_props` always contains "i" (node index)
    * `edge_props` always contains "i" (edge index)
    """

    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        network: Network,
        params={},
        node_props={},
        edge_props={},
        record_chunks=(),
        process=None,
        updated_edges=None,
    ):
        """
        Internal constructor, use NetworkProcess.new_state() or self.copy_updated to create.

        Needs a `network` reference even if the edges were modified (considered to be the original).
        """
        self.rng_key = rng_key
        # Optional NetworkProcess reference
        self._process = process
        # Network - a read only (!) reference
        self._network = network
        # Data - copied and converted to ndarrays in _ensure_ndarrays()
        self.edges = self._network.edges if updated_edges is None else updated_edges
        self.params = params
        self.node_props = node_props
        self.edge_props = edge_props
        # Data sizes
        self.n = jnp.array(self._network.n, dtype=jnp.int32)
        self.params.setdefault("n", self.n)
        self.m = jnp.array(self.edges.shape[0], dtype=jnp.int32)
        self.params.setdefault("m", self.m)
        # Default step sounting
        self.params.setdefault("step", jnp.array(0, dtype=jnp.int32))
        # Ensure nodes and edges have numbers
        if "i" not in self.edge_props:
            self.edge_props["i"] = jnp.arange(self.m, dtype=jnp.int32)
        if "i" not in self.node_props:
            self.node_props["i"] = jnp.arange(self.n, dtype=jnp.int32)
        if "out_deg" not in self.node_props:
            self.node_props["out_deg"] = jnp.bincount(self.edges[:, 0], length=self.n)
        if "in_deg" not in self.node_props:
            self.node_props["in_deg"] = jnp.bincount(self.edges[:, 1], length=self.n)

        # Chunked stats records
        self._record_chunks = list(record_chunks)
        self._ensure_ndarrays()
        self._check_data()

    def _ensure_ndarrays(self, concretize_types=True):
        """Ensure all the data in pytrees are DeviceArrays.

        With concretize_types=True converts all weak_types to strong types,
        integers to int32 and floats to float32, bools to bool_.
        """
        self.edges = jax_utils.ensure_array(self.edges, dtype=jnp.int32)
        self.m = jax_utils.ensure_array(self.m, dtype=jnp.int32)
        self.n = jax_utils.ensure_array(self.n, dtype=jnp.int32)
        self.params = jax_utils.ensure_pytree(
            self.params, concretize_types=concretize_types
        )
        self.node_props = jax_utils.ensure_pytree(
            self.node_props, concretize_types=concretize_types
        )
        self.edge_props = jax_utils.ensure_pytree(
            self.edge_props, concretize_types=concretize_types
        )

    def copy_updated(self, new_state: ProcessStateData, new_records=()):
        """
        Return a copy of self updated with given ProcessStateData.

        Filters out underscored pytree entries of the update and checks
        that the updated pytree elements (top-level ones) exist in old state.

        `new_records` is an iterable of `record_pytree`s.
        Note that state `m`, `n` and `edges` must stay the same.
        """

        assert isinstance(new_state, ProcessStateData)
        assert self.__class__ == ProcessState  # Any extended children may need changes
        assert self.n == new_state.n
        assert self.m == new_state.m
        return ProcessState(
            rng_key=new_state.rng_key,
            network=self._network,
            updated_edges=new_state.edges,
            params=_filter_check_merge(self.params, new_state.params, "params"),
            node_props=_filter_check_merge(
                self.node_props, new_state.node_props, "node_props"
            ),
            edge_props=_filter_check_merge(
                self.edge_props, new_state.edge_props, "edge_props"
            ),
            record_chunks=list(self._record_chunks) + list(new_records),
            process=self._process,
        )

    def _check_data(self):
        assert self.edges.shape == (self.m, 2)
        for a in jax.tree_util.tree_leaves(self.params):
            assert isinstance(a, jnp.DeviceArray)
        for a in jax.tree_util.tree_leaves(self.node_props):
            assert isinstance(a, jnp.DeviceArray)
            assert a.shape[0] == self.n
        for a in jax.tree_util.tree_leaves(self.edge_props):
            assert isinstance(a, jnp.DeviceArray)
            assert a.shape[0] == self.m

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

    def as_pytree(self):
        """
        Return the state data as a named tuple "view" - to use as pytree.

        Does not contain the history.
        """
        return ProcessStateData(
            rng_key=self.rng_key,
            edges=self.edges,
            params=self.params,
            node_props=self.node_props,
            edge_props=self.edge_props,
            n=self.n,
            m=self.m,
        )

    def block_on_all(self):
        """
        Block until all arrays are actually computed (does not copy them to CPU).
        """
        for pt in (self.params, self.edge_props, self.node_props):
            for v in jax.tree_util.tree_leaves(pt):
                v.block_until_ready()
        if len(self._record_chunks) > 0:
            for v in jax.tree_util.tree_leaves(self._record_chunks[-1]):
                v.block_until_ready()


def _filter_check_merge(orig: PytreeDict, update: PytreeDict, name: str):
    "Merge `update` items into a copy of `orig`, skip underscored, check existence."
    update = {k: v for k, v in update.items() if not k.startswith("_")}
    tgt = jax_utils.tree_copy(orig)
    for k, v in update.items():
        if k not in orig:
            raise ValueError(
                f"Key {k} of {name} update not present in orig: {list(orig.keys())}"
            )
        tgt[k] = jax_utils.tree_copy(v)
    return tgt
