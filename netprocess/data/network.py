import json
import pickle
from typing import Any

import attr
import jax.numpy as jnp
import networkx as nx
import numpy as np

from .data_utils import load_object, save_object
from .network_stats import powerlaw_exp_fp, powerlaw_exp_vl


@attr.s
class Network:
    """
    Saved as:

    "NAME.pickle[.zstd]" - pickle-v4 serialized dict of:
        * `edges: np.ndarray` shaped (M, 2) (edges in both directions for undirected graphs)
        * `params_pytree: dict` of np.ndarrays, any shape
        * `nodes_pytree: dict` of np.ndarrays, shaped (N, ...)
        * `edges_pytree: dict` of np.ndarrays, shaped (M, ...)
        * `meta: dict` of anything, in particular has:
            * 'n', 'm'
            * 'gen_seed' used for generating
            * 'gen_type' name of generator
            * 'gen_*' any generator params
            * 'degree_mean'
            * 'degree_var'
            * 'clustering_mean'
            * 'clustering_var'
            * 'clustering_global'
    """

    edges = attr.ib(type=np.ndarray)
    meta = attr.ib(type=dict)
    params_pytree = attr.ib(factory=dict, type=dict)
    nodes_pytree = attr.ib(factory=dict, type=dict)
    edges_pytree = attr.ib(factory=dict, type=dict)
    ATTRS = ("params_pytree", "nodes_pytree", "edges_pytree", "meta", "edges")

    @property
    def n(self):
        return self.meta["n"]

    @property
    def m(self):
        return self.meta["m"]

    def __attrs_post_init__(self):
        assert "n" in self.meta
        self.meta.setdefault("m", self.edges.shape[0])
        assert self.edges.shape == (self.m, 2)

    @classmethod
    def from_graph(cls, g, meta=None, dtype=np.int32, with_stats=True):
        """
        Convert a graph or a digraph to (from, to) pairs in Mx2 numpy array.

        Graph nodes must be integers 0..n-1 already.
        Edges are sorted by "from".
        Undirected graphs get both edge directions added.
        """
        # TODO: assert all nodes are integers 0..n-1, or remap to this

        edges = np.array(g.edges(), dtype=dtype)
        if not isinstance(g, nx.DiGraph):
            edges = np.concatenate((edges, edges[:, 1::-1]))
        eorder = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[eorder]

        if meta is None:
            meta = {}
        else:
            meta = dict(meta)

        n = g.order()
        meta["n"] = n
        meta["m"] = edges.shape[0]

        s = cls(meta=meta, edges=edges)
        if with_stats:
            s.compute_stats(g)
        return s

    @classmethod
    def load(cls, path):
        d = load_object(path)
        assert frozenset(d.keys()) == frozenset(cls.ATTRS)
        # Set it here if the network was moved after generation, also to be more
        # useful for storing resulting state
        d["meta"]["network_path"] = path
        return cls(**d)

    def write(self, path):
        save_object({k: getattr(self, k) for k in self.ATTRS}, path)

    def compute_stats(self, g):
        assert not isinstance(g, nx.DiGraph)

        # triangle and degree count for every vertex
        ts, ds = np.zeros(self.n), np.zeros(self.n, dtype=self.edges.dtype)
        for v, d, t, _ in nx.algorithms.cluster._triangles_and_degree_iter(g):
            ts[v] = t
            ds[v] = d

        self.meta["degree_mean"] = np.mean(ds)
        self.meta["degree_var"] = np.var(ds)

        self.meta.setdefault("powerlaw_min_k", max(np.min(ds), 1.0))
        self.meta["powerlaw_exp_vl"] = powerlaw_exp_vl(
            ds, self.meta["powerlaw_min_k"], discrete=True
        )
        self.meta["powerlaw_exp_fp"] = powerlaw_exp_fp(
            ds, self.meta["powerlaw_min_k"], discrete=True
        )

        cs = ts / np.maximum(ds * (ds - 1), 1.0)
        self.meta["clustering_mean"] = np.mean(cs)
        self.meta["clustering_var"] = np.var(cs)
        self.meta["clustering_global"] = np.sum(ts) / np.maximum(
            np.sum(ds * (ds - 1)), 1.0
        )
