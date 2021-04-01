import json
import pickle
from typing import Any

import jax.numpy as jnp
import networkx as nx
import numpy as np

# import powerlaw
import zstd


def powerlaw_exp_vl(degrees, k_min=1.0, discrete=False):
    """
    Estimate powerlaw exponent using "vannila" sampling from https://arxiv.org/pdf/1908.00310.pdf
    """
    ds = np.array(degrees)
    if discrete:
        k_min = k_min - 0.5
    ds = ds[ds >= k_min]
    return len(ds) / np.sum(np.log(ds / k_min)) + 1.0


def powerlaw_exp_fp(degrees, k_min=1.0, discrete=False):
    """
    Estimate powerlaw exponent using "friendship paradox" sampling from https://arxiv.org/pdf/1908.00310.pdf
    """
    ds = np.array(degrees)
    if discrete:
        k_min = k_min - 0.5
    ds = ds[ds >= k_min]
    return np.sum(ds) / np.sum(ds * np.log(ds / k_min)) + 2.0


class Network:
    """
    Saved as:

    "NAME.pickle[.zstd]" - pickle-v4 serialized dict of:
        * `edges: np.ndarray` shaped (M, 2) (edges in both directions for undirected graphs)
        * `params_pytree: dict` of np.ndarrays
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

    def __init__(self, data):
        self.data = dict(data)
        assert "edges" in self.data
        assert isinstance(self.data["edges"], np.ndarray)
        assert "meta" in self.data
        assert self.data["edges"].shape == (self.data["meta"]["m"], 2)
        self.data.setdefault("params_pytree", {})
        self.data.setdefault("nodes_pytree", {})
        self.data.setdefault("edges_pytree", {})

    def __getattribute__(self, name: str) -> Any:
        if name in ("params_pytree", "nodes_pytree", "edges_pytree", "meta", "edges"):
            return self.data[name]
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any):
        if name in ("params_pytree", "nodes_pytree", "edges_pytree", "meta", "edges"):
            self.data[name] = value
        super().__setattr__(name, value)

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

        if with_stats:
            assert not isinstance(g, nx.DiGraph)

            # triangle and degree count for every vertex
            ts, ds = np.zeros(n), np.zeros(n, dtype=dtype)
            for v, d, t, _ in nx.algorithms.cluster._triangles_and_degree_iter(g):
                ts[v] = t
                ds[v] = d

            meta["degree_mean"] = np.mean(ds)
            meta["degree_var"] = np.var(ds)

            pf = powerlaw.Fit(
                ds,
                discrete=True,
                xmin=meta.get("powerlaw_min_k"),
                xmax=meta.get("powerlaw_max_k"),
            )
            meta["powerlaw_jeff_exp"] = pf.power_law.alpha
            meta["powerlaw_jeff_xmin"] = pf.power_law.xmin
            meta["powerlaw_jeff_xmax"] = pf.power_law.xmax

            meta.setdefault("powerlaw_min_k", max(np.min(ds), 1.0))
            meta["powerlaw_exp_vl"] = powerlaw_exp_vl(
                ds, meta["powerlaw_min_k"], discrete=True
            )
            meta["powerlaw_exp_fp"] = powerlaw_exp_fp(
                ds, meta["powerlaw_min_k"], discrete=True
            )

            cs = ts / np.maximum(ds * (ds - 1), 1.0)
            meta["clustering_mean"] = np.mean(cs)
            meta["clustering_var"] = np.var(cs)
            meta["clustering_global"] = np.sum(ts) / np.maximum(
                np.sum(ds * (ds - 1)), 1.0
            )

        return cls(dict(meta=meta, edges=edges))

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            if path.endswith(".zstd"):
                return cls(pickle.loads(zstd.decompress(f.read())))
            else:
                return cls(pickle.load(f.read))

    def write(self, path):
        with open(path, "wb") as f:
            if path.endswith(".zstd"):
                f.write(zstd.compress(pickle.dumps(self.data, protocol=4)))
            else:
                pickle.dump(self.data, f, protocol=4)
