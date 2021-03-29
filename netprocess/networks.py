import json

import attr
import jax.numpy as jnp
import networkx as nx
import numpy as np
import filelock

from .utils import is_integer


@attr.s(auto_attribs=True)
class Network:
    """
    Saved as:
    "NAME.json" - metadata dict
    "NAME.npy" - numpy array shaped (N, 2)
    "NAME.graphml.bz2" - compressed GraphML (optional)
    "NAME.json.lock" - metadata lockfile
    """

    _nx_graph: nx.Graph = None
    meta: dict = attr.Factory(dict)
    edges: np.ndarray = None
    base_path: str = None

    @classmethod
    def load(cls, base_path):
        pass

    @classmethod
    def from_nx_graph(cls, g, base_path):
        pass

    def nx_graph(self):
        if self._nx_graph is None:
            # nx.readwrite.graphml.write_graphml
            self._nx_graph = nx.readwrite.graphml.read_graphml(
                f"{self.base_path}.graphml.bz2"
            )
        return self._nx_graph

    def write_meta(self):
        with open(f"{self.base_path}.json", "wt") as f:
            json.dump(self.meta, f)

    def write_np(self):
        np.save(f"{self.base_path}.npy", self.edges)


def nx_graph_to_edges(graph: nx.Graph, dtype=jnp.int32, sort="from") -> jnp.DeviceArray:
    """
    Convert a graph or a digraph to (from, to) pairs in Nx2 numpy array.

    Graph nodes must be integers 0..n-1 already.
    Edges are sorted by "from".
    Undirected graphs get both edge directions added.
    """

    def s(a):
        if sort == "from":
            return sorted(a)
        elif sort == "to":
            return sorted(a, key=lambda x: x[1])
        elif sort is None or sort == False:
            return a
        else:
            raise ValueError("Invalus `sort` param")

    n = graph.order()
    assert all(is_integer(x) for x in graph.nodes)
    assert all(x >= 0 and x < n for x in graph.nodes)
    if graph.is_directed():
        return jnp.array(s(graph.edges()))
    return jnp.array(s(list(graph.edges()) + [(y, x) for x, y in graph.edges()]))
