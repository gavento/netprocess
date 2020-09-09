import jax.numpy as jnp
import networkx as nx
from .utils import is_integer


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
