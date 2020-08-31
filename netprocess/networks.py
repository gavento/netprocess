import networkx as nx
import jax.numpy as jnp
import jax.lax as lax
from .utils import is_integer


def nx_graph_to_edges(graph: nx.Graph, dtype=jnp.int32) -> jnp.ndarray:
    """
    Convert a graph or a digraph to (from, to) pairs in Nx2 numpy array.

    Graph nodes must be integers already (but do not have to be a full range).
    Edges are sorted by "from".
    Undirected graphs get both edge directions added.
    """
    assert all(is_integer(x) for x in graph.nodes)
    if graph.is_directed():
        return jnp.array(sorted(graph.edges()))
    return jnp.array(sorted(graph.edges()) + sorted((y, x) for x, y in graph.edges()))


def sum_from_inneighbors(edges: jnp.ndarray, node_data: jnp.ndarray) -> jnp.ndarray:
    """
    Edges is a Nx2 array of (from, to), node_data is Nxk array, result in Nxk.
    """
    n, k = node_data.shape
    assert edges.shape[1] == 2
    return lax.scatter_add(
        jnp.zeros((n, k), dtype=node_data.dtype),
        edges[:, [1]],
        node_data[edges[:, 0]],
        lax.ScatterDimensionNumbers((1,), (0,), (0,)),
    )
