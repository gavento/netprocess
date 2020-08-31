import jax
import jax.numpy as jnp
from netprocess import networks
import networkx as nx


def test_nx_and_gather_neighbors():
    g = nx.Graph([(0, 2), (2, 1), (2, 3)])
    n = g.order()
    assert n == 4
    edges = networks.nx_graph_to_edges(g)
    assert edges.shape == (6, 2)
    node_data = jnp.array([(0, 1), (1, 1), (2, 1), (3, 1)])
    s = networks.sum_from_inneighbors(edges, node_data)
    assert (s == jnp.array([(2, 1), (2, 1), (4, 3), (2, 1)])).all()


def test_count_states():
    g = nx.Graph([(0, 2), (2, 1), (2, 3), (0, 1)])
    edges = networks.nx_graph_to_edges(g)
    node_states = jnp.array([0, 1, 0, 2])
    s = networks.count_inneighbor_states(edges, node_states, 3)
    assert (s == jnp.array([(1, 1, 0), (2, 0, 0), (1, 1, 1), (1, 0, 0)])).all()
