import jax
import jax.numpy as jnp
import netprocess
import pytest
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
