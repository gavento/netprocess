import jax.numpy as jnp
from netprocess import seir, networks
import networkx as nx
import jax


def test_sir_model():
    upf = seir.build_sir_update_function(0.1, 0.03)
    upf_jit = jax.jit(upf)

    N = 30
    g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=42)
    edges = networks.nx_graph_to_edges(g)
    node_states = jnp.zeros(N, dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)

    # Few passes without any infections
    for i in range(10):
        k, rng = jax.random.split(rng)
        node_states = upf_jit(k, edges, node_states)
    assert sum(node_states) == 0

    # Infect a single node
    node_states = jax.ops.index_update(node_states, 0, 1)
    for i in range(100):
        k, rng = jax.random.split(rng)
        n2j = upf_jit(k, edges, node_states)
        if i % 20 == -1:
            n2 = upf(k, edges, node_states)
            assert (n2 == n2j).all()
        node_states = n2j
        # print(node_states)
