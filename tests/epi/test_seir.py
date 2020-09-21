import jax
import jax.numpy as jnp
import networkx as nx
from netprocess import epi, network_process


def test_sir_model():
    N = 30
    g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=42)
    np = network_process.NetworkProcess([epi.SIRUpdateOp()])

    s = np.new_state(g, params_pytree={"edge_beta": 0.05, "gamma": 0.1}, seed=43)

    # Few passes without any infections
    s = np.run(s, steps=2)
    print(np.trace_log())
    assert sum(s.nodes_pytree["compartment"]) == 0

    # Infect a single high-degree node
    s.nodes_pytree["compartment"] = jax.ops.index_update(
        s.nodes_pytree["compartment"], 0, 1
    )

    # Infection spread
    for i in range(5):
        s = np.run(s, steps=5)
        print(s.nodes_pytree["compartment"])
