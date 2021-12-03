import jax.numpy as jnp
import networkx as nx
from netprocess import NetworkProcess, Network, epidemics, operations


def test_si_model():
    N = 10
    g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=42)
    net = Network.from_graph(g)
    np = NetworkProcess([epidemics.SIUpdateOp()])
    s = np.new_state(net, params={"edge_infection_rate": 0.3}, seed=43)

    # Few passes without any infections
    print(s.nodes_pytree["compartment"])
    s = np.run(s, steps=3)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"]) == 0

    # Infect a single high-degree node
    s.nodes_pytree["compartment"] = jnp.array([1] + [0] * (s.n - 1))

    # Infection spread
    s = np.run(s, steps=3)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"]) < 10
    assert sum(s.nodes_pytree["compartment"]) > 4
    assert np._traced == 1  ## Not essential, testing tracing-once property


def test_sir_model():
    N = 10
    g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=46)
    net = Network.from_graph(g)
    np = NetworkProcess([epidemics.SIRUpdateOp(), operations.AdvanceTimeOp()])
    s = np.new_state(
        net, params={"edge_infection_rate": 0.6, "recovery_rate": 0.5}, seed=43
    )

    # Few passes without any infections
    print(s.nodes_pytree["compartment"])
    s = np.run(s, steps=4)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"]) == 0

    # Infect a single high-degree node
    s.nodes_pytree["compartment"] = jnp.array([1] + [0] * (s.n - 1))

    # Infection spread
    s = np.run(s, steps=4)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"] == 0) in range(2, 5)
    assert sum(s.nodes_pytree["compartment"] == 1) in range(3, 7)
    assert sum(s.nodes_pytree["compartment"] == 2) in range(1, 6)
    assert abs(s.params_pytree["t"] - 8.0) < 1e-3
    assert np._traced == 1  ## Not essential, testing tracing-once property


def test_seir_model():
    N = 10
    g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=47)
    net = Network.from_graph(g)
    np = NetworkProcess([epidemics.SEIRUpdateOp(immunity_loss=True)])
    s = np.new_state(
        net,
        params={
            "edge_expose_rate": 0.7,
            "infectious_rate": 1.0,
            "recovery_rate": 0.8,
            "immunity_loss_rate": 0.1,
        },
        seed=46,
    )

    # Few passes without any infections
    print(s.nodes_pytree["compartment"])
    s = np.run(s, steps=5)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"]) == 0

    # Infect a single high-degree node
    s.nodes_pytree["compartment"] = jnp.array([1] + [0] * (s.n - 1))

    # Infection spread
    s = np.run(s, steps=5)
    print(s.nodes_pytree["compartment"])
    assert sum(s.nodes_pytree["compartment"] == 0) in range(2, 5)
    assert sum(s.nodes_pytree["compartment"] == 1) in range(2, 5)
    assert sum(s.nodes_pytree["compartment"] == 2) in range(2, 5)
    assert sum(s.nodes_pytree["compartment"] == 3) in range(1, 3)
    assert np._traced == 1  ## Not essential, testing tracing-once property
