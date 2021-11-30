import jax
import jax.numpy as jnp
from jax.numpy import array as a
import networkx as nx
import pytest
from netprocess import network_process
from netprocess.data import Network, network
from netprocess.network_process import operations
from netprocess.network_process.operations import games


def test_policies():
    p = games.EpsilonErrorPolicy(epsilon="eps")
    assert p.compute_policy(
        a([0.0, 1, 2, 1]), None, {"eps": 0.4}, {}, {}, {}
    ) == pytest.approx(a([0.1, 0.1, 0.7, 0.1]))
    assert p.compute_policy(
        a([0.0, 1, 2, 1, 0]), None, {"eps": 0.0}, {}, {}, {}
    ) == pytest.approx(a([0.0, 0.0, 1.0, 0.0, 0.0]))

    p = games.EpsilonErrorPolicy(epsilon=0.8)
    assert p.compute_policy(
        a([-10.0, -1, -2, -4]), None, {}, {}, {}, {}
    ) == pytest.approx([0.2, 0.4, 0.2, 0.2])

    p = games.SoftmaxPolicy(beta=0.0)
    assert p.compute_policy(
        a([0.0, -10, -1, -2, -4]), None, {}, {}, {}, {}
    ) == pytest.approx(a([0.2, 0.2, 0.2, 0.2, 0.2]))


def test_payoffs():
    p = games.EpsilonErrorPolicy()
    g1 = games.DiscreteGame(["C", "D"], jnp.array([[4, 0], [5, 1]]), p)
    g2 = games.DiscreteGame(
        ["C", "D"], jnp.array([[[4, 4], [0, 5]], [[5, 0], [1, 1]]]), p
    )
    assert g1.payouts.value.shape != g2.payouts.value.shape
    assert g1.get_payoff(0, 0, 0) == 4
    assert g1.get_payoff(0, 1, 0) == 0
    assert g1.get_payoff(0, 1, 1) == 5
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            for pl in [0, 1]:
                assert g1.get_payoff(a1, a2, pl) == g2.get_payoff(a1, a2, pl)


def test_discrete_game():
    N = 30
    net_g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=42)
    net = Network.from_graph(net_g)

    p = games.SoftmaxPolicy(beta="beta")
    g = games.DiscreteGame(["C", "D"], jnp.array([[4, 0], [5, 1]]), p)
    np = network_process.NetworkProcess([g])
    s = np.new_state(net, seed=43, params_pytree={"beta": 1.0})
    s = np.run(s, steps=10, jit=True)
    assert sum(s.nodes_pytree["action"]) > 0.99 * N
    s.params_pytree["beta"] = 0.05
    s = np.run(s, steps=10, jit=True)
    print(s.nodes_pytree["action"])
    assert sum(s.nodes_pytree["action"]) < 0.8 * N
    assert sum(s.nodes_pytree["action"]) > 0.5 * N
