import jax
import jax.numpy as jnp
import networkx as nx
import pytest
from jax.numpy import array as a
from netprocess import Network, NetworkProcess, games


def test_policies():
    p = games.EpsilonErrorPolicy(epsilon="eps")
    assert p.compute_policy(
        a([0.0, 1, 2, 1]), NodeUpdateData(state={"eps": 0.4})
    ) == pytest.approx(a([0.1, 0.1, 0.7, 0.1]))
    assert p.compute_policy(
        a([0.0, 1, 2, 1, 0]), NodeUpdateData(state={"eps": 0.0})
    ) == pytest.approx(a([0.0, 0.0, 1.0, 0.0, 0.0]))

    p = games.EpsilonErrorPolicy(epsilon=0.8)
    assert (
        p.compute_policy(
            a([-10.0, -1, -2, -4]),
            NodeUpdateData(state={}),
        )
        == pytest.approx([0.2, 0.4, 0.2, 0.2])
    )

    p = games.SoftmaxPolicy(beta=0.0)
    assert (
        p.compute_policy(
            a([0.0, -10, -1, -2, -4]),
            NodeUpdateData(state={}),
        )
        == pytest.approx(a([0.2, 0.2, 0.2, 0.2, 0.2]))
    )


def test_payoffs():
    p = games.EpsilonErrorPolicy()
    g1 = games.games.NormalFormGameBase(["C", "D"], jnp.array([[4, 0], [5, 1]]), p)
    g2 = games.games.NormalFormGameBase(
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


def test_best_response_game():
    N = 30
    net_g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=42)
    net = Network.from_graph(net_g)

    p = games.SoftmaxPolicy(beta="beta")
    g = games.BestResponseGame(["C", "D"], jnp.array([[4, 0], [5, 1]]), p)
    np = NetworkProcess([g])
    s = np.new_state(net, seed=43, props={"beta": 1.0, "node.next_action": [0] * N})
    s = np.run(s, steps=10, jit=True)
    assert sum(s.node["action"]) > 0.9 * N
    s["beta"] = 0.05
    s = np.run(s, steps=10, jit=True)
    print(s.node["action"])
    assert sum(s.node["action"]) < 0.8 * N
    assert sum(s.node["action"]) > 0.4 * N


def test_best_response_game():
    N = 30
    net_g = nx.random_graphs.barabasi_albert_graph(N, 3, seed=43)
    net = Network.from_graph(net_g)

    g = games.RegretMatchingGame(["C", "D"], jnp.array([[4, 0], [5, 1]]))
    np = NetworkProcess([g])
    s = np.new_state(net, seed=47, props={"node.next_action": [0] * N})
    s = np.run(s, steps=10, jit=True)
    assert sum(s.node["action"]) > 0.9 * N
    s = np.run(s, steps=10, jit=True)
    print(s.node["action"])
    assert sum(s.node["action"]) == N
