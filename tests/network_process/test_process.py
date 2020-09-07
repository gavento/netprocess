from netprocess import network_process
import networkx as nx


def test_nop_process():
    np = network_process.NetworkProcess([])
    np.warmup()
    _s0 = np.new_state([(0, 1), (1, 2)], n=3)
    s1 = np.new_state(nx.complete_graph(4), seed=32, params_pytree={"beta": 1.5})
    s2 = np.run(s1, steps=4)
    assert s2.params_pytree["beta"] == 1.5
