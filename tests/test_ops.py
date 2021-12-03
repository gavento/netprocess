import networkx as nx
from netprocess import Network, operations, NetworkProcess


def test_time_advancing_op():
    np = NetworkProcess([operations.AdvanceTimeOp()])

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, params={"delta_t": 0.1})
    sa1 = np.run(sa0, steps=4, jit=False)
    assert abs(sa1.params_pytree["t"] - 0.4) < 1e-6

    sb0 = np.new_state(n0, seed=42)
    sb1 = np.run(sb0, steps=5, jit=False)
    assert abs(sb1.params_pytree["t"] - 5.0) < 1e-6
