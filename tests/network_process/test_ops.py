import jax
import jax.numpy as jnp
import networkx as nx
from netprocess import network_process
from netprocess.data import Network, network
from netprocess.network_process import operations


def test_time_advancing_op():
    np = network_process.NetworkProcess([operations.AdvanceTimeOp()])

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, params_pytree={"delta_t": 0.1})
    sa1 = np.run(sa0, steps=4, jit=False)
    assert abs(sa1.params_pytree["t"] - 0.4) < 1e-6

    sb0 = np.new_state(n0, seed=42)
    sb1 = np.run(sb0, steps=5, jit=False)
    assert abs(sb1.params_pytree["t"] - 5.0) < 1e-6
