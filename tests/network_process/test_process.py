import jax.numpy as jnp
import jax
import networkx as nx
from netprocess import network_process
from netprocess.data import Network, network


def _new_state(process):
    net = Network(
        edges=jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)]), meta=dict(n=4)
    )

    return process.new_state(
        net,
        seed=42,
        nodes_pytree={
            "x": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y": jnp.array([0.1, 0.2, 0.3, 0.4]),
            "indeg": jnp.zeros(net.n, dtype=jnp.int32),
            "outdeg": jnp.zeros(net.n, dtype=jnp.int32),
        },
        edges_pytree={
            "aa": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "stat": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        },
    )


def test_nop_process():
    np = network_process.NetworkProcess([network_process.OperationBase()])

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, params_pytree={"beta": 1.5})
    sa1 = np.run(sa0, steps=4, jit=False)
    assert sa1.params_pytree["beta"] == 1.5

    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=2, jit=False)

    # Look at step separately, set to 0 to allow comparison
    assert sb1.params_pytree["step"] == 2
    sb1.params_pytree["step"] = 0

    for xpt, ypt in (
        (sb0.nodes_pytree, sb1.nodes_pytree),
        (sb0.edges_pytree, sb1.edges_pytree),
        (sb0.params_pytree, sb1.params_pytree),
    ):
        assert jax.tree_structure(xpt) == jax.tree_structure(ypt)
        for x, y in zip(jax.tree_leaves(xpt), jax.tree_leaves(ypt)):
            assert (x == y).all()


def test_custom_process():
    # Full message passing
    class TestOp(network_process.OperationBase):
        def update_edge(self, rng_key, params, edge, from_node, to_node):
            return {
                "aa": edge["aa"] + from_node["y"],
                "_nope": 1,
                "_tgt_x": to_node["x"],
                "_src_x": from_node["x"],
                "_deg": 1,
            }

        def update_node(self, rng_key, params, node, in_edges, out_edges):
            return {
                "x": in_edges["sum"]["_src_x"],
                "y": node["y"]
                + jax.lax.cond(
                    out_edges["sum"]["_deg"] >= 2,
                    lambda _: 100.0,
                    lambda _: jnp.float32(jax.random.randint(rng_key, (), 200, 300)),
                    None,
                ),
                "indeg": in_edges["sum"]["_deg"],
                "outdeg": out_edges["sum"]["_deg"],
                "_nope": 0,
            }

        def update_params(self, rng_key, state, orig_state):
            return {"_a": jnp.sum(state.nodes_pytree["indeg"])}

        def create_record(self, rng_key, state, orig_state):
            return {"a_rec": state.params_pytree["_a"] * state.edges_pytree["aa"][0]}

    np = network_process.NetworkProcess([TestOp()])
    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=1)
    print(np.trace_log())

    assert (sb1.nodes_pytree["indeg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.nodes_pytree["outdeg"] == jnp.array([2, 1, 2, 0])).all()
    assert (sb1.nodes_pytree["x"] == jnp.array([[3, 4], [6, 8], [1, 2], [5, 6]])).all()
    # NB: this one depends on the RNG for reproducibility
    assert (sb1.nodes_pytree["y"] == jnp.array([100.1, 298.2, 100.3, 285.4])).all()
    assert (sb1.edges_pytree["stat"] == sb1.edges_pytree["stat"]).all()
    assert (sb1.edges_pytree["aa"] == jnp.array([1.1, 2.3, 3.3, 4.1, 5.2])).all()
    assert (sb1.all_records()["a_rec"] == jnp.array([5.5])).all()
    # Check underscored are ommited
    assert "_nope" not in sb1.nodes_pytree
    assert "_nope" not in sb1.edges_pytree
