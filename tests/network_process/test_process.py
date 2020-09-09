import jax.numpy as jnp
import jax
import networkx as nx
from netprocess import network_process


def _new_state(process):
    n = 4
    return process.new_state(
        jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)]),
        n=4,
        seed=42,
        nodes_pytree={
            "x": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y": jnp.array([0.1, 0.2, 0.3, 0.4]),
            "indeg": jnp.zeros(4, dtype=jnp.int32),
            "outdeg": jnp.zeros(4, dtype=jnp.int32),
        },
        edges_pytree={
            "aa": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "stat": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        },
    )


def test_nop_process():
    np = network_process.NetworkProcess([network_process.OperationBase()])
    np.warmup()

    np.new_state([(0, 1), (1, 2)], n=3)

    sa0 = np.new_state(nx.complete_graph(4), seed=32, params_pytree={"beta": 1.5})
    sa1 = np.run(sa0, steps=4)
    assert sa1.params_pytree["beta"] == 1.5

    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=1)
    for xpt, ypt in (
        (sb0.nodes_pytree, sb1.nodes_pytree),
        (sb0.edges_pytree, sb1.edges_pytree),
        (sb0.params_pytree, sb1.params_pytree),
    ):
        assert jax.tree_structure(xpt) == jax.tree_structure(ypt)
        for x, y in zip(jax.tree_leaves(xpt), jax.tree_leaves(ypt)):
            assert (x == y).all()


def test_message_process():
    # Full message passing
    class TestOp(network_process.OperationBase):
        def update_edge(self, rng_key, params, edge, from_node, to_node):
            return (
                {"aa": edge["aa"] + from_node["y"]},
                {"other": to_node["x"], "deg": 1},
                {"other": from_node["x"], "deg": 1},
            )

        def update_node(self, rng_key, params, node, in_edges, out_edges):
            return {
                "x": in_edges["sum"]["other"],
                "y": node["y"]
                + jax.lax.cond(
                    out_edges["sum"]["deg"] >= 2,
                    lambda _: 100.0,
                    lambda _: jnp.float32(jax.random.randint(rng_key, (), 200, 300)),
                    None,
                ),
                "indeg": in_edges["sum"]["deg"],
                "outdeg": out_edges["sum"]["deg"],
            }

    np = network_process.NetworkProcess([TestOp()])
    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=1)

    assert (sb1.nodes_pytree["indeg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.nodes_pytree["outdeg"] == jnp.array([2, 1, 2, 0])).all()
    assert (sb1.nodes_pytree["x"] == jnp.array([[3, 4], [6, 8], [1, 2], [5, 6]])).all()
    # NB: this one depends on the RNG for reproducibility
    assert (sb1.nodes_pytree["y"] == jnp.array([100.1, 287.2, 100.3, 227.4])).all()
    assert (sb1.edges_pytree["stat"] == sb1.edges_pytree["stat"]).all()
    assert (sb1.edges_pytree["aa"] == jnp.array([1.1, 2.3, 3.3, 4.1, 5.2])).all()
