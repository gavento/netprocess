import jax.numpy as jnp
import jax
import networkx as nx
from networkx.generators import directed
from netprocess import Network, NetworkProcess
from netprocess.operations import (
    OperationBase,
    NodeUpdateData,
    EdgeUpdateData,
    ParamUpdateData,
)


def _new_state(process):
    net = Network.from_edges(
        n=4,
        edges=jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)]),
        directed=True,
    )

    return process.new_state(
        net,
        seed=42,
        node_props={
            "x": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y": jnp.array([0.1, 0.2, 0.3, 0.4]),
            "indeg": jnp.zeros(net.n, dtype=jnp.int32),
            "outdeg": jnp.zeros(net.n, dtype=jnp.int32),
        },
        edge_props={
            "aa": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "stat": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        },
    )


def test_nop_process():
    np = NetworkProcess([OperationBase()])

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, params={"beta": 1.5})
    sa1 = np.run(sa0, steps=4, jit=False)
    assert sa1.params["beta"] == 1.5

    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=2, jit=False)

    # Look at step separately, set to 0 to allow comparison
    assert sb1.params["step"] == 2
    sb1.params["step"] = 0

    for xpt, ypt in (
        (sb0.node_props, sb1.node_props),
        (sb0.edge_props, sb1.edge_props),
        (sb0.params, sb1.params),
    ):
        assert jax.tree_structure(xpt) == jax.tree_structure(ypt)
        for x, y in zip(jax.tree_leaves(xpt), jax.tree_leaves(ypt)):
            assert (x == y).all()


def test_custom_process():
    # Full message passing
    class TestOp(OperationBase):
        def update_edge(self, data: EdgeUpdateData):
            return {
                "aa": data.edge["aa"] + data.from_node["y"],
                "_nope": 1,
                "_tgt_x": data.to_node["x"],
                "_src_x": data.from_node["x"],
                "_deg": 1,
            }

        def update_node(self, data: NodeUpdateData):
            return {
                "x": data.in_edges["sum"]["_src_x"],
                "y": data.node["y"]
                + jax.lax.cond(
                    data.out_edges["sum"]["_deg"] >= 2,
                    lambda _: 100.0,
                    lambda _: jnp.float32(
                        jax.random.randint(data.rng_key, (), 200, 300)
                    ),
                    None,
                ),
                "indeg": data.in_edges["sum"]["_deg"],
                "outdeg": data.out_edges["sum"]["_deg"],
                "_nope": 0,
            }

        def update_params(self, data: ParamUpdateData):
            a = jnp.sum(data.state.node_props["indeg"])
            return {
                "_a": a,
                "_a_rec": a * data.state.edge_props["aa"][0],
            }

    np = NetworkProcess([TestOp()], record=["_a_rec"])
    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=1)
    print(np.trace_log())

    assert (sb1.node_props["indeg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.node_props["outdeg"] == jnp.array([2, 1, 2, 0])).all()
    assert (sb1.node_props["x"] == jnp.array([[3, 4], [6, 8], [1, 2], [5, 6]])).all()
    # NB: this one depends on the RNG for reproducibility
    assert (sb1.node_props["y"] == jnp.array([100.1, 298.2, 100.3, 285.4])).all()
    assert (sb1.edge_props["stat"] == sb1.edge_props["stat"]).all()
    assert (sb1.edge_props["aa"] == jnp.array([1.1, 2.3, 3.3, 4.1, 5.2])).all()
    assert (sb1.all_records()["_a_rec"] == jnp.array([5.5])).all()
    # Check underscored are ommited
    assert "_nope" not in sb1.node_props
    assert "_nope" not in sb1.edge_props

    # Also test the computed degrees
    assert (sb1.node_props["in_deg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.node_props["out_deg"] == jnp.array([2, 1, 2, 0])).all()
