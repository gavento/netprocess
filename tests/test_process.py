import collections.abc

import jax
import jax.numpy as jnp
import networkx as nx
from netprocess import Network, NetworkProcess
from netprocess.operations import (
    EdgeUpdateData,
    NodeUpdateData,
    OperationBase,
    ParamUpdateData,
)
from networkx.generators import directed


def _new_state(process):
    net = Network.from_edges(
        n=4,
        edges=jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)]),
        directed=True,
    )

    return process.new_state(
        net,
        seed=42,
        props={
            "node": {
                "x": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                "y": jnp.array([0.1, 0.2, 0.3, 0.4]),
                "indeg": jnp.zeros(net.n, dtype=jnp.int32),
                "outdeg": jnp.zeros(net.n, dtype=jnp.int32),
            },
            "edge": {
                "aa": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                "stat": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            },
        },
    )


def test_nop_process():
    np = NetworkProcess([OperationBase()])

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, props={"beta": 1.5})
    sa1 = np.run(sa0, steps=4, jit=False)
    assert sa1.data["beta"] == 1.5

    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=2, jit=False)

    # Look at step separately, set to 0 to allow comparison
    assert sb1.step == 2
    assert not (sb0.prng_key == sb1.prng_key).all()
    sb1.data["step"] = 0
    sb1.data["prng_key"] = sb0.prng_key

    xpt, ypt = sb0.data, sb1.data
    assert jax.tree_structure(xpt) == jax.tree_structure(ypt)
    for x, y in zip(jax.tree_leaves(xpt), jax.tree_leaves(ypt)):
        assert (x == y).all()


def test_custom_process():
    # Full message passing
    class TestOp(OperationBase):
        def update_edge(self, data: EdgeUpdateData):
            return {
                "aa": data.edge["aa"] + data.src_node["y"],
                "_nope": 1,
                "_tgt_x": data.tgt_node["x"],
                "_src_x": data.src_node["x"],
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
                        jax.random.randint(data.prng_key, (), 200, 300)
                    ),
                    None,
                ),
                "indeg": data.in_edges["sum"]["_deg"],
                "outdeg": data.out_edges["sum"]["_deg"],
                "_nope": 0,
            }

        def update_params(self, data: ParamUpdateData):
            a = jnp.sum(data.state.node["indeg"])
            return {
                "_a": a,
                "_a_rec": a * data.state.edge["aa"][0],
            }

    np = NetworkProcess([TestOp()], record_keys=["_a_rec"])
    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=1)
    print(np.trace_log())

    assert (sb1.node["indeg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.node["outdeg"] == jnp.array([2, 1, 2, 0])).all()
    assert (sb1.node["x"] == jnp.array([[3, 4], [6, 8], [1, 2], [5, 6]])).all()
    # NB: this one depends on the RNG for reproducibility
    assert (sb1.node["y"] == jnp.array([100.1, 298.2, 100.3, 285.4])).all()
    assert (sb1.edge["stat"] == sb1.edge["stat"]).all()
    assert (sb1.edge["aa"] == jnp.array([1.1, 2.3, 3.3, 4.1, 5.2])).all()
    assert (sb1.all_records()["_a_rec"] == jnp.array([5.5])).all()
    # Check underscored are ommited
    assert "_nope" not in sb1.node
    assert "_nope" not in sb1.edge
    assert "_nope" not in sb1.data

    # Also test the computed degrees
    assert (sb1.node["in_deg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.node["out_deg"] == jnp.array([2, 1, 2, 0])).all()
