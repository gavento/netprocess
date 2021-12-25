import jax
import jax.numpy as jnp
import networkx as nx
from netprocess import Network, NetworkProcess, operations
from netprocess.operations import OperationBase
from netprocess.process import ProcessRecords, ProcessState
from netprocess.utils.prop_tree import PropTree


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


def test_active_flag():
    net = Network.from_edges(
        n=4,
        edges=jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)]),
        directed=True,
    )

    def op(state: ProcessState):
        state.apply_edge_fn(lambda state, edge, src, tgt: {"si": 1})
        # state.apply_edge_fn(lambda state, edge, src, tgt: {"si": 1 + src["i"]})
        # state.apply_node_fn(lambda state, node, edges: {"x": edges["in.sum.si"]})

    np = NetworkProcess([op])
    s0 = np.new_state(
        net,
        props={
            "node.x": jnp.zeros(net.n, dtype=jnp.int32),
            "edge.si": jnp.zeros(net.m, dtype=jnp.int32),
        },
    )
    s1 = np.run(s0, steps=1, jit=False)
    assert (s1.node["x"] == jnp.array([2, 4, 1, 3])).all()

    s1.edge["active"] = jnp.array([True, False, True, True, False])
    s2 = np.run(s1, steps=1, jit=False)
    assert (s2.node["x"] == jnp.array([0, 1, 1, 3])).all()


def test_state_as_pytree():
    s = ProcessState(x=(1, 2), node={}, edge={"weight": (1, 1, 1, 1)}, n=3, m=4)
    s._network = {"foo": "bar"}
    s._records = ProcessRecords()
    s2 = jax.tree_util.tree_map(lambda x: x, s)
    assert type(s) == type(s2)
    assert s._network is s2._network
    assert (s["x"] == s2["x"]).all()
    assert (s.edge["weight"] == s2.edge["weight"]).all()


def test_nop_process():
    np = NetworkProcess(
        [OperationBase(), lambda state: None, lambda state, old_state: None]
    )

    n0 = Network.from_graph(nx.complete_graph(4))
    sa0 = np.new_state(n0, seed=32, props={"beta": 1.5})

    sa1 = np.run(sa0, steps=4, jit=False)
    assert sa1["beta"] == 1.5

    sb0 = _new_state(np)
    sb1 = np.run(sb0, steps=2, jit=False)

    # Look at step separately, set to 0 to allow comparison
    assert sb1.step == 2
    assert (sb0.prng_key == sb1.prng_key).all()
    sb1["step"] = 0
    sb1["prng_key"] = sb0.prng_key
    assert sb0.data_eq(sb1)


def test_records():
    pr = ProcessRecords(stride=1)
    pr.add_record(PropTree(a=[1, 2, 3]))
    assert len(pr) == 3
    assert pr.steps == 3
    pr.add_record(PropTree(a=[4, 5]))
    pr.add_record(PropTree(a=[]))
    assert len(pr) == 5
    assert pr.last_record()["a"] == 5
    pra = pr.all_records()
    assert isinstance(pra, PropTree)
    assert pra.data_eq(PropTree(a=[1, 2, 3, 4, 5]))

    print("second")
    pr = ProcessRecords(stride=3)
    pr.add_record(PropTree(a=[0, 1, 2, 3, 4]))
    assert len(pr) == 2
    assert pr.steps == 5
    pr.add_record(PropTree(a=[]))
    pr.add_record(PropTree(a=[5]))
    pr.add_record(PropTree(a=[6]))
    pr.add_record(PropTree(a=[7, 8, 9]))
    pr.add_record(PropTree(a=[10]))
    pr.add_record(PropTree(a=[11, 12, 13]))
    pra = pr.all_records()
    print(pra)
    assert pra.data_eq(PropTree(a=[0, 3, 6, 9, 12]))


def test_branching_records():
    np = NetworkProcess(
        [operations.Fun(params_f=lambda data: {"_x": data.state.step})],
        record_keys=["_x"],
    )
    n0 = Network.from_graph(nx.complete_graph(4))
    s0 = np.new_state(n0, seed=32)
    s1 = np.run(s0, steps=3, jit=False)
    s2a = np.run(s1, steps=2, jit=False)
    s2b = np.run(s1, steps=1, jit=False)
    assert (s2a.records.all_records()["_x"] == jnp.array([0, 1, 2, 3, 4])).all()
    assert (s2b.records.all_records()["_x"] == jnp.array([0, 1, 2, 3])).all()


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
    assert (sb1.records.all_records()["_a_rec"] == jnp.array([5.5])).all()
    # Check underscored are ommited
    assert "_nope" not in sb1.node
    assert "_nope" not in sb1.edge
    assert "_nope" not in sb1

    # Also test the computed degrees
    assert (sb1.node["in_deg"] == jnp.array([1, 2, 1, 1])).all()
    assert (sb1.node["out_deg"] == jnp.array([2, 1, 2, 0])).all()
