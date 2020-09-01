import jax
import jax.numpy as jnp
from netprocess import networks
import networkx as nx


def test_step_edge_node_messages():
    # Explicit graph edge order
    edges = jnp.array([(0, 2), (2, 1), (2, 3), (0, 1), (1, 0)])
    n = 4
    rng = jax.random.PRNGKey(42)
    nd1 = {
        "x": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "y": jnp.array([0.1, 0.2, 0.3, 0.4]),
        "indeg": jnp.zeros(n, dtype=jnp.int32),
        "outdeg": jnp.zeros(n, dtype=jnp.int32),
    }
    ed1 = {
        "aa": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "stat": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }

    # Smoketest: NOP update
    def ef_a(r, ed, fnd, tnd):
        return {}, {}, {}

    def nf_a(r, nd, ied, oed):
        return {}

    nd_a2, ed_a2 = networks.step_edge_node_messages(rng, nd1, ed1, edges, ef_a, nf_a)
    assert nd_a2.keys() == nd1.keys()
    assert ed_a2.keys() == ed1.keys()
    for k in nd1:
        assert (nd1[k] == nd_a2[k]).all()
    for k in ed1:
        assert (ed1[k] == ed_a2[k]).all()

    # Full message passing
    def ef_b(r, ed, fnd, tnd):
        return (
            {"aa": ed["aa"] + fnd["y"]},
            {"other": tnd["x"], "deg": 1},
            {"other": fnd["x"], "deg": 1},
        )

    def nf_b(r, nd, ieds, oeds):
        return {
            "x": ieds["other"],
            "y": nd["y"]
            + jax.lax.cond(
                oeds["deg"] >= 2,
                lambda _: 100.0,
                lambda _: jnp.float32(jax.random.randint(r, (), 200, 300)),
                None,
            ),
            "indeg": ieds["deg"],
            "outdeg": oeds["deg"],
        }

    nd_b2, ed_b2 = networks.step_edge_node_messages(rng, nd1, ed1, edges, ef_b, nf_b)
    assert nd_b2.keys() == nd1.keys()
    assert ed_b2.keys() == ed1.keys()
    assert (nd_b2["indeg"] == jnp.array([1, 2, 1, 1])).all()
    assert (nd_b2["outdeg"] == jnp.array([2, 1, 2, 0])).all()
    assert (nd_b2["x"] == jnp.array([[3, 4], [6, 8], [1, 2], [5, 6]])).all()
    # NB: this one depends on the RNG for reproducibility
    assert (nd_b2["y"] == jnp.array([100.1, 298.2, 100.3, 228.4])).all()
    assert (ed_b2["stat"] == ed1["stat"]).all()
    assert (ed_b2["aa"] == jnp.array([1.1, 2.3, 3.3, 4.1, 5.2])).all()
