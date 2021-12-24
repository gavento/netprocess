from attr import frozen
import jax
import jax.numpy as jnp
import pytest
import functools
from netprocess.utils import jax_utils, utils
from netprocess.utils import prop_tree
from netprocess.utils.prop_tree import PropTree
import numpy as np


def a(*els, **kws):
    return jnp.array(els, **kws)


class PT1(PropTree):
    a: PropTree
    b: jnp.ndarray


class PT0(PropTree):
    _OTHER_PROPS = False
    x: PT1
    y: PT1
    w: int


def test_prop_tree_types():
    p1 = PT1(b=(1, 2), a={})
    assert isinstance(p1, PT1)
    assert not isinstance(p1.a, PT1)
    assert isinstance(p1.b, jnp.ndarray)
    p1["nonex"] = 13

    p0 = PT0()
    p0["x.a.foo"] = 42
    assert isinstance(p0.x, PT1)
    assert not isinstance(p0.x, PT0)
    assert not isinstance(p0.x.a, PT1)
    assert not isinstance(p0.x.a, PT0)
    assert p0.x.a["foo"] == 42
    with pytest.raises(AttributeError):
        p0["nonex"] = 13


def test_prop_tree():
    p = PropTree()
    assert len(p) == 0

    p = PropTree({"a.b": 42, "a.c": (2, 3, 4)}, x=False)
    assert p["a"]["b"] == 42
    assert p["a", "b"] == 42
    assert p["a.b"] == 42
    assert p["a.c"][0] == 2
    assert p.top_len() == 2
    assert len(p) == 3
    assert set(p.keys()) == set(["a.b", "a.c", "x"])
    assert len(p["a"]) == 2
    assert len(jax.tree_util.tree_leaves(p)) == 3
    assert jax.tree_util.tree_map(lambda x: x + 1, p)["a.c"][0] == 3

    p["c.d"] = (13,)
    assert p["c"]["d"][0] == 13
    assert "a" in p
    assert "b" in p["a"]
    assert "z" not in p

    def uncalled():
        assert False

    assert p.setdefault("e-2/3", lambda: 18) == 18
    assert p.setdefault("e-2/3", 19) == 18
    assert p.setdefault("e-2/3", uncalled) == 18

    p2 = jax.tree_util.tree_map(lambda x: x, p)
    assert list(p2.items()) == list(p.items())

    assert p.data_eq(p2)
    assert not PropTree(a=1, b=2).data_eq(PropTree(a=1))
    assert not PropTree(a=1).data_eq(PropTree(a=1, b=2))
    assert PropTree(a={"x": 1}).data_eq(PropTree(a={"x": 1}, b={"c": {}}))
    assert PropTree(a={"x": True}).data_eq(PT1(a={"x": True}, z={}))
    assert PropTree(a=1.13).data_eq(PropTree(a=1.13))
    assert not PropTree(a=1.13).data_eq(PropTree(a=1.13001))
    assert PropTree(a=1.13).data_eq(PropTree(a=1.13001), eps=0.001)

    p3 = PropTree(a=1)
    p3["b.c"] = [2, 3]
    p3 = p3.copy(frozen=True)
    p3b = p3["b"]
    with pytest.raises(Exception):
        p3["c"] = 9
    with pytest.raises(Exception):
        p3b["d"] = 3
    assert (p3b["c"] == jnp.array([2, 3])).all()
    p3 = p3.copy(frozen=False)
    p3["c"] = 9
    p3["b"]["e"] = 42
    p3["b.f"] = 43


def test_integrality():
    assert not utils.is_integer(1.0)
    assert utils.is_integer(1)
    assert utils.is_integer(jnp.int16(1))
    assert not utils.is_integer(jnp.float16(1))


def test_update_dicts():
    d = {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        utils.update_dict_disjoint(d, {"a": 2})
    with pytest.raises(ValueError):
        utils.update_dict_present(d, {"c": 2})
    utils.update_dict_disjoint(d, {"c": 2})
    utils.update_dict_present(d, {"a": 3})
    assert d == {"a": 3, "b": 2, "c": 2}


def test_jax_random_choice_range():
    """Ensure that random.choice normalizes the probabilities."""
    k = jax.random.PRNGKey(42)
    s = jax.random.choice(k, 2, (1000,), p=jnp.array([1000.0, 2000.0]))
    assert sum(s) < 900
    assert sum(s) > 300


def test_switch():
    s0 = functools.partial(jax_utils.switch, [lambda: [41, 42], [43, 44]])
    assert s0(0) == [41, 42]
    assert s0(1) == [43, 44]

    s1 = functools.partial(jax_utils.switch, [lambda x: x + 1, lambda _: 42, 43])
    assert s1(0, 3) == 4
    assert s1(1, 0) == 42
    assert s1(2, 0) == 43

    s3 = functools.partial(
        jax_utils.switch, [lambda x, y, z: x + y + z, lambda *_: 42, 43]
    )
    assert s3(0, 3, 4, 5) == 12
    assert s3(1, 0, 1, 2) == 42
    assert s3(2, 0, 0, 0) == 43


def test_apply_scatter_op():
    T, F = True, False
    r = jax_utils.apply_scatter_op(
        jax.lax.scatter_add, 3, a(2, 1, -3, 3), a(2, 0, 0, 2), active=a(1, 1, 1, 0)
    )
    assert (r == a(-2, 0, 2)).all()
    r = jax_utils.apply_scatter_op(
        jax.lax.scatter_max, 3, a(F, T, F, T), a(2, 0, 0, 2), active=a(1, 1, 1, 0)
    )
    assert (r == a(T, F, F)).all()
    r = jax_utils.apply_scatter_op(
        jax.lax.scatter_add, 3, a(F, T, T, T), a(2, 0, 0, 2), active=a(1, 1, 1, 0)
    )
    assert (r == a(2, 0, 0)).all()
    r = jax_utils.apply_scatter_op(jax.lax.scatter_mul, 2, a(2.0), a(1))
    assert (r == a(1.0, 2.0)).all()
    r = jax_utils.apply_scatter_op(jax.lax.scatter_min, 2, a(2.0, 3.0), a(1, 1))
    assert (r == a(np.inf, 2.0)).all()
    assert (r == a(jnp.inf, 2.0)).all()


def test_create_scatter_aggregates():
    T, F = True, False
    pt = prop_tree.PropTree(
        x=a(2, 1, -3, 3), y=a(F, T, T, T), z=a(2.0, 3.0, -np.inf, 1.0)
    )
    r = jax_utils.create_scatter_aggregates(3, pt, a(2, 0, 0, 2), active=a(1, 1, 1, 0))
    assert (r["sum.x"] == a(-2, 0, 2)).all()
    assert (r["sum.y"] == a(2, 0, 0)).all()
    assert (r["max.y"] == a(T, F, F)).all()
    assert (r["min.z"] == a(-np.inf, np.inf, 2.0)).all()
