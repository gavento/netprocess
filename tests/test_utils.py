import jax
import jax.numpy as jnp
import pytest
import functools
from netprocess.utils import jax_utils, utils
from netprocess.utils.prop_tree import PropTree


def test_prop_tree():
    p = PropTree()
    assert len(p) == 0

    p = PropTree({"a.b": 42, "a.c": (2, 3, 4)}, x=False)
    assert p["a"]["b"] == 42
    assert p["a", "b"] == 42
    assert p["a.b"] == 42
    assert p["a.c"][0] == 2
    assert len(p) == 2
    assert p.leaf_len() == 3
    assert set(p.leaf_keys()) == set(["a.b", "a.c", "x"])
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
    assert list(p2.leaf_items()) == list(p.leaf_items())


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
