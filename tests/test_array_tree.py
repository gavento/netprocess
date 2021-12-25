import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attr import frozen
from netprocess.utils import ArrayTree, jax_utils, utils


def a(*els, **kws):
    return jnp.array(els, **kws)


class AT1(ArrayTree):
    pass


def test_unit():
    p = ArrayTree()
    assert len(p) == 0
    p.update({"a.b": 42, "a.c": (2, 3, 4)}, x=False)
    assert p["a"]["b"] == 42
    assert p["a", "b"] == 42
    assert p["a.b"] == 42
    assert p["a.c"][0] == 2
    assert len(p) == 2
    assert p.leaf_count() == 3
    assert set(p.leaf_keys()) == set(["a.b", "a.c", "x"])
    assert len(p["a"]) == 2
    assert len(jax.tree_util.tree_leaves(p)) == 3
    assert jax.tree_util.tree_map(lambda x: x + 1, p)["a.c"][0] == 3

    p["c.d"] = (13,)
    assert p["c"]["d"][0] == 13
    assert "a" in p
    assert "b" in p["a"]
    assert "z" not in p
    assert p.setdefault("e-2/3", 18) == 18
    assert p.setdefault("e-2/3", 19) == 18

    p2 = jax.tree_util.tree_map(lambda x: x, p)
    assert list(p2.leaf_items()) == list(p.leaf_items())


def test_data_eq():
    with pytest.raises(ValueError):
        ArrayTree(a=1, b=2).data_eq(ArrayTree(a=1))
    with pytest.raises(ValueError):
        ArrayTree(a=1).data_eq(ArrayTree(a=1, b=2))
    with pytest.raises(TypeError):
        ArrayTree(a=[1, 3, 2]).data_eq(ArrayTree(a=[1, 2]))

    assert ArrayTree(a={"x": 1}).data_eq(ArrayTree(a={"x": 1}, b={"c": {}}))
    assert ArrayTree(a={"x": True}).data_eq(AT1(a={"x": True}, z={}))
    assert ArrayTree(a=1.13).data_eq(ArrayTree(a=1.13))
    assert not ArrayTree(a=1.13).data_eq(ArrayTree(a=1.13001))
    assert ArrayTree(a=1.13).data_eq(ArrayTree(a=1.13001), eps=0.001)


def test_frozen():
    p = ArrayTree(a=1)
    p["b.c"] = [2, 3]
    pf = p.copy(frozen=True)
    p["x"] = 1
    assert pf._frozen
    pfb = pf["b"]
    assert pfb._frozen

    pfb = pf["b"]
    assert pfb._frozen
    with pytest.raises(Exception):
        pf["b"] = 9
    with pytest.raises(Exception):
        pfb["d"] = 3
    assert (pfb["c"] == jnp.array([2, 3])).all()
    pu = pf.copy(frozen=False)
    assert not pu["b"]._frozen
    pu["c"] = 9
    pu["b"]["e"] = 42
    pu["b.f"] = 43

    pf2 = ArrayTree(pf)
    assert pf2._frozen
    assert pf2["b"]._frozen

    # Constructor preserves frozen status
    p2 = ArrayTree({"a": ArrayTree(b=2).copy(frozen=True)})
    assert not p2._frozen
    assert p2["a"]._frozen
    p3 = ArrayTree(p2)
    assert not p3._frozen
    assert p3["a"]._frozen


def test_replacing():
    pf = ArrayTree({"x.y": 42, "u.v.w": 43}).copy(frozen=True)
    p = ArrayTree(a=pf, b={"c": 13, "d.e.f": 14})
    assert not p._frozen
    assert p["a"]._frozen

    p2 = p.copy(
        replacing={
            "a.x.y": 44,
            "foo": {},
            "a.x.z.q.f": 1.0,
            "a.u": ArrayTree({"z.zz": 7}),
            "b.d": ArrayTree(e=0).copy(frozen=True),
        }
    )
    print(p2)
    assert p2["a.x.y"] == 44
    assert p2["a.x.z.q.f"] == 1.0
    assert p2["a.u.z.zz"] == 7
    assert p2["a.u.v.w"] == 43
    assert "foo" not in p2
    assert p2["b.c"] == 13
    assert p2["b.d.e"] == 0
    assert p2.leaf_count() == 6

    assert not p2._frozen
    assert p2["a"]._frozen
    assert p2["a.u"]._frozen
    assert p2["a.u.z"]._frozen
    assert not p2["b"]._frozen
    # assert p2["b.d"]._frozen ## Subtree frozenness is lost
