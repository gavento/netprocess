import jax.numpy as jnp
from netprocess import utils
import pytest


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
