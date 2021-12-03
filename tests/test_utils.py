import jax
import jax.numpy as jnp
import pytest
import functools
from netprocess.utils import jax_utils, utils


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
