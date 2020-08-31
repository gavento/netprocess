import jax.numpy as jnp
from netprocess import utils


def test_integrality():
    assert not utils.is_integer(1.0)
    assert utils.is_integer(1)
    assert utils.is_integer(jnp.int16(1))
    assert not utils.is_integer(jnp.float16(1))
