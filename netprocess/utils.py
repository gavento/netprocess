import typing

import jax.numpy as jnp

ArrayDict = typing.Dict[str, jnp.ndarray]


def is_integer(x):
    if isinstance(x, int):
        return True
    if isinstance(x, jnp.ndarray):
        return jnp.issubdtype(x.dtype, jnp.integer)
    return False
