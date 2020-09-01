import typing

import jax.numpy as jnp
import jax

ArrayDict = typing.Dict[str, jnp.ndarray]


def is_integer(x):
    if isinstance(x, int):
        return True
    if isinstance(x, jnp.ndarray):
        return jnp.issubdtype(x.dtype, jnp.integer)
    return False


def cond(val, true_res, false_res):
    """
    Convenience wrapper around `jax.lax.cond`, coercng values and functions.

    `true_res` and `false_res` may be a 0-param function or a value.
    """
    if callable(true_res):
        tr = lambda _: true_res()
    else:
        tr = lambda _: true_res

    if callable(false_res):
        fr = lambda _: false_res()
    else:
        fr = lambda _: false_res

    return jax.lax.cond(val, tr, fr, None)
