import typing

import jax
import jax.numpy as jnp


def ensure_array(a, dtype=None):
    """Return a `DeviceArray` form of `a` with given dtype, NOP if already is."""
    if not isinstance(a, jnp.DeviceArray):
        return jnp.array(a, dtype=dtype)
    elif dtype is not None and a.dtype != dtype:
        return jnp.array(a, dtype=dtype)
    return a


def ensure_pytree(pt, dtype=None, ensure_dict=True):
    """
    Return a JAX pytree of `DeviceArray`s (`dtype` if given).

    By default also check that the top-level is a dict.
    """
    if not isinstance(pt, dict):
        raise TypeError(f"Expected PyTree dict, got {type(pt)}")
    return jax.tree_util.tree_map(lambda a: ensure_array(a, dtype=dtype), pt)


def concatenate_pytrees(pytrees: typing.List, check_treedefs: bool = True, axis=0):
    arrays, treedefs = jax.tree_util.unzip2(
        jax.tree_util.tree_flatten(pt) for pt in pytrees
    )
    if check_treedefs:
        for td in treedefs[1:]:
            if td != treedefs[0]:
                raise TypeError(f"Record treedefs do not match: {treedefs[0]} vs {td}")
    concats = [
        jnp.concatenate([a[i] for a in arrays], axis=axis)
        for i in range(len(arrays[0]))
    ]
    return jax.tree_util.tree_unflatten(treedefs[0], concats)


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


def switch(i: jnp.int32, funs: List[Any], *args):
    """
    Return `funs[i](*args)`
    """
    f = funs[-1]
    if not callable(f):
        f = lambda *_: funs[-1]
    if len(funs) == 1:
        return f(*args)
    return jax.lax.cond(
        i >= len(funs) - 1,
        lambda _: f(*args),
        lambda _: switch(i, funs[:-1], *args),
        None,
    )
