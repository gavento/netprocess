import typing

import jax
import jax.numpy as jnp

from .utils import PytreeDict, Pytree


def ensure_array(a, dtype=None) -> jnp.DeviceArray:
    """Return a `DeviceArray` form of `a` with given dtype, NOP if already is."""
    if not isinstance(a, jnp.DeviceArray):
        return jnp.array(a, dtype=dtype)
    elif dtype is not None and a.dtype != dtype:
        return jnp.array(a, dtype=dtype)
    return a


def ensure_pytree(pt: Pytree, dtype=None, ensure_dict=True) -> Pytree:
    """
    Return a JAX pytree of `DeviceArray`s (`dtype` if given).

    By default also check that the top-level is a dict.
    """
    if ensure_dict and not isinstance(pt, dict):
        raise TypeError(f"Expected PyTree dict, got {type(pt)}")
    return jax.tree_util.tree_map(lambda a: ensure_array(a, dtype=dtype), pt)


def tree_copy(pytree):
    """
    Return a copy of the given pytree.

    Note that structure is copied, leaves are not (as they are immutable).
    """
    return jax.tree_util.tree_map(lambda x: x, pytree)


def concatenate_pytrees(
    pytrees: typing.Iterable[Pytree], check_treedefs: bool = True, axis=0
) -> Pytree:
    """
    Concatenate the matching ndarrays in an iterable of pytrees.

    Returns pytree of the same tree type.
    By default checks that the treedefs actually match
    (otherwise only leaf count and order matters).
    """
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


def switch(funs: typing.List, i: jnp.int32, *args) -> typing.Any:
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
        lambda _: switch(funs[:-1], i, *args),
        None,
    )
