import typing

import jax
import jax.numpy as jnp
import numpy as np

from .types import Pytree
from .prop_tree import PropTree


def ensure_array(a, dtype=None, concretize_types=True) -> jnp.ndarray:
    """Return a `DeviceArray` form of `a` with given dtype, NOP if already is."""
    if not isinstance(a, jnp.ndarray):
        a = jnp.array(a, dtype=dtype)
    elif dtype is not None and a.dtype != dtype:
        a = jnp.array(a, dtype=dtype)
    if concretize_types and a.weak_type:
        if a.dtype == jnp.bool_:
            a = jnp.array(a, dtype=jnp.bool_)
        elif jnp.issubdtype(a.dtype, jnp.integer):
            a = jnp.array(a, dtype=jnp.int32)
        elif jnp.issubdtype(a.dtype, jnp.floating):
            a = jnp.array(a, dtype=jnp.float32)
        else:
            raise TypeError(f"Can't concretize weak_type'd {a.dtype}")
    return a


def ensure_pytree(
    pt: Pytree, dtype=None, ensure_dict=True, concretize_types=True
) -> Pytree:
    """
    Return a JAX pytree of `DeviceArray`s (`dtype` if given).

    By default also check that the top-level is a dict.
    With `concretize_types` also converts weak_types to concrete types (32bit).
    """
    if ensure_dict and not isinstance(pt, dict):
        raise TypeError(f"Expected PyTree dict, got {type(pt)}")
    return jax.tree_util.tree_map(
        lambda a: ensure_array(a, dtype=dtype, concretize_types=concretize_types), pt
    )


def pad_pytree_to(_cls, pt: Pytree, old_n: jnp.integer, new_n: jnp.integer) -> Pytree:
    """
    Pad or shrink first dim of all pytree leaves.
    Return the new pytree.
    """
    if old_n < new_n:

        def ext(a):
            return jnp.pad(a, [(0, n - old_n)] + ([(0, 0)] * (len(a.shape) - 1)))

        return jax.tree_map(ext, pt)
    else:
        return jax.tree_map(lambda a: a[:n], pt)


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
    arrays, treedefs = jax.util.unzip2(jax.tree_util.tree_flatten(pt) for pt in pytrees)
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


def _op_neutral(op, dtype):
    if np.issubdtype(dtype, np.bool_):
        return op in (jax.lax.scatter_mul, jax.lax.scatter_min)
    isint = np.issubdtype(dtype, np.integer)
    return dtype.type(
        {
            jax.lax.scatter_add: 0,
            jax.lax.scatter_mul: 1,
            jax.lax.scatter_min: np.iinfo(dtype).max if isint else np.inf,
            jax.lax.scatter_max: np.iinfo(dtype).min if isint else -np.inf,
        }[op]
    )


def apply_scatter_op(
    scatter_agg_op,
    n: int,
    values: jnp.ndarray,
    targets: jnp.ndarray,
    active: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Apply given scatter aggregate operation on `values` with their target indices `targets`

    `scatter_agg_op` is one of `jax.lax.scatter_*`. `n` is the result size, target indices outside the range are dropped.
    If `active` is given, only `active[i]==True` positions are taken into account.
    """
    if np.issubdtype(values.dtype, np.bool_) and scatter_agg_op in (
        jax.lax.scatter_add,
        jax.lax.scatter_mul,
    ):
        values = jnp.int32(values)
    neutral_value = _op_neutral(scatter_agg_op, values.dtype)
    # Array of neutral values
    z = jnp.full((n,) + values.shape[1:], neutral_value, dtype=values.dtype)
    if active is not None:
        targets = jnp.where(active, targets, n + 1)
    targets = jnp.expand_dims(targets, 1)
    dims = jax.lax.ScatterDimensionNumbers(
        tuple(range(1, len(values.shape))), (0,), (0,)
    )
    return scatter_agg_op(z, targets, values, dims, mode="drop")


def create_scatter_aggregates(
    n: int,
    edge: PropTree,
    targets: jnp.ndarray,
    active: jnp.ndarray = None,
) -> PropTree:
    apply_op_to_tree = lambda op: jax.tree_util.tree_map(
        lambda a: apply_scatter_op(op, n, a, targets, active), edge
    )
    return PropTree(
        sum=apply_op_to_tree(jax.lax.scatter_add),
        prod=apply_op_to_tree(jax.lax.scatter_mul),
        min=apply_op_to_tree(jax.lax.scatter_min),
        max=apply_op_to_tree(jax.lax.scatter_max),
    )
