import contextlib
import logging
import time
import typing

import jax.numpy as jnp
import jax

Pytree = typing.Any
PytreeDict = typing.Dict[str, typing.Any]
PRNGKey = jnp.ndarray

log = logging.getLogger(__name__)


def update_dict_disjoint(d: dict, update: dict):
    """Update dictionary in-place, raise ValueError on key conflict."""
    for k, v in update.items():
        if k in d:
            raise ValueError(f"Update key {k} already present")
        d[k] = v


def update_dict_present(d: dict, update: dict):
    """Update dictionary in-place, raise ValueError on new key."""
    for k, v in update.items():
        if k not in d:
            raise ValueError(f"Update key {k} not present in update dict")
        d[k] = v


def is_integer(x) -> bool:
    """Is `x` a simple integer - an `int` or ()-shaped ndarray?"""
    if isinstance(x, int):
        return True
    if isinstance(x, jnp.ndarray):
        return jnp.issubdtype(x.dtype, jnp.integer) and x.shape == ()
    return False


@contextlib.contextmanager
def logged_time(name, level=logging.INFO):
    """
    Context manager to measure and log operation time.
    """
    t0 = time.time()
    yield
    t1 = time.time()
    log.log(level, f"{name} took {t1-t0:.3g} s")


class TracingDict(dict):
    """
    Utility dict wrapper to track accessed keys.

    Iteration is ignored.
    """

    def __init__(self, d: dict, target: set, prefix: str = "", suffix: str = ""):
        super().__init__(d)
        self._target = target
        self._prefix = prefix
        self._suffix = suffix

    def _as_dict(self):
        return dict(self.items())

    def __getitem__(self, key):
        r = super().__getitem__(key)
        self._target.add(f"{self._prefix}{key!s}{self._suffix}")
        return r


# jax.tree_util.register_pytree_node(
#    TracingDict, lambda td: ((td._as_dict(),), None), lambda _, d: d[0]
# )
