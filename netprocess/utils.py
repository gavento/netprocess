import contextlib
import logging
import time
import typing

import jax.numpy as jnp
import numpy as np

Pytree = typing.Union[
    typing.List["Pytree"], typing.Tuple["Pytree"], "PytreeDict", np.ndarray
]
PytreeDict = typing.Dict[str, Pytree]
PRNGKey = jnp.ndarray

log = logging.getLogger(__name__)


class KeyOrValue:
    """
    Wraps a Jax numpy value or a key (to be fetched from a pytreedict).

    With an optional default (to fill the pytree on creation, not used on get_from)
    """

    def __init__(self, key_or_value, default=None, dtype=None):
        self.key = None
        self.value = None
        self.default = None

        if isinstance(key_or_value, KeyOrValue):
            self.key = key_or_value.key
            self.value = key_or_value.value
            self.default = key_or_value.default
            assert default is None
            assert dtype is None or dtype == key_or_value.value.dtype
        elif isinstance(key_or_value, str):
            self.key = key_or_value
            if default is not None:
                self.default = jnp.array(default, dtype=dtype)
        else:
            self.value = jnp.array(key_or_value, dtype=dtype)
            if default is not None:
                raise Exception("Warning: Do not combine a value with a default")

    def get_from(self, pytree: PytreeDict):
        if self.value is not None:
            return self.value
        return pytree[self.key]

    def ensure_in(self, pytree: PytreeDict):
        if (
            self.key is not None
            and self.key not in pytree
            and not self.key.startswith("_")
        ):
            if self.default is not None:
                pytree[self.key] = self.default
            raise Exception(
                f"Property {self.key:r} without a default missing in pytree"
            )

    def __str__(self):
        if self.key is None:
            return f"{self.value}"
        return self.key


KeyOrValueT = typing.Union[KeyOrValue, str, int, float, np.ndarray]


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


# jax.tree_util.register_pytree_node(
#    TracingDict, lambda td: ((td._as_dict(),), None), lambda _, d: d[0]
# )
