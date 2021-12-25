import logging
import typing

import jax.numpy as jnp
import numpy as np

from netprocess.utils.array_tree import ArrayTree

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

    def get_from(self, pt: ArrayTree):
        if self.value is not None:
            return self.value
        return pt[self.key]

    def ensure_in(self, pt: ArrayTree, repeat_times=None):
        if self.key is not None and self.key not in pt and not self.key.startswith("_"):
            if self.default is not None:
                if repeat_times is None:
                    pt[self.key] = self.default
                else:
                    pt[self.key] = jnp.expand_dims(self.default, 0).repeat(
                        repeat_times, 0
                    )
            else:
                raise Exception(
                    f"Property {self.key!r} without a default missing in PropTree"
                )

    def __str__(self):
        if self.key is None:
            return f"{self.value}"
        return self.key


KeyOrValueT = typing.Union[KeyOrValue, str, int, float, np.ndarray]
