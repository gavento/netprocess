import typing
from collections.abc import MutableMapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_utils


@jax.tree_util.register_pytree_node_class
class PropTree(MutableMapping):
    """
    A tree-of-dictionaries structure, all leaves being JAX arrays or scalars.
    """

    ATYPES = (
        jnp.ndarray,
        np.ndarray,
        int,
        float,
        bool,
        np.integer,
        np.floating,
        np.bool_,
        tuple,
        list,
    )

    def __init__(
        self, items: typing.Union[dict, "PropTree"] = {}, **kw_items: dict
    ) -> None:
        self._items = {}
        if isinstance(items, PropTree):
            items = items._items
        for k, v in items.items():
            self[k] = v
        for k, v in kw_items.items():
            self[k] = v

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        k, pt = self._rec(key)
        return pt._items[k]

    def __setitem__(self, key: typing.Union[str, tuple], val: Any) -> Any:
        k, pt = self._rec(key, creating=True)
        pt._items[k] = self._convert_val(val)

    def __len__(self):
        return len(tuple(iter(self)))

    def __delitem__(self, key: typing.Union[str, tuple]):
        k, pt = self._rec(key)
        del pt._items[k]

    def __iter__(self):
        for k, v in self._items.items():
            if isinstance(v, PropTree):
                for k2 in v:
                    yield f"{k}.{k2}"
            else:
                yield k

    @classmethod
    def _convert_val(cls, v: Any) -> Any:
        if isinstance(v, PropTree):
            return PropTree(**v._items)
        elif isinstance(v, dict):
            return PropTree(**v)
        elif isinstance(v, cls.ATYPES):
            return jax_utils.ensure_array(v)
        else:
            raise TypeError(
                f"Invalid type {type(v)} passed to {self.__class__.__name__}: {v!r}"
            )

    def copy(self) -> "PropTree":
        """Return a deep copy of self"""
        s = self.__class__()
        for k, v in self.leaf_items():
            s[k] = v
        return s

    def _replace(self, updates: dict = {}, **kw_updates) -> "PropTree":
        """Return a deep copy with some replaced properties"""
        s2 = self.copy()
        for k, v in updates.items():
            s2[k] = v
        for k, v in kw_updates.items():
            s2[k] = v
        return s2

    def _rec(
        self, key: typing.Union[str, tuple], creating: bool = False
    ) -> typing.Tuple[str, "PropTree"]:
        if isinstance(key, str):
            key = key.split(".")
        assert len(key) >= 1
        if len(key) == 1:
            return key[0], self
        if key[0] not in self._items:
            if creating:
                self._items[key[0]] = self.__class__()
            else:
                raise KeyError(f"key {key[0]!r} not found in PropTree")
        if not isinstance(self._items[key[0]], PropTree):
            raise TypeError(
                f"Indexing into both PropTree and the contained array is forbidden"
            )
        return self._items[key[0]]._rec(key[1:], creating=creating)

    def setdefault(self, key: typing.Union[str, tuple], val: Any) -> Any:
        k, pt = self._rec(key, creating=True)
        if k not in pt:
            if callable(val):
                val = val()
            pt[k] = self._convert_val(val)
        return pt[k]

    def top_keys(self):
        return self._items.keys()

    def top_len(self):
        return len(self._items)

    def top_items(self):
        return self._items.items()

    def top_values(self):
        return self._items.values()

    def nice_str(self, indent=2, _i0=0) -> str:
        s = []
        for k, v in self._items.items():
            if isinstance(v, PropTree):
                s.append(f"{' '*_i0}{k}:")
                s.append(v.nice_str(indent, _i0 + indent))
            else:
                vs = "[...]"
                if v.shape == () or (len(v.shape) == 1 and v.shape[0] <= 3):
                    vs = str(v)
                s.append(f"{' '*_i0}{k}: {v.dtype}{v.shape} = {vs}")
        return "\n".join(s) + "\n"

    def __str__(self):
        def s(v):
            if v.shape == () or (len(v.shape) == 1 and v.shape[0] <= 3):
                return str(v)
            return f"{v.dtype}{v.shape}"

        a = (f"{k!r}: {s(v)}" for k, v in self.items())
        return f"{'{'}{', '.join(a)}{'}'}"

    def _as_rec_dict(self):
        return {
            k: v._as_rec_dict() if isinstance(v, PropTree) else v
            for k, v in self._items.items()
        }

    def tree_flatten(self):
        ks = list(self._items.keys())
        return (self._items[k] for k in ks), ks

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls({k: v for k, v in zip(aux_data, children)})
