import typing
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
class PropTree:
    """
    A tree-of-dictionaries structure, all leaves being JAX arrays or scalars.
    """

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

    @classmethod
    def _convert_val(cls, v: Any) -> Any:
        if isinstance(v, PropTree):
            return PropTree(**v._items)
        elif isinstance(v, dict):
            return PropTree(**v)
        elif isinstance(v, jnp.ndarray):
            return v
        elif isinstance(v, np.ndarray):
            return jnp.array(v)
        elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            return jnp.array(v)
        elif isinstance(v, tuple):
            return jnp.array(v)
        else:
            raise TypeError(
                f"Invalid type {type(v)} passed to {self.__class__.__name__}: {v!r}"
            )

    def copy(self) -> "PropTree":
        """Return a deep copy of self"""
        return self.__class__(**self._items)

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
                self._items[key[0]] = PropTree()
            else:
                raise KeyError(f"key {key[0]!r} not found in PropTree")
        if not isinstance(self._items[key[0]], PropTree):
            raise TypeError(
                f"Indexing into both PropTree and the contained array is forbidden"
            )
        return self._items[key[0]]._rec(key[1:], creating=creating)

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        k, pt = self._rec(key)
        return pt._items[k]

    def __setitem__(self, key: typing.Union[str, tuple], val: Any) -> Any:
        k, pt = self._rec(key, creating=True)
        pt._items[k] = self._convert_val(val)

    def __contains__(self, key: typing.Union[str, tuple]) -> bool:
        try:
            _ = self[key]
        except KeyError:
            return False
        return True

    def setdefault(self, key: typing.Union[str, tuple], val: Any) -> Any:
        k, pt = self._rec(key, creating=True)
        if k not in pt:
            if callable(val):
                val = val()
            pt[k] = self._convert_val(val)
        return pt[k]

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()

    def __len__(self):
        return len(self._items)

    def leaf_items(self):
        for k, v in self._items.items():
            if isinstance(v, PropTree):
                for k2, v2 in v.leaf_items():
                    yield f"{k}.{k2}", v2
            else:
                yield k, v

    def leaf_keys(self):
        for k, _v in self.leaf_items():
            yield k

    def leaf_len(self):
        return len(tuple(self.leaf_items()))

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
