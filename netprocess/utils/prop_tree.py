import typing
from collections.abc import MutableMapping
from typing import Any, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_utils


@jax.tree_util.register_pytree_node_class
class PropTree(MutableMapping):
    """
    A tree-of-dictionaries structure, all leaves being JAX arrays or scalars.
    """

    __slots__ = ("_items",)
    _FROZEN = False
    _OTHER_PROPS = True
    _ATYPES = (
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
            self._setitem_f(k, v)
        for k, v in kw_items.items():
            self._setitem_f(k, v)

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        k, pt = self._rec(key)
        return pt._items[k]

    def _setitem_f(self, key: typing.Union[str, tuple], val: Any):
        "Internal operation, also works on FROZEN instances"
        k, pt = self._rec(key, creating=True)
        t = pt._type_for_child(k)
        if t is None and not pt._OTHER_PROPS:
            raise AttributeError(
                f"Key {k!r} is not among allowed props for {pt.__class__}"
            )
        if (
            isinstance(val, (dict, PropTree))
            and (t is not None)
            and not issubclass(t, PropTree)
        ):
            raise TypeError(
                f"Trying to create a dict-like node {k} but annotated type is {t}"
            )
        pt._items[k] = pt._convert_val(val, dict_type=t)

    def __setitem__(self, key: typing.Union[str, tuple], val: Any):
        if self._FROZEN:
            raise Exception(
                f"Trying to set item {key!r} in frozen {self.__class__.__name__}"
            )
        self._setitem_f(key, val)

    def __getattr__(self, key):
        th = get_type_hints(self.__class__)
        if key in th:
            return self[key]
        raise AttributeError(
            f"Key {key!r} is not a valid attribute for class {self.__class__}"
        )

    def _delitem_f(self, key: typing.Union[str, tuple]):
        "Internal operation, also works on FROZEN instances"
        k, pt = self._rec(key)
        del pt._items[k]

    def __delitem__(self, key: typing.Union[str, tuple]):
        if self._FROZEN:
            raise Exception(
                f"Trying to delete item {key!r} in frozen {self.__class__.__name__}"
            )
        self._delitem_f(key)

    def __iter__(self):
        for k, v in self._items.items():
            if isinstance(v, PropTree):
                for k2 in v:
                    yield f"{k}.{k2}"
            else:
                yield k

    def __len__(self):
        return len(tuple(iter(self)))

    def data_eq(self, other: "PropTree", eps: float = None) -> bool:
        """
        Check equality of PropTrees by leaf data only (can have diffrent types or other attributes)

        Requires the same shapes and dtypes. Ignores any empty (leaf-less) subtrees.
        Optionally checks with some tolerance for floats, using `eps` for both absolute and relative error.
        """
        assert isinstance(other, PropTree)
        k1 = set(self.keys())
        k2 = set(other.keys())
        if k1 != k2:
            return False
        for k in k1:
            d1 = self[k]
            d2 = other[k]
            if d1.dtype != d2.dtype or d1.shape != d2.shape:
                return False
            if jnp.issubdtype(d1.dtype, jnp.floating) and eps is not None:
                d1m = d1 - eps - jnp.abs(d1) * eps
                d1p = d1 + eps + jnp.abs(d1) * eps
                if not ((d1m <= d2).all() and (d1p >= d2).all()):
                    return False
            else:
                if not (d1 == d2).all():
                    return False
        return True

    @classmethod
    def _convert_val(cls, v: Any, dict_type=None) -> Any:
        if isinstance(v, PropTree):
            return v.copy()
        elif isinstance(v, dict):
            return (dict_type or PropTree)(v)
        elif isinstance(v, cls._ATYPES):
            return jax_utils.ensure_array(v)
        else:
            return v  ####TODO
            raise TypeError(f"Invalid type {type(v)} passed to {cls.__name__}: {v!r}")

    def copy(self) -> "PropTree":
        f, a = self.tree_flatten()
        return self.__class__.tree_unflatten(a, f)

    def _replace(self, updates: dict = {}, **kw_updates) -> "PropTree":
        """Return a deep copy with some replaced properties"""
        s2 = self.copy()
        for k, v in updates.items():
            s2._setitem_f(k, v)
        for k, v in kw_updates.items():
            s2._setitem_f(k, v)
        return s2

    def _type_for_child(self, key):
        th = get_type_hints(self.__class__)
        return th.get(key)

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
                t = self._type_for_child(key[0])
                if not self._OTHER_PROPS and t is None:
                    raise AttributeError(
                        f"Key {key[0]!r} is not among allowed props for {self.__class__}"
                    )
                if t is not None:
                    assert issubclass(t, PropTree)
                else:
                    t = PropTree
                self._items[key[0]] = t()  ## NB: Somehow resolve sub-tree types?
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
            if not pt._OTHER_PROPS and pt._type_for_child(k) is None:
                raise AttributeError(
                    f"Key {k!r} is not among allowed props for {pt.__class__}"
                )
            if callable(val):
                val = val()
            pt[k] = pt._convert_val(val, dict_type=pt._type_for_child(k))
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
        return tuple(self._items[k] for k in ks), ks

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        s = cls()
        for k, v in zip(aux_data, children):
            s._items[k] = v
        return s
