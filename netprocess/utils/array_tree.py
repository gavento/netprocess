from re import L
import typing
from collections.abc import MutableMapping
from typing import Any, Callable, Iterable, Union

import jax
from jax._src.api import F
import jax.numpy as jnp
import numpy as np

from . import jax_utils


@jax.tree_util.register_pytree_node_class
class ArrayTree(MutableMapping):
    """
    A tree-of-dictionaries structure, all leaves being JAX arrays or scalars.

    In addition to tree of dictionaries:

    * All leaves are guaranteed to be JAX `ndarray`s, even scalars.
    * Supports indexing via dotted strings and tuples: `a["x.y"]==a["x","y"]==a["x"]["y"]`.
    * All subtrees can be frozen or mutable (default). A frozen subtree can be unfrozen
        only by a copy. All nodes of a frozen node must also be frozen.
    * All non-leaf nodes are `ArrayTree`s.

    Copies are cheap since the contained arrays are not copied (being immutable).

    NB: ArrayTrees are also copied when you **assign** them as a value in another ArrayTree -
    this way subtrees are never shared between trees and mutations on one won't show in the other.
    However, taking a subtree allows you to mutate the original tree,
    e.g. `b=a['b']; b['x']=42` and you will also see `a['b.x']==42`.
    """

    __slots__ = ("_items", "_frozen")
    _CONVERTED_TYPES = (
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
        self,
        items: typing.Union[dict, "ArrayTree"] = {},
        **kw_items: dict,
    ) -> None:
        self._items = {}
        self._frozen = False
        for k, v in items.items():
            self[k] = v
        for k, v in kw_items.items():
            self[k] = v
        if isinstance(items, ArrayTree) and items._frozen:
            self._frozen = True
            for _k, v in self._iter_subtrees():
                v._frozen = True

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        if isinstance(key, str):
            key = key.split(".")
        key = tuple(key)
        if len(key) == 0:
            return self
        child = self._items[key[0]]
        if len(key) == 1:
            return child
        if not isinstance(child, ArrayTree):
            raise TypeError(
                f"Child node {key[0]} is a leaf, can't index it with {key[1:]}"
            )
        return child[key[1:]]

    def __setitem__(self, key: typing.Union[str, tuple], val: Any):
        if self._frozen:
            raise Exception(
                f"Trying to set item {key!r} in frozen {self.__class__.__name__}"
            )
        if isinstance(key, str):
            key = key.split(".")
        key = tuple(key)
        assert len(key) > 0
        if len(key) == 1:
            self._items[key[0]] = self._convert_val(val)
        else:
            if key[0] not in self._items:
                self._items[key[0]] = ArrayTree()
            # Conversion will happen in recursion
            child = self._items[key[0]]
            if not isinstance(child, ArrayTree):
                raise TypeError(
                    f"Child node {key[0]} is a leaf, can't index it with {key[1:]}"
                )
            child[key[1:]] = val

    def __delitem__(self, key: typing.Union[str, tuple]):
        if self._frozen:
            raise Exception(
                f"Trying to delete item {key!r} in frozen {self.__class__.__name__}"
            )
        if isinstance(key, str):
            key = key.split(".")
        key = tuple(key)
        assert len(key) > 0
        if len(key) == 1:
            del self._items[key[0]]
        else:
            child = self._items[key[0]]
            if not isinstance(child, ArrayTree):
                raise TypeError(
                    f"Child node {key[0]} is a leaf, can't index it with {key[1:]}"
                )
            del child[key[1:]]

    def __iter__(self):
        return self._items.__iter__()

    def __len__(self):
        return len(self._items)

    def leaf_items(self):
        """Iterator over all leaf pairs `(dotted_string_key, ndarray)`.

        Ignores any empty subtrees."""
        for k, v in self.items():
            if isinstance(v, ArrayTree):
                for k2, v2 in v.leaf_items():
                    yield (f"{k}.{k2}", v2)
            else:
                yield (k, v)

    def leaf_keys(self):
        """Iterator over all leaf keys as dotted strings."""
        for k, _v in self.leaf_items():
            yield k

    def leaf_values(self):
        """Iterator over all leaf ndarrays."""
        for _k, v in self.leaf_items():
            yield v  #        elif isinstance(v, ()):

    def leaf_count(self):
        """Number of ndarray leaves."""
        return len(tuple(self.leaf_items()))

    def _iter_subtrees(self):
        """Iterate over all non-leaf subtrees, yielding `(key_str, ArrayTree)`.

        Does not return `self`. Also returnd empty subtrees."""
        for k, v in self.items():
            if isinstance(v, ArrayTree):
                yield (k, v)
                for k2, v2 in v._iter_subtrees():
                    yield (f"{k}.{k2}", v2)

    def copy(self, frozen=None, replacing={}) -> "ArrayTree":
        """Copy the tree, optionaly un/freezing and setting given values.

        The copy is cheap as the ndarrays are immutable and shared.
        Preserves subtree frozen status by default.

        With `replacing`, sets the given key-value pairs.
        Works if the some subtrees of `self` are frozen, preserving the frozenness.

        Note that only leaf arrays are replaced, not entire subtrees!
        Frozenness of any subtrees of `replacing` is currently ignored.

        Note: Copy is performed through jax PyTree flattening & unflattening, so subclasses
        only need to modify tree_`(un)flatten`.
        """
        s = jax.tree_util.tree_map(lambda x: x, self)
        if frozen is not None:
            s._frozen = frozen
            for _k, v in s._iter_subtrees():
                v._frozen = frozen
        if replacing:
            replacing = ArrayTree(replacing)
            for k, v in replacing.leaf_items():
                k = tuple(k.split("."))
                r = s
                while len(k) > 1 and k[0] in r:
                    r = r[k[0]]
                    k = k[1:]
                if not r._frozen:
                    r[k] = v
                else:
                    if len(k) > 1:
                        v = ArrayTree({k[1:]: v})
                        k = k[:1]
                    if isinstance(v, ArrayTree):
                        v = v.copy(frozen=True)
                    r._frozen = False
                    r[k[0]] = v
                    r._frozen = True
        return s

    @classmethod
    def _convert_val(cls, v: Any) -> Any:
        if isinstance(v, cls._CONVERTED_TYPES):
            return jax_utils.ensure_array(v)
        elif isinstance(v, ArrayTree):
            return v.copy()
        elif isinstance(v, dict):
            return ArrayTree(v).copy()
        #        elif isinstance(v, ()):
        # This is necessary for JAX tracing
        #            return v
        else:
            raise TypeError(f"Invalid type {type(v)} passed to {cls.__name__}: {v!r}")

    def map_tree_arrays(
        self, f: Callable, *others: Iterable[Union["ArrayTree", dict]]
    ) -> "ArrayTree":
        """Apply f to all array leaves of the tree, returning the resulting tree.

        Ignores subtree types (even converting `dict`s to `ArrayTree`s) and frozen status.
        Raises `ValueError` when the trees don't have the same leaf keys.
        """
        ks = set(self.leaf_keys())
        others = [ArrayTree(t) for t in others]
        for s in others:
            if set(s.leaf_keys()) != ks:
                raise ValueError(
                    f"Can't map trees with different key sets: {sorted(ks)} vs {sorted(s.leaf_keys())}"
                )
        return ArrayTree({k: f(self[k], *[o[k] for o in others]) for k in ks})

    def data_eq(
        self, other: Union["ArrayTree", dict], eps: float = None
    ) -> jnp.ndarray:
        """
        Check equality of ArrayTrees by leaf data only.

        Ignores subtree types (even converting `dict`s to `ArrayTree`s) and frozen status.
        Raises `ValueError` when the trees don't have the same leaf keys.
        JAX raises `TypeError` when incompatible shpaes are compared.

        Optionally checks equality with some tolerance for floats, using `eps`
        for both absolute `(+- eps)` and relative error `(*/ (1+eps))`.

        Returns:
            jnp.ndarray bool scalar
        """

        def compare(a1, a2):
            if jnp.issubdtype(a2.dtype, jnp.floating) and eps is not None:
                a1, a2 = a2, a1
            if jnp.issubdtype(a1.dtype, jnp.floating) and eps is not None:
                a1m = a1 - eps - jnp.abs(a1) * eps
                a1p = a1 + eps + jnp.abs(a1) * eps
                return (a1m <= a2).all() and (a1p >= a2).all()
            else:
                return (a1 == a2).all()

        eqs = self.map_tree_arrays(compare, other)
        return jax.tree_util.tree_reduce(lambda a, b: a * b, eqs, jnp.array(True))

    def nice_str(self, indent: int = 2, _i0: int = 0) -> str:
        s = []
        for k, v in self.items():
            if isinstance(v, ArrayTree):
                s.append(f"{' '*_i0}{k}:")
                s.append(v.nice_str(indent, _i0 + indent))
            else:
                vs = "[...]"
                if v.shape == () or (len(v.shape) == 1 and v.shape[0] <= 3):
                    vs = str(v)
                s.append(f"{' '*_i0}{k}: array({v.shape}, {v.dtype}) = {vs}")
        return "\n".join(s)

    def __str__(self):
        def format(v):
            if v.shape == () or (len(v.shape) == 1 and v.shape[0] <= 4):
                return str(v)
            return f"array({v.shape}, {v.dtype})"

        s = (
            f"{k}: {str(v) if isinstance(v, ArrayTree) else format(v)}"
            for k, v in self.items()
        )
        return f"{{ {', '.join(s)} }}"

    def as_rec_dict(self):
        """Return a simple tree-of-dictionaries."""
        return {
            k: v._as_rec_dict() if isinstance(v, ArrayTree) else v
            for k, v in self._items.items()
        }

    def tree_flatten(self):
        """Flattening method for JAX Pytrees"""
        ks = tuple(self._items.keys())
        return tuple(self._items[k] for k in ks), (ks, self._frozen)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        s = cls()
        ks, frozen = aux_data
        for k, v in zip(ks, children):
            s._items[k] = v
        s._frozen = frozen
        return s
