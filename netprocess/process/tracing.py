import logging

import typing
from typing import Any

from jax._src.tree_util import tree_unflatten
from ..utils.prop_tree import PropTree

log = logging.getLogger(__name__)


class TracingPropTreeWrapper(PropTree):
    """
    Utility PropTree wrapper to track accessed keys.
    """

    __slots__ = ("_pt", "_target", "_prefix")

    def __init__(
        self,
        prop_tree: PropTree,
        *,
        _target: set = None,
        _prefix: str = "",
    ):
        super().__init__()  ## The PropTree itself is empty, we hold reference to the wrapped object
        self._pt = prop_tree
        self._target = _target
        self._prefix = _prefix

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        v = self._pt[k]
        p = f"{self._prefix}.{'.'.join(key)}".removeprefix(".")
        if not isinstance(v, PropTree):
            self._target.add(p)
            return v
        else:
            return TracingPropTreeWrapper(v, _target=self._target, _prefix=p)

    def copy(self):
        raise NotImplemented("Forbidden for TracingPropTreeWrapper")

    def tree_flatten(self):
        f, a = self._pt.tree_flatten()
        return (f, (a, self._pt.__class__, self._target, self._prefix))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        a, c, _target, _prefix = aux_data
        pt = c.tree_unflatten(a, children)
        return cls(pt, _target=_target, _prefix=_prefix)


class Tracer:
    """
    A utility class to hold the results of tracing and wrap data for tracing.
    """

    def __init__(self, tracing=True):
        self.accessed = set()
        self.tracing = tracing
        self.trace_log = []

    def reset_access(self):
        self.accessed = set()

    def wrap(self, pt: PropTree, prefix: str = "") -> TracingPropTreeWrapper:
        "Wrap a PropTree as immutable TracingPropTreeWrapper"
        return TracingPropTreeWrapper(pt, _target=self.accessed, _prefix=prefix)

    def log_line(self, msg):
        self.trace_log.append(msg)

    def log_operation_io(self, op_name, output_pytree, reset_access=True):
        if output_pytree and self.tracing:
            self.trace_log.append(
                f"  {op_name}: {', '.join(sorted(self.accessed))} -> {', '.join(output_pytree.keys())}"
            )
        if reset_access:
            self.reset_access()

    def get_log(self):
        return "\n".join(self.trace_log)
