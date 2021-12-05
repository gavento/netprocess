import logging
import typing
from typing import Any

from ..utils.prop_tree import PropTree

log = logging.getLogger(__name__)


class TracingPropTreeWrapper(PropTree):
    """
    Utility PropTree wrapper to track accessed keys.
    """

    _FROZEN = True
    __slots__ = ("_pt", "_target", "_prefix")

    def __init__(
        self,
        prop_tree: PropTree,
        *,
        _target: set = None,
        _prefix: str = "",
    ):
        super().__init__()  ## The PropTree itself is empty, we defer to the wrapped object
        self._pt = prop_tree
        self._target = _target
        self._prefix = _prefix

    def __getitem__(self, key: typing.Union[str, tuple]) -> Any:
        if isinstance(key, str):
            key = key.split(".")
        v = self._pt[key]
        p = f"{self._prefix}.{'.'.join(key)}".lstrip(".")
        if not isinstance(v, PropTree):
            self._target.add(p)
            return v
        else:
            return TracingPropTreeWrapper(v, _target=self._target, _prefix=p)

    def __getattr__(self, key):
        return self[key]

    def copy(self):
        return TracingPropTreeWrapper(
            self._pt.copy(), _target=self._target, _prefix=self._prefix
        )

    def _replace(self, updates: dict = {}, **kw_updates) -> PropTree:
        return TracingPropTreeWrapper(
            self._pt._replace(updates, **kw_updates),
            _target=self._target,
            _prefix=self._prefix,
        )

    def tree_flatten(self):
        f, a = self._pt.tree_flatten()
        return (f, (a, self._pt.__class__, self._target, self._prefix))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        a, c, _target, _prefix = aux_data
        pt = c.tree_unflatten(a, children)
        return cls(pt, _target=_target, _prefix=_prefix)

    def __str__(self):
        return f"{self.__class__.__name__}({self._pt}, _prefix={self._prefix!r})"


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
