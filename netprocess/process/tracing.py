import logging

log = logging.getLogger(__name__)


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

    def TD(self, d, label, depth=1):
        "Wrap a PytreeDict in TracingDict, recursively if `depth`>1"
        if (not self.tracing) or depth <= 0:
            return d
        if depth == 1:
            return TracingDict(d, self.accessed, f"{label}[", "]")
        return {k: self.TD(v, f"{label}[{k}]", depth - 1) for k, v in d.items()}

    def TS(self, state, label=""):
        "Wrap all ProcessStateData fields in TacingDict, return a new state."
        if not self.tracing:
            return state
        return state._replace(
            **{
                k: self.TD(getattr(state, k), label + k[0].upper())
                for k in ["nodes_pytree", "edges_pytree", "params_pytree"]
            }
        )

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
