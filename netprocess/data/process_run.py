import copy

import attr

from ..network_process.state import State
from .data_utils import load_object, save_object
from .network import Network


@attr.s
class ProcessRun:
    """
    Record and storage of a finished network process run.
    """

    network = attr.ib(default=None, type=Network)
    meta = attr.ib(type=dict)
    records = attr.ib(type=dict)
    ATTRS = ("meta", "records")

    @classmethod
    def from_state(cls, state: State):
        return cls(
            meta=copy.depcopy(state.meta),
            network=state.network,
            records=state.all_records(),
        )

    @classmethod
    def load(cls, path):
        d = load_object(path)
        assert frozenset(d.keys()) == frozenset(cls.ATTRS)
        # Set it here if the network was moved after generation, also to be more
        # useful for storing resulting state
        d["meta"]["process_run_path"] = path
        return cls(**d)

    def load_network(self, path_prefix=None):
        p = self.meta["network_path"]
        self.network = Network.load(
            os.path.join(path_prefix, p) if path_prefix is not None else p
        )

    def write(self, path):
        save_object({k: getattr(self, k) for k in self.ATTRS}, path)
