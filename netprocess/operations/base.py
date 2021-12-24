from ..process.state import ProcessState
from ..utils.prop_tree import PropTree

from ..utils import PRNGKey


class OperationBase:
    """Base class for an operation in the network process."""

    def init_state(self, state: ProcessState):
        """Initialize any operation intial data in the state.

        This is never JIT-ted. May also check for presence and
        shapes of parameters required for the operation."""
        pass

    def __call__(self, state: ProcessState, orig_state: ProcessState):
        """Main entry point of the operation.

        May take 1 or 2 parameters (i.e. `prev_state` is optional).
        For modifying the edges and nodes in a vectorized way,
        consider calling `state.apply_node_fn` or `state.apply_edge_fn`.

        Args:
            state: Current state to be modified.
            orig_state: State at the start of the step (optional, frozen).
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
