import inspect
import logging
import random
from typing import Callable, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..network import Network
from ..utils import ArrayTree
from .state import ProcessState

log = logging.getLogger(__name__)


class NetworkProcess:
    """
    A network process is an immutable sequence of operations on `ProcessState`.

    The state is created for a network via `state0 = process.new_state(net, ...)` and then
    eveolved with e.g. `state1 = process.run(state0, steps=10)`.
    State also accumulates any gathered recorded properties on every step.
    """

    def __init__(
        self,
        operations: Iterable[Union["netprocess.operations.OperationBase", Callable]],
        record_keys=(),
    ):
        self.operations = tuple(operations)
        self._run_jit = jax.jit(self._run, static_argnames=["tracing", "jit"])
        self._traced = 0
        self.record_keys = record_keys

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.operations}>"

    def run(self, state: ProcessState, steps=1, jit=True) -> ProcessState:
        """
        Run `steps` steps of the process on the given `state`, returning a new state.

        By default, JIT-compiles the operations for GPU or CPU.
        Recorded keys are automatically added to the new state `records`.
        """
        assert len(state._record_set) == 0
        steps_array = jnp.arange(state.step, state.step + steps, dtype=jnp.int32)
        if jit:
            state2, records = self._run_jit(
                state._bare_copy(), steps_array, tracing=True, jit=True
            )
        else:
            state2, records = self._run(
                state._bare_copy(), steps_array, tracing=False, jit=False
            )
        state2._network, state2._records = state._network, state._records
        if len(records) > 0:
            state2.records.add_record(records)
        return state2

    def warmup_jit(self, state=None, n=None, m=None, steps=1, block=True):
        """
        Force the compilation of the JITted run() function for given `n`, `m` and `steps`, and wait for it (if `block`).
        """
        if state is None:
            assert n >= 2 and m >= 1
            state = self.new_state([[0, 1]] * m, n=n, seed=42)
        else:
            assert n is None and m is None
        # Run the jitted function
        new_state = self.run(state, steps=steps)
        # Wait for all computations
        if block:
            new_state.block_on_all()

    def _run(
        self,
        state: ProcessState,
        steps_array: jnp.DeviceArray,
        tracing: bool,
        jit: bool,
    ) -> Tuple[ArrayTree, ProcessState]:
        """Returns (new_state, all_records_pytree). JIT-able when jit=True."""
        if tracing:
            self._traced += 1
            log.debug(
                f"Tracing {self} {self._traced}th time with n={state.n}, m={state.m}, steps={steps_array.shape[0]}, keys={list(state.keys())}"
            )
        if jit:
            return jax.lax.scan(
                lambda s, i: self._run_step(s, i),
                state,
                steps_array,
            )
        else:
            rs = []
            for si in steps_array:
                state, r = self._run_step(state, si)
                rs.append(r)
            return state, jax.tree_multimap(lambda *r: jnp.stack(list(r)), *rs)

    def _run_step(self, state: ProcessState, step: jnp.int32):
        """Returns (new_state, record_pytree). JIT-able."""

        # Original state, never updated
        prev_state = state.copy(frozen=True)
        state = state.copy(frozen=False)
        # Set step number
        # NB: this shuld be a noop with correct external step numbering
        state["step"] = step
        state._record_set = ArrayTree()

        # Run all the update steps, updating the staself._run_step(s, si)te
        for op in self.operations:
            params = len(inspect.signature(op).parameters)
            if params == 1:
                op(state)
            elif params == 2:
                op(state, prev_state)
            else:
                raise TypeError(f"Operation must be callable with 1 or 2 arguments")
        # Record named parameters
        for rk in self.record_keys:
            state.record(rk)
        # Collect record set into PropTree
        records = state._take_record_set()
        state._record_set = None

        # Create the new state, filtering underlines and checking
        # that we only update existing keys
        state = state._filter_underscored()
        for k in state.keys():
            if k not in prev_state:
                raise Exception(
                    f"New key {k!r} found in state after run but the key set must be stable. "
                    "Either add it to original state or mark it as temporary with leading '_' in name."
                )
        # Finally, increment the step number and update the PRNG
        state["step"] = state["step"] + 1
        return state, records

    def new_state(
        self,
        network: Network,
        props: ArrayTree = {},
        *,
        seed=None,
        record_stride: int = 1,
    ) -> ProcessState:
        """
        Create a new ProcessState, also ensuring the required initial properties for all operations.

        State properties are taken from `props` or using defaults.
        `seed` may be jax PRNGKey, a 64 bit number (used as a seed) or None (randomize).
        The given param/sprops are used as overrides over the Network parameters (if any; those are unmodified).
        """
        if seed is None:
            seed = random.randint(0, 1 << 64 - 1)
        if isinstance(seed, (jnp.ndarray, np.ndarray)):
            assert seed.shape == (2,)
            assert seed.dtype == jnp.int32
            prng_key = jnp.array(seed)
        else:
            prng_key = jax.random.PRNGKey(seed)

        # Note: all pytree elements are converted to jax arrays later in the state constructor
        # Note: all pytrees are properly copied later in the state constructor
        sd = ProcessState.from_network(
            network, prng_key, props=props, record_stride=record_stride
        )
        # Prepare state for operations
        from ..operations.base import OperationBase

        for op in self.operations:
            if isinstance(op, OperationBase):
                op.init_state(sd)
        sd._assert_shapes()
        return sd
