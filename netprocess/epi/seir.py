import jax
import jax.numpy as jnp

from ..jax_utils import cond, switch
from ..network_process import OperationBase, ProcessStateData
from ..network_process.compartmental import GenericCompartmentalUpdateOp, Transition
from ..utils import PRNGKey, PytreeDict


class _SIUpdateOp(GenericCompartmentalUpdateOp):
    def __init__(
        self, prefix="", delta_t_key="delta_t", state_key="compartment", reinfect=False
    ):
        transitions = (
            Transition("S", "I", "I", binary=True, param_key=f"{prefix}edge_beta"),
        )
        if reinfect:
            transitions = transitions + (
                Transition("R", "S", param_key=f"{prefix}rho"),
            )
        super().__init__(
            compartments=("S", "I"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )


class _SIRUpdateOp(GenericCompartmentalUpdateOp):
    def __init__(
        self, prefix="", delta_t_key="delta_t", state_key="compartment", reinfect=False
    ):
        transitions = (
            Transition("S", "I", "I", binary=True, param_key=f"{prefix}edge_beta"),
            Transition("I", "R", param_key=f"{prefix}gamma"),
        )
        if reinfect:
            transitions = transitions + (
                Transition("R", "S", param_key=f"{prefix}rho"),
            )
        super().__init__(
            compartments=("S", "I", "R"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )


class _SEIRUpdateOp(GenericCompartmentalUpdateOp):
    def __init__(
        self, prefix="", delta_t_key="delta_t", state_key="compartment", reinfect=False
    ):
        transitions = (
            Transition("S", "E", "I", binary=True, param_key=f"{prefix}edge_beta"),
            Transition("E", "I", param_key=f"{prefix}alpha"),
            Transition("I", "R", param_key=f"{prefix}gamma"),
        )
        if reinfect:
            transitions = transitions + (
                Transition("R", "S", param_key=f"{prefix}rho"),
            )
        super().__init__(
            compartments=("S", "E", "I", "R"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )


class SIRUpdateOp(OperationBase):
    def __init__(
        self,
        state_key="compartment",
        infected_key="infected",
        infection_key="_infection",
    ):
        self.STATE = state_key
        self.INFECTED = infected_key
        self.INFECTION = infection_key

    def prepare_state_pytrees(self, state):
        state.nodes_pytree.setdefault(self.STATE, jnp.zeros(state.n, dtype=jnp.int32))
        if not self.INFECTED.startswith("_"):
            state.nodes_pytree.setdefault(
                self.INFECTED, jnp.zeros(state.n, dtype=jnp.int32)
            )
        state.params_pytree.setdefault("edge_beta", 0.05)
        state.params_pytree.setdefault("gamma", 0.1)
        state.params_pytree.setdefault("delta_t", 1.0)
        assert state.params_pytree["delta_t"] < 1.000001

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        infection = cond(
            jnp.logical_and(from_node[self.STATE] == 1, to_node[self.STATE] == 0),
            lambda: jnp.int32(jax.random.bernoulli(rng_key, params["edge_beta"])),
            0,
        )
        return {self.INFECTION: infection}

    def update_node(self, rng_key, params, node, in_edges, out_edges):
        compartment = switch(
            [
                # S -> {S, I}:
                lambda: cond(in_edges["sum"][self.INFECTION] > 0, 1, 0),
                # I -> {I, R}:
                lambda: cond(jax.random.bernoulli(rng_key, params["gamma"]), 2, 1),
                # R -> {R}:
                2,
            ],
            node[self.STATE],
        )
        return {
            self.STATE: compartment,
            self.INFECTED: node[self.INFECTED] + out_edges["sum"][self.INFECTION],
        }


class ComputeImmediateROp:
    def create_record(
        self, rng_key: PRNGKey, state: ProcessStateData, orig_state: ProcessStateData
    ) -> PytreeDict:
        return {}, {"immediate_R_mean": mean, "immediate_R_samples": count}
