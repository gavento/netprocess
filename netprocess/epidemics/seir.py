import jax
import jax.numpy as jnp

from ..operations import OperationBase
from ..process.state import ProcessStateData
from ..utils import PRNGKey, PytreeDict
from ..utils.jax_utils import cond, switch
from .compartmental import (
    BinaryPoissonTransition,
    PoissonCompartmentalUpdateOp,
    PoissonTransition,
)


class SIUpdateOp(PoissonCompartmentalUpdateOp):
    def __init__(self, prefix="", delta_t_key="delta_t", state_key="compartment"):
        transitions = (
            BinaryPoissonTransition(
                "S", "I", "I", rate_key=f"{prefix}edge_infection_rate"
            ),
        )
        super().__init__(
            compartments=("S", "I"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )


class SIRUpdateOp(PoissonCompartmentalUpdateOp):
    def __init__(
        self,
        prefix="",
        delta_t_key="delta_t",
        state_key="compartment",
        immunity_loss=False,
    ):
        transitions = (
            BinaryPoissonTransition(
                "S", "I", "I", rate_key=f"{prefix}edge_infection_rate"
            ),
            PoissonTransition("I", "R", rate_key=f"{prefix}recovery_rate"),
        )
        if immunity_loss:
            transitions = transitions + (
                PoissonTransition("R", "S", rate_key=f"{prefix}immunity_loss_rate"),
            )
        super().__init__(
            compartments=("S", "I", "R"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )


class SEIRUpdateOp(PoissonCompartmentalUpdateOp):
    def __init__(
        self,
        prefix="",
        delta_t_key="delta_t",
        state_key="compartment",
        immunity_loss=False,
    ):
        transitions = (
            BinaryPoissonTransition(
                "S", "E", "I", rate_key=f"{prefix}edge_expose_rate"
            ),
            PoissonTransition("E", "I", rate_key=f"{prefix}infectious_rate"),
            PoissonTransition("I", "R", rate_key=f"{prefix}recovery_rate"),
        )
        if immunity_loss:
            transitions = transitions + (
                PoissonTransition("R", "S", rate_key=f"{prefix}immunity_loss_rate"),
            )
        super().__init__(
            compartments=("S", "E", "I", "R"),
            delta_t_key=delta_t_key,
            state_key=state_key,
            aux_prefix=prefix,
            transitions=transitions,
        )