import copy
from typing import Tuple

import attr
import jax
import jax.numpy as jnp
import numpy as np

from ...jax_utils import cond, switch
from ...utils import PRNGKey, PytreeDict, KeyOrValue
from .base import OperationBase


class SoftmaxGame(OperationBase):
    """
    For directed or undirected graphs
    For games on undirected graphs, use antisymmetric payoffs, e.g. U1(a1,a2)=U1(a2,a1).

    Given:
    * Payout matrix, shape either:
    ** U[A, A, 2], indexed as [action_from, action_to, 0] for payout of "from" player and [action_from, action_to, 1] for the "to" player
    ** U[A, A] is treated as antisymmetric game, U'[a1, a2, 0]=U[a1,a2] and U'[a1,a2,1]=U[a2,a1]
    * Update probability (per step)
    * Temperature (irrationality)

    State:
    * Node
    ** `action` - last action taken by node player

    Stats:
    * Node
    ** `_{ap}current_payoff` - the player total payoff (sum)
    ** `_{ap}actions_payoff` - array of payoffs for each action (the player could have played) (sums)
    ** `_{ap}switched_action` - 1 if the player has switched action
    ** `_{ap}current_regret` - the current action regret
    * Edge
    ** `_{ap}actions_payoff_from_node`
    ** `_{ap}actions_payoff_to_node`
    """

    def __init__(
        self,
        action_set,
        payouts,
        action_key="action",
        update_probability=0.5,
        temperature=1.0,
        aux_prefix="",
    ):
        self.action_set = action_set
        self.action_key = action_key
        self.payouts = KeyOrValue(payouts)
        self.update_probability = KeyOrValue(update_probability, default=0.5)
        self.temperature = KeyOrValue(temperature, default=1.0)

        self.aux_prefix = aux_prefix
        self._current_payoff_key = f"_{self.aux_prefix}current_payoff"
        self._actions_payoff_key = f"_{self.aux_prefix}actions_payoff"
        self._current_regret_key = f"_{self.aux_prefix}current_regret"
        self._switched_action_key = f"_{self.aux_prefix}switched_action"
        self._actions_payoff_from_node_key = (
            f"_{self.aux_prefix}actions_payoff_from_node"
        )
        self._actions_payoff_to_node_key = f"_{self.aux_prefix}actions_payoff_to_node"

    def prepare_state_pytrees(self, state):
        """
        Prepare the state for running the process.

        Ensures the state has all state variables, missing parameters get defaults (and raise errors if none),
        probabilities are adjusted to delta_t, jax arrays are ensured for all pytree vals.
        """
        if self.action_key not in state.nodes_pytree:
            state.nodes_pytree[self.action_key] = jnp.zeros(state.n, dtype=jnp.int32)

        self.payouts.ensure_in(state.params_pytree)
        self.update_probability.ensure_in(state.params_pytree)
        self.temperature.ensure_in(state.params_pytree)

    def update_edge(self, rng_key, params, edge, from_node, to_node):
        """
        Compute the payoffs for all actions vs the the current action.
        """
        payouts = self.payouts.get_from(params)
        return {
            self._actions_payoff_from_node_key: payouts[:, to_node[self.action_key], 0],
            self._actions_payoff_to_node_key: payouts[from_node[self.action_key], :, 1],
        }

    def update_node(self, rng_key, params, node, in_edges, out_edges):
        actions_payoff = (
            in_edges[self._actions_payoff_to_node_key]["sum"]
            + out_edges[self._actions_payoff_from_node_key]["sum"]
        )
        deg = node["in_deg"] + node["out_deg"]
        actions_payoff_mean = actions_payoff / deg

        ### TODO!
        action = node[self.action_key]

        return {
            self._current_payoff_key: actions_payoff[action],
            self._actions_payoff_key: actions_payoff,
            self._current_regret_key: jnp.max(actions_payoff) - actions_payoff[action],
            self.action_key: action,
            self._switched_action_key: jnp.int32(action != node[self.action_key]),
        }
