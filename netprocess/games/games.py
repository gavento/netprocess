from typing import Tuple

import jax
import jax.numpy as jnp

from ..operations import EdgeUpdateData, NodeUpdateData, OperationBase
from ..process import ProcessState
from ..utils import KeyOrValue, KeyOrValueT, PropTree
from .policies import PlayerPolicyBase


class PureStrategyGame(OperationBase):
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
        action_set: Tuple[str],
        payouts: KeyOrValueT,
        player_policy: PlayerPolicyBase,
        action_key: str = "action",
        update_probability: KeyOrValue = 0.5,
        aux_prefix="",
        player_mean_payout: bool = False,
    ):
        self.action_set = tuple(action_set)
        self.payouts = KeyOrValue(payouts)
        assert isinstance(player_policy, PlayerPolicyBase)
        self.player_policy = player_policy
        self.action_key = action_key
        self.update_probability = KeyOrValue(update_probability)
        self.player_mean_payout = player_mean_payout

        self.aux_prefix = aux_prefix
        self._current_payoff_key = f"_{self.aux_prefix}current_payoff"
        self._actions_payoff_key = f"_{self.aux_prefix}actions_payoff"
        self._current_regret_key = f"_{self.aux_prefix}current_regret"
        self._updated_action_key = f"_{self.aux_prefix}updated_action"
        self._changed_action_key = f"_{self.aux_prefix}changed_action"
        self._actions_payoff_from_node_key = (
            f"_{self.aux_prefix}actions_payoff_from_node"
        )
        self._actions_payoff_to_node_key = f"_{self.aux_prefix}actions_payoff_to_node"

    @property
    def n(self):
        return len(self.action_set)

    def get_payoff(self, a1, a2, player, data: PropTree = None) -> jnp.DeviceArray:
        p = self.payouts.get_from(data)
        assert p.shape == (self.n, self.n, 2) or p.shape == (self.n, self.n)
        if len(p.shape) == 3:
            return p[a1, a2, player]
        else:
            return jax.lax.cond(
                player == 0, lambda _: p[a1, a2], lambda _: p[a2, a1], None
            )

    def prepare_state_data(self, state: ProcessState):
        """
        Prepare the state for running the process.

        Ensures the state has all state variables, missing parameters get defaults (and raise errors if none),
        probabilities are adjusted to delta_t, jax arrays are ensured for all pytree vals.
        """
        if self.action_key not in state.node:
            state.node[self.action_key] = jnp.zeros(state.n, dtype=jnp.int32)

        self.payouts.ensure_in(state)
        ps = self.payouts.get_from(state).shape
        assert ps == (self.n, self.n, 2) or ps == (self.n, self.n)
        self.update_probability.ensure_in(state)
        self.player_policy.prepare_state_data(state)

    def update_edge(self, data: EdgeUpdateData) -> PropTree:
        """
        Compute the payoffs for all actions vs the the current action.
        """
        return {
            self._actions_payoff_from_node_key: self.get_payoff(
                slice(None), data.tgt_node[self.action_key], 0
            ),
            self._actions_payoff_to_node_key: self.get_payoff(
                data.src_node[self.action_key], slice(None), 1
            ),
        }

    def update_node(self, data: NodeUpdateData) -> PropTree:
        actions_payoff = (
            data.in_edges["sum"][self._actions_payoff_to_node_key]
            + data.out_edges["sum"][self._actions_payoff_from_node_key]
        )
        if self.player_mean_payout:
            deg = jnp.maximum(data.node["deg"], 1)
            actions_payoff = actions_payoff / deg
        prng_key0, prng_key1, prng_key2 = jax.random.split(data.prng_key, 3)
        do_update = jax.random.bernoulli(
            prng_key0, self.update_probability.get_from(data)
        )
        action_probs = self.player_policy.compute_policy(
            actions_payoff, data._replace(prng_key=prng_key1)
        )
        new_action = jax.random.choice(prng_key2, self.n, p=action_probs)
        action = jax.lax.cond(
            do_update, lambda _: new_action, lambda _: data.node[self.action_key], None
        )

        return {
            self._current_payoff_key: actions_payoff[action],
            self._actions_payoff_key: actions_payoff,
            self._current_regret_key: jnp.max(actions_payoff) - actions_payoff[action],
            self.action_key: action,
            self._updated_action_key: jnp.int32(do_update),
            self._changed_action_key: jnp.int32(action != data.node[self.action_key]),
        }
