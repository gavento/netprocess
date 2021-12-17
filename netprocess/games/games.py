from typing import Tuple

import jax
import jax.numpy as jnp
from numpy import float32

from ..operations import EdgeUpdateData, NodeUpdateData, OperationBase
from ..process import ProcessState
from ..utils import KeyOrValue, KeyOrValueT, PropTree
from .policies import PlayerPolicyBase, EpsilonErrorPolicy


class NormalFormGameBase(OperationBase):
    """
    A base class for normal-form (matrix) 2-player games.

    For directed or undirected graphs. On undirected graphs, use antisymmetric
    payoffs to get meaningful results (U1(a1,a2)=U1(a2,a1)).

    Given:
    * Payout matrix, shape either:
    ** U[A, A, 2], indexed as [action_from, action_to, 0] for payout of "from"
       player and [action_from, action_to, 1] for the "to" player
    ** U[A, A] specifies an antisymmetric game: U'[a1, a2, 0]=U[a1,a2] and U'[a1,a2,1]=U[a2,a1]

    Variables:
    * Node
    ** `action` - the action played this round (for which other stats are computed)
    ** `_current_payoff` - the player payoff in this turn
    ** `_action_payoffs` - array of payoffs for each possible action
    ** `next_action` - the action selected for next turn (not evaluated yet)
    * Edge
    ** `_src_node_action_payoffs`
    ** `_tgt_node_action_payoffs`
    """

    def __init__(
        self,
        action_set: Tuple[str],
        payouts: KeyOrValueT,
        path: str = "",
        player_payout_is_mean: bool = False,
    ):
        self.action_set = tuple(action_set)
        self.payouts = KeyOrValue(payouts)
        self.path = path
        self.action = KeyOrValue(self.key("action"), default=0)
        self.next_action = KeyOrValue(self.key("next_action"))
        self.player_payout_is_mean = player_payout_is_mean
        self.action_counts = KeyOrValue(
            self.key("action_counts"), default=jnp.zeros(self.n, dtype=jnp.int32)
        )
        self._action_payoffs = KeyOrValue(self.key("_action_payoffs"))
        self._current_payoff = KeyOrValue(self.key("_current_payoff"))
        self.cummulative_regret = KeyOrValue(
            self.key("cummulative_regret"), default=0.0
        )
        self._empirical_strategy = KeyOrValue(self.key("_empirical_strategy"))

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

    def key(self, key):
        "Return key prefixed with `self.path`"
        return f"{self.path}.{key}" if self.path else key

    def prepare_state_data(self, state: ProcessState):
        """
        Prepare the state for running the process.

        Ensures the state has all state variables, missing parameters get defaults (and raise errors if none),
        probabilities are adjusted to delta_t, jax arrays are ensured for all pytree vals.
        """
        self.payouts.ensure_in(state)
        ps = self.payouts.get_from(state).shape
        assert ps == (self.n, self.n, 2) or ps == (self.n, self.n)
        self.action.ensure_in(state.node, repeat_times=state.n)
        self.next_action.ensure_in(state.node, repeat_times=state.n)
        self.action_counts.ensure_in(state.node, repeat_times=state.n)
        self.cummulative_regret.ensure_in(state.node, repeat_times=state.n)

    def update_edge(self, data: EdgeUpdateData) -> PropTree:
        """
        Compute the payoffs for all actions vs the the other player's `next_action`.
        """
        return {
            self.key("_src_node_action_payoffs"): self.get_payoff(
                slice(None), self.next_action.get_from(data.tgt_node), 0
            ),
            self.key("_tgt_node_action_payoffs"): self.get_payoff(
                self.next_action.get_from(data.src_node), slice(None), 1
            ),
        }

    def update_node(self, data: NodeUpdateData) -> PropTree:
        actions_payoff = (
            data.in_edges["sum"][self.key("_tgt_node_action_payoffs")]
            + data.out_edges["sum"][self.key("_src_node_action_payoffs")]
        )
        if self.player_payout_is_mean:
            deg = jnp.maximum(data.node["deg"], 1)
            actions_payoff = actions_payoff / deg
        next_action = self.next_action.get_from(data.node)
        action_counts = self.action_counts.get_from(data.node).at[next_action].add(1)
        return {
            self._action_payoffs.key: actions_payoff,
            self._current_payoff.key: actions_payoff[next_action],
            self.action_counts.key: action_counts,
            self.action.key: next_action,
            self._empirical_strategy.key: action_counts / jnp.sum(action_counts),
        }


class BestResponseGame(NormalFormGameBase):
    """"""

    def __init__(
        self,
        action_set: Tuple[str],
        payouts: KeyOrValueT,
        player_policy: PlayerPolicyBase = None,
        path: str = "",
        update_probability: KeyOrValue = 1.0,
    ):
        super().__init__(action_set, payouts=payouts, path=path)
        if player_policy is None:
            player_policy = EpsilonErrorPolicy(0)
        assert isinstance(player_policy, PlayerPolicyBase)
        self.player_policy = player_policy
        self.update_probability = KeyOrValue(update_probability)

    def prepare_state_data(self, state: ProcessState):
        super().prepare_state_data(state)
        self.update_probability.ensure_in(state)
        self.player_policy.prepare_state_data(state)

    def update_node(self, data: NodeUpdateData) -> PropTree:
        up = super().update_node(data)
        actions_payoff = self._action_payoffs.get_from(up)
        prng_key0, prng_key1, prng_key2 = jax.random.split(data.prng_key, 3)
        do_update = jax.random.bernoulli(
            prng_key0, self.update_probability.get_from(data)
        )
        action_probs = self.player_policy.compute_policy(
            actions_payoff, data._replace(prng_key=prng_key1)
        )
        new_action = jax.random.choice(prng_key2, self.n, p=action_probs)
        action = jax.lax.cond(
            do_update, lambda _: new_action, lambda _: data.node[self.action.key], None
        )
        return dict(**up, **{self.next_action.key: action})


class RegretMatchingGame(NormalFormGameBase):
    """"""

    def __init__(
        self,
        action_set: Tuple[str],
        payouts: KeyOrValueT,
        path: str = "",
        mu: KeyOrValue = None,
    ):
        super().__init__(action_set, payouts=payouts, path=path)
        if mu is None:
            mu = (self.n + 1) * (jnp.max(payouts) - jnp.min(payouts))
        self.mu = KeyOrValue(mu)
        # action_counterfactual_payoffs is indexed by (played_action, counterfactual_action)
        self.action_counterfactual_payoffs = KeyOrValue(
            self.key("action_counterfactual_payoffs"),
            default=jnp.zeros((self.n, self.n), dtype=jnp.float32),
        )
        self._action_probabilities = KeyOrValue(self.key("_action_probabilities"))
        self._current_regrets = KeyOrValue(self.key("_current_regrets"))

    def prepare_state_data(self, state: ProcessState):
        super().prepare_state_data(state)
        self.mu.ensure_in(state)
        self.action_counterfactual_payoffs.ensure_in(state.node, repeat_times=state.n)

    def update_node(self, data: NodeUpdateData) -> PropTree:
        up = super().update_node(data)
        last_action = self.action.get_from(up)
        action_payoffs = self._action_payoffs.get_from(up)
        action_counterfactual_payoffs = (
            self.action_counterfactual_payoffs.get_from(data.node)
            .at[last_action, :]
            .add(action_payoffs)
        )
        current_regrets = jnp.maximum(
            0.0,
            (
                action_counterfactual_payoffs[last_action, :]
                - action_counterfactual_payoffs[last_action, last_action]
            )
            / (1 + data.state.step),
        )
        probs = current_regrets / self.mu.get_from(data)
        probs = probs.at[last_action].add(1.0 - jnp.sum(probs))
        probs = jnp.maximum(0.0, probs)
        probs = probs / jnp.sum(probs)
        new_action = jax.random.choice(data.prng_key, self.n, p=probs)
        return dict(
            **up,
            **{
                self.next_action.key: new_action,
                self.action_counterfactual_payoffs.key: action_counterfactual_payoffs,
                self._action_probabilities.key: probs,
                self._current_regrets.key: current_regrets,
            },
        )
