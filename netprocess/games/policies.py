import jax.numpy as jnp
import numpy as np

from ..utils import KeyOrValue, KeyOrValueT, PRNGKey, PytreeDict


class PlayerPolicyBase:
    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        rng_key: PRNGKey,
        params: PytreeDict,
        node: PytreeDict,
        in_edges: PytreeDict,
        out_edges: PytreeDict,
    ) -> jnp.DeviceArray:
        raise NotImplemented

    def prepare_state_pytrees(self, state):
        pass


class SoftmaxPolicy(PlayerPolicyBase):
    def __init__(self, beta: KeyOrValueT):
        self.beta = KeyOrValue(beta)

    def prepare_state_pytrees(self, state):
        self.beta.ensure_in(state.params)

    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        rng_key: PRNGKey,
        params: PytreeDict,
        node: PytreeDict,
        in_edges: PytreeDict,
        out_edges: PytreeDict,
    ) -> jnp.DeviceArray:
        beta = self.beta.get_from(params)
        e = jnp.exp(action_utilities * beta)
        return e / jnp.sum(e)


class EpsilonErrorPolicy(PlayerPolicyBase):
    """
    Plays the best action with prob. (1-epsilon) and a uniformly action otherwise.
    """

    def __init__(self, epsilon: KeyOrValueT = 0.0):
        self.epsilon = KeyOrValue(epsilon)

    def prepare_state_pytrees(self, state):
        self.epsilon.ensure_in(state.params)

    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        rng_key: PRNGKey,
        params: PytreeDict,
        node: PytreeDict,
        in_edges: PytreeDict,
        out_edges: PytreeDict,
    ) -> jnp.DeviceArray:
        epsilon = self.epsilon.get_from(params)
        z = jnp.zeros_like(action_utilities)
        z = z.at[jnp.argmax(action_utilities)].set(1.0 - epsilon)
        z = z + epsilon / action_utilities.shape[0]
        assert jnp.abs(jnp.sum(z) - 1.0) < 1e-3
        return z
