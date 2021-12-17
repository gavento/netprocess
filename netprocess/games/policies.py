import jax.numpy as jnp

from ..process import ProcessState
from ..utils import KeyOrValue, KeyOrValueT
from ..operations import NodeUpdateData


class PlayerPolicyBase:
    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        data: NodeUpdateData,
    ) -> jnp.ndarray:
        raise NotImplemented

    def prepare_state_data(self, state: ProcessState):
        pass


class SoftmaxPolicy(PlayerPolicyBase):
    def __init__(self, beta: KeyOrValueT):
        self.beta = KeyOrValue(beta)

    def prepare_state_data(self, state: ProcessState):
        self.beta.ensure_in(state)

    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        data: NodeUpdateData,
    ) -> jnp.ndarray:
        beta = self.beta.get_from(data.state)
        e = jnp.exp(action_utilities * beta)
        return e / jnp.sum(e)


class EpsilonErrorPolicy(PlayerPolicyBase):
    """
    Plays the best action with prob. (1-epsilon) and a uniformly action otherwise.
    """

    def __init__(self, epsilon: KeyOrValueT = 0.0):
        self.epsilon = KeyOrValue(epsilon)

    def prepare_state_data(self, state: ProcessState):
        self.epsilon.ensure_in(state)

    def compute_policy(
        self,
        action_utilities: jnp.DeviceArray,
        data: NodeUpdateData,
    ) -> jnp.ndarray:
        epsilon = self.epsilon.get_from(data.state)
        z = jnp.zeros_like(action_utilities)
        z = z.at[jnp.argmax(action_utilities)].set(1.0 - epsilon)
        z = z + epsilon / action_utilities.shape[0]
        # assert jnp.abs(jnp.sum(z) - 1.0) < 1e-3
        return z
