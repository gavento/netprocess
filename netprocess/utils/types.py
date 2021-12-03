import jax.numpy as jnp
import typing


Pytree = typing.Union[
    typing.List["Pytree"], typing.Tuple["Pytree"], "PytreeDict", jnp.ndarray
]

PytreeDict = typing.Dict[str, Pytree]

PRNGKey = jnp.ndarray
