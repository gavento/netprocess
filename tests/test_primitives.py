import jax
import jax.numpy as jnp
import netprocess
import pytest
from netprocess import primitives


def test_jax_random_choice_range():
    """Ensure that random.choice normalizes the probabilities."""
    k = jax.random.PRNGKey(42)
    s = jax.random.choice(k, 2, (1000,), p=[1000.0, 2000.0])
    assert sum(s) < 900
    assert sum(s) > 300


def test_build_switch():
    s0 = primitives.build_switch([lambda: [41, 42], [43, 44]])
    assert s0(0) == [41, 42]
    assert s0(1) == [43, 44]

    s1 = primitives.build_switch([lambda x: x + 1, lambda _: 42, 43])
    assert s1(0, 3) == 4
    assert s1(1, 0) == 42
    assert s1(2, 0) == 43

    s3 = primitives.build_switch([lambda x, y, z: x + y + z, lambda *_: 42, 43])
    assert s3(0, 3, 4, 5) == 12
    assert s3(1, 0, 1, 2) == 42
    assert s3(2, 0, 0, 0) == 43


def test_build_update_function():
    k = jax.random.PRNGKey(42)

    # Builds oscillators with different periods
    tf1 = primitives.build_update_function(
        [
            lambda t, limit: jax.lax.cond(
                t >= limit - 1, lambda _: [0.0, 1.0], lambda _: [1.0, 0.0], None
            ),
            lambda t, limit: jax.lax.cond(
                t >= limit - 1, lambda _: [1.0, 0.0], lambda _: [0.0, 1.0], None
            ),
        ]
    )
    periods = jnp.array([1, 2, 3])
    s0 = jnp.array([0, 0, 1])
    t0 = jnp.array([0, 0, 2])
    s1, t1 = tf1(k, s0, t0, periods)
    assert list(s1) == [1, 0, 0]
    assert list(t1) == [0, 1, 0]
    s2, t2 = tf1(k, s1, t1, periods)
    assert list(s2) == [0, 1, 0]
    assert list(t2) == [0, 0, 1]

    # Builds random coins
    tf2 = primitives.build_update_function(
        [jnp.array([0.5, 0.5]), jnp.array([0.5, 0.5])]
    )
    s0 = jnp.zeros(10000, dtype=jnp.int32)
    t0 = jnp.zeros(10000, dtype=jnp.int32)
    s1, t1 = tf2(k, s0, t0)
    assert all(s1 == 1 - t1)
    assert sum(s1) > 4000
    assert sum(s1) < 6000
