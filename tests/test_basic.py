import jax
import pytest


def test_jax_random_choice_range():
    """Ensure that random.choice normalizes the probabilities."""
    k = jax.random.PRNGKey(42)
    s = jax.random.choice(k, 2, (1000,), p=[1000.0, 2000.0])
    assert sum(s) < 800
    assert sum(s) > 400
