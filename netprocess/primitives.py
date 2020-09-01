from typing import List, Callable, Any, Union
import jax


def build_switch(funs: List[Any]) -> Callable[..., Any]:
    """
    Build a switch statement from jax primitives.

    Given a list of functions (or constants), return a function
    `switch(x, *args)` that calls the `x`-th function and returns the result
    (resp. returns the constant)
    """
    assert len(funs) > 0
    f = funs[-1]
    if not callable(f):
        f = lambda *_: funs[-1]

    if len(funs) == 1:
        return lambda _, *a: f(*a)

    return lambda x, *a: jax.lax.cond(
        x >= len(funs) - 1,
        lambda _: f(*a),
        lambda _: build_switch(funs[:-1])(x, *a),
        None,
    )


def switch(i: int, funs: List[Any], *args):
    """
    Return `funs[i](*args)`
    """
    f = funs[-1]
    if not callable(f):
        f = lambda *_: funs[-1]
    return jax.lax.cond(
        i >= len(funs) - 1,
        lambda _: f(*args),
        lambda _: switch(i, funs[:-1], *args),
        None,
    )


def update_node_states(prop_dict_v, update_funs):
    """
    Given
    """

    def update_node_state(prop_dict):
        "prop_dict -> update_dict"
        state = prop_dict["state"]
        return switch(state, update_funs, prop_dict)

    update_node_state = jax.vmap(update_node_state)


def update_node_state(
    prop_dict,
    update_funs,
):
    state = prop_dict["state"]
    updates = switch(state, update_funs, prop_dict)
    res = {}
    for k in prop_dict:
        if k in updates:
            res[k] = updates[k]
        else:
            res[k] = prop_dict[k]


def build_update_function(transition_likelihood_functions):
    """
    Build a JAX update function for a vector of nodes.

    Input: list of function (or constants), where i-th returns likelihoods of transition
    from state i to all states 0..k-1 as a verctor. The functions have signature:
    `(property_dict)->[likelihood_of_states]`.

    Returns: a function that computes the next-tick value of the graph state structure:
    `(rng_key, state, time_in_state, *node_properties)->(state2, time_in_state2)`

    The time_in_state is the number of steps that the node stayed in the same state.
    I.e. it is 0 right after state change.
    """
    switch = build_switch(transition_likelihood_functions)

    def update_node(args):
        "(key, state, time_in_state, *node_properties) -> (state2, time_in_state2)"
        key, state, time_in_state, *node_properties = args
        probs = switch(state, time_in_state, *node_properties)
        # Note: the probs are not normalized, this is done by jax.random.choice
        state2 = jax.random.choice(key, len(transition_likelihood_functions), p=probs)
        return (
            state2,
            jax.lax.cond(
                state2 == state, lambda _: time_in_state + 1, lambda _: 0, None
            ),
        )

    update_node_vmap = jax.vmap(update_node)

    def update_all(rng_key, state_v, time_in_state_v, *node_properties_v):
        "Returns `(state2_v, time_in_state2_v)`"
        rng_v = jax.random.split(rng_key, num=state_v.shape[0])
        state2_v, time2_v = update_node_vmap(
            (rng_v, state_v, time_in_state_v, *node_properties_v)
        )
        return (state2_v, time2_v)

    return update_all
