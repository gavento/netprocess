import logging
import random
import time
import tqdm

import jax
from jax import lax
from jax import numpy as jnp

from . import messages, networks, primitives
from .utils import ArrayDict, cond

log = logging.getLogger(__name__)


class CompartmentProcess:
    def warmup(self, report=False):
        t0 = time.time()
        nd = self.initial_node_data([0, 0])
        edges = jnp.array([[0, 1]])
        rng = jax.random.PRNGKey(0)
        nd = self.update_fn(rng, nd, edges)
        _ = nd["state"][0]
        if report:
            log.info(f"JIT warmup of {self!r} took {time.time() - t0} s")

    def new_instance(self, n, graph_or_edges, rng_key=None):
        if isinstance(graph_or_edges, nx.Graph):
            graph_or_edges = networks.nx_graph_to_edges(graph_or_edges)
        return ProcessInstance(self, graph_or_edges, [0] * n, rng_key=rng_key)

    def initial_node_data(self, initial_states):
        initial_states = jnp.array(initial_states, dtype=jnp.int32)
        return {"state": initial_states}

    def instance_stats(self, instance):
        return {
            "compartments": jnp.sum(
                jax.nn.one_hot(instance.history[-1], len(states)), axis=0
            )
            / n
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class SIRProcess(CompartmentProcess):
    COMPARTMENTS = ("S", "I", "R")

    def initial_node_data(self, initial_states):
        node_data = super().initial_node_data(initial_states)
        n = node_data["state"].shape[0]
        if self.count_time:
            node_data["time"] = jnp.zeros(n, dtype=jnp.int32)
        if self.count_infected:
            node_data["infected"] = jnp.zeros(n, dtype=jnp.int32)
        return node_data

    def __init__(
        self, edge_beta, gamma, count_time=False, count_infected=False, jit=True
    ):
        self.count_time = count_time
        self.count_infected = count_infected
        self.edge_beta = edge_beta
        self.gamma = gamma
        self.update_fn = build_sir_update(
            self.edge_beta,
            self.gamma,
            count_time=self.count_time,
            count_infected=self.count_infected,
            jit=jit,
        )


class ProcessInstance:
    def __init__(self, process: CompartmentProcess, edges, initial_states, rng=None):
        self.process = process
        self.step = 0
        self.edges = edges
        self.node_data = self.process.initial_node_data(initial_states)
        self.history = [self.node_data]
        self.stats = [self.process.instance_stats(self)]

    def step(self, log_every=None):
        if log_every is not None and self.step % log_every == 0:
            while len(states) <= max(node_data["state"]):
                states.append(f"S{len(states)}")
            sc = jnp.sum(jax.nn.one_hot(node_data["state"], len(states)), axis=0) / n
            log.info(
                f"Step {self.step:-4d}  "
                + "".join(f"{s}: {v:.3f}  " for s, v in zip(states, sc))
            )
        self.history.append(node_data)
        r, rng_key = jax.random.split(rng_key)
        node_data = update_fn(r, node_data, edges)


def run_compartment_process(
    update_fn,
    steps,
    initial_states,
    edges,
    log_every=None,
    rng_key=None,
    states=(),
    progress=True,
):
    initial_states = jnp.array(initial_states, dtype=jnp.int32)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(random.randint(0, 1 << 63))
    n = initial_states.shape[0]
    node_data = {
        "state": initial_states,
        "time": jnp.zeros(n, dtype=jnp.int32),
        "infected": jnp.zeros(n, dtype=jnp.int32),
    }
    history = []
    stats = []

    def step(i):
        nonlocal node_data, history, stats, rng_key
        if log_every is not None and i % log_every == 0:
            while len(states) <= max(node_data["state"]):
                states.append(f"S{len(states)}")
            sc = jnp.sum(jax.nn.one_hot(node_data["state"], len(states)), axis=0) / n
            log.info(
                f"Step {i:-4d}  "
                + "".join(f"{s}: {v:.3f}  " for s, v in zip(states, sc))
            )
        history.append(node_data)
        r, rng_key = jax.random.split(rng_key)
        node_data = update_fn(r, node_data, edges)

    for i in tqdm.trange(0, 1, desc="modelling (JIT step)", disable=not progress):
        step(i)

    for i in tqdm.trange(1, steps, desc="modelling", disable=not progress):
        step(i)

    return history, stats


def build_sir_update(
    edge_beta: float,
    gamma: float,
    count_time=False,
    count_infected=False,
    jit=True,
):
    """
    Returns a SIR-update function with optional stats keeping.

    States: 0=S, 1=I, 2=R
    Returns: `update_SIR(rng_key, nodes_dict, edges_from_to) -> node_dict`.
    """

    def sir_ef(r, ed, fnd, tnd):
        infection = cond(
            jnp.logical_and(fnd["state"] == 1, tnd["state"] == 0),
            lambda: jnp.int32(jax.random.bernoulli(r, edge_beta)),
            0,
        )
        return {}, {"infection": infection}, {"infection": infection}

    def sir_nf(r, nd, ied, oed):
        state = primitives.switch(
            nd["state"],
            [
                # S -> {S, I}:
                lambda: cond(ied["infection"] > 0, 1, 0),
                # I -> {I, R}:
                lambda: cond(jax.random.bernoulli(r, gamma), 2, 1),
                # R -> {R}:
                2,
            ],
        )
        d = {"state": state}

        if count_time:
            d["time"] = cond(state == nd["state"], lambda: nd["time"] + 1, 0)
        if count_infected:
            d["infected"] = nd["infected"] + oed["infection"]
        return d

    f = messages.build_step_edge_node_messages(sir_ef, sir_nf, jit=jit)

    def update_SIR(rng_key, nodes_dict, edges_from_to):
        return f(rng_key, nodes_dict, {}, edges_from_to)[0]

    return update_SIR
