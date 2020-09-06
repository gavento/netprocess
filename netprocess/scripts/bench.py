import logging

import click
import tqdm

import jax
import networkx as nx
from jax import lax
from jax import numpy as jnp
from netprocess import networks, seir

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.option("-N", default=1000)
@click.option("-K", default=3)
@click.option("-s", "--steps", default=1000)
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
def bench_sir2(n, k, steps, edge_beta, gamma):

    for _ in tqdm.trange(1, desc="create graph"):
        g = nx.random_graphs.barabasi_albert_graph(n, k)
        log.info(
            f"Graph: n={g.order()}, m={g.size()}, avg_deg={2 * g.size() / g.order()}"
        )
    for _ in tqdm.trange(1, desc="edge list"):
        edges = networks.nx_graph_to_edges(g)

    nd = {
        "state": jnp.zeros(n, dtype=jnp.int32),
        "time": jnp.zeros(n, dtype=jnp.int32),
        "infected": jnp.zeros(n, dtype=jnp.int32),
    }
    nd["state"] = jax.ops.index_update(nd["state"], n // 2, 1)

    update = seir.build_sir_update(
        edge_beta, gamma, count_time=True, count_infected=True
    )

    seir.run_compartment_process(
        update,
        steps,
        [1] + [0] * (n - 1),
        edges,
        log_every=max(steps // 20, 1),
        states=("S", "I", "R"),
    )


@cli.command()
@click.option("-N", default=1000)
@click.option("-K", default=3)
@click.option("-s", "--steps", default=1000)
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
def bench_sir(n, k, steps, edge_beta, gamma):
    rng = jax.random.PRNGKey(42)

    for _ in tqdm.trange(1, desc="create graph"):
        g = nx.random_graphs.barabasi_albert_graph(n, k)
        log.info(
            f"Graph: n={g.order()}, m={g.size()}, avg_deg={2 * g.size() / g.order()}"
        )
    for _ in tqdm.trange(1, desc="edge list"):
        edges = networks.nx_graph_to_edges(g)

    node_states = jnp.zeros(n, dtype=jnp.int32)
    node_states = jax.ops.index_update(node_states, 0, 1)

    for _ in tqdm.trange(1, desc="build update function"):
        upf = seir.build_sir_update_function(edge_beta, gamma)
    for _ in tqdm.trange(1, desc="JIT"):
        upf_jit = jax.jit(upf)
        upf_jit(rng, edges, node_states)[0]

    for i in tqdm.trange(steps, desc="SIR steps"):
        if i % max(steps // 50, 1) == 0:
            counts = jnp.sum(jax.nn.one_hot(node_states, 3), axis=0) / n
            log.info(
                f"Step {i:-4d}   S: {counts[0]:.3f}   I: {counts[1]:.3f}   R: {counts[2]:.3f}"
            )
        rng2, rng = jax.random.split(rng)
        node_states = upf_jit(rng2, edges, node_states)
