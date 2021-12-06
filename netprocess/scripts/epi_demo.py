import logging
import time

import click
import jax
import networkx as nx
from jax import numpy as jnp
from netprocess import Network, NetworkProcess, epidemics, operations, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
@click.option("-I", "--infect", default=100)
@click.option("-n", "--nodes", default=100000)
@click.option("-s", "--steps", default=100)
def epi_demo(edge_beta, gamma, infect, nodes, steps):
    k = 3
    np = NetworkProcess(
        [
            epidemics.SIRUpdateOp(),
            # operations.CountNodeStatesOp(states=3, key="compartment"),
            # operations.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    params = {"edge_infection_rate": edge_beta, "recovery_rate": gamma}

    log.info(
        f"Network: Barabasi-Albert. n={nodes}, k={k}, cca {nodes*k*2:.2e} directed edges"
    )
    with utils.logged_time("  Creating graph", logger=log):
        g = nx.random_graphs.barabasi_albert_graph(nodes, k)
    with utils.logged_time("  Creating state", logger=log):
        net = Network.from_graph(g)
        state = np.new_state(net, props=params, seed=42)
        rng = jax.random.PRNGKey(43)
        comp = jnp.int32(jax.random.bernoulli(rng, infect / nodes, shape=[nodes]))
        state.node["compartment"] = comp
    with utils.logged_time("  Running model", logger=log):
        t0 = time.time()
        state2 = np.run(state, steps=steps)
        state2.block_on_all()
        t1 = time.time()

    log.info(np.trace_log())
    sps = steps / (t1 - t0)
    log.info(
        f"{steps} steps took {t1-t0:.2g} s,  {sps:.3g} steps/s,  "
        + f"{sps*state.m:.3g} edge_ops/s,  {sps * state.n:.3g} node_ops/s"
    )

    # rs = state2.records.all_records()
    # print(rs["compartment_count"])
    # print(rs["compartment_transitions"])
