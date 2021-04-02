import logging

import click
import jax
import networkx as nx
from jax import numpy as jnp
from netprocess import epi, network_process, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
@click.option("-I", "--infect", default=100)
@click.option("-n", "--nodes", default=100000)
def epi_demo1(edge_beta, gamma, infect, nodes):
    k = 3

    np = network_process.NetworkProcess(
        [
            epi.SIRUpdateOp(),
            network_process.CountNodeStatesOp(states=3, key="compartment"),
            network_process.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    params = {"edge_beta": edge_beta, "gamma": gamma}

    log.info(
        f"Network: Barabasi-Albert. n={nodes}, k={k}, cca {nodes*k*2:.2e} directed edges"
    )
    with utils.logged_time("  Creating graph"):
        g = nx.random_graphs.barabasi_albert_graph(nodes, k)
    with utils.logged_time("  Creating state"):
        state = np.new_state(g, params_pytree=params, seed=42)
        rng = jax.random.PRNGKey(43)
        comp = jnp.int32(jax.random.bernoulli(rng, infect / nodes, shape=[nodes]))
        state.nodes_pytree["compartment"] = comp
    with utils.logged_time("  Running model"):
        state2 = np.run(state, steps=100)
        state2.block_on_all()
    log.info(np.trace_log())
    rs = state2.all_records()
    print(rs["compartment_count"])
    print(rs["compartment_transitions"])
