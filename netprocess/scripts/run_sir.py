import logging
import time

import click
import jax
import networkx as nx
import numpy as np
import tqdm
from jax import lax
from jax import numpy as jnp
from netprocess import epi, network_process, data, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.option("-b", "--edge_beta", default=0.05, type=float)
@click.option("-g", "--gamma", default=0.07, type=float)
@click.option(
    "-I",
    "--infect",
    default=1e-3,
    type=float,
    help="Mean fraction infected (IID Bernoulli)",
)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-O", "--output_prefix", default="sim_SEIR_", type=str)
@click.option("-S", "--steps", default=1000, type=int)
@click.option("-D", "--delta_t", default=0.1, type=float)
@click.argument("network")
def run_sir(edge_beta, gamma, infect, seed, output_prefix, network, steps, delta_t):
    if seed is None:
        seed = np.random.randint(1000000)

    net = data.Network.load(network)
    log.info(f"Loaded {network!r} n={net.meta['n']}, m={net.meta['m']} (directed)")

    net.params["edge_beta"] = edge_beta
    net.params["gamma"] = gamma
    net.params["delta_t"] = delta_t

    net.meta["sim_edge_beta"] = edge_beta
    net.meta["sim_gamma"] = edge_beta
    net.meta["sim_seed"] = seed
    net.meta["sim_steps"] = steps
    net.meta["sim_delta_t"] = delta_t

    process = network_process.NetworkProcess(
        [
            epi.SIRUpdateOp(),
            network_process.CountNodeStatesOp(states=3, key="compartment"),
            network_process.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    net.node_props["compartment"] = jnp.int32(
        jax.random.bernoulli(
            jax.random.PRNGKey(seed + 1), infect, shape=[net.meta["n"]]
        )
    )
    state = process.new_state(net, seed=seed)

    with utils.logged_time("Running simulation"):
        state2 = process.run(state, steps=steps)
        state2.block_on_all()

    # TODO: Make SIR model aware of delta_t
    # TODO: Store as a ProcessRun instance
    log.info(process.trace_log())
    rs = state2.all_records()
    print(rs["compartment_count"])
    print(rs["compartment_transitions"])
