import logging
import time

import click
import tqdm

import jax
import networkx as nx
from jax import lax
from jax import numpy as jnp
from netprocess import epi, network_process, networks, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
def bench_sir(edge_beta, gamma):

    np = network_process.NetworkProcess(
        [
            epi.SIRUpdateOp(),
            network_process.CountNodeStatesOp(states=3, key="compartment"),
            network_process.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    params = {"edge_beta": edge_beta, "gamma": gamma}

    for n in [100, 10000, 1000000]:
        for k in [3, 10]:
            log.info(
                f"Network: Barabasi-Albert. n={n}, k={k}, cca {n*k*2:.2e} directed edges"
            )
            with utils.logged_time("  Creating graph"):
                g = nx.random_graphs.barabasi_albert_graph(n, k)
            with utils.logged_time("  Creating state"):
                state = np.new_state(g, params_pytree=params, seed=42)
            with utils.logged_time("  Warming up JIT"):
                np.warmup_jit(state)
            with utils.logged_time("  One iteration"):
                state2 = np.run(state)
                state2.block_on_all()
            for steps in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
                state2 = np.run(state, steps=steps)
                state2.block_on_all()
                t0 = time.time()
                state2 = np.run(state, steps=steps)
                state2.block_on_all()
                t1 = time.time()
                log.info(f"    {steps} took {t1-t0:.3g} s")

                if t1 - t0 > 0.5:
                    break
            sps = steps / (t1 - t0)
            log.info(
                f"  {steps} steps took {t1-t0:.2g} s,  {sps:.3g} steps/s,  "
                + f"{sps*state.m:.3g} edge_ops/s,  {sps * state.n:.3g} node_ops/s"
            )
