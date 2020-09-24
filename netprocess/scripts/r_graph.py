import logging
import time
import re

import click
import tqdm

import jax
import networkx as nx
from jax import lax
from jax import numpy as jnp
from netprocess import epi, network_process, networks, utils
from matplotlib import pyplot as plt


from .cli import cli

log = logging.getLogger(__name__)


def prepare_plot(rs, nodes, serial_interval):
    ever_infected = jnp.add(nodes, -rs["compartment_count"]._value[:, 0])
    ever_infected = jnp.divide(ever_infected, nodes)

    newly_infected = rs["compartment_transitions"]._value[:, 1, 0]
    infected = rs["compartment_count"]._value[:, 1]
    susceptible = rs["compartment_count"]._value[:, 0]
    susceptible_part = jnp.divide(susceptible, nodes)

    r = serial_interval * jnp.divide(newly_infected[1:], jnp.multiply(infected[:-1], susceptible_part[:-1]))
    return r, ever_infected


@cli.command()
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
@click.option("-I", "--infect", default=100)
@click.option("-n", "--nodes", default=100000)
@click.option("-si", "--serial_interval", default=4)
@click.option("-steps", "--steps", default=100)
@click.option("-k", "--k", default=3)
def r_graph(edge_beta, gamma, infect, nodes, serial_interval, steps, k):
    np = network_process.NetworkProcess(
        [
            epi.SIRUpdateOp(),
            network_process.CountNodeStatesOp(states=3, key="compartment"),
            network_process.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    params = {"edge_beta": edge_beta, "gamma": gamma}
    name = f"Barabasi-Albert_n={nodes}_k={k}_steps={steps}_SI={serial_interval}_infect={infect}_beta={params['edge_beta']}_gamma={params['gamma']}"
    name = re.sub('\.', ',', name)


    log.info(
        f"Network: Barabasi-Albert. n={nodes}, k={k}, cca {nodes*k*2:.2e} directed edges"
    )
    with utils.logged_time("  Creating graph"):
        g = nx.random_graphs.barabasi_albert_graph(nodes, k)
        g2 = nx.random_graphs.gnm_random_graph(nodes, nodes*k)
    with utils.logged_time("  Creating state"):
        state = np.new_state(g, params_pytree=params, seed=42)
        state2 = np.new_state(g2, params_pytree=params, seed=42)
        rng = jax.random.PRNGKey(43)
        comp = jnp.int32(jax.random.bernoulli(rng, infect / nodes, shape=[nodes]))
        state.nodes_pytree["compartment"] = comp
        state2.nodes_pytree["compartment"] = comp
    with utils.logged_time("  Running model"):
        state = np.run(state, steps=steps)
        state2 = np.run(state2, steps=steps)
        state.block_on_all()
        state2.block_on_all()
    log.info(np.trace_log())
    rs = state.all_records()
    rs2 = state2.all_records()

    r1, ever_infected1 = prepare_plot(rs, nodes, serial_interval)
    r2, ever_infected2 = prepare_plot(rs2, nodes, serial_interval)

    plt.subplot(211, title='Barabasi-Albert', xlabel='% already infected', ylabel='R')
    plt.plot(ever_infected1[:-1]*100, r1)

    plt.subplot(212, title = "random_regular", xlabel='% already infected', ylabel='R')
    plt.plot(ever_infected2[:-1] * 100, r2)

    #plt.xlabel('% already infected')
    #plt.ylabel('R')
    #plt.title(name)
    plt.savefig(name)
    plt.show()

