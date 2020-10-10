import logging
import time
import re

import click
import tqdm
import plotly
import plotly.graph_objects as go
import scipy.stats as st

import jax
import networkx as nx
from jax import lax
from jax import numpy as jnp
from netprocess import epi, network_process, networks, utils
from matplotlib import pyplot as plt


from .cli import cli

log = logging.getLogger(__name__)


def prepare_plot(rs, nodes, gamma):
    ever_infected = jnp.add(nodes, -rs["compartment_count"]._value[:, 0])
    ever_infected = jnp.divide(ever_infected, nodes)

    newly_infected = rs["compartment_transitions"]._value[:, 1, 0]
    infected = rs["compartment_count"]._value[:, 1]
    susceptible = rs["compartment_count"]._value[:, 0]
    susceptible_part = jnp.divide(susceptible, nodes)

    r = newly_infected[1:] / (gamma * infected[:-1] * susceptible_part[:-1])
    return r, ever_infected


def make_simulation(np, g, comp, params, steps):
    state = np.new_state(g, params_pytree=params, seed=42)
    state.nodes_pytree["compartment"] = comp
    state = np.run(state, steps=steps)
    state.block_on_all()
    rs = state.all_records()
    return rs


def interpolate(x_1, x_2, y, r_1, r_2):
    t = (x_2 - y) / (x_2 - x_1)
    r = t*r_1 + (1-t)*r_2
    return r


def aggregate_graphs(list_of_graphs):
    union = sorted(list(set().union(*[list_of_graphs[i]["ever_infected"] for i in range(len(list_of_graphs))])))
    new_graphs = []
    new_r = [0] * len(union)
    for graph in list_of_graphs:
        j = 0
        new_graph = [0] * len(union)
        for i in range(len(union)):
            while (graph["ever_infected"][j] < union[i]) and j < len(graph["ever_infected"])-1:
                j += 1
            if graph["ever_infected"][j] == union[i] or j == len(graph["ever_infected"])-1:
                new_graph[i] = graph["r"][j]
            else:
                new_graph[i] = interpolate(graph["ever_infected"][j-1], graph["ever_infected"][j], union[i], graph["r"][j-1], graph["r"][j])
        new_graphs.append({"r": new_graph, "ever_infected": union})
    for i in range(len(union)):
        new_r[i] = sum([new_graphs[j]["r"][i] / len(list_of_graphs) for j in range(len(new_graphs))])
    return new_r, union, new_graphs


def remove_hubs(g, max_degree):
    to_delete = []
    for i in range(len(g.nodes)):
        if g.degree[i] > max_degree:
            to_delete.append(i)
    for i in to_delete:
        g.remove_node(i)
    g = nx.relabel.convert_node_labels_to_integers(g)
    return g

@cli.command()
@click.option("-b", "--edge_beta", default=0.05)
@click.option("-g", "--gamma", default=0.07)
@click.option("-I", "--infect", default=100)
@click.option("-n", "--nodes", default=100000)
@click.option("-steps", "--steps", default=200)
@click.option("-k", "--k", default=3)
@click.option("-pt", "--p_triangle", default=0.5)
@click.option("-ci", "--confidence_interval", default=0.05)
@click.option("-md", "--max_degree", default=300)
def r_graph(edge_beta, gamma, infect, nodes, steps, k, p_triangle, confidence_interval, max_degree):
    np = network_process.NetworkProcess(
        [
            epi.SIRUpdateOp(),
            network_process.CountNodeStatesOp(states=3, key="compartment"),
            network_process.CountNodeTransitionsOp(states=3, key="compartment"),
        ]
    )
    params = {"edge_beta": edge_beta, "gamma": gamma}
    name = f"Barabasi-Albert_n={nodes}_k={k}_steps={steps}_infect={infect}_beta={params['edge_beta']}_gamma={params['gamma']}"
    name = re.sub('\.', ',', name)


    log.info(
        f"Network: Barabasi-Albert. n={nodes}, k={k}, cca {nodes*k*2:.2e} directed edges"
    )
    # g = nx.random_graphs.powerlaw_cluster_graph(nodes, k, p_triangle) #nx.random_graphs.barabasi_albert_graph(nodes, k)
    # g2 = nx.random_graphs.random_regular_graph(2*k, nodes)#gnm_random_graph(nodes, nodes*k)

    rng = jax.random.PRNGKey(43)
    # comp = jnp.int32(jax.random.bernoulli(rng, infect / nodes, shape=[nodes]))
    fig = go.Figure()

    for i in [1, 2]:
        k = i * k
        params["edge_beta"] = params["edge_beta"] / i
        for p in [0, 0.5, 0.9]:
            list_of_graphs = []
            for repetition in range(10):
                g = nx.random_graphs.powerlaw_cluster_graph(nodes, k, p)
                g = remove_hubs(g, max_degree)
                new_nodes = len(g.nodes)
                comp = jnp.int32(jax.random.bernoulli(rng, infect / new_nodes, shape=[new_nodes]))
                rs = make_simulation(np, g, comp, params, steps)
                r, ever_infected = prepare_plot(rs, new_nodes, gamma)
                list_of_graphs.append({"r": r, "ever_infected": ever_infected[:-1]})
                #fig.add_trace(go.Scatter(x=ever_infected[:-1] * 100, y=r, name=f"repetition={repetition}, p={p}, k={k}"))
            r_final, ever_infected_final, new_graphs = aggregate_graphs(list_of_graphs)
            std = [jnp.std(jnp.array([new_graphs[j]["r"][m] for j in range(len(new_graphs))], dtype=jnp.float32)) for m in range(len(r_final))]
            intervals = [st.t.interval(1-confidence_interval, len(list_of_graphs) - 1, loc=r_final[m], scale=std[m]) for m in range(len(std))]
            fig.add_trace(go.Scatter(x=ever_infected_final * 100, y=r_final, name=f"p={p}, k={k}", line=dict(width=5)))
            fig.add_trace(go.Scatter(
                name='Upper Bound',
                x=ever_infected_final * 100,
                y=[intervals[m][1] for m in range(len(std))],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                name='Lower Bound',
                x=ever_infected_final * 100,
                y=[intervals[m][0] for m in range(len(std))],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
            #for i in range(len(new_graphs)):
             #   fig.add_trace(go.Scatter(x=new_graphs[i]["ever_infected"] * 100, y=new_graphs[i]["r"], name=f"repetition={i}, mean:p={p}, k={k}"))

    fig.update_layout(
        xaxis_title="Already infected (%)",
        yaxis_title="R",
    )
    fig.show()
    plotly.offline.plot(fig, filename='/home/lenka/Projects/netprocess/netprocess/scripts/figure.html')

