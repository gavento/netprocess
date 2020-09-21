import logging
import time

import click
import tqdm

import jax
import networkx as nx
from jax import lax
from jax import numpy as jnp
from netprocess import epi, network_process, networks, utils
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)
edge_beta = 0.05
gamma = 0.07
np = network_process.NetworkProcess([epi.SIRUpdateOperation()])
params = {"edge_beta": edge_beta, "gamma": gamma}
n = 100
k = 3
print("Network: Barabasi-Albert. n={}, k={}, cca {:.2e} directed edges".format(n, k, n*k*2))
g = nx.random_graphs.barabasi_albert_graph(n, k)
state = np.new_state(g, params_pytree=params, seed=42)
np.warmup_jit(state)
n_steps = 100
infected_each_day = []
compartment = state.nodes_pytree["compartment"]._value
infected = jnp.equal(compartment, 1)
n_infected = jnp.sum(infected)
for _ in range(n_steps):
	state2 = np.run(state, steps=1)
	state2.block_on_all()
	new_compartment = state2.nodes_pytree["compartment"]._value
	new_infected = jnp.equal(new_compartment, 1)
	n_new_infected = jnp.sum(infected)
	infected_each_day.append(new_infected)

plt.plot(infected_each_day)
plt.xlabel('time (days)')
plt.ylabel('newly infected')
plt.title('Number of newly infected per day')
plt.savefig("newly_infected")
plt.show()


print("Finished")

            # for steps in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            #     state2 = np.run(state, steps=steps)
            #     state2.block_on_all()
            #     t0 = time.time()
            #     state2 = np.run(state, steps=steps)
            #     state2.block_on_all()
            #     t1 = time.time()
            #     log.info(f"    {steps} took {t1-t0:.3g} s")
			#
            #     if t1 - t0 > 0.5:
            #         break
            # sps = steps / (t1 - t0)
            # log.info(
            #     f"  {steps} steps took {t1-t0:.2g} s,  {sps:.3g} steps/s,  "
            #     + f"{sps*state.m:.3g} edge_ops/s,  {sps * state.n:.3g} node_ops/s"
            # )
