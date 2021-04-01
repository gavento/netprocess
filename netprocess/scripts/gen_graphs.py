import logging
import os.path

import click
import networkx as nx
import numpy as np
from netprocess import data, utils

from .cli import cli

log = logging.getLogger(__name__)


def write_network_multi(g, meta, base_path):
    data.Network.from_graph(g, meta=meta).write(f"{base_path}.pickle.zstd")
    nx.write_graphml(g, f"{base_path}.graphml.gz")
    nx.write_gpickle(g, f"{base_path}.gpickle.gz", protocol=4)


@cli.command()
@click.argument("n", type=int)
@click.argument("m", type=int)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-o", "--output_dir", type=str, default=None)
def gen_barabasi_albert(n, m, seed, output_dir):
    if seed is None:
        seed = np.random.randint(1000000)
    if not output_dir:
        output_dir = "."
    base_path = os.path.join(output_dir, f"barabasi_albert-n_{n}-m_{m}-s_{seed:06}")
    with utils.logged_time(f"Creating {base_path!r}"):
        g = nx.random_graphs.barabasi_albert_graph(n, m, seed=seed)
        write_network_multi(
            g,
            dict(
                gen_type="barabasi_albert",
                gen_n=n,
                gen_m=m,
                gen_seed=seed,
                powerlaw_min_k=m,
                powerlaw_true_exp=3.0,
            ),
            base_path,
        )
