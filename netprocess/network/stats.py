import io
import itertools
import sys

import networkx as nx
import numpy as np
import powerlaw


def _stat_dict(prefix, data, qs=True):
    data = np.array(data).flatten()
    assert len(data) > 0
    sts = {}
    sts[f"{prefix}_mean"] = np.mean(data)
    sts[f"{prefix}_median"] = np.median(data)
    sts[f"{prefix}_min"] = np.min(data)
    sts[f"{prefix}_max"] = np.max(data)
    sts[f"{prefix}_std"] = np.std(data)
    if qs:
        sts[f"{prefix}_q05"] = np.quantile(data, 0.05)
        sts[f"{prefix}_q25"] = np.quantile(data, 0.25)
        sts[f"{prefix}_q75"] = np.quantile(data, 0.75)
        sts[f"{prefix}_q95"] = np.quantile(data, 0.95)
    return sts


def compute_degree_powerlaw(g, xmin=None):
    """Compute degree distribution powerlaw exponent with `powerlaw.Fit`.

    The range is automatically determined if `xmin==None`, use `xmin=number` for a concrete
    range, or `xmin=True` to use the smallest degree as `xmin`.
    """
    sts = {}
    degs = [k for _, k in g.degree()]
    if xmin == True:
        xmin = np.min(degs)
    try:
        # powerlaw.Fit always prints progress messages to stdout :/
        s0 = sys.stdout
        sys.stdout = io.StringIO()
        fit = powerlaw.Fit(degs, discrete=True, verbose=False, xmin=xmin)
    finally:
        sys.stdout = s0
    sts["degree_powerlaw_alpha"] = fit.power_law.alpha
    sts["degree_powerlaw_xmin"] = fit.power_law.xmin
    sts["degree_powerlaw_error"] = fit.power_law.sigma
    return sts


def compute_graph_stats(g):
    """
    Compute a and return a dictionary of some network statistisc of the given graph.

    For most statistics, returns "X_mean", "X_median", "X_min", "X_max", "X_std",
    and for some also "X_q05", "X_q25", "X_q75", "X_q95".
    """
    sts = {}
    # Degrees & power law
    degs = [k for _, k in g.degree()]
    sts.update(_stat_dict("degree", degs))
    # Clustering & transitivity
    clust = list(nx.algorithms.cluster.clustering(g).values())
    sts.update(_stat_dict("clustering", clust))
    sts["transitivity"] = nx.algorithms.cluster.transitivity(g)
    # Path lengths & connectivity
    dist_dict = nx.algorithms.shortest_paths.all_pairs_shortest_path_length(g)
    dists = list(itertools.chain(*[v[1].values() for v in dist_dict]))
    sts.update(_stat_dict("distances", dists))
    sts["components"] = nx.algorithms.components.number_connected_components(g)
    return sts
