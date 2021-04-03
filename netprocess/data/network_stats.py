import numpy as np


def powerlaw_exp_vl(degrees, k_min=1.0, discrete=False):
    """
    Estimate powerlaw exponent using "vannila" sampling from https://arxiv.org/pdf/1908.00310.pdf
    """
    ds = np.array(degrees)
    if discrete:
        k_min = k_min - 0.5
    ds = ds[ds >= k_min]
    return len(ds) / np.sum(np.log(ds / k_min)) + 1.0


def powerlaw_exp_fp(degrees, k_min=1.0, discrete=False):
    """
    Estimate powerlaw exponent using "friendship paradox" sampling from https://arxiv.org/pdf/1908.00310.pdf
    """
    ds = np.array(degrees)
    if discrete:
        k_min = k_min - 0.5
    ds = ds[ds >= k_min]
    return np.sum(ds) / np.sum(ds * np.log(ds / k_min)) + 2.0
