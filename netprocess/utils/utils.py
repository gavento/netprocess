import contextlib
import datetime
import inspect
import logging
import time
from typing import Any

import numpy as np


def now_isofmt() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def update_dict_disjoint(d: dict, update: dict):
    """Update dictionary in-place, raise ValueError on key conflict."""
    for k, v in update.items():
        if k in d:
            raise ValueError(f"Update key {k} already present")
        d[k] = v


def update_dict_present(d: dict, update: dict):
    """Update dictionary in-place, raise ValueError on new key."""
    for k, v in update.items():
        if k not in d:
            raise ValueError(f"Update key {k} not present in update dict")
        d[k] = v


def is_integer(x) -> bool:
    """Is `x` a simple integer - an `int` or ()-shaped ndarray?"""
    if isinstance(x, int):
        return True
    if isinstance(x, jnp.ndarray):
        return jnp.issubdtype(x.dtype, jnp.integer) and x.shape == ()
    return False


def get_caller_logger(name="log", levels=2, *, frame=None):
    """Get logger from the caller"""
    if frame is None:
        frame = inspect.currentframe()
    l = None
    if levels > 0 and frame.f_back:
        l = get_caller_logger(name=name, levels=levels - 1, frame=frame.f_back)
    if not l:
        if name in frame.f_globals:
            l = frame.f_globals[name]
    return l


def jsonize(d: Any) -> Any:
    """Recursively transform object to JSON-compatible types.

    Also transforms `nan` to None."""
    if isinstance(d, dict):
        return {k: jsonize(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return [jsonize(v) for v in d]
    elif isinstance(d, (int, bool, str)) or d is None:
        return d
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating) or isinstance(d, float):
        if np.isnan(d):
            return None
        return float(d)
    else:
        raise TypeError(f"Unable to JSONize {d!r} of type {type(d)}")


@contextlib.contextmanager
def logged_time(name, level=logging.INFO, logger=None):
    """
    Context manager to measure and log operation time.
    """
    if logger is None:
        logger = get_caller_logger()
    t0 = time.time()
    yield
    t1 = time.time()
    logger.log(level, f"{name} took {t1-t0:.3g} s")
