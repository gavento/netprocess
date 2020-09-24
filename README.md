# NetProcess

NetProcess is a framework for modelling processes on general networks. Initially, it was motivated by modelling epidemics and information spread in networks but is written with network science and statistical physics research in mind, and is able to should e.g. Forward flux sampling.

It is designed for speed and written using [JAX](https://jax.readthedocs.io/en/latest/) wiht its underlying XLA JIT compiler - all the operations are compiled to native code and optimized to be executed in parallel on CPU or GPU.

See e.g. the epidemic process [demo script](https://github.com/gavento/netprocess/blob/master/netprocess/scripts/epi_demo.py) for a quickstart.

## Network process model

The **state** of a process, [`ProcessState` and `ProcessStateData](https://github.com/gavento/netprocess/blob/master/netprocess/network_process/state.py) consists of:

* A fixed network (with directed links)
* A set of per-node named properties (each an array with shape `[N,...]`)
* A set of per-edge named properties (each an array with shape `[M,...]`)
* A set of global named parameters (each an array with any shape)
* A pseudo-random generator state (so the runs are reproducible up to numeric unstability).
* A set of named output records (each an array with any shape), one set for each step taken so far.

The **process**, [`NetworkProcess`](https://github.com/gavento/netprocess/blob/master/netprocess/network_process/process.py) is defined by a set of **operations** derived from [`OperationBase`](https://github.com/gavento/netprocess/blob/master/netprocess/network_process/operation.py).

On each step, first the edge update function of every operation is ran, then all node update functions, then all parameter update functions, finally record generating functions. (An operation does not have to define all of those.)

```python
def update_edge(self, rng_key: PRNGKey, params: PytreeDict, edge: PytreeDict,
                from_node: PytreeDict, to_node: PytreeDict) -> PytreeDict:
    """Gets a slice of edge properties for the edge, and slices for in-node and
    for out-node, plus PRGN state and global params."""
    return {}

def update_node(self, rng_key: PRNGKey, params: PytreeDict, node: PytreeDict,
                in_edges: PytreeDict, out_edges: PytreeDict) -> PytreeDict:
    """Gets a slice of the node properties for the node, plus aggregates of all in-edges
    and all out-edges.
    The aggregates are two-level dictionaries: the first level is the aggregation oeration
    from ['sum', 'prod', 'max', 'min'], the second is the property name.
    The function also gets the PRGN state and global params."""
    return {}

def update_params(self, rng_key: PRNGKey, state: ProcessStateData,
                  orig_state: ProcessStateData) -> PytreeDict:
    """Gets the current state (as a named tuple) and the pre-step state.
    Use `state.rng_key` for any randomness!"""
    return {}

def create_record(self, rng_key: PRNGKey, state: ProcessStateData,
                  orig_state: ProcessStateData) -> PytreeDict:
    """Gets the new state (as a named tuple) and the pre-step state.
    Does *not* get access to already generated records.
    Use `state.rng_key` for any randomness!"""
    return {}
```

All update functions return a dictionary of
  1. State updates (these must be existing properties and of matchong shapes). A node update can only update per-node properties etc.
  2. Temporary values that are available to later operations but only within the same step. Temporary values start with an underscore `_`.

A record generating function returns a dictionalry of arrays too, these are then stacked in the state along a new first axis.

## JAX notes

* All JAX arrays are immutable. Do not try to modify the `state` by assignment etc.
* XLA is very good at pruning unused computations, e.g. only used edge aggregations are actually being computed after JIT. Do not worry about creating unused temporary properties. (State properties are always computed, however.)
* The random generator in JAX has immutable state, the state is not *updated* by using it but rather gets *split* before being passed to update functions. See the [JAX.random docs](https://jax.readthedocs.io/en/latest/jax.random.html) and [PRNG design docs](https://github.com/google/jax/blob/master/design_notes/prng.md) for more info.
* Read [Jax - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) if in doubt.

## Speed

On a NVIDIA Tesla K80 GPU in Google Colab, a simple epidemiological SIR process recording the transition rates on a network with N=1000000, M=6000000 (Barabási-Albert graph with k=3 and bidirectional edges) performs *7.8e+08 edge updates per second* (the edge updates and value shuffling likely dominate the computation time).

On a 2 core Intel(R) Xeon(R) CPU @ 2.20GHz in Google Colab, the same process on the same graph performs *6.8e+06 edge updates per second* (the edge updates likely dominate the computation).


## Limitations

* All the sizes need to be known in advance for XLA JIT. This means that the computation is re-compiled whenever the graph sizes (nodes, edges) or number of steps in batch change.
(This can be mitigated with masking, and this is WIP.)

* All the sizes and created propertis must be the same in all steps. (E.g. when updating a property, you can not occasionally skip the update by not returning it in the updates dictionary; just return the same value.)

* All user-defined operations must be XLA compilable. If you need python/compiled code interop, you need to do it on step boundary. This may be slow if it requires GPU<->CPU data transfer, but fast enough for GPU-only.

* The network is assumed to be static. This is actually possible to circumvent with some care - you can modify the state between the steps if you take care to prperly update all the data (mainly node data, edge data, existing record shapes).
