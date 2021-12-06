# NetProcess

NetProcess is a framework for modelling general processes on networks. Initially, it was motivated by modelling epidemics and information spread in networks but is written with network science, game theory and statistical physics research in mind, and should be suitable for e.g. forward flux sampling.

See e.g. the epidemic process [demo script](https://github.com/gavento/netprocess/blob/master/netprocess/scripts/epi_demo.py) for a quickstart.

## JAX

It is designed for speed and written using [JAX](https://jax.readthedocs.io/en/latest/) wiht its underlying XLA JIT compiler - all the operations are compiled to native code and optimized to be executed in parallel on CPU or GPU. JAX is modelled after NumPy and has a very similar API, with the benefit of (immensely) improved performance. This has some limitations when writing operations (all arrays are immutable, functions are pure (no side-effects beyond the returned value), you can't use all of python's control flow, random generator seed is passed along, etc. see (the sharp bits)[https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html]), but outside of `process.run()`, you can mostly just use it as NumPy.

## Network process model

### Process `NetworkProcess`

The structure describing a process, wrapping a set of operations to be applied to a state on every step, and set of properties to record during simulation.
`NetworkProcess` is not tied to any one state or network, and it is immutable after creation.

### Network instance `Network`

An instance of a network (a graph). The nodes must be numbered 0..(n-1), the edges are numbered 0..(m-1).
It is stored in a combination of a JSON file (for properties) and Hdf5 file (for array data, incl. any node or edge properties).

Network may be directed or undirected (bool flag `network.directed`). However, directedness is not checked in any way and the operations for undirected graphs should be written to ignore the direction of the edges (i.e. treat `src` and `tgt` as interchangeable).

### State `ProcessState`

The *state* of a process is stored as [`ProcessState`](https://github.com/gavento/netprocess/blob/master/netprocess/network_process/state.py), consisting of _state properties_ (`ProcessState is also a `PropTree`) and of `ProcessRecords` object to record evolution of some of the properties.

State contains:
* A set of _per-node_ properties (each an array with shape `[n,...]`) in `state.node`
  * In particular, every edge has properties: indices `i`, `deg`, `in_deg`, `out_deg`, `weight` (default 1.0), and `active` (default `True`)
* A set of _per-edge_ properties (each an array with shape `[m,...]`) in `state.edge`
  * In particular, every edge has properties: indices `src`, `tgt`, `i`, `weight` (default 1.0), and `active` (default `True`)
* A set of global properties (each an array with any shape or a scalar), directly under `state`
  * In particular: `n`, `m`, `steps` (step counter), `prng_key` (JAX pseudorandom seed)
* A pseudo-random generator state in `state.prng_key`. See [JAX and randomness](https://jax.readthedocs.io/en/latest/jax.random.html).
* A fixed `Network` in `state.network`. (Do not use it within the operations, use `edge.src` and others instead)

A state is considered immutable, after an updated state is obtained by `s2=process.run(s1)`, `s1` is still valid including its history.

The set of state properties need to stay the same through every step, including shapes and data type (a fundamental requirement of JAX and JIT compiler). However, on every step some temporary properties can be generated. Their names need to start with `_` and are discarded at the end of every step. (They can still be recorded and passed between operations in one step, though.)

Note on `active`: Inactive edges do not pass along data (i.e. do not contribute to `data.in_edges["max.something"]` etc.) but still update their state. Node inactivity is mostly just a marker - the updates of inactive nodes are still computed. *Active edges also pass along values from inactive nodes.* Your operations, records, and aggregations need to ignore inactive edges/nodes as appropriate.

### Pytrees and `PropTree`

JAX has a neat concept of a (Pytree)[https://jax.readthedocs.io/en/latest/pytrees.html]: any tree-like structure of dictionaries, lists and simiar containers eventually holding some JAX arrays. This has the advantage of being ble to pass several array around in JIT-ed functions in a convenient structure (rather than as extremely long lists of individual array parameters). Somewhat similar to [`tf.nest`](https://www.tensorflow.org/api_docs/python/tf/nest) if you happen to know it.

[`PropTree`](https://github.com/gavento/netprocess/blob/master/netprocess/utils/prop_tree.py) is a tree-of-dictionaries that additionally supports nested indexing: you can directly write `ptree["foo.bar"]` or ptree["foo", "bar"]` in place of `ptree["foo"]["bar"]`. This also creates any intermediate tree nodes on assignment, and makes copy-on-write immutability easier (e.g. `ptree2=ptree1._replace({"a.b.c": 42, "a.x": some_array}, baz=0)`). Moreover, `PropTree` ensures any tree leaves are JAX `ndarray`s and converts your data into them for you.

`ProcessState` is also a `PropTree`, as are operation update data (`EdgeUpdateData`, `NodeUpdateData` and `ParamUpdateData`). All the subtrees of a `PropTree` are also `PropTree`s.

For convenience, some `PropTrees` have distinguished properties that can be also accessed as attributes, e.g. with `ProcessState` you can write `state.n` as well as `state["n"]`, and can use `state.edge["src"]` or `state["edge.src"]`.

### Operations derived from `OperationBase`

Every oeration is derived from [`OperationBase`](https://github.com/gavento/netprocess/blob/master/netprocess/network_process/operation.py) and passes along messages between nodes via edges and updates some node, edge and global propertie of the state. Operations consist of methods `update_edge`, `update_node`, `update_params`, and state-initialization helper `prepare_state_data`.

On each step:

* First the edge update function `update_edge` of every operation is ran on every edge. For every edge, the function gets `EdgeUpdateData` with the properties of both endpoint nodes, of the edge, and global state properties (e.g. parameters). It only updates the properties of the edge, though.
* Then the `update_node` function of every operation is ran on evey node. For every node, it gets `NodeUpdateData` with properties of the node, global state properties (e.g. parameters) and aggregated data from all incoming and outgoing edges. Aggregates are used as `data.in_edges["max.some_edge_property"]`, `max`, `min`, `sum` and `prod` are available. Only properties of each node are updated.
* Then `update_params` is ran for all the operations, updating _any_ state properties. This can also read and update all edge and node data, but may be inefficient (and cumbersome to e.g. traverse edges).
* Lastly, selected properties are collected into state record (a snapshot of these properties at every step).

For simple operations, consider using `operations.Fun` with lambda functions. For example `Fun(node_f = lambda data: {"_mean_in_index": data.in_edges["sum.i"] / data.node["in_deg"]} )` is an peration that computes the mean index of incoming edges of every node, storing it as a (temporary) property `state.node._mean_in_index` (of shape `[n]` and dtype `jnp.float32`).

### JAX notes

* All JAX arrays are immutable. Do not try to modify the `state` by assignment etc.
* XLA is very good at pruning unused computations, e.g. only used edge aggregations are actually being computed after JIT. Do not worry about creating unused temporary properties. (State properties are always computed, however.)
* The random generator in JAX has immutable state, the state is not *updated* by using it but rather gets *split* before being passed to update functions. See the [JAX.random docs](https://jax.readthedocs.io/en/latest/jax.random.html) and [PRNG design docs](https://github.com/google/jax/blob/master/design_notes/prng.md) for more info.
* Read [Jax - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) if in doubt.

## Speed

On a NVIDIA Tesla K80 GPU in Google Colab, a simple epidemiological SIR process recording the transition rates on a network with N=1000000, M=6000000 (Barabási-Albert graph with k=3 and bidirectional edges) performs *7.8e+08 edge updates per second* (the edge updates and value shuffling likely dominate the computation time).

On a 2 core Intel(R) Xeon(R) CPU @ 2.20GHz in Google Colab, the same process on the same graph performs *6.8e+06 edge updates per second* (again, the edge updates likely dominate the computation).


## Limitations

* All the sizes need to be known in advance for XLA JIT. This means that the computation is re-compiled whenever the graph sizes (nodes, edges) or number of steps in batch change.
(This can be mitigated with masking, and this is WIP.)

* All the sizes and created propertis must be the same in all steps. (E.g. when updating a property, you can not occasionally skip the update by not returning it in the updates dictionary; just return the same value.)

* All user-defined operations must be XLA compilable. If you need python/compiled code interop, you need to do it on step boundary. This may be slow if it requires GPU<->CPU data transfer, but fast enough for GPU-only.

* The network is assumed to be static. This is actually possible to circumvent with some care - you can modify the state between the steps if you take care to prperly update all the data (mainly node data, edge data, existing record shapes).
