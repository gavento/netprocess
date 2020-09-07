import jax
import jax.lax as lax
import jax.numpy as jnp
import networkx as nx
import typing
from .utils import is_integer, ArrayDict


def nx_graph_to_edges(graph: nx.Graph, dtype=jnp.int32, sort="from") -> jnp.ndarray:
    """
    Convert a graph or a digraph to (from, to) pairs in Nx2 numpy array.

    Graph nodes must be integers 0..n-1 already.
    Edges are sorted by "from".
    Undirected graphs get both edge directions added.
    """

    def s(a):
        if sort == "from":
            return sorted(a)
        elif sort == "to":
            return sorted(a, key=lambda x: x[1])
        elif sort is None or sort == False:
            return a
        else:
            raise ValueError("Invalus `sort` param")

    n = graph.order()
    assert all(is_integer(x) for x in graph.nodes)
    assert all(x >= 0 and x < n for x in graph.nodes)
    if graph.is_directed():
        return jnp.array(s(graph.edges()))
    return jnp.array(s(list(graph.edges()) + [(y, x) for x, y in graph.edges()]))


def sum_from_inneighbors(edges: jnp.ndarray, node_data: jnp.ndarray) -> jnp.ndarray:
    """
    Edges is a Nx2 array of (from, to), node_data is Nxk array, result in Nxk.
    """
    n, k = node_data.shape
    assert edges.shape[1] == 2
    return lax.scatter_add(
        jnp.zeros((n, k), dtype=node_data.dtype),
        edges[:, [1]],
        node_data[edges[:, 0]],
        lax.ScatterDimensionNumbers((1,), (0,), (0,)),
    )


def count_inneighbor_states(
    edges: jnp.ndarray, node_states: jnp.ndarray, states: int, dtype=jnp.int32
) -> jnp.ndarray:
    (n,) = node_states.shape
    assert edges.shape[1] == 2
    node_states_1hot = jax.nn.one_hot(node_states, states, dtype=dtype)
    return sum_from_inneighbors(edges, node_states_1hot)


def step_edge_node_messages(
    rng_key: jax.random.PRNGKey,
    nodes_dict: ArrayDict,
    edges_dict: ArrayDict,
    edges_from_to: jnp.ndarray,
    edge_f: typing.Callable,
    node_f: typing.Callable,
):
    """
    `edge_f(
        rng_key: PRNGKey,
        edge_dict: ArrayDict,
        from_node_dict: ArrayDict,
        to_node_dict: ArrayDict,
        ) -> (
            edge_update: ArrayDict,
            from_node_update: ArrayDict,
            to_node_update: ArrayDict
            )`

    `node_f(
        rng_key: PRNGKey,
        node_dict: ArrayDict,
        inedge_updates_sum: ArrayDict,
        outedge_updates_sum: ArrayDict,
        ) -> node_update: ArrayDict`

    Returns: (new_node_dict: ArrayDict, new_edge_dict: ArrayDict)
    """
    e_from = edges_from_to[:, 0]
    e_to = edges_from_to[:, 1]
    n = next(iter(nodes_dict.values())).shape[0]
    m = edges_from_to.shape[0]
    # Randomness
    edge_rng_key, node_rng_key = jax.random.split(rng_key)
    edge_rng_keys = jax.random.split(edge_rng_key, num=m)
    node_rng_keys = jax.random.split(node_rng_key, num=n)
    # Extract node values for edges
    n2e_from_dict = {k: nodes_dict[k][e_from] for k in nodes_dict}
    n2e_to_dict = {k: nodes_dict[k][e_to] for k in nodes_dict}
    # Compute edge updates and messages
    edge_update_dict, from_update_dict, to_update_dict = jax.vmap(edge_f)(
        edge_rng_keys, edges_dict, n2e_from_dict, n2e_to_dict
    )
    # Compute node input values
    def collect_sum(e_vals, e_endpoints, k=None):
        z = jnp.zeros((n,) + e_vals.shape[1:], dtype=e_vals.dtype)
        e_endpoints_exp = jnp.expand_dims(e_endpoints, 1)
        dims = lax.ScatterDimensionNumbers(
            tuple(range(1, len(e_vals.shape))), (0,), (0,)
        )
        # print("CSUM", k, e_vals.shape, e_endpoints.shape, z.shape, e_endpoints_exp.shape, dims)
        return lax.scatter_add(z, e_endpoints_exp, e_vals, dims)

    inedge_updates_sum = {
        k: collect_sum(to_update_dict[k], e_to, k) for k in to_update_dict
    }
    outedge_updates_sum = {
        k: collect_sum(from_update_dict[k], e_from, k) for k in from_update_dict
    }
    # Compute edge updates and messages
    node_update_dict = jax.vmap(node_f)(
        node_rng_keys, nodes_dict, inedge_updates_sum, outedge_updates_sum
    )

    # Combine node values
    new_nodes_dict = dict(nodes_dict)
    new_nodes_dict.update(node_update_dict)
    new_edges_dict = dict(edges_dict)
    new_edges_dict.update(edge_update_dict)
    return new_nodes_dict, new_edges_dict


def build_step_edge_node_messages(node_f, edge_f, jit=True, **jit_kwargs):
    """
    Return a jitted intance of step_edge_node_messages (see its docs).

    JIT can be bypassed by an option `jit=False` (for uniform API).
    Returns: `step_edge_node_messages_jitted(
        rng_key, nodes_dict, edges_dict, edges_from_to
        ) -> (new_nodes_dict, new_edges_dict)`
    """
    if jit:
        f = jax.jit(step_edge_node_messages, (4, 5), **jit_kwargs)
    else:
        f = step_edge_node_messages

    def step_edge_node_messages_jitted(
        rng_key: jax.random.PRNGKey,
        nodes_dict: ArrayDict,
        edges_dict: ArrayDict,
        edges_from_to: jnp.ndarray,
    ) -> (ArrayDict, ArrayDict):
        return f(rng_key, nodes_dict, edges_dict, edges_from_to, node_f, edge_f)

    return step_edge_node_messages_jitted
