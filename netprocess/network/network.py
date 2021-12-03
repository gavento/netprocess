import io
import json
from pathlib import Path

import attr
import h5py
import hdf5plugin
import networkx as nx
import numpy as np
from netprocess.utils.types import PytreeDict

from ..utils import file_utils, utils


@attr.s
class Network:
    base_path: Path = attr.ib()
    json_path: Path = attr.ib()
    h5_path: Path = attr.ib()
    _h5_file: h5py.File = attr.ib(default=None)
    _network: nx.Graph = attr.ib(default=None)
    attribs: dict = attr.ib(factory=dict)

    REQUIRED_ATTRIBS = ["n", "m", "directed"]
    EDGES_PATH = "edges"

    @classmethod
    def open(cls, json_path: Path):
        """Open and load the given JSON. H5 file not checked and only loaded lazily."""
        net = cls._new_only_paths(json_path)

        with file_utils.open_file(net.json_path, mode="r") as f:
            d = json.load(f)
            for a in cls.REQUIRED_ATTRIBS:
                if a not in d:
                    raise KeyError(
                        f"Network at {net.json_path!r} missing attribute {a}"
                    )
            net.attribs = dict(d)
        return net

    @property
    def n(self):
        return self.attribs["n"]

    @property
    def m(self):
        return self.attribs["m"]

    @property
    def edges(self):
        return self.h5_file[self.EDGES_PATH]

    @property
    def h5_file(self):
        """Lazily opened H5 file for network data."""
        if self._h5_file is None:
            if self.h5_path is None:
                self._h5_file = h5py.File(io.BytesIO(), mode="w")
            else:
                self._h5_file = h5py.File(self.h5_path, mode="a")
        return self._h5_file

    @property
    def network(self):
        """Lazily loaded (Di)Graph instance from the H5 file."""
        if self._network is None:
            self._network = nx.DiGraph() if self["directed"] else nx.Graph()
            self._network.add_nodes_from(range(self["n"]))
            self._network.add_edges_from(self.edges)
        return self._network

    def write(self, indent=2):
        """
        Write/update JSON and flush the H5 file without closing it (if open).
        """
        if self.json_path is None:
            raise ValueError(f"Error: can't write a Network created without path.")

        with utils.open_file(self.json_path, mode="wt") as f:
            json.dump(utils.jsonize(self.attribs), f, indent=indent)
            f.write("\n")

        if self._h5_file is not None:
            self._h5_file.flush()

    def export_graphml(self, compress_gzip=True) -> Path:
        """
        Export the graph as Graphml, compressed by default.

        Returns the path of the file.
        """
        path = self.base_path.with_name(self.base_path.name + ".graphml.gz")
        if not compress_gzip:
            path = path.with_suffix("")
        nx.write_graphml(self.network, path)
        return path

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: np.ndarray,
        directed: bool,
        json_path: Path = None,
        *,
        label: str = "",
        origin: dict = {},
    ) -> "Network":
        """Create a Network instance from a list of edges.

        Does not write the instance to disk yet.
        """
        if not json_path is None:
            json_path = Path(json_path)
            assert not json_path.exists()
        net = cls._new_only_paths(json_path)
        m = edges.shape[0]
        assert edges.shape == (m, 2)
        edges = np.int32(edges)
        assert np.all(edges >= 0)
        assert np.all(edges < n)

        net.attribs["created"] = utils.now_isofmt()
        net.attribs["n"] = n
        net.attribs["m"] = m
        net.attribs["directed"] = directed
        net.attribs["label"] = label
        net.attribs["stats"] = {}
        net.attribs["origin"] = origin
        net.add_array(cls.EDGES_PATH, edges)
        return net

    @classmethod
    def from_graph(
        cls, g: nx.Graph, json_path: Path = None, *, label: str = ""
    ) -> "Network":
        g = nx.convert_node_labels_to_integers(g)
        if g.size() == 0:
            edges = np.zeros((0, 2), dtype=np.int32)
        else:
            edges = np.array(g.edges(), dtype=np.int32)
        net = cls.from_edges(
            n=g.order(),
            edges=edges,
            directed=isinstance(g, nx.DiGraph),
            json_path=json_path,
            label=label,
        )
        net._network = g
        return net

    def __getitem__(self, name: str) -> dict:
        if name in self.attribs:
            return self.attribs[name]
        elif name in self.h5_file:
            return self.h5_file[name][()]
        else:
            raise KeyError(f"{name!r} not an attribute nor a data array")

    @classmethod
    def _new_only_paths(cls, json_path: Path):
        """Return a Network instance without opening the H5 data file nor the JSON info file."""
        if json_path is None:
            return cls(
                base_path=None,
                json_path=None,
                h5_path=None,
            )

        json_path = Path(json_path)
        base_path = file_utils.file_basic_path(json_path, ".json")
        h5_path = base_path.with_name(base_path.name + ".h5")
        return cls(
            base_path=base_path,
            json_path=json_path,
            h5_path=h5_path,
        )

    def h5_as_pytree(self, root_path="/", required=False) -> PytreeDict:
        """Get (a subtree of) the h5 file as a nested dict of ndarrays.

        Returns {} if group not found (unless `required` is set)."""
        if root_path not in self.h5_file and not required:
            return {}

        d = {}
        g = self.h5_file[root_path]
        for k, i in g.items():
            if isinstance(i, h5py._hl.group.Group):
                d[k] = self.h5_as_pytree(i.name)
            else:
                d[k] = i[:]

    def add_array(self, name: str, array_data: np.ndarray, compress: bool = True):
        """Add an array to the H5 data file"""
        c = hdf5plugin.Blosc(cname="zstd") if compress else None
        if array_data.nbytes < 1024:
            c = None
        self.h5_file.create_dataset(name, data=array_data, compression=c)
