from __future__ import annotations

from collections import defaultdict, Counter
from enum import IntEnum
from typing import NamedTuple, Optional, Tuple, List, Dict, FrozenSet, Set
import glob
import random

import networkx as nx
import numpy as np
from lxml import etree
from lxml.etree import Element
from tqdm import tqdm
import pandas as pd

from nml2catmaid.constants import DATA_ROOT


def id_gen(start=0):
    while True:
        yield start
        start += 1


ID_GEN = id_gen(1)


def find_all(element: Element, key: str) -> List[Element]:
    elems = key.split(':')
    nskey = None if len(elems) == 1 else elems[0]
    name = elems[-1]

    return element.findall(f"{{{element.nsmap[nskey]}}}{name}")


def find(element: Element, key: str) -> Optional[Element]:
    try:
        return find_all(element, key)[0]
    except IndexError:
        return None


class Node(NamedTuple):
    id: int
    x: float
    y: float
    z: float
    radius: Optional[float] = None
    iid: Optional[int] = None
    iname: Optional[str] = None

    def location(self, order='xyz') -> np.ndarray:
        return np.array([getattr(self, key) for key in order])

    def distance_from(self, other: Node):
        return np.linalg.norm(self.location() - other.location())

    @classmethod
    def from_segment(cls, seg: Element) -> Tuple[Node, int]:
        distal = find(seg, 'mml:distal')
        parent_iid = seg.attrib.get("parent")

        if parent_iid is not None:
            parent_iid = int(parent_iid)

        return Node(
            next(ID_GEN),
            float(distal.attrib["x"]),
            float(distal.attrib["y"]),
            float(distal.attrib["z"]),
            float(distal.attrib["diameter"]) / 2,
            int(seg.attrib["id"]),
            seg.attrib["name"],
        ), parent_iid

    @classmethod
    def between_nodes(cls, proximal: Node, distal: Node, fraction=0.5) -> Node:
        prox_loc = proximal.location()
        dist_loc = distal.location()

        x, y, z = (dist_loc - prox_loc) * fraction + prox_loc

        return Node(next(ID_GEN), x, y, z)


class Skeleton:
    node_headers = ("id", "parent_id", "skeleton_id", "x", "y", "z", "radius")

    def __init__(self, name, graph, ntype, iid_to_node, skid=None):
        self.id = next(ID_GEN) if skid is None else skid
        self.name = name
        self.graph = graph
        self.ntype = ntype

        self.iid_to_node = iid_to_node
        self._iid_to_edge = dict()

        for iid, node in self.iid_to_node.items():
            preds = list(self.graph.predecessors(node))
            if preds:
                self._iid_to_edge[iid] = (preds.pop(), node)

    def _make_iid_to_edge(self):
        return

    def node_from_iid(self, node_iid):
        return self.iid_to_node[node_iid]

    def edge_from_iid(self, segment_iid):
        return self._iid_to_edge[segment_iid]

    def interpose_node(self, segment_iid, fraction_between=0.5):
        proximal, distal = self.edge_from_iid(segment_iid)
        new_node = Node.between_nodes(proximal, distal, fraction=fraction_between)
        g = self.graph
        g.add_node(new_node, fraction=fraction_between)

        path = nx.shortest_path(g, proximal, distal)

        parent_iter = iter(path)
        child_iter = iter(path)
        next(child_iter)

        for parent, child in zip(parent_iter, child_iter):
            prox_frac = g.node[parent].get("fraction", 0)
            dist_frac = g.node[child].get("fraction", 1)
            if prox_frac < fraction_between < dist_frac:
                g.add_edge(parent, new_node, length=parent.distance_from(new_node))
                g.add_edge(new_node, child, length=new_node.distance_from(child))
                g.remove_edge(parent, child)
                return new_node
            elif dist_frac == fraction_between:
                return child

        raise ValueError("Node did not fit between existing nodes")

    @classmethod
    def from_nml(cls, fpath):
        root: Element = etree.parse(str(fpath)).getroot()
        cell = root[0][0]
        notes = find(cell, "meta:notes")
        ntype = notes.text.strip()
        name = cell.attrib["name"]
        segs = find(cell, "mml:segments")

        iid_to_node = dict()

        g = nx.DiGraph()

        for seg in segs:
            n, parent_iid = Node.from_segment(seg)
            iid_to_node[n.iid] = n
            g.add_node(n, orig_parent_iid=parent_iid)

        for node, parent_iid in g.nodes(data="orig_parent_iid"):
            if parent_iid is None:
                continue
            parent_node = iid_to_node[parent_iid]
            g.add_edge(parent_node, node, length=node.distance_from(parent_node))

        return cls(name, g, ntype, iid_to_node)

    def to_table(self):
        rows = []

        parents = set()
        children = set()
        for parent, child in sorted(self.graph.edges()):
            parents.add(parent)
            children.add(child)
            rows.append([
                child.id, parent.id, self.id, child.x, child.y, child.z, child.radius or None
            ])

        return pd.DataFrame(data=rows, columns=self.node_headers)


class Connector(NamedTuple):
    node: Node
    is_gapjunction: bool
    neurotransmitter: str


class RelationId(IntEnum):
    PRESYNAPTIC_TO = 0
    POSTSYNAPTIC_TO = 1
    GAPJUNCTION_WITH = 2

    def __bool__(self):
        return True


def counter_diff(c1: Counter, c2: Counter):
    total = 0
    for key, count1 in c1.items():
        if key in c2:
            total += abs(count1 - c2.pop(key))
        else:
            total += count1

    return total + sum(c2.values())


class Network:
    skeleton_header = ("id", "name", "ntype")
    connector_header = ("treenode_id", "relation_id", "connector_id", "x", "y", "z", "neurotransmitter")

    def __init__(self, cells_by_name, synapses, gap_junctions):
        self.cells_by_name = cells_by_name
        self.synapses = synapses
        self.gap_junctions = gap_junctions

    @classmethod
    def from_nml(cls, cells_by_name: Dict[str, Skeleton], fpath):
        synapses = nx.MultiDiGraph()
        gap_junctions = nx.MultiGraph()

        for name, cell in cells_by_name.items():
            synapses.add_node(name, obj=cell)
            gap_junctions.add_node(name, obj=cell)

        root = etree.parse(str(fpath)).getroot()
        projections = find(root, "projections")

        # unordered partners to source node to count
        gj_edges: Dict[
            FrozenSet[Tuple[str, Node], Tuple[str, Node]],
            Dict[Tuple[str, Node], List[str]]
        ] = defaultdict(lambda: defaultdict(lambda: []))

        total_gj = 0

        for projection in projections:
            is_gj = projection.attrib["name"].endswith("_GJ")

            src_name = projection.attrib["source"]
            src = cells_by_name[src_name]
            tgt_name = projection.attrib["target"]
            tgt = cells_by_name[tgt_name]

            neurotransmitter = find(projection, "synapse_props").attrib["synapse_type"].rstrip('_GJ')

            for connection in find_all(find(projection, "connections"), "connection"):
                pre_seg = int(connection.attrib["pre_segment_id"])
                pre_frac = 1 - float(connection.attrib["pre_fraction_along"])
                pre_node = src.interpose_node(pre_seg, pre_frac)

                post_seg = int(connection.attrib["post_segment_id"])
                post_frac = 1 - float(connection.attrib["post_fraction_along"])
                post_node = tgt.interpose_node(post_seg, post_frac)

                if is_gj:
                    src_key = (src_name, pre_node)
                    gj_edges[frozenset([src_key, (tgt_name, post_node)])][src_key].append(neurotransmitter)
                    total_gj += 1
                else:
                    synapses.add_edge(
                        src, tgt, pre_node=pre_node, post_node=post_node,
                        connector=Connector(Node.between_nodes(pre_node, post_node), False, neurotransmitter)
                    )

        asymmetric_gj = 0

        for partners, neurotransmitter_by_source in gj_edges.items():
            nt_lists = list(neurotransmitter_by_source.values())

            if len(nt_lists) == 0:
                continue

            assert len(nt_lists) <= 2, "wtf"

            if len(nt_lists) != 2:
                asymmetric_gj += len(nt_lists[0])
                # raise ValueError(f"One-way gap junction for {partners}")
            elif Counter(nt_lists[0]) != Counter(nt_lists[1]):
                asymmetric_gj += counter_diff(Counter(nt_lists[0]), Counter(nt_lists[1]))
                # raise ValueError(f"Asymmetric gap junctions for {partners}:\n\t{sorted(nt_lists[0])}\n\t{sorted(nt_lists[1])}")

            # src and tgt are arbitrary but deterministic
            (src_skel_name, src_node), (tgt_skel_name, tgt_node) = sorted(partners)
            for neurotransmitter in nt_lists[0]:
                gap_junctions.add_edge(
                    src_skel_name, tgt_skel_name, pre_node=src_node, post_node=tgt_node,
                    connector=Connector(Node.between_nodes(src_node, tgt_node), True, neurotransmitter)
                )

        if asymmetric_gj:
            raise ValueError(f"Of {total_gj} gap junctions, {asymmetric_gj} were asymmetric")

        return cls(cells_by_name, synapses, gap_junctions)

    def _skels_to_table(self) -> pd.DataFrame:
        skel_rows = []
        for _, skel in sorted(self.cells_by_name.items(), key=lambda x: x[0]):
            skel_rows.append([skel.id, skel.name, skel.ntype])

        return pd.DataFrame(data=skel_rows, columns=self.skeleton_header)

    def _nodes_to_table(self) -> pd.DataFrame:
        node_tables = []
        for _, skel in sorted(self.cells_by_name.items(), key=lambda x: x[0]):
            node_tables.append(skel.to_table())

        return pd.concat(node_tables, 'columns', ignore_index=True)

    def _connectors_to_table(self) -> pd.DataFrame:
        rows = []
        for pre_skel_name, post_skel_name, data in self.synapses.edges(data=True):
            pre_node: Node = data["pre_node"]
            post_node: Node = data["post_node"]
            connector: Connector = data["connector"]
            rows.append([
                pre_node.id, RelationId.PRESYNAPTIC_TO.value,
                connector.node.id, connector.node.x, connector.node.y, connector.node.z, connector.neurotransmitter
            ])
            rows.append([
                post_node.id, RelationId.POSTSYNAPTIC_TO.value,
                connector.node.id, connector.node.x, connector.node.y, connector.node.z, connector.neurotransmitter
            ])

        for pre_skel_name, post_skel_name, data in self.gap_junctions.edges(data=True):
            connector: Connector = data["connector"]
            for node in [data["pre_node"], data["post_node"]]:
                rows.append([
                    node.id, RelationId.GAPJUNCTION_WITH.value,
                    connector.node.id, connector.node.x, connector.node.y, connector.node.z, connector.neurotransmitter
                ])

        return pd.DataFrame(data=rows, columns=self.connector_header)

    def to_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        skels = self._skels_to_table()
        nodes = self._nodes_to_table()
        connectors = self._connectors_to_table()

        return skels, nodes, connectors


def main(data_root):
    cell_fpaths = sorted(glob.glob(str(data_root / "*.morph.xml")))

    skel_by_name = dict()

    for cell_fpath in tqdm(cell_fpaths):
        sk = Skeleton.from_nml(cell_fpath)
        skel_by_name[sk.name] = sk

    network_fpath = data_root / "Generated.net.xml"

    network = Network.from_nml(skel_by_name, network_fpath)

    network.to_tables()


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(1)

    main(DATA_ROOT)
