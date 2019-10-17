from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import Dict, NamedTuple, Optional, Iterator, Tuple

import neuroml
import numpy as np
import pandas as pd


def id_generator(x=1):
    while True:
        yield x
        x += 1


ID_GEN = id_generator()


class Point(NamedTuple):
    x: float
    y: float
    z: float

    @classmethod
    def from_nml(cls, point: neuroml.Point3DWithDiam) -> Point:
        return Point(point.x, point.y, point.z)

    def distance_to(self, other: Point) -> float:
        return np.linalg.norm(np.array(self) - np.array(other))


class Node(NamedTuple):
    id: int
    location: Point
    radius: Optional[float] = None

    @classmethod
    def from_location(cls, xyz):
        return Node(next(ID_GEN), Point(*xyz))


class Treenode(NamedTuple):
    node: Node
    parent: Optional[Node] = None

    def flatten(self):
        return (
            self.node.id, self.parent.id,
            self.node.location.x, self.node.location.y, self.node.location.z,
            np.NaN if self.node.radius is None else self.node.radius
        )

    @staticmethod
    def keys():
        return (
            "treenode_id", "parent_id",
            "location_x", "location_y", "location_z",
            "radius"
        )


class Edge:
    def __init__(self, proximal: Node, distal: Node):
        self.proximal: Node = proximal
        self.distal: Node = distal
        self.interposed: Dict[float, Node] = {
            None: self.distal, 0: self.proximal, 1: self.distal
        }

    @property
    def is_point(self):
        return self.proximal.location == self.distal.location

    def _interpose(self, fraction: float) -> np.ndarray:
        prox_loc = np.array(self.proximal.location)
        dist_loc = np.array(self.distal.location)
        return (dist_loc - prox_loc) * fraction + prox_loc

    def add_node(self, fraction_along=1.0) -> Node:
        if self.is_point:
            return self.distal
        if fraction_along in self.interposed:
            return self.interposed[fraction_along]
        node = Node.from_location(self._interpose(fraction_along))
        self.interposed[fraction_along] = node
        return node

    @property
    def key(self) -> Tuple[Node, Node]:
        return self.proximal, self.distal

    def splinter(self) -> Iterator[Edge]:
        """Split edge with temporary interposed nodes into a number of edges"""
        prox = self.proximal
        for _, dist in sorted(self.interposed.items(), key=lambda x: x[0]) + [(1.0, self.distal)]:
            yield Edge(prox, dist)
            prox = dist


class MorphologyConverter:
    def __init__(self, morphology: neuroml.Morphology):
        self.morphology = morphology
        self.segments: Dict[int, neuroml.Segment] = {s.id: s for s in self.morphology.segments}
        self.point_to_node: Dict[Point, Node] = dict()
        self.seg_id_to_edge: Dict[int, Edge] = dict()

        self._populate_coord_ids()

    def to_skeleton(self, skid: Optional[int]=None):
        if not skid:
            skid = next(ID_GEN)

        data = list(tn.flatten() for tn in self._generate_treenodes())
        df = pd.DataFrame(data, columns=Treenode.keys())
        df.index = df["treenode_id"]
        df["skeleton_id"] = [skid for _ in range(len(df))]
        return df

    def _register_node(self, node):
        self.point_to_node[node.location] = node

    def treenodes(self):
        yield from self._generate_treenodes()

    def _source_points(self) -> Iterator[neuroml.Point3DWithDiam]:
        for seg in self.morphology.segments:
            if seg.proximal:
                yield seg.proximal
            if seg.distal:
                yield seg.distal

    def _populate_coord_ids(self):
        diameters = defaultdict(list)
        point_ids = dict()
        for point_nml in self._source_points():
            point = Point.from_nml(point_nml)
            if point not in point_ids:
                point_ids[point] = next(ID_GEN)
            if point_nml.diameter:
                diameters[point].append(point_nml.diameter)

        diameters = {key: np.max(value) if value else None for key, value in diameters.items()}

        for point, node_id in point_ids:
            diameter = diameters.get(point)
            radius = None if diameter is None else diameter / 2
            if radius is not None:
                radius = diameter / 2
            self.point_to_node[point] = Node(node_id, point, radius)

    def _toposort_segments(self) -> Iterator[neuroml.Segment]:
        root_seg = None
        prox_to_dists = defaultdict(set)
        for dist_id, seg in self.segments.items():
            this = seg.parent
            if this is None:
                if root_seg is None:
                    root_seg = dist_id
                else:
                    raise ValueError(f"Morphology {self.morphology.id} has more than one root segment: {(root_seg, dist_id)}")
            else:
                prox_to_dists[this.segments] = dist_id

        if root_seg is None:
            ValueError(f"Morphology {self.morphology.id} has no root segment")
        to_visit = [root_seg]

        while to_visit:
            this = to_visit.pop()
            to_visit.append(sorted(prox_to_dists[this]))
            yield this

    def _generate_treenodes(self):
        """Assume the root has prox and dist"""
        it = self._toposort_segments()
        root_seg = next(it)
        root = self.point_to_node[Point.from_nml(root_seg.proximal)]
        yield Treenode(root)
        extra_edges = []
        self.seg_id_to_edge[root_seg.id] = Edge(root, self.point_to_node[Point.from_nml(root_seg.distal)])
        for dist_seg in it:
            parent = dist_seg.parent
            prox_seg_id = parent.segments
            prox_edge = self.seg_id_to_edge[prox_seg_id]

            prox_node = prox_edge.add_node(parent.fraction_along)
            self._register_node(prox_node)
            if dist_seg.proximal:
                extra_edges.append(Edge(
                    prox_node,
                    self.point_to_node[Point.from_nml(dist_seg.proximal)]
                ))
                prox_node = extra_edges[-1].distal

            self.seg_id_to_edge[dist_seg.id] = Edge(
                prox_node,
                self.point_to_node[Point.from_nml(dist_seg.distal)]
            )

        all_edges = chain([])
        for edge in chain(extra_edges, self.seg_id_to_edge.values()):
            all_edges = chain(all_edges, edge.splinter())

        for edge in all_edges:
            if not edge.is_point:
                yield Treenode(edge.distal, edge.proximal)

