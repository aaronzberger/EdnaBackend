from __future__ import annotations

import itertools
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from nptyping import Float32, NDArray, Shape
from tqdm import tqdm

from config import (ARBITRARY_LARGE_DISTANCE, MAX_NODE_STORAGE_DISTANCE,
                    MINS_PER_HOUSE, WALKING_M_PER_S, node_distance_table_file,
                    segment_distance_matrix_file)
from gps_utils import Point, great_circle_distance
from route import get_distance


@dataclass
class Segment():
    id: str
    start: Point
    end: Point
    num_houses: int
    all_points: list[Point]

    def __post_init__(self):
        self.length: float = 0.0
        for first, second in itertools.pairwise(self.all_points):
            self.length += get_distance(first, second)
        self.time_to_walk = self.num_houses * MINS_PER_HOUSE + (self.length / WALKING_M_PER_S * (1/60))

    def get_node_ids(self) -> tuple[str, str]:
        split = self.id.find(':')
        # TODO: Maybe make these separate in the json
        return self.id[:split], self.id[split + 1:self.id.find(':', split + 1)]

    def reversed(self):
        '''Reverse this segment. Completes in linear time with the number of points'''
        new_segment = deepcopy(self)
        new_segment.start = deepcopy(new_segment.end)
        new_segment.end = deepcopy(self.start)
        new_segment.all_points.reverse()
        node_ids = new_segment.get_node_ids()
        new_segment.id = node_ids[1] + ':' + node_ids[0] + ':' + self.id[-1]
        return new_segment


'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'


class NodeDistances():
    _node_distances: dict[str, dict[str, Optional[float]]] = {}

    @classmethod
    def _insert_pair(cls, node_1: Point, node_2: Point):
        # If this pair already exists in the opposite order, skip
        try:
            cls._node_distances[node_2.id][node_1.id]
        except KeyError:
            # Calculate a fast great circle distance
            distance = great_circle_distance(node_1, node_2)

            # Only calculate and insert the routed distance if needed
            cls._node_distances[node_1.id][node_2.id] = None if distance > MAX_NODE_STORAGE_DISTANCE \
                else get_distance(node_1, node_2)

    @classmethod
    def __init__(cls, segments: list[Segment]):
        print('Beginning node distances generation... ')

        cls.all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))

        if os.path.exists(node_distance_table_file):
            print('Node distance table file found.')
            cls._node_distances = json.load(open(node_distance_table_file))
            return

        print('No node distance table file found at {}. Generating now...'.format(node_distance_table_file))
        cls._node_distances = {}
        with tqdm(total=len(cls.all_nodes) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for node in cls.all_nodes:
                cls._node_distances[node.id] = {}
                for other_node in cls.all_nodes:
                    cls._insert_pair(node, other_node)
                    progress.update()

            print('Saving to {}'.format(node_distance_table_file))
            json.dump(cls._node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def add_nodes(cls, nodes: list[Point]):
        with tqdm(total=len(nodes) * len(cls.all_nodes), desc='Adding Nodes', unit='pairs', colour='green') as progress:
            for node in nodes:
                if node.id not in cls._node_distances:
                    cls._node_distances[node.id] = {}
                    for other_node in cls.all_nodes:
                        cls._insert_pair(node, other_node)
                        progress.update()
                else:
                    progress.update(len(cls.all_nodes))

        json.dump(cls._node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point) -> Optional[float]:
        '''
        Get the distance between two nodes by their coordinates

        Parameters:
            p1 (Point): the first point
            p2 (Point): the second point

        Returns:
            float: distance between the two nodes

        Raises:
            KeyError: if the pair does not exist in the table
        '''
        try:
            return cls._node_distances[p1.id][p2.id]
        except KeyError:
            return cls._node_distances[p2.id][p1.id]


'-----------------------------------------------------------------------------------------'
'                                Segment Distances Generation                                '
'-----------------------------------------------------------------------------------------'


class SegmentDistances():
    _segment_distances: dict[str, dict[str, Optional[float]]] = {}
    _segments: list[Segment] = []

    @classmethod
    def _insert_pair(cls, s1: Segment, s2: Segment):
        # If this pair already exists in the opposite order, skip
        try:
            cls._segment_distances[s2.id][s1.id]
        except KeyError:
            routed_distances = \
                [NodeDistances.get_distance(i, j) for i, j in
                    [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]
            existing_distances = [i for i in routed_distances if i is not None]
            cls._segment_distances[s1.id][s2.id] = None if len(existing_distances) == 0 else min(existing_distances)

    @classmethod
    def __init__(cls, segments: list[Segment]):
        print('Beginning segment distances generation... ')

        cls._segments = deepcopy(segments)

        if os.path.exists(segment_distance_matrix_file):
            print('Segment distance table file found.')
            cls._segment_distances = json.load(open(segment_distance_matrix_file))
            return

        print('No segment distance table file found at {}. Generating now...'.format(segment_distance_matrix_file))
        cls._segment_distances = {}
        with tqdm(total=len(segments) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for segment in segments:
                cls._segment_distances[segment.id] = {}
                for other_segment in segments:
                    cls._insert_pair(segment, other_segment)
                    progress.update()

            print('Saving to {}'.format(segment_distance_matrix_file))
            json.dump(cls._segment_distances, open(segment_distance_matrix_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, s1: Segment, s2: Segment) -> Optional[float]:
        '''
        Get the distance between two segments

        Parameters:
            s1 (Segment): the first segment
            s2 (Segment): the second segment

        Returns:
            float: distance between the two segments

        Raises:
            KeyError: if the pair does not exist in the table
        '''
        try:
            return cls._segment_distances[s1.id][s2.id]
        except KeyError:
            return cls._segment_distances[s2.id][s1.id]

    @classmethod
    def get_distance_matrix(cls) -> NDArray[Shape[len(_segments), len(_segments)], Float32]:
        matrix = np.empty((len(cls._segments), len(cls._segments)), dtype=np.float32)
        for r, segment in enumerate(cls._segments):
            for c, other_segment in enumerate(cls._segments):
                distance = cls.get_distance(segment, other_segment)
                matrix[r][c] = ARBITRARY_LARGE_DISTANCE if distance is None else distance
        return matrix
