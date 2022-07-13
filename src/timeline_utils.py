from __future__ import annotations

import itertools
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field

from tqdm import tqdm

from gps_utils import BASE_DIR, Point
from route import get_distance


@dataclass
class Segment():
    id: str
    start: Point
    end: Point
    num_houses: int
    all_points: list[Point]

    def __post_init__(self):
        self.length = 0
        for first, second in itertools.pairwise(self.all_points):
            self.length += get_distance(first, second)


'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'


class NodeDistances():
    node_distance_table = {}

    @classmethod
    def __init__(cls, segments: list[Segment]):
        print('Beginning node distances generation... ')

        cls.all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))

        cls.node_distance_table_file = os.path.join(BASE_DIR, 'store', 'node_distance_table.pkl')
        if os.path.exists(cls.node_distance_table_file):
            print('Node distance table file found.')
            cls.node_distance_table = pickle.load(open(cls.node_distance_table_file, 'rb'))
        else:
            print('No node distance table file found at {}. Generating now...'.format(cls.node_distance_table_file))
            cls.node_distance_table = {}
            with tqdm(total=len(cls.all_nodes) ** 2 / 2, desc='Generating', unit='pairs', colour='green') as progress:
                for node in cls.all_nodes:
                    node_id = str(node.lat) + ':' + str(node.lon)
                    cls.node_distance_table[node_id] = {}
                    for other_node in cls.all_nodes:
                        other_id = str(other_node.lat) + ':' + str(other_node.lon)
                        # If this pair already exists in the opposite order, skip
                        try:
                            cls.node_distance_table[other_id][node_id]
                            continue
                        except KeyError:
                            cls.node_distance_table[node_id][other_id] = get_distance(node, other_node)
                            progress.update()

                print('Saving to {}'.format(cls.node_distance_table_file))
                with open(cls.node_distance_table_file, 'wb') as output:
                    pickle.dump(cls.node_distance_table, output)

    @classmethod
    def add_nodes(cls, nodes: list[Point]):
        # First, check if these nodes have already been added
        with tqdm(total=len(nodes) * len(cls.all_nodes), desc='Adding Nodes', unit='pairs', colour='green') as progress:
            for node in nodes:
                node_id = str(node.lat) + ':' + str(node.lon)
                if node_id not in cls.node_distance_table:
                    cls.node_distance_table[node_id] = {}
                for other_node in cls.all_nodes:
                    other_id = str(other_node.lat) + ':' + str(other_node.lon)
                    try:
                        cls.node_distance_table[node_id][other_id]
                    except KeyError:
                        cls.node_distance_table[node_id][other_id] = get_distance(node, other_node)
                    progress.update()

        with open(cls.node_distance_table_file, 'wb') as output:
            pickle.dump(cls.node_distance_table, output)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point):
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
        p1_id = str(p1.lat) + ':' + str(p1.lon)
        p2_id = str(p2.lat) + ':' + str(p2.lon)
        try:
            return cls.node_distance_table[p1_id][p2_id]
        except KeyError:
            return cls.node_distance_table[p2_id][p1_id]


@dataclass
class Timeline():
    start: Point
    end: Point
    deltas: list[float] = field(default_factory=lambda: [0.0])
    segments: list[Segment] = field(default_factory=list)
    total_time: float = 0.0

    def calculate_time(self):
        self.total_time = 0.0
        self.total_time += sum([segment.length * (1/60) for segment in self.segments])  # Walk the segments
        self.total_time += sum([segment.num_houses * 1.5 for segment in self.segments])  # Talk to voters
        self.total_time += sum([dist * (1/60) for dist in self.deltas])  # Walk between segments

    def _insert(self, segment: Segment, index: int):
        self.segments.insert(index, segment)
        del self.deltas[index]

        # Add the distance from the point before this segment to the start
        point_before = self.start if index == 0 else self.segments[index - 1].end
        self.deltas.insert(index, NodeDistances.get_distance(point_before, segment.start))

        # Add the distance from the end of this segment to the point after
        point_after = self.end if index == len(self.segments) - 1 else self.segments[index + 1].start
        self.deltas.insert(index + 1, NodeDistances.get_distance(segment.end, point_after))
        self.calculate_time()

    def insert(self, segment: Segment, index: int) -> bool:
        # Test if we can fit this segment
        theoretical_timeline = deepcopy(self)
        theoretical_timeline._insert(segment, index)
        if theoretical_timeline.total_time > 180:
            return False

        self._insert(segment, index)
        return True

    def get_bid(self, segment: Segment, index: int) -> float | None:
        theoretical_timeline = deepcopy(self)
        possible = theoretical_timeline.insert(segment, index)
        delta_delta = sum(theoretical_timeline.deltas) - sum(self.deltas)
        if not possible or delta_delta > 200:
            return None
        return delta_delta
