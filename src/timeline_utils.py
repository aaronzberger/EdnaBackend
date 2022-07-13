from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass, field
import os

from gps_utils import Point, BASE_DIR
from route import get_distance
import pickle
from tqdm import tqdm


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

    def extract_end_ids(self):
        '''Get a dict of the two node ids of this segment'''
        first_break = self.id.find(':')
        return {
            'start': self.id[:first_break],
            'end': self.id[first_break + 1:self.id.find(':', first_break + 1)]
        }


'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'


class NodeDistances():
    node_distance_table = {}

    @classmethod
    def __init__(cls, segments: list[Segment]):
        print('Beginning node distances generation... ')

        all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))
        all_ids = list(itertools.chain.from_iterable(i.extract_end_ids().values() for i in segments))

        node_distance_table_file = os.path.join(BASE_DIR, 'node_distance_table.pkl')
        if os.path.exists(node_distance_table_file):
            print('Node distance table file found.')
            cls.node_distance_table = pickle.load(open(node_distance_table_file, 'rb'))
        else:
            print('No node distance table file found at {}. Generating now...'.format(node_distance_table_file))
            cls.node_distance_table = {}
            with tqdm(total=len(all_ids) ** 2 / 2, desc='Generating Table', unit='iters', colour='green') as progress:
                def distance_metric(p1, p2):
                    progress.update()
                    return get_distance(p1, p2)

                for node, id in zip(all_nodes, all_ids):
                    cls.node_distance_table[id] = {}
                    for other_node, other_id in zip(all_nodes, all_ids):
                        # If this pair already exists in the opposite order, skip
                        if other_id not in cls.node_distance_table:
                            cls.node_distance_table[id][other_id] = distance_metric(node, other_node)

                print('Saving to {}'.format(node_distance_table_file))
                with open(node_distance_table_file, 'wb') as output:
                    pickle.dump(cls.node_distance_table, output)

    @classmethod
    def get_distance_by_id(cls, id1: str, id2: str):
        '''
        Get the distance between two nodes by their IDs

        Parameters:
            id1 (str): the first node ID
            id2 (str): the second node ID

        Returns:
            float: distance between the two nodes

        Raises:
            KeyError: if the pair does not exist in the table
        '''
        try:
            return cls.node_distance_table[id1][id2]
        except KeyError:
            return cls.node_distance_table[id2][id1]


@dataclass
class Timeline():
    start: Point
    end: Point
    deltas: list[float] = field(default_factory=lambda: [0.0])
    segments: list[Segment] = field(default_factory=list)
    total_time: float = 0.0

    def calculate_time(self):
        self.total_time = 0.0
        self.total_time += sum([segment.length * (1/60) for segment in self.segments])
        self.total_time += sum([segment.num_houses * 1.5 for segment in self.segments])
        self.total_time += sum([dist * (1/60) for dist in self.deltas])

    def _insert(self, segment: Segment, index: int):
        self.segments.insert(index, segment)
        del self.deltas[index]
        if index == 0:
            self.deltas.insert(index, get_distance(self.start, segment.start))
        else:
            self.deltas.insert(index, NodeDistances.get_distance_by_id(
                self.segments[index - 1].extract_end_ids()['end'], segment.extract_end_ids()['start']))
        if index == len(self.segments) - 1:
            self.deltas.insert(index + 1, get_distance(segment.end, self.end))
        else:
            self.deltas.insert(index + 1, NodeDistances.get_distance_by_id(
                segment.extract_end_ids()['end'], self.segments[index + 1].extract_end_ids()['start']))
        self.calculate_time()

    def insert(self, segment: Segment, index: int) -> bool:
        # Test if we can fit this segment
        assert index < len(self.deltas) and index >= 0, "Index is {} with {} deltas, {} segments".format(
            index, len(self.deltas), len(self.segments))
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
        if not possible or delta_delta > 50:
            return None
        return delta_delta
