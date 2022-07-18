from __future__ import annotations

import itertools
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from config import BASE_DIR, MINS_PER_HOUSE
from gps_utils import Point
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
        self.time_to_walk = self.num_houses * MINS_PER_HOUSE + self.length * (1/60)

    def get_node_ids(self) -> tuple[str, str]:
        split = self.id.find(':')
        # TODO: Maybe make these separate in the json
        return self.id[:split], self.id[split + 1:self.id.find(':', split + 1)]

    def reversed(self):
        '''Reverse this segment. Completes in linear time with the number of points'''
        new_segment = deepcopy(self)
        new_segment.start = new_segment.end
        new_segment.end = self.start
        new_segment.all_points.reverse()
        node_ids = new_segment.get_node_ids()
        new_segment.id = node_ids[1] + ':' + node_ids[0] + ':' + self.id[-1]
        return new_segment


'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'


class NodeDistances():
    node_distance_table: dict[str, dict[str, float]] = {}

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
    def get_distance(cls, p1: Point, p2: Point) -> float:
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
    # TODO: Calculate distance for default value
    deltas: list[float] = field(default_factory=lambda: [0.0])
    segments: list[Segment] = field(default_factory=list)
    total_time: float = 0.0
    # TODO: Make class variable
    max_minutes: int = 180

    def __post_init__(self):
        self.insertion_queue: list[str] = []

    def get_segment_times(self) -> tuple[list[list[float]], list[list[float]]]:
        # For each segment or delta, the start and end time
        segment_times: list[list[float]] = []
        delta_times: list[list[float]] = []

        running_time = 0
        for delta, segment in zip(self.deltas, self.segments):
            segment_times.append([delta * (1/60) + running_time, delta * (1/60) + running_time + segment.time_to_walk])
            delta_times.append([running_time, running_time + delta * (1/60)])
            running_time += delta * (1/60) + segment.time_to_walk
        return segment_times, delta_times

    def calculate_time(self):
        self.total_time = 0.0
        self.total_time += sum([segment.time_to_walk for segment in self.segments])  # Walk it and talk to voters
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
        # Test the bid of inserting this segment forward
        theoretical_timeline = deepcopy(self)
        theoretical_timeline._insert(segment, index)
        bid_foward = theoretical_timeline.total_time

        # Test the bid of inserting this segment backwards
        theoretical_timeline = deepcopy(self)
        theoretical_timeline._insert(segment.reversed(), index)
        bid_backwards = theoretical_timeline.total_time

        forward = bid_foward < bid_backwards

        if (bid_foward if forward else bid_backwards) > self.max_minutes:
            return False

        # Insert the segment in the correct direction
        self._insert(segment if forward else segment.reversed(), index)
        self.insertion_queue.append(segment.id if forward else segment.reversed().id)
        return True

    def get_bid(self, segment: Segment, index: int) -> Optional[float]:
        theoretical_timeline = deepcopy(self)
        possible = theoretical_timeline.insert(segment, index)
        delta_delta = sum(theoretical_timeline.deltas) - sum(self.deltas)
        if not possible or delta_delta > 200:
            return None
        return delta_delta

    def generate_report(self) -> dict[str, dict[str, int]]:
        '''
        Generate a report of how this list was generated

        Returns:
            dict: a dictionary that maps the segment id (key) to the
                (1) route index and (2) insertion index (in a list)
        '''
        segment_ids = [segment.id for segment in self.segments]
        report: dict[str, dict[str, int]] = {}
        for i, id in enumerate(segment_ids):
            report[id] = {
                'route_order': i + 1,
                'insertion_order': self.insertion_queue.index(id) + 1
            }
        return report
