from __future__ import annotations

import itertools
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from config import (MAX_NODE_STORAGE_DISTANCE, MAX_TIMELINE_MINS,
                    MINS_PER_HOUSE, WALKING_M_PER_S, node_distance_table_file)
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
    node_distances: dict[str, dict[str, Optional[float]]] = {}

    @classmethod
    def _insert_pair(cls, node_1: Point, node_2: Point):
        node_1_id = str(node_1.lat) + ':' + str(node_1.lon)
        node_2_id = str(node_2.lat) + ':' + str(node_2.lon)
        # If this pair already exists in the opposite order, skip
        try:
            cls.node_distances[node_2_id][node_1_id]
        except KeyError:
            # Calculate a fast great circle distance
            distance = great_circle_distance(node_1, node_2)

            # Only calculate and insert the routed distance if needed
            cls.node_distances[node_1_id][node_2_id] = None if distance > MAX_NODE_STORAGE_DISTANCE \
                else get_distance(node_1, node_2)

    @classmethod
    def __init__(cls, segments: list[Segment]):
        print('Beginning node distances generation... ')

        cls.all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))

        if os.path.exists(node_distance_table_file):
            print('Node distance table file found.')
            cls.node_distances = json.load(open(node_distance_table_file, 'r'))
            return

        print('No node distance table file found at {}. Generating now...'.format(node_distance_table_file))
        cls.node_distances = {}
        with tqdm(total=len(cls.all_nodes) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for node in cls.all_nodes:
                node_id = str(node.lat) + ':' + str(node.lon)
                cls.node_distances[node_id] = {}
                for other_node in cls.all_nodes:
                    cls._insert_pair(node, other_node)
                    progress.update()

            print('Saving to {}'.format(node_distance_table_file))
            json.dump(cls.node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def add_nodes(cls, nodes: list[Point]):
        with tqdm(total=len(nodes) * len(cls.all_nodes), desc='Adding Nodes', unit='pairs', colour='green') as progress:
            for node in nodes:
                node_id = str(node.lat) + ':' + str(node.lon)
                if node_id not in cls.node_distances:
                    cls.node_distances[node_id] = {}
                    for other_node in cls.all_nodes:
                        cls._insert_pair(node, other_node)
                        progress.update()
                else:
                    progress.update(len(cls.all_nodes))

        json.dump(cls.node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

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
        p1_id = str(p1.lat) + ':' + str(p1.lon)
        p2_id = str(p2.lat) + ':' + str(p2.lon)
        try:
            return cls.node_distances[p1_id][p2_id]
        except KeyError:
            return cls.node_distances[p2_id][p1_id]


class Timeline():
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

        # Calculate the distance from the start to end point. This is the first delta
        start_distance = 0.0 if start == end else NodeDistances.get_distance(start, end)
        self.deltas: list[float] = [start_distance if start_distance is not None else get_distance(start, end)]

        self.segments: list[Segment] = []
        self.total_time: float = 0.0
        self.insertion_queue: list[str] = []

    def get_segment_times(self) -> tuple[list[list[float]], list[list[float]]]:
        # For each segment or delta, the start and end time
        segment_times: list[list[float]] = []
        delta_times: list[list[float]] = []

        running_time = 0
        for delta, segment in zip(self.deltas, self.segments):
            segment_times.append([delta / WALKING_M_PER_S * (1/60) + running_time,
                                  delta / WALKING_M_PER_S * (1/60) + running_time + segment.time_to_walk])
            delta_times.append([running_time, running_time + delta / WALKING_M_PER_S * (1/60)])
            running_time += delta / WALKING_M_PER_S * (1/60) + segment.time_to_walk
        return segment_times, delta_times

    def calculate_time(self):
        self.total_time = 0.0
        self.total_time += sum([segment.time_to_walk for segment in self.segments])  # Walk it and talk to voters
        self.total_time += sum([dist / WALKING_M_PER_S * (1/60) for dist in self.deltas])  # Walk between segments

    def _insert(self, segment: Segment, index: int):
        self.segments.insert(index, segment)
        del self.deltas[index]

        # Add the distance from the point before this segment to the start
        point_before = self.start if index == 0 else self.segments[index - 1].end
        distance_before = NodeDistances.get_distance(point_before, segment.start)

        # Add the distance from the end of this segment to the point after
        point_after = self.end if index == len(self.segments) - 1 else self.segments[index + 1].start
        distance_after = NodeDistances.get_distance(segment.end, point_after)

        # If the distance doesn't exist in the distance table, this segment is too far anyway. Leave a terminal bid
        if distance_before is None or distance_after is None:
            self.total_time = MAX_TIMELINE_MINS
            self.deltas.insert(index, MAX_TIMELINE_MINS * 60 * WALKING_M_PER_S)
            return

        self.deltas.insert(index, distance_before)
        self.deltas.insert(index + 1, distance_after)
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

        if (bid_foward if forward else bid_backwards) > MAX_TIMELINE_MINS:
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
