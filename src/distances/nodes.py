
import itertools
import json
import os
import random
from typing import Optional

from tqdm import tqdm

from src.config import MAX_NODE_STORAGE_DISTANCE, node_distance_table_file
from src.gps_utils import great_circle_distance, Point
from src.route import get_distance
from src.timeline_utils import Segment


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
        cls.all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))

        if os.path.exists(node_distance_table_file):
            print('Node distance table file found.')
            cls._node_distances = json.load(open(node_distance_table_file))
            num_samples = min(len(cls.all_nodes), 1000)
            need_regeneration = False
            for node in random.sample(cls.all_nodes, num_samples):
                if node.id not in cls._node_distances:
                    need_regeneration = True
                    break
            if not need_regeneration:
                return
            else:
                print('The saved node distance table did not include all requested nodes. Regenerating...')
        else:
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
