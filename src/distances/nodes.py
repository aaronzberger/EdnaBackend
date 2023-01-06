
import itertools
import json
import os
import random
from typing import Optional

from tqdm import tqdm

from src.config import AnyPoint, Point, blocks_file_t, node_distance_table_file, pt_id
from src.gps_utils import great_circle_distance
from src.route import get_distance


class NodeDistances():
    MAX_STORAGE_DISTANCE = 1600
    _node_distances: dict[str, dict[str, float]] = {}

    @classmethod
    def _insert_pair(cls, node_1: AnyPoint, node_2: AnyPoint):
        # If this pair already exists in the opposite order, skip
        try:
            cls._node_distances[pt_id(node_2)][pt_id(node_1)]
        except KeyError:
            # Calculate a fast great circle distance
            distance = great_circle_distance(node_1, node_2)

            # Only calculate and insert the routed distance if needed
            if distance <= cls.MAX_STORAGE_DISTANCE:
                cls._node_distances[pt_id(node_1)][pt_id(node_2)] = get_distance(node_1, node_2)

    @classmethod
    def __init__(cls, blocks: blocks_file_t):
        cls.all_nodes = list(itertools.chain.from_iterable((i['nodes'][0], i['nodes'][-1]) for i in blocks.values()))

        if os.path.exists(node_distance_table_file):
            print('Node distance table file found.')
            cls._node_distances = json.load(open(node_distance_table_file))

            # If the file was found, make sure it includes all the records we need
            num_samples = min(len(cls.all_nodes), 1000)
            need_regeneration = False
            for node in random.sample(cls.all_nodes, num_samples):
                if pt_id(node) not in cls._node_distances:
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
                cls._node_distances[pt_id(node)] = {}
                for other_node in cls.all_nodes:
                    cls._insert_pair(node, other_node)
                    progress.update()

            print('Saving to {}'.format(node_distance_table_file))
            json.dump(cls._node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def add_nodes(cls, nodes: list[Point]):
        with tqdm(total=len(nodes) * len(cls.all_nodes), desc='Adding Nodes', unit='pairs', colour='green') as progress:
            for node in nodes:
                if pt_id(node) not in cls._node_distances:
                    cls._node_distances[pt_id(node)] = {}
                    for other_node in cls.all_nodes:
                        cls._insert_pair(node, other_node)
                        progress.update()
                else:
                    progress.update(len(cls.all_nodes))

        json.dump(cls._node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: AnyPoint, p2: AnyPoint) -> Optional[float]:
        '''
        Get the distance between two nodes by their coordinates

        Parameters:
            p1 (Point): the first point
            p2 (Point): the second point

        Returns:
            float | None: distance between the two points if it exists, None otherwise
        '''
        try:
            return cls._node_distances[pt_id(p1)][pt_id(p2)]
        except KeyError:
            try:
                return cls._node_distances[pt_id(p2)][pt_id(p1)]
            except KeyError:
                return None
