
import itertools
import json
import math
import os
import random
from typing import Optional

from tqdm import tqdm

from src.config import Point, blocks_file_t, node_distance_table_file, pt_id
from src.gps_utils import great_circle_distance
from src.route import get_distance


class NodeDistances():
    MAX_STORAGE_DISTANCE = 1600
    _node_distances: dict[str, dict[str, float]] = {}

    @classmethod
    def _insert_pair(cls, node_1: Point, node_2: Point):
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
            unseen_node: Optional[Point] = None
            for node in random.sample(cls.all_nodes, num_samples):
                if pt_id(node) not in cls._node_distances:
                    need_regeneration = True
                    unseen_node = node
                    break
            if not need_regeneration:
                return
            print('The saved node distance table did not include all requested nodes ({}). Regenerating...'.format(unseen_node))
        else:
            print('No node distance table file found at {}. Generating now...'.format(node_distance_table_file))

        cls._node_distances = {}
        with tqdm(total=math.comb(len(cls.all_nodes), 2), desc='Generating', unit='pairs', colour='green') as progress:
            for i, node in enumerate(cls.all_nodes):
                cls._node_distances[pt_id(node)] = {}
                for other_node in cls.all_nodes[i:]:
                    cls._insert_pair(node, other_node)
                    progress.update()

            print('Saving to {}'.format(node_distance_table_file))
            json.dump(cls._node_distances, open(node_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point) -> Optional[float]:
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
