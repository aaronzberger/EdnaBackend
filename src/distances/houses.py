
import itertools
import json
import os
import random
from statistics import mean
from typing import Optional

from src.config import (BASE_DIR, DIFFERENT_SIDE_COST, DIFFERENT_SIDE_TIME_DIVISION, KEEP_APARTMENTS,
                        blocks_file, blocks_file_t)
from src.distances.nodes import NodeDistances
from src.gps_utils import Point
from src.route import get_distance
from src.timeline_utils import Segment
from tqdm import tqdm


class HouseDistances():
    MAX_STORAGE_DISTANCE = 1600
    _house_distances: dict[str, dict[str, tuple[float, float]]] = {}
    _save_file = os.path.join(BASE_DIR, 'store', 'house_distances.json')
    _blocks: blocks_file_t = json.load(open(blocks_file))

    @classmethod
    def _insert_point(cls, pt: Point, s: Segment):
        s_houses = cls._blocks[s.id]['addresses']

        if len(s_houses) == 0:
            return

        # Calculate the distances between the segment endpoints
        distance_to_start = NodeDistances.get_distance(pt, s.start)
        if distance_to_start is None:
            distance_to_start = get_distance(pt, s.start)
        distance_to_end = NodeDistances.get_distance(pt, s.end)
        if distance_to_end is None:
            distance_to_end = get_distance(pt, s.end)
        end_distances = [distance_to_start, distance_to_end]

        if pt.id not in cls._house_distances:
            cls._house_distances[pt.id] = {pt.id: (0, 0)}

        for address, info in s_houses.items():
            if not KEEP_APARTMENTS and ' APT ' in address:
                continue

            through_start = end_distances[0] + info['distance_to_start']
            through_end = end_distances[1] + info['distance_to_end']

            cls._house_distances[pt.id][address] = (round(min([through_start, through_end])), 0)

    @classmethod
    def _insert_pair(cls, s1: Segment, s2: Segment):
        s1_houses = cls._blocks[s1.id]['addresses']
        s2_houses = cls._blocks[s2.id]['addresses']

        if len(s1_houses) == 0 or len(s2_houses) == 0:
            return

        # If any combination of houses on these two segments is inserted, they all are
        try:
            cls._house_distances[next(iter(s2_houses))][next(iter(s1_houses))]
            return
        except KeyError:
            pass

        # Check if the segments are the same
        if s1.id == s2.id:
            for (address_1, info_1), (address_2, info_2) in itertools.product(s1_houses.items(), s2_houses.items()):
                if not KEEP_APARTMENTS and ' APT ' in address_1:
                    continue
                if address_1 not in cls._house_distances:
                    cls._house_distances[address_1] = {}

                if address_1 == address_2:
                    cls._house_distances[address_1][address_2] = (0, 0)
                else:
                    cost = 0
                    # Simply use the difference of the distances to the start
                    distance = round(
                        abs(info_1['distance_to_start'] - info_2['distance_to_start']))
                    if info_1['side'] != info_2['side']:
                        # Add the actual distance to cross the road
                        distance += (info_1['distance_to_road'] + info_2['distance_to_road']) / DIFFERENT_SIDE_TIME_DIVISION

                        # Add the cost of crossing (depending on the type of road)
                        try:
                            cost = DIFFERENT_SIDE_COST[s1.type]
                        except KeyError:
                            print('Unable to find penalty for crossing {} street. Adding none'.format(s1.type))
                    cls._house_distances[address_1][address_2] = (distance, cost)
            return

        # Calculate the distances between the segment endpoints
        end_distances = [NodeDistances.get_distance(i, j) for i, j in
                         [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]
        end_distances = [d for d in end_distances if d is not None]

        # If this pair is too far away, don't add to the table.
        if len(end_distances) != 4 or min(end_distances) > cls.MAX_STORAGE_DISTANCE:
            return

        # Iterate over every possible pair of houses
        for (address_1, info_1), (address_2, info_2) in itertools.product(s1_houses.items(), s2_houses.items()):
            if not KEEP_APARTMENTS and ' APT ' in address_1:
                continue
            if address_1 not in cls._house_distances:
                cls._house_distances[address_1] = {}

            start_start = end_distances[0] + info_1['distance_to_start'] + info_2['distance_to_start']
            start_end = end_distances[1] + info_1['distance_to_start'] + info_2['distance_to_end']
            end_start = end_distances[2] + info_1['distance_to_end'] + info_2['distance_to_start']
            end_end = end_distances[3] + info_1['distance_to_end'] + info_2['distance_to_end']

            distance = round(min([start_start, start_end, end_start, end_end]))
            cost = 0
            try:
                cost = mean([DIFFERENT_SIDE_COST[s1.type], DIFFERENT_SIDE_COST[s2.type]])
            except KeyError:
                print('Unable to find penalty for crossing {} street. Adding none'.format(s1.type))
            cls._house_distances[address_1][address_2] = (distance, cost)

    @classmethod
    def __init__(cls, cluster: list[Segment], center: Point):
        if os.path.exists(cls._save_file):
            need_regeneration = False
            print('House distance table file found. Loading may take a while...')
            cls._house_distances = json.load(open(cls._save_file))
            num_samples = min(len(cluster), 100)

            # Sample random segments from the input to check if they are already stored
            for segment in random.sample(cluster, num_samples):
                houses = cls._blocks[segment.id]['addresses']
                try:
                    cls._house_distances[next(iter(houses))]
                except StopIteration:
                    # There are no houses in this segment
                    continue
                except KeyError:
                    # This house was not already stored
                    need_regeneration = True
                    break
            if center.id not in cls._house_distances:
                need_regeneration = True
            if not need_regeneration:
                return
            else:
                print('The saved distance table did not include all requested segments. Regenerating...')
        else:
            print('No house distance table file found at {}. Generating now...'.format(cls._save_file))

        cls._house_distances = {}
        with tqdm(total=len(cluster) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for segment in cluster:
                cls._insert_point(center, segment)
                for other_segment in cluster:
                    cls._insert_pair(segment, other_segment)
                    progress.update()

        print('Saving to {}'.format(cls._save_file))

        json.dump(cls._house_distances, open(cls._save_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point) -> Optional[tuple[float, float]]:
        '''
        Get the distance between two houses by their coordinates

        Parameters:
            p1 (Point): the first point
            p2 (Point): the second point

        Returns:
            float | None: distance between the two points if it exists, None otherwise
        '''
        try:
            return cls._house_distances[p1.id][p2.id]
        except KeyError:
            try:
                return cls._house_distances[p2.id][p1.id]
            except KeyError:
                return None
