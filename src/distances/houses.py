
import itertools
import json
import os
import random
from statistics import mean
from typing import Optional

from tqdm import tqdm

from src.config import (DIFFERENT_SIDE_COST, DISTANCE_TO_ROAD_MULTIPLIER,
                        KEEP_APARTMENTS, USE_COST_METRIC, Block, Point, blocks_file,
                        blocks_file_t, house_distance_table_file, pt_id)
from src.distances.nodes import NodeDistances
from src.route import get_distance


def store(distance: int, cost: int) -> int:
    '''
    Convert a distance and cost into storage form (compressing into one variable)

    Parameters:
        distance (int): the distance to store (must be less than 10000)
        cost (int): the cost to store (must be less than 10000)

    Returns:
        int: the storable integer representing the inputs
    '''
    if distance >= 10000 or cost >= 10000:
        raise ValueError('Cannot store value more than 9999 in house distance matrix')

    return distance * 10000 + cost


def unstore(value: int) -> tuple[int, int]:
    '''
    Convert a stored value into the original values for time and cost

    Parameters:
        value (int): the value stored in the table

    Returns:
        (int, int): a tuple of the distance and cost encoded by this value
    '''
    return (value // 10000, value % 10000)


class HouseDistances():
    MAX_STORAGE_DISTANCE = 1600
    _house_dcs: dict[str, dict[str, int]] = {}
    _blocks: blocks_file_t = json.load(open(blocks_file))

    @classmethod
    def _insert_point(cls, pt: Point, b: Block):
        b_houses = b['addresses']

        if len(b_houses) == 0:
            return

        # Calculate the distances between the segment endpoints
        dc_to_start = NodeDistances.get_distance(pt, b['nodes'][0])
        if dc_to_start is None:
            dc_to_start = get_distance(pt, b['nodes'][0]), 0 if USE_COST_METRIC else get_distance(pt, b['nodes'][0])
        dc_to_end = NodeDistances.get_distance(pt, b['nodes'][-1])
        if dc_to_end is None:
            dc_to_end = get_distance(pt, b['nodes'][-1]), 0 if USE_COST_METRIC else get_distance(pt, b['nodes'][-1])

        if pt_id(pt) not in cls._house_dcs:
            cls._house_dcs[pt_id(pt)] = {pt_id(pt): store(0, 0)} if USE_COST_METRIC else {pt_id(pt): 0}

        for address, info in b_houses.items():
            if not KEEP_APARTMENTS and ' APT ' in address:
                continue

            if USE_COST_METRIC:
                through_start = (dc_to_start[0] + info['distance_to_start'], dc_to_start[1])
                through_end = (dc_to_end[0] + info['distance_to_end'], dc_to_end[1])

                distance, cost = through_start if through_start[0] < through_end[0] else through_end

                cls._house_dcs[pt_id(pt)][address] = store(round(distance), round(cost))
            else:
                through_start = dc_to_start + info['distance_to_start']
                through_end = dc_to_end + info['distance_to_end']

                cls._house_dcs[pt_id(pt)][address] = round(min(through_start, through_end))

    @classmethod
    def _insert_pair(cls, b1: Block, b1_id: str, b2: Block, b2_id: str):
        def crossing_penalty(block: Block) -> int:
            try:
                return DIFFERENT_SIDE_COST[block['type']]
            except KeyError:
                print('Unable to find penalty for crossing {} street. Adding none'.format(b1['type']))
                return 0

        b1_houses = b1['addresses']
        b2_houses = b2['addresses']

        if len(b1_houses) == 0 or len(b2_houses) == 0:
            return

        # If any combination of houses on these two segments is inserted, they all are
        try:
            cls._house_dcs[next(iter(b2_houses.keys()))][next(iter(b1_houses.keys()))]
            return
        except KeyError:
            pass

        # Check if the segments are the same
        if b1_id == b2_id:
            for (address_1, info_1), (address_2, info_2) in itertools.product(b1_houses.items(), b2_houses.items()):
                if not KEEP_APARTMENTS and ' APT ' in address_1:
                    continue
                if address_1 not in cls._house_dcs:
                    cls._house_dcs[address_1] = {}

                if address_1 == address_2:
                    cls._house_dcs[address_1][address_2] = store(0, 0) if USE_COST_METRIC else 0
                else:
                    distance_to_road = (info_1['distance_to_road'] + info_2['distance_to_road']) * DISTANCE_TO_ROAD_MULTIPLIER
                    # For primary and secondary roads, crossing in the middle of the block is not possible
                    # Go to the nearest crosswalk and back
                    if b1['type'] in ['motorway', 'trunk', 'primary', 'secondary']:
                        distance = min([info_1['distance_to_start'] + info_2['distance_to_start'] + distance_to_road,
                                        info_1['distance_to_end'] + info_2['distance_to_end'] + distance_to_road])
                    else:
                        # Simply use the difference of the distances to the start
                        distance = round(
                            abs(info_1['distance_to_start'] - info_2['distance_to_start']) + distance_to_road)

                    if not USE_COST_METRIC:
                        cls._house_dcs[address_1][address_2] = distance
                    else:
                        cost = 0
                        if info_1['side'] != info_2['side']:
                            cost += crossing_penalty(b1)

                        cls._house_dcs[address_1][address_2] = store(round(distance), cost)
            return

        # Calculate the distances between the segment endpoints
        end_dcs = [NodeDistances.get_distance(i, j) for i, j in
                   [(b1['nodes'][0], b2['nodes'][0]), (b1['nodes'][0], b2['nodes'][-1]),
                    (b1['nodes'][-1], b2['nodes'][0]), (b1['nodes'][-1], b2['nodes'][1])]]
        end_dcs = [d for d in end_dcs if d is not None]

        # If this pair is too far away, don't add to the table.
        if len(end_dcs) != 4 or min([d[0] for d in end_dcs]) > cls.MAX_STORAGE_DISTANCE:
            return

        # Iterate over every possible pair of houses
        for (address_1, info_1), (address_2, info_2) in itertools.product(b1_houses.items(), b2_houses.items()):
            if not KEEP_APARTMENTS and ' APT ' in address_1:
                continue
            if address_1 not in cls._house_dcs:
                cls._house_dcs[address_1] = {}

            distances_to_road = (info_1['distance_to_road'] + info_2['distance_to_road']) * DISTANCE_TO_ROAD_MULTIPLIER

            # The format of the list is: [start_start, start_end, end_start, end_end]
            distances: list[float] = \
                [end_dcs[0][0] + info_1['distance_to_start'] + info_2['distance_to_start'] + distances_to_road,
                 end_dcs[1][0] + info_1['distance_to_start'] + info_2['distance_to_end'] + distances_to_road,
                 end_dcs[2][0] + info_1['distance_to_end'] + info_2['distance_to_start'] + distances_to_road,
                 end_dcs[3][0] + info_1['distance_to_end'] + info_2['distance_to_end'] + distances_to_road]

            # Add the street crossing penalties
            if USE_COST_METRIC:
                cost = mean([crossing_penalty(b1), crossing_penalty(b2)])
                distance = min(distances)

                cls._house_dcs[address_1][address_2] = store(round(distance), round(cost))
            else:
                cls._house_dcs[address_1][address_2] = round(min(distances))

    @classmethod
    def __init__(cls, blocks: blocks_file_t, depot: Optional[Point] = None):
        if os.path.exists(house_distance_table_file):
            need_regeneration = False
            print('House distance table file found. Loading may take a while...')
            cls._house_dcs = json.load(open(house_distance_table_file))
            num_samples = min(len(blocks), 100)

            # Sample random segments from the input to check if they are already stored
            for block in random.sample(list(blocks.values()), num_samples):
                try:
                    cls._house_dcs[next(iter(block['addresses'].keys()))]
                except StopIteration:
                    # There are no houses in this segment
                    continue
                except KeyError:
                    # This house was not already stored
                    need_regeneration = True
                    break
            if not need_regeneration and (depot is None or pt_id(depot) in cls._house_dcs):
                return
            print('The saved distance table did not include all requested segments. Regenerating...')
        else:
            print('No house distance table file found at {}. Generating now...'.format(house_distance_table_file))

        cls._house_dcs = {}
        with tqdm(total=len(blocks) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for b_id, block in blocks.items():
                if depot is not None:
                    cls._insert_point(depot, block)
                for other_b_id, other_block in blocks.items():
                    cls._insert_pair(block, b_id, other_block, other_b_id)
                    progress.update()

        print('Saving to {}'.format(house_distance_table_file))

        json.dump(cls._house_dcs, open(house_distance_table_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point) -> Optional[tuple[float, float]]:
        '''
        Get the distance between two houses by their coordinates

        Parameters:
            p1 (Point): the first point
            p2 (Point): the second point

        Returns:
            tuple[float, float] | None: distance, cost between the two points if it exists, None otherwise
        '''
        if len(cls._house_dcs) == 0:
            raise ValueError('House distance table has not been initialized')
        try:
            return unstore(cls._house_dcs[pt_id(p1)][pt_id(p2)])
        except KeyError:
            try:
                return unstore(cls._house_dcs[pt_id(p2)][pt_id(p1)])
            except KeyError:
                return None
