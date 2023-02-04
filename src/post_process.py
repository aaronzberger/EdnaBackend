import csv
import itertools
import json
import os
from copy import deepcopy
import pickle
from sys import argv

from src.config import (BASE_DIR, TURF_SPLIT, Tour, blocks_file, blocks_file_t,
                        houses_file, houses_file_t, Point)
from src.distances.blocks import BlockDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.gps_utils import SubBlock, project_to_line
from src.viz_utils import display_walk_lists


class PostProcess():
    def __init__(self, blocks: blocks_file_t, points: list[Point]):
        self.address_to_segment_id: houses_file_t = json.load(open(houses_file))

        self.blocks = blocks
        self.points = points

    def _calculate_exit(self, final_house: Point, next_house: Point) -> Point:
        '''
        Calculate the optimal exit point for a subsegment given the final house and the next house

        Parameters:
            final_house (Point): the final house on the segment
            next_house (Point): the next house to visit after this segment

        Returns:
            Point: the exit point of this segment, which is either the start or endpoint of final_house's segment
        '''
        # Determine the exit direction, which will either be the start or end of the segment
        origin_block_id = self.address_to_segment_id[final_house['id']]
        origin_block = self.blocks[origin_block_id]

        through_end = self.blocks[origin_block_id]['addresses'][final_house['id']]['distance_to_end']
        through_end += min(MixDistances.get_distance_through_ends(node=origin_block['nodes'][-1], house=next_house))

        through_start = self.blocks[origin_block_id]['addresses'][final_house['id']]['distance_to_start']
        through_start += min(MixDistances.get_distance_through_ends(node=origin_block['nodes'][0], house=next_house))

        return origin_block['nodes'][-1] if through_end < through_start else origin_block['nodes'][0]

    def _calculate_entrance(self, intersection: Point, next_house: Point) -> Point:
        '''
        Calculate the optimal entrance point for a sub-block given the running intersection and the next house

        Parameters:
            intersection (Point): the current location of the walker
            next_house (Point): the first house to visit on the next segment

        Returns:
            Point: the entrance point of the next segment, which is either the start or endpoint of next_house's segment
        '''
        # Determine the exit direction, which will either be the start or end of the segment
        destination_block_id = self.address_to_segment_id[next_house['id']]
        destination_block = self.blocks[destination_block_id]

        through_end = self.blocks[destination_block_id]['addresses'][next_house['id']]['distance_to_end']
        try:
            to_end = NodeDistances.get_distance(intersection, destination_block['nodes'][-1])
            through_end += 1600 if to_end is None else to_end[0]
        except TypeError:
            print('Unable to find distance through end of block in post-processing')
            through_end += 1600

        through_start = self.blocks[destination_block_id]['addresses'][next_house['id']]['distance_to_start']
        try:
            to_start = NodeDistances.get_distance(intersection, destination_block['nodes'][0])
            through_start += 1600 if to_start is None else to_start[0]
        except TypeError:
            print('Unable to find distance through end of block in post-processing')
            through_start += 1600

        return destination_block['nodes'][-1] if through_end < through_start else destination_block['nodes'][0]

    def _process_sub_block(self, houses: list[Point], block_id: str, entrance: Point, exit: Point) -> SubBlock:
        assigned_addresses = [i['id'] for i in houses]
        block_addresses = {add: inf for add, inf in self.blocks[block_id]['addresses'].items() if add in assigned_addresses}
        block = self.blocks[block_id]

        # Find the first subsegment containing houses
        first_subsegment = min(block_addresses.values(), key=lambda a: a['subsegment'][1])['subsegment']

        # print(assigned_addresses, [i for i in block_addresses.keys()])

        # Determine which houses are on this subsegment
        houses_on_first = [i for i in block_addresses.values() if i['subsegment'] == first_subsegment]

        # Find the house on this subsegment closest to the start
        closest_to_start = min(houses_on_first, key=lambda d: d['distance_to_start'])
        closest_to_start = Point(lat=closest_to_start['lat'], lon=closest_to_start['lon'])  # type: ignore

        # Repeat for the furthest subsegment
        last_subsegment = max(block_addresses.values(), key=lambda a: a['subsegment'][1])['subsegment']
        houses_on_last = [i for i in block_addresses.values() if i['subsegment'] == last_subsegment]
        closest_to_end = max(houses_on_last, key=lambda d: d['distance_to_start'])
        closest_to_end = Point(lat=closest_to_end['lat'], lon=closest_to_end['lon'])  # type: ignore

        # Project these houses onto their respective subsegments to obtain the furthest points on the segment
        extremum = (
            project_to_line(
                p1=closest_to_start,
                p2=block['nodes'][first_subsegment[0]], p3=block['nodes'][first_subsegment[1]]),
            project_to_line(
                p1=closest_to_end,
                p2=block['nodes'][last_subsegment[0]], p3=block['nodes'][last_subsegment[1]]),
        )

        # TODO: Correct extremum by making the extremum the entry or exit points if they apply

        # Calculate the navigation points
        middle_points = block['nodes'][first_subsegment[1]:last_subsegment[0] + 1]
        navigation_points = [extremum[0]] + [Point(**p) for p in middle_points] + [extremum[1]]

        return SubBlock(
            block=block, start=entrance, end=exit, extremum=extremum,
            houses=houses, navigation_points=navigation_points)

    def post_process(self, tour: Tour) -> list[SubBlock]:
        # Iterate through the solution and add subsegments
        walk_list: list[SubBlock] = []

        tour_stops = [self.points[h['location']['index']] for h in tour['stops']]
        depot, houses = tour_stops[0], tour_stops[1:-1]

        current_sub_block_points: list[Point] = []

        # Take the side closest to the first house (likely where a canvasser would park)
        running_intersection = depot
        running_block_id = self.address_to_segment_id[houses[0]['id']]

        # Process the list
        for house, next_house in itertools.pairwise(houses):
            next_block_id = self.address_to_segment_id[next_house['id']]
            current_sub_block_points.append(house)

            if next_block_id != running_block_id:
                entrance_pt = self._calculate_entrance(running_intersection, current_sub_block_points[0])
                exit_pt = self._calculate_exit(house, next_house)
                subsegment = self._process_sub_block(
                    current_sub_block_points, running_block_id, entrance=entrance_pt, exit=exit_pt)

                current_sub_block_points = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_pt
                running_block_id = next_block_id

                walk_list.append(subsegment)

        # Since we used pairwise, the last house is never evaluated
        current_sub_block_points.append(houses[-1])

        # Determine the final intersection the canvasser will end up at to process the final subsegment
        exit_point = self._calculate_entrance(depot, current_sub_block_points[-1])

        walk_list.append(self._process_sub_block(
            current_sub_block_points, running_block_id, entrance=running_intersection, exit=exit_point))

        return walk_list


if __name__ == '__main__':
    # region Handle universe file
    all_blocks: blocks_file_t = json.load(open(blocks_file))

    if len(argv) == 2:
        # Ensure the provided file exists
        if not os.path.exists(argv[1]):
            raise FileExistsError('Usage: make_walk_lists.py [UNIVERSE FILE]')

        reader = csv.DictReader(open(argv[1]))
        houses_to_id: houses_file_t = json.load(open(houses_file))
        requested_blocks: blocks_file_t = {}
        total_houses = failed_houses = 0

        # Process each requested house
        for house in reader:
            formatted_address = house['Address'].upper()
            total_houses += 1
            if formatted_address not in houses_to_id:
                failed_houses += 1
                continue
            block_id = houses_to_id[formatted_address]
            house_info = deepcopy(all_blocks[block_id]['addresses'][formatted_address])

            if block_id in requested_blocks:
                requested_blocks[block_id]['addresses'][formatted_address] = house_info
            else:
                requested_blocks[block_id] = deepcopy(all_blocks[block_id])
                requested_blocks[block_id]['addresses'] = {formatted_address: house_info}
        print('Failed on {} of {} houses'.format(failed_houses, total_houses))
    else:
        requested_blocks: blocks_file_t = json.load(open(blocks_file))
    # endregion

    NodeDistances(requested_blocks)
    BlockDistances(requested_blocks)
    MixDistances()

    # Load the solution file
    solution = json.load(open(os.path.join(BASE_DIR, 'optimize', 'solution.json')))

    points = pickle.load(open(os.path.join(BASE_DIR, 'optimize', 'points.pkl'), 'rb'))

    post_processor = PostProcess(requested_blocks, points=points)
    walk_lists: list[list[SubBlock]] = []
    for i, tour in enumerate(solution['tours']):
        # Do not count the starting location service at the start or end
        tour['stops'] = tour['stops'][1:-1] if TURF_SPLIT else tour['stops']

        if len(tour['stops']) == 0:
            print('List {} has 0 stops'.format(i))
            continue

        walk_lists.append(post_processor.post_process(tour))

    # Save the walk lists
    display_walk_lists(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'walk_lists.html'))
