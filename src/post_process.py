import csv
import itertools
import json
import os
import pickle
from copy import deepcopy
from random import randint
from sys import argv

import names

from src.config import (BASE_DIR, TURF_SPLIT, HousePeople, Person, Point, Tour,
                        blocks_file, blocks_file_t, houses_file, houses_file_t,
                        pt_id)
from src.distances.blocks import BlockDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.gps_utils import SubBlock, project_to_line
from src.route import get_route, RouteMaker
from src.viz_utils import display_individual_walk_lists, display_walk_lists


class PostProcess():
    def __init__(self, blocks: blocks_file_t, points: list[Point]):
        self.address_to_segment_id: houses_file_t = json.load(open(houses_file))

        self.blocks = blocks
        self.points = points
        RouteMaker()

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

        extremum: tuple[Point, Point] = (entrance, exit)

        # region: Calculate the navigation points
        if pt_id(entrance) != pt_id(exit):
            # Order the navigation points
            navigation_points = block['nodes'] if pt_id(entrance) == pt_id(block['nodes'][0]) else block['nodes'][::-1]

        elif pt_id(entrance) == pt_id(exit) == pt_id(block['nodes'][0]):
            # Find the last subsegment containing houses
            last_subsegment = max(block_addresses.values(), key=lambda a: a['subsegment'][1])['subsegment']

            # Find the last house on that segment
            houses_on_last = [i for i in block_addresses.values() if i['subsegment'] == last_subsegment]
            closest_to_end = max(houses_on_last, key=lambda d: d['distance_to_start'])
            closest_to_end = Point(lat=closest_to_end['lat'], lon=closest_to_end['lon'])  # type: ignore

            # Project that house onto the segment
            end_extremum = project_to_line(
                p1=closest_to_end,
                p2=block['nodes'][last_subsegment[0]], p3=block['nodes'][last_subsegment[1]])
            extremum = (extremum[0], end_extremum)

            navigation_points = block['nodes'][:last_subsegment[0] + 1] + [end_extremum]

        elif pt_id(entrance) == pt_id(exit) == pt_id(block['nodes'][-1]):
            # Find the first subsegment containing houses
            first_subsegment = min(block_addresses.values(), key=lambda a: a['subsegment'][1])['subsegment']

            # Find the first house on that segment
            houses_on_first = [i for i in block_addresses.values() if i['subsegment'] == first_subsegment]
            closest_to_start = min(houses_on_first, key=lambda d: d['distance_to_start'])
            closest_to_start = Point(lat=closest_to_start['lat'], lon=closest_to_start['lon'])  # type: ignore

            # Project that house onto the segment
            start_extremum = project_to_line(
                p1=closest_to_start,
                p2=block['nodes'][first_subsegment[0]], p3=block['nodes'][first_subsegment[1]])
            extremum = (start_extremum, extremum[1])

            navigation_points = [start_extremum] + block['nodes'][first_subsegment[1]:]
        # endregion

        # region: Order the houses
        if pt_id(entrance) != pt_id(exit):
            running_side = block_addresses[houses[0]['id']]['side']
            start = 0
            for i, house in enumerate(houses):
                if block_addresses[house['id']]['side'] != running_side:
                    houses[start:i] = sorted(houses[start:i], key=lambda h: block_addresses[h['id']]['distance_to_start'],
                                             reverse=pt_id(entrance) != pt_id(block['nodes'][0]))
                    running_side = block_addresses[house['id']]['side']
                    start = i
            # Now, sort the last side
            houses[start:] = sorted(houses[start:], key=lambda h: block_addresses[h['id']]['distance_to_start'],
                                    reverse=pt_id(entrance) != pt_id(block['nodes'][0]))
        elif pt_id(entrance) == pt_id(exit):
            # We can assume that the first house is on the "out" side
            out_side = [h for h in houses if block_addresses[h['id']]['side'] == block_addresses[houses[0]['id']]['side']]
            back_side = [h for h in houses if block_addresses[h['id']]['side'] != block_addresses[houses[0]['id']]['side']]

            # Put the "out" side houses first, then the "back" side houses
            houses = sorted(out_side, key=lambda h: block_addresses[h['id']]['distance_to_start'],
                            reverse=pt_id(entrance) != pt_id(block['nodes'][0])) + \
                sorted(back_side, key=lambda h: block_addresses[h['id']]['distance_to_end'],
                       reverse=pt_id(entrance) != pt_id(block['nodes'][0]))

        # endregion

        return SubBlock(
            block=block, start=entrance, end=exit, extremum=extremum,
            houses=houses, navigation_points=navigation_points)

    def fill_holes(self, walk_list: list[SubBlock]) -> list[SubBlock]:
        '''
        Fill in intermediate blocks between sub-blocks, wherever the end of one block
        is not the same as the start of the next one

        Parameters:
            walk_list (list[SubBlock]): The list of sub-blocks to fill in

        Returns:
            list[SubBlock]: The list of sub-blocks with holes filled in
        '''
        # Iterate through the solution and add subsegments
        new_walk_list: list[SubBlock] = []

        ids = []
        for subblock in walk_list:
            ids.append(pt_id(subblock.start))
            ids.append(pt_id(subblock.end))
        print("IDs: {}".format(ids))

        for first, second in itertools.pairwise(walk_list):
            new_walk_list.append(first)

            # If the end of the first subblock is not the same as the start of the second subblock, add a new subblock
            if pt_id(first.end) != pt_id(second.start):
                # directions = get_route(first.end, second.start)

                # print("Directions from {} to {}: {}".format(first.end, second.start, directions['route']))

                # for start_pt, end_pt in itertools.pairwise(directions['route']):
                #     start = Point(lat=start_pt[0], lon=start_pt[1])
                #     end = Point(lat=end_pt[0], lon=end_pt[1])

                # TODO: DEBUG and test here
                block_ids = RouteMaker.get_route(first.end, second.start)
                for block_id in block_ids:
                    block = self.blocks[block_id]

                    # NOTE: For now, the order of the block does not matter
                    start = Point(lat=block['nodes'][0]['lat'], lon=block['nodes'][0]['lon'])
                    end = Point(lat=block['nodes'][-1]['lat'], lon=block['nodes'][-1]['lon'])

                    new_walk_list.append(SubBlock(
                        block=block_id, start=start, end=end, extremum=(start, end),
                        houses=[], navigation_points=block['nodes']))

                print('Starting point ({}, {}) and ending point ({}, {}) are not the same'.format(
                    first.end['lat'], first.end['lon'], second.start['lat'], second.start['lon']))
                print('So, adding sub-blocks between them with nodes: {}'.format(block_ids))

        new_walk_list.append(walk_list[-1])

        return new_walk_list

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
        entrance_point = self._calculate_entrance(running_intersection, current_sub_block_points[0])

        walk_list.append(self._process_sub_block(
            current_sub_block_points, running_block_id, entrance=entrance_point, exit=exit_point))

        # Fill in any holes
        walk_list = self.fill_holes(walk_list)

        return walk_list

    def generate_file(self, walk_list: list[SubBlock], output_file: str):
        '''
        Generate a JSON file with the walk list (for front-end)

        Parameters:
            walk_list (list[SubBlock]): The walk list to generate the file for
            output_file (str): The file to write to
        '''
        # Generate the JSON file
        list_out = {}
        list_out['blocks'] = []
        for sub_block in walk_list:
            nodes = []
            for nav_pt in sub_block.navigation_points:
                nodes.append({
                    'lat': nav_pt['lat'],
                    'lon': nav_pt['lon']
                })
            houses = []
            for house in sub_block.houses:
                houses.append(HousePeople(
                    address=house['id'],
                    coordinates={
                        'lat': house['lat'],
                        'lon': house['lon']
                    },
                    voter_info=[Person(
                        name=names.get_full_name(),
                        age=randint(18, 95))
                            for _ in range(randint(1, 5))]
                ))
            list_out['blocks'].append({
                'nodes': nodes,
                'houses': houses
            })

        # Write the file
        json.dump(list_out, open(output_file, 'w'))


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

    list_visualizations = display_individual_walk_lists(walk_lists)
    for i, walk_list in enumerate(list_visualizations):
        walk_list.save(os.path.join(BASE_DIR, 'viz', 'walk_lists', '{}.html'.format(i)))

    for i in range(len(walk_lists)):
        post_processor.generate_file(walk_lists[i], os.path.join(BASE_DIR, 'viz', 'files', f'{i}.json'))
