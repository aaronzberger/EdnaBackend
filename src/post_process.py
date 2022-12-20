# TODO: If a segment is traversed more than once, make sure it matches (down one back on other side)

import itertools
import json
from copy import deepcopy
from typing import Optional

from src.config import (Tour, blocks_file, blocks_file_t, houses_file,
                        houses_file_t)
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.gps_utils import Point, project_to_line
from src.timeline_utils import Segment, SubSegment


class PostProcess():
    def __init__(self, segments: list[Segment], points: list[Point], canvas_start: Optional[Point] = None):
        self.blocks: blocks_file_t = json.load(open(blocks_file))
        self.address_to_segment_id: houses_file_t = json.load(open(houses_file))

        self.id_to_segment = {s.id: s for s in segments}

        self.points = points
        self.canvas_start = canvas_start

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
        origin_segment = self.id_to_segment[self.address_to_segment_id[final_house.id]]

        through_end = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_end']
        through_end += min(MixDistances.get_distance_through_ends(
            node=origin_segment.end, house=next_house))

        through_start = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_start']
        through_start += min(MixDistances.get_distance_through_ends(
            node=origin_segment.start, house=next_house))

        return origin_segment.end if through_end < through_start else origin_segment.start

    def _calculate_entrance(self, intersection: Point, next_house: Point) -> Point:
        '''
        Calculate the optimal entrance point for a subsegment given the running intersection and the next house

        Parameters:
            intersection (Point): the current location of the walker
            next_house (Point): the first house to visit on the next segment

        Returns:
            Point: the entrance point of the next segment, which is either the start or endpoint of next_house's segment
        '''
        # Determine the exit direction, which will either be the start or end of the segment
        destination_segment = self.id_to_segment[self.address_to_segment_id[next_house.id]]

        through_end = self.blocks[destination_segment.id]['addresses'][next_house.id]['distance_to_end']
        try:
            to_end = NodeDistances.get_distance(intersection, destination_segment.end)
            through_end += 1600 if to_end is None else to_end
        except TypeError:
            through_end += 1600

        through_start = self.blocks[destination_segment.id]['addresses'][next_house.id]['distance_to_start']
        try:
            to_start = NodeDistances.get_distance(intersection, destination_segment.start)
            through_start += 1600 if to_start is None else to_start
        except TypeError:
            through_start += 1600

        return destination_segment.end if through_end < through_start else destination_segment.start

    def _process_subsegment(self, houses: list[Point], segment: Segment, entrance: Point, exit: Point) -> SubSegment:
        original_houses = deepcopy(houses)
        block_info = self.blocks[segment.id]['addresses']

        # Find the first subsegment containing houses
        first_subsegment = min(block_info.values(), key=lambda a: a['subsegment'][1])['subsegment']

        # Determine which houses are on this subsegment
        houses_on_first = [i for _, i in block_info.items() if i['subsegment'] == first_subsegment]

        # Find the house on this subsegment closest to the start
        closest_to_start = min(houses_on_first, key=lambda d: d['distance_to_start'])
        closest_to_start = Point(closest_to_start['lat'], closest_to_start['lon'])

        # Repeat for the furthest subsegment
        last_subsegment = max(block_info.values(), key=lambda a: a['subsegment'][1])['subsegment']
        houses_on_last = [i for _, i in block_info.items() if i['subsegment'] == last_subsegment]
        closest_to_end = max(houses_on_last, key=lambda d: d['distance_to_start'])
        closest_to_end = Point(closest_to_end['lat'], closest_to_end['lon'])

        # print('Found that segment {} had first add {} and last {}, on subsegments {} and {}'.format(
        #     segment.id, closest_to_start, closest_to_end, first_subsegment, last_subsegment
        # ))

        # Project these houses onto their respective subsegments to obtain the furthest points on the segment
        extremum = (
            project_to_line(
                p1=closest_to_start,
                p2=segment.navigation_points[first_subsegment[0]], p3=segment.navigation_points[first_subsegment[1]]),
            project_to_line(
                p1=closest_to_end,
                p2=segment.navigation_points[last_subsegment[0]], p3=segment.navigation_points[last_subsegment[1]])
        )

        # TODO: Fix extremum calculation
        # extremum = (
        #     project_to_line(
        #         p1=closest_to_start,
        #         p2=Point(**self.blocks[segment.id]['nodes'][first_subsegment[0]]), p3=Point(**self.blocks[segment.id]['nodes'][first_subsegment[1]])),
        #     project_to_line(
        #         p1=closest_to_end,
        #         p2=Point(**self.blocks[segment.id]['nodes'][last_subsegment[0]]), p3=Point(**self.blocks[segment.id]['nodes'][last_subsegment[1]]))
        # )

        # Calculate the navigation points
        middle_points = self.blocks[segment.id]['nodes'][first_subsegment[1] + 1:last_subsegment[0]]
        navigation_points = [extremum[0]] + [Point(**p, type='other') for p in middle_points] + [extremum[1]]

        # If the start and end are different, this is a forward or backward bouncing block
        if entrance != exit:
            houses = sorted(
                houses, key=lambda h: block_info[h.id]['distance_to_start'], reverse=entrance != segment.start)
        else:
            # First, order all houses on the same side of the street as the first
            first_half = [h for h in houses if block_info[h.id]['side'] == block_info[houses[0].id]['side']]
            first_half = sorted(first_half, key=lambda h: block_info[h.id]['distance_to_start']
                                if entrance == segment.start else block_info[h.id]['distance_to_end'])

            # Next, backtrack: order all houses on the other side going back towards the entrance/exit
            second_half = [h for h in houses if block_info[h.id]['side'] != block_info[houses[0].id]['side']]
            second_half = sorted(second_half, key=lambda h: block_info[h.id]['distance_to_start']
                                 if entrance == segment.start else block_info[h.id]['distance_to_end'], reverse=True)

            houses = first_half + second_half

        return SubSegment(
            segment=segment, start=entrance, end=exit, extremum=extremum,
            houses=houses, navigation_points=navigation_points)

    def post_process(self, tour: Tour) -> list[SubSegment]:
        # Iterate through the solution and add subsegments
        walk_list: list[SubSegment] = []

        current_subsegment_points: list[Point] = []

        try:
            houses = [self.points[h['location']['index']] for h in tour['stops']]
        except IndexError:
            print(tour['stops'])
            print(len(self.points))
            print([h['location']['index']] for h in tour['stops'])

        # Determine the starting intersection for the walk list
        if self.canvas_start is None:
            # Take the side closest to the first house (likely where a canvasser would park)
            first_segment_id = self.address_to_segment_id[houses[0].id]
            running_intersection = self.id_to_segment[first_segment_id].start if \
                self.blocks[first_segment_id]['addresses'][houses[0].id]['distance_to_start'] < \
                self.blocks[first_segment_id]['addresses'][houses[0].id]['distance_to_end'] else \
                self.id_to_segment[first_segment_id].end
            parking_location = deepcopy(running_intersection)
        else:
            running_intersection: Point = self.canvas_start

        # Process the list
        for house, next_house in itertools.pairwise(houses):
            segment_id = self.address_to_segment_id[house.id]
            next_segment_id = self.address_to_segment_id[next_house.id]
            current_subsegment_points.append(house)

            if next_segment_id != segment_id:
                exit_point = self._calculate_exit(house, next_house)
                subsegment = self._process_subsegment(
                    current_subsegment_points, self.id_to_segment[segment_id],
                    entrance=self._calculate_entrance(running_intersection, current_subsegment_points[0]),
                    exit=exit_point)

                current_subsegment_points = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_point

                # If this is a single-house segment on the side
                if subsegment.start == subsegment.end and len(subsegment.houses) == 1:
                    print('Removed a stray segment for walkability')
                    continue

                walk_list.append(subsegment)

        # Since we used pairwise, the last house is never evaluated
        current_subsegment_points.append(houses[-1])

        # Determine the final intersection the canvasser will end up at to process the final subsegment
        if self.canvas_start is None:
            exit_point = self._calculate_entrance(parking_location, current_subsegment_points[-1])
        else:
            exit_point = self._calculate_entrance(self.canvas_start, current_subsegment_points[-1])

        walk_list.append(self._process_subsegment(
            current_subsegment_points, self.id_to_segment[self.address_to_segment_id[houses[-1].id]],
            entrance=running_intersection, exit=exit_point))
        
        # Now, do inter-segment ordering

        return walk_list
