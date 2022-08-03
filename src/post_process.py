import itertools
import json
from copy import deepcopy
from typing import Optional

from src.config import (Tour, blocks_file, blocks_file_t, houses_file,
                        houses_file_t)
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

    def distance_through_ends(self, start: Point, house: Point, segment: Segment) -> tuple[float, float]:
        '''
        Determine the distances from an intersection point to a house through the two ends of the house's segments

        Parameters:
            start (Point): the start point, which should be an intersection
            house (Point): the house, which should lie on the segment (the next parameter)
            segment (Segment): the segment on which the house lies

        Returns:
            float: the distance from the intersection to the house through the start of the segment
            float: the distance from the intersection to the house through the end of the segment
        '''
        try:
            through_start = NodeDistances.get_distance(start, segment.start)
            through_start = through_start if through_start is not None else 1600
        except KeyError:
            through_start = 1600
        through_start += self.blocks[segment.id]['addresses'][house.id]['distance_to_start']
        try:
            through_end = NodeDistances.get_distance(start, segment.end)
            through_end = through_end if through_end is not None else 1600
        except KeyError:
            through_end = 1600
        through_end += self.blocks[segment.id]['addresses'][house.id]['distance_to_end']
        return through_start, through_end

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
        destination_segment = self.id_to_segment[self.address_to_segment_id[next_house.id]]

        through_end = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_end']
        through_end += min(self.distance_through_ends(
            start=origin_segment.end, house=next_house, segment=destination_segment))

        through_start = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_start']
        through_start += min(self.distance_through_ends(
            start=origin_segment.start, house=next_house, segment=destination_segment))

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
            through_end += NodeDistances.get_distance(intersection, destination_segment.end)  # type: ignore
        except (KeyError, TypeError):
            through_end += 1600

        through_start = self.blocks[destination_segment.id]['addresses'][next_house.id]['distance_to_start']
        try:
            through_start += NodeDistances.get_distance(intersection, destination_segment.start)  # type: ignore
        except (KeyError, TypeError):
            through_start += 1600

        return destination_segment.end if through_end < through_start else destination_segment.start

    def _process_subsegment(self, houses: list[Point], segment: Segment, entrance: Point, exit: Point) -> SubSegment:
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

        # Project these houses onto their respective subsegments to obtain the furthest points on the segment
        extremum = (
            project_to_line(
                p1=closest_to_start,
                p2=segment.navigation_points[first_subsegment[0]], p3=segment.navigation_points[first_subsegment[1]]),
            project_to_line(
                p1=closest_to_end,
                p2=segment.navigation_points[last_subsegment[0]], p3=segment.navigation_points[last_subsegment[1]])
        )

        # Calculate the navigation points
        middle_points = self.blocks[segment.id]['nodes'][first_subsegment[1]:last_subsegment[0]]
        navigation_points = [closest_to_start] + [Point(**p) for p in middle_points] + [closest_to_end]

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

        houses = [self.points[h['location']['index']] for h in tour['stops']]

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
                walk_list.append(self._process_subsegment(
                    current_subsegment_points, self.id_to_segment[segment_id],
                    entrance=self._calculate_entrance(running_intersection, current_subsegment_points[0]),
                    exit=exit_point))

                current_subsegment_points = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_point

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

        return walk_list
