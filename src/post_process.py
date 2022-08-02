import itertools
import json
import random

from src.config import (Tour, blocks_file, blocks_file_t, houses_file,
                        houses_file_t)
from src.distances.nodes import NodeDistances
from src.gps_utils import EmptyPoint, Point, project_to_line, tsp
from src.timeline_utils import Segment, SubSegment


class PostProcess():
    def __init__(self, segments: list[Segment]):
        self.blocks: blocks_file_t = json.load(open(blocks_file))
        self.house_to_id: houses_file_t = json.load(open(houses_file))

        self.id_to_segment = {s.id: s for s in segments}

    def calculate_exit(self, final_house: Point, next_house: Point) -> Point:
        '''
        Calculate the exit optimal exit point for a subsegment given the final house and the next house

        Parameters:
            final_house (Point): the final house on the segment
            next_house (Point): the next house to visit after this segment

        Returns:
            Point: the exit point of this segment, which is either the start or endpoint of final_house's segment
        '''
        def distance_through_ends(start: Point, house: Point, segment: Segment) -> tuple[float, float]:
            '''
            Determine the distances from an intersection point to a house through the two ends of the house's segments

            Parameters:
                start (Point): the start point, which should be an intersection
                house (Point): the final house, which should lie on the segment (the next parameter)
                segment (Segment): the segment on which the final house lies

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

        # Determine the exit direction, which will either be the start or end of the segment
        origin_segment = self.id_to_segment[self.house_to_id[final_house.id]]
        destination_segment = self.id_to_segment[self.house_to_id[next_house.id]]

        through_end = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_end']
        through_end += min(distance_through_ends(
            start=origin_segment.end, house=final_house, segment=destination_segment))

        through_start = self.blocks[origin_segment.id]['addresses'][final_house.id]['distance_to_start']
        through_start += min(distance_through_ends(
            start=origin_segment.start, house=final_house, segment=destination_segment))

        return origin_segment.end if through_end < through_start else origin_segment.start

    def process_subsegment(self, houses: list[Point], segment: Segment, entrance: Point, exit: Point) -> SubSegment:
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

    def post_process(self, tour: Tour, points: list[Point]) -> list[SubSegment]:
        # Iterate through the solution and add subsegments
        walk_list: list[SubSegment] = []

        current_subsegment_points: list[Point] = []

        # To determine starting location, simply run TSP to see the fastest order to arrive at the last house
        running_intersection: Point = EmptyPoint

        houses = [points[h['location']['index']] for h in tour['stops']]
        for house, next_house in itertools.pairwise(houses):
            segment_id = self.house_to_id[house.id]
            next_segment_id = self.house_to_id[next_house.id]

            if next_segment_id != segment_id:
                if running_intersection == EmptyPoint:
                    # We cannot run tsp on all the houses if there are too many
                    subsegment_sample = random.sample(current_subsegment_points, min(len(current_subsegment_points), 6))
                    ordered_sample = tsp(subsegment_sample, lock_last=True)

                    # Now, whichever side the first house is closest to is the most efficient entrance
                    running_intersection = self.id_to_segment[segment_id].start if \
                        self.blocks[segment_id]['addresses'][ordered_sample[0].id]['distance_to_start'] < \
                        self.blocks[segment_id]['addresses'][ordered_sample[0].id]['distance_to_end'] else \
                        self.id_to_segment[segment_id].end

                walk_list.append(self.process_subsegment(
                    current_subsegment_points, self.id_to_segment[segment_id],
                    entrance=running_intersection, exit=self.calculate_exit(house, next_house)))
            else:
                current_subsegment_points.append(house)

        # To determine the exiting location, run TSP
        subsegment_sample = random.sample(current_subsegment_points, min(len(current_subsegment_points), 6))
        ordered_sample = tsp(subsegment_sample, lock_first=True)

        segment_id = subsegment_sample[0].id

        # Now, whichever side the first house is closest to is the most efficient exit
        exit_point = self.id_to_segment[segment_id].start if \
            self.blocks[segment_id]['addresses'][ordered_sample[-1].id]['distance_to_start'] < \
            self.blocks[segment_id]['addresses'][ordered_sample[-1].id]['distance_to_end'] else \
            self.id_to_segment[segment_id].end

        walk_list.append(self.process_subsegment(
            current_subsegment_points, self.id_to_segment[self.house_to_id[houses[-1].id]],
            entrance=running_intersection, exit=exit_point))

        return walk_list
