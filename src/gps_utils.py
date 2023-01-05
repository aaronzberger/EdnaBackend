from __future__ import annotations
from dataclasses import dataclass
import itertools

from math import acos, asin, cos, radians, sin

import utm
from geographiclib.geodesic import Geodesic
from haversine import Unit, haversine

from src.config import MINS_PER_HOUSE, WALKING_M_PER_S, Block, Point

converter: Geodesic = Geodesic.WGS84  # type: ignore


def along_track_distance(p1: Point, p2: Point, p3: Point) -> tuple[float, float]:
    '''
    Calculate the along-track distances from a point to a line in GPS coordinates

    Parameters:
        p1 (Point): coordinates of the single point
        p2 (Point): the first point in the line
        p3 (Point): the second point in the line

    Returns:
        float: the distance (in meters) on the line from the point (p1) to the beginning of the line (p2)
        float: the distance (in meters) on the line from the point (p1) to the end of the line (p3)
    '''
    # (ald = along track distance)              |                     ald P2
    #                                           |    ⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⌉
    #     ald P2            ald P3              |    |                                    ald P3
    #   ⌈‾‾‾‾‾‾‾‾‾‾‾⌉⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⌉     |    |                                 ⌈‾‾‾‾‾‾‾‾‾‾‾⌉
    #   P2 —————————————————————————————— P3    |   P2 —————————————————————————————— P3 - - - - -︱
    #               ︱                          ︱                                                 ︱
    #               ︱                          ︱                                                 ︱
    #               ︱                          ︱                                                 ︱
    #               P1                          |                                                 P1

    line_p2_p1 = converter.InverseLine(lat1=p2['lat'], lon1=p2['lon'], lat2=p1['lat'], lon2=p1['lon'])
    line_p3_p1 = converter.InverseLine(lat1=p3['lat'], lon1=p3['lon'], lat2=p1['lat'], lon2=p1['lon'])
    line_p2_p3 = converter.InverseLine(lat1=p2['lat'], lon1=p2['lon'], lat2=p3['lat'], lon2=p3['lon'])
    earth_radius = line_p2_p3.a

    bearing_p1_p2 = radians(line_p2_p1.azi1)  # P1-P2 bearing
    bearing_p2_p3 = radians(line_p2_p3.azi1)  # P2-P3 bearing
    distance_p1_p2 = line_p2_p1.s13  # P1-P2 distance

    cross_track_distance = asin(sin(distance_p1_p2 / earth_radius) * sin(bearing_p1_p2 - bearing_p2_p3)) * earth_radius
    ald_p2 = acos(cos(distance_p1_p2 / earth_radius) / cos(cross_track_distance / earth_radius)) * earth_radius

    distance_p1_p3 = line_p3_p1.s13  # P1-P3 distance
    ald_p3 = acos(cos(distance_p1_p3 / earth_radius) / cos(cross_track_distance / earth_radius)) * earth_radius

    return ald_p2, ald_p3


def cross_track_distance(p1: Point, p2: Point, p3: Point, debug: bool = False) -> float:
    '''
    Calculate the cross-track distance from a point to a line in GPS coordinates

    Parameters:
        p1 (Point): coordinates of the single point
        p2 (Point): the first point in the line
        p3 (Point): the second point in the line
        debug (bool): whether to print debug statements

    Returns:
        float: the distance (in meters) from the point (p1) to the line
    '''
    #                                           |
    #   P2 —————————————————————————————— P3    |   P2 —————————————————————————————— P3 - - - -︱ ⎤
    #               ︱ ⎤                        ︱                                               ︱ ⎥
    #               ︱ ⎥ cross track distance   ︱                                               ︱ ⎥ cross track distance
    #               ︱ ⎦                        ︱                                               ︱ ⎦
    #               P1                          |                                               P1

    line_p2_p1 = converter.InverseLine(lat1=p2['lat'], lon1=p2['lon'], lat2=p1['lat'], lon2=p1['lon'])
    line_p3_p1 = converter.InverseLine(lat1=p3['lat'], lon1=p3['lon'], lat2=p1['lat'], lon2=p1['lon'])
    line_p2_p3 = converter.InverseLine(lat1=p2['lat'], lon1=p2['lon'], lat2=p3['lat'], lon2=p3['lon'])
    earth_radius = line_p2_p3.a

    bearing_p1_p2 = radians(line_p2_p1.azi1)  # P1-P2 bearing
    bearing_p2_p3 = radians(line_p2_p3.azi1)  # P2-P3 bearing
    distance_p1_p2 = line_p2_p1.s13  # P1-P2 distance

    cross_track_distance = asin(sin(distance_p1_p2 / earth_radius) * sin(bearing_p1_p2 - bearing_p2_p3)) * earth_radius
    ald_p2 = acos(cos(distance_p1_p2 / earth_radius) / cos(cross_track_distance / earth_radius)) * earth_radius

    distance_p1_p3 = line_p3_p1.s13  # P1-P3 distance
    ald_p3 = acos(cos(distance_p1_p3 / earth_radius) / cos(cross_track_distance / earth_radius)) * earth_radius

    distance_p2_p3 = line_p2_p3.s13  # P2-P3 distance

    # If either of the along track distances is longer than the distance of
    # the line, the point must be off to one side of the line
    if abs(ald_p2) > abs(distance_p2_p3) or abs(ald_p3) > abs(distance_p2_p3):
        if debug:
            print('Point not on line, since along track distance {:.2f} or {:.2f}'.format(
                    ald_p2, ald_p3) +
                  ' is greater than the block distance {:.2f}'.format(
                    distance_p2_p3))
        return distance_p1_p2 if abs(distance_p1_p2) < abs(distance_p1_p3) else distance_p1_p3
    else:
        if debug:
            print('Point is on line, since along track distance {:.2f} and {:.2f}'.format(
                    ald_p2, ald_p3) +
                  'are less than block distance {:.2f}'.format(
                    distance_p2_p3))
        return cross_track_distance


def great_circle_distance(p1: Point, p2: Point) -> float:
    '''
    Calculate the distance "as the crow flies" between two points in GPS coordinates

    Parameters:
        p1 (Point): the first point
        p2 (Point): the second point

    Returns:
        float: the distance (in meters) between the points
    '''
    return haversine((p1['lat'], p1['lon']), (p2['lat'], p2['lon']), unit=Unit.METERS)


def middle(p1: Point, p2: Point) -> Point:
    '''
    Calculate the midpoint between two points in GPS coordinates

    Parameters:
        p1 (Point): the first point
        p2 (Point): the second point

    Returns:
        Point: the midpoint between the two provided points
    '''
    path_between = converter.InverseLine(lat1=p1['lat'], lon1=p1['lon'], lat2=p2['lat'], lon2=p2['lon'])
    middle = path_between.Position(path_between.s13 / 2.0)
    return Point(lat=middle['lat2'], lon=middle['lon2'], type='other', id=None)


def pt_to_utm(pt: Point) -> tuple[float, float, int, str]:
    '''
    Convert a point in GPS coordinates to UTM x-y grid coordinates

    Parameters:
        pt: the point

    Returns:
        float: the x-component
        float: the y-component
        int: the zone
        str: the letter
    '''
    return utm.from_latlon(pt['lat'], pt['lon'])  # type: ignore


def utm_to_pt(x: float, y: float, zone: int, letter: str) -> Point:
    '''
    Convert UTM x-y grid coordinates to a point in GPS coordinates

    Parameters:
        x (float): x-component of point
        y (float): y-component of point

    Returns:
        Point: the geographic point with its GPS coordinates
    '''
    return Point(*utm.to_latlon(x, y, zone, letter))  # type: ignore


def angle_between_pts(p1: Point, p2: Point) -> float:
    '''
    Calculate the angle between two points

    Parameters:
        p1 (Point): the first point
        p2 (Point): the second point

    Returns:
        float: the angle (in degrees) between the two provided points
    '''
    return converter.InverseLine(lat1=p1['lat'], lon1=p1['lon'], lat2=p2['lat'], lon2=p2['lon']).azi1 - 90


def project_to_line(p1: Point, p2: Point, p3: Point) -> Point:
    '''
    Project a point to the line spanned by two points

    Parameters:
        p1 (Point): the individual point
        p2 (Point): the first point in the line
        p3 (Point): the second point in the line

    Returns:
        Point: the result of projecting p1 onto the line spanned by p2 and p3
    '''
    path_between = converter.InverseLine(lat1=p2['lat'], lon1=p2['lon'], lat2=p3['lat'], lon2=p3['lon'])
    ald_p1 = along_track_distance(p1, p2, p3)[0]
    projected = path_between.Position(ald_p1)
    return Point(lat=projected['lat2'], lon=projected['lon2'], type='other', id=None)


@dataclass
class SubBlock():
    block: Block

    # Note that start and end could be the same point.
    start: Point
    end: Point

    # Furthest points in each direction the houses span
    extremum: tuple[Point, Point]

    houses: list[Point]
    navigation_points: list[Point]

    def __post_init__(self):
        self.length: float = 0.0
        self.length += great_circle_distance(self.start, self.navigation_points[0])
        for first, second in itertools.pairwise(self.navigation_points):
            self.length += great_circle_distance(first, second)
        self.length += great_circle_distance(self.end, self.navigation_points[-1])

        # TODO: Time to walk depends on walk_method and should likely be iterated through
        self.time_to_walk = len(self.houses) * MINS_PER_HOUSE + (self.length / WALKING_M_PER_S * (1/60))
