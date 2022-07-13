from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, cos, radians, sin

import utm
from geographiclib.geodesic import Geodesic
from haversine import Unit, haversine

BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'


@dataclass
class Point():
    lat: float
    lon: float


def along_track_distance(p1: Point, p2: Point, p3: Point):
    '''
    Calculate the distance from a point to a line in GPS coordinates
    The line is made up of points P2 and P3, and the individual point is P1

    Parameters:
        p1 (Point): coordinates of the point P1
        p2 (Point): coordinates of P2
        p3 (Point): coordinates of P3

    Returns:
        float: the distance in meters along the road from the house to the first point
        float: the distance in meters along the road from the house to the second point
    '''
    # (ald = along track distance)              |                     ald P1
    #                                           |    ⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⌉
    #     ald P1            ald P2              |    |                                    ald P2
    #   ⌈‾‾‾‾‾‾‾‾‾‾‾⌉⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⌉     |    |                                 ⌈‾‾‾‾‾‾‾‾‾‾‾⌉
    #   P1 —————————————————————————————— P2    |   P1 —————————————————————————————— P2 - - - - -︱
    #               ︱                          ︱                                                 ︱
    #               ︱                          ︱                                                 ︱
    #               ︱                          ︱                                                 ︱
    #               P3                          |                                                 P3

    EARTH_R = 6371e3  # meters
    converter = Geodesic.WGS84

    θ13 = radians(converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p1.lat, lon2=p1.lon)['azi1'])  # P1-P3 bearing
    θ12 = radians(converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p3.lat, lon2=p3.lon)['azi1'])  # P1-P2 bearing
    δ13 = converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p1.lat, lon2=p1.lon)['s12']  # P1-P3 distance

    cross_track_distance = asin(sin(δ13 / EARTH_R) * sin(θ13 - θ12)) * EARTH_R
    along_track_distance_P1 = acos(cos(δ13 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # ALD wrt P1

    δ23 = converter.Inverse(lat1=p3.lat, lon1=p3.lon, lat2=p1.lat, lon2=p1.lon)['s12']  # P2-P3 distance
    along_track_distance_P2 = acos(cos(δ23 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # ALD wrt P2

    return along_track_distance_P1, along_track_distance_P2


def cross_track_distance(p1: Point, p2: Point, p3: Point, debug=False):
    '''
    Calculate the distance from a point to a line in GPS coordinates
    The line is made up of points P2 and P3, and the individual point is P1

    Parameters:
        p1 (Point): coordinates of the point P1
        p2 (Point): coordinates of P2
        p3 (Point): coordinates of P3
        debug (bool): whether to print debug statements

    Returns:
        float: the distance in meters from the point to the line
    '''
    #                                           |
    #   P1 —————————————————————————————— P2    |   P1 —————————————————————————————— P2 - - - -︱ ⎤
    #               ︱ ⎤                        ︱                                               ︱ ⎥
    #               ︱ ⎥ cross track distance   ︱                                               ︱ ⎥ cross track distance
    #               ︱ ⎦                        ︱                                               ︱ ⎦
    #               P3                          |                                               P3

    EARTH_R = 6371e3  # meters
    converter = Geodesic.WGS84

    θ13 = radians(converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p1.lat, lon2=p1.lon)['azi1'])  # P1-P3 bearing
    θ12 = radians(converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p3.lat, lon2=p3.lon)['azi1'])  # P1-P2 bearing
    δ13 = converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p1.lat, lon2=p1.lon)['s12']  # P1-P3 distance

    cross_track_distance = asin(sin(δ13 / EARTH_R) * sin(θ13 - θ12)) * EARTH_R
    along_track_distance_P1 = acos(cos(δ13 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # ALD wrt P1

    δ23 = converter.Inverse(lat1=p3.lat, lon1=p3.lon, lat2=p1.lat, lon2=p1.lon)['s12']  # P2-P3 distance
    along_track_distance_P2 = acos(cos(δ23 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # ALD wrt P2

    δ12 = converter.Inverse(lat1=p2.lat, lon1=p2.lon, lat2=p3.lat, lon2=p3.lon)['s12']  # P1-P2 distance (line length)

    # If either of the along track distances is longer than the distance of
    # the line, the point must be off to one side of the line
    if abs(along_track_distance_P1) > abs(δ12) or abs(along_track_distance_P2) > abs(δ12):
        if debug:
            print('Point not on line, since along track distance {:.2f} or {:.2f}'.format(
                    along_track_distance_P1, along_track_distance_P2) +
                  ' is greater than the block distance {:.2f}'.format(
                    δ12))
        return δ13 if abs(δ13) < abs(δ23) else δ23
    else:
        if debug:
            print('Point is on line, since along track distance {:.2f} and {:.2f}'.format(
                    along_track_distance_P1, along_track_distance_P2) +
                  'are less than block distance {:.2f}'.format(
                    δ12))
        return cross_track_distance


def distance(p1: Point, p2: Point):
    '''
    Calculate the distance between points in GPS coordinates

    Parameters:
        p1 (Point): first point
        p2 (Point): second point

    Returns:
        float: the distance in meters between the points
    '''
    return haversine((p1.lat, p1.lon), (p2.lat, p2.lon), unit=Unit.METERS)


def pt_to_utm(pt: Point):
    '''
    Converts GPS coordinates to X-Y grid coordinates

    Parameters:
        pt: the point

    Returns:
        float: the x-component
        float: the y-component
        int: the zone
        char: the letter
    '''
    return utm.from_latlon(pt.lat, pt.lon)


def utm_to_pt(x, y, zone, letter):
    '''
    Converts GPS coordinates to X-Y grid coordinates

    Parameters:
        x (float): x-component of point
        y (float): y-component of point

    Returns:
        Point: the Point
    '''
    return Point(utm.to_latlon(x, y, zone, letter))


def angle_between_pts(p1: Point, p2: Point):
    '''Calculate the bearing from p1 to p2'''
    converter = Geodesic.WGS84
    return converter.Inverse(lat1=p1.lat, lon1=p1.lon, lat2=p2.lat, lon2=p2.lon)['azi1'] - 90
