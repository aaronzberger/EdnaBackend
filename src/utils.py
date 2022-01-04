from enum import Enum
from geographiclib.geodesic import Geodesic
from math import radians, sin, cos, asin, acos

class Colors(Enum):
    '''FOr coloring terminal text'''
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def cross_track_distance(ptlat, ptlon, lat1, lon1, lat2, lon2, debug=False):
    '''
    Calculate the distance from a point to a line in GPS coordinates
    As used below, the line is made up of points P1 and P2, and the individual point is P3

    Parameters:
        ptlat (float): latitude of P3
        ptlon (float): longitude of P3
        lat1 (float): latitude of P1
        lon1 (float): longitude of P1
        lat2 (float): latitude of P2
        lon2 (float): longitude of P2
        debug (bool): whether to print debug statements

    Returns:
        float: the distance in meters from the point to the line
    '''

    # Examples: (ald = along track distance)
    #                                           |
    #   P1 —————————————————————————————— P2    |   P1 —————————————————————————————— P2 - - - -︱ ⎤
    #               ︱ ⎤                        ︱                                               ︱ ⎥                                 
    #               ︱ ⎥ cross track distance   ︱                                               ︱ ⎥ cross track distance
    #               ︱ ⎦                        ︱                                               ︱ ⎦
    #               P3                          |                                               P3
    # ――――――――――――――――――――――――――――――――――――――――――✛―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
    #                                           |                     ald P1          
    #     ald P1            ald P2              |    ⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ald P2‾‾‾‾⌉
    #   ⌈‾‾‾‾‾‾‾‾‾‾‾⌉⌈‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾⌉     |                                      ⌈‾‾‾‾‾‾‾‾‾‾‾⌉
    #   P1 —————————————————————————————— P2    |   P1 —————————————————————————————— P2 - - - - -︱ 
    #               ︱                          ︱                                                 ︱                                  
    #               ︱                          ︱                                                 ︱
    #               ︱                          ︱                                                 ︱ 
    #               P3                          |                                                 P3
    #
    EARTH_R = 6371e3;  # meters
    converter = Geodesic.WGS84
    
    θ13 = radians(converter.Inverse(lat1=lat1, lon1=lon1, lat2=ptlat, lon2=ptlon)['azi1'])  # bearing from P1 to P3
    θ12 = radians(converter.Inverse(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)['azi1'])  # bearing from P1 to P2
    δ13 = converter.Inverse(lat1=lat1, lon1=lon1, lat2=ptlat, lon2=ptlon)['s12']  # distance from P1 to P3

    cross_track_distance = asin(sin(δ13 / EARTH_R) * sin(θ13 - θ12)) * EARTH_R
    along_track_distance_P1 = acos(cos(δ13 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # along track distance wrt P1
    
    δ23 = converter.Inverse(lat1=lat2, lon1=lon2, lat2=ptlat, lon2=ptlon)['s12']  # distance from P2 to P3
    along_track_distance_P2 = acos(cos(δ23 / EARTH_R) / cos(cross_track_distance / EARTH_R)) * EARTH_R  # along track distance wrt P2

    δ12 = converter.Inverse(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)['s12']  # distance from P1 to P2 (length of the line)

    # If either of the along track distances is longer than the distance of the line, the point must be off to one side of the line
    if abs(along_track_distance_P1) > abs(δ12) or abs(along_track_distance_P2) > abs(δ12):
        if debug: print('Point not on line, since along track distance {:.2f} or {:.2f} is greater than the block distance {:.2f}'.format(
            along_track_distance_P1, along_track_distance_P2, δ12))
        return δ13 if abs(δ13) < abs(δ23) else δ23
    else:
        if debug: print('Point is on line, since along track distance {:.2f} and {:.2f} are less than block distance {:.2f}'.format(
            along_track_distance_P1, along_track_distance_P2, δ12))
        return cross_track_distance