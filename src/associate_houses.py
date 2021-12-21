# -*- encoding: utf-8 -*-

import csv
import json
from tqdm import tqdm
import pickle
from geographiclib.geodesic import Geodesic
from math import radians, sin, cos, asin, acos
from copy import deepcopy

EARTH_R = 6371e3;  # meters
MAX_DISTANCE = 500  # meters

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

# Load the hash table containing node coordinates hashed by ID
node_coords_table = pickle.load(open('/home/aaron/walk_list_creator/input/hash_nodes.pkl', 'rb'))

# This file contains the coordinates of every building in the county
house_points_file = open('/home/aaron/walk_list_creator/input/address_pts.csv', 'r')
num_rows = -1
for line in house_points_file: num_rows += 1
house_points_file.seek(0)
next(house_points_file)
house_points_reader = csv.reader(house_points_file)

# Load the block_output file, which contains the blocks returned from the OSM query
blocks = json.load(open('/home/aaron/walk_list_creator/input/block_output.json', 'r'))
block_associations = []

with tqdm(total=num_rows, desc='Matching', unit='rows', colour='green') as progress:
    for item in house_points_reader:
        progress.update()
        if item[18].strip().upper() != 'PITTSBURGH' or int(item[21]) != 15217:
            continue
        house_lat, house_lon = item[-2], item[-1]
        street_name = item[12].split(' ')[0].upper()
        best_segment = None

        debug = False  # Add a debug condition here

        # Iterate through the blocks looking for possible matches
        for start_node in blocks:
            for block in blocks[start_node]:
                if block[2] is None:
                    continue
                possible_street_names = [block[2]['ways'][i][1]['name'].split(' ')[0].upper() for i in range(len(block[2]['ways']))]
                if street_name in possible_street_names:

                    # Iterate through the block segments within this block (it may curve around)
                    for idx in range(len(block[2]['nodes']) - 1):
                        node_1 = node_coords_table.get(block[2]['nodes'][idx])
                        node_2 = node_coords_table.get(block[2]['nodes'][idx + 1])

                        if node_1 is None or node_2 is None:
                            continue

                        house_to_segment = cross_track_distance(
                            float(house_lat), float(house_lon), node_1['lat'], node_1['lon'], node_2['lat'], node_2['lon'])

                        if debug: print('With Node ID {} and {}, distance is {:.2f}.'.format(node_1['id'], node_2['id'], house_to_segment))

                        if best_segment is None or (best_segment is not None and abs(house_to_segment) < abs(best_segment[3])):
                            if debug: print('Replacing best segment, for which distance was {:.2f}'.format(-1 if best_segment is None else best_segment[3]))
                            #               First node     sub node 1        sub node 2            distance      second node    id
                            best_segment = [start_node, deepcopy(node_1), deepcopy(node_2), abs(house_to_segment), block[0], block[1]]

        if best_segment is not None and best_segment[3] <= MAX_DISTANCE:
            block_associations.append([house_lat, house_lon, str(best_segment[0]) + str(best_segment[4]) + str(best_segment[5])])
        if debug: print('best block for {}, {} is {}.'.format(house_lat, house_lon, best_segment))

print('Writing...')
output_writer = csv.writer(open('/home/aaron/walk_list_creator/associated.csv', 'w'))
output_writer.writerow(['Lat', 'Lon', 'ID'])
output_writer.writerows(block_associations)