# -*- encoding: utf-8 -*-

import math
import csv
import json
from tqdm import tqdm
import pickle
import sys

EARTH_R = 6371e3;  # meters

def distance(lat1, lon1, lat2, lon2):
    φ1 = lat1 * math.pi/180 # φ, λ in radians
    φ2 = lat2 * math.pi/180
    Δφ = (lat2-lat1) * math.pi/180
    Δλ = (lon2-lon1) * math.pi/180

    a = math.sin(Δφ/2) * math.sin(Δφ/2) + \
            math.cos(φ1) * math.cos(φ2) * \
            math.sin(Δλ/2) * math.sin(Δλ/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return EARTH_R * c; # in metres

def bearing(lat1, lon1, lat2, lon2):
    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - \
          math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
    θ = math.atan2(y, x)
    return (θ + math.pi) % math.pi
    # brng = (θ*180/math.pi + 360) % 360; # in degrees

def cross_track_distance(ptlat, ptlon, lat1, lon1, lat2, lon2):
    θ13 = bearing(lat1, lon1, ptlat, ptlon)
    θ12 = bearing(lat1, lon1, lat2, lon2)
    δ13 = distance(lat1, lon1, ptlat, ptlon)
    δ13 = δ13 / EARTH_R
    return math.asin(math.sin(δ13)*math.sin(θ13-θ12)) * EARTH_R


al_node1 = (40.4397215, -79.9348697)
al_node2 = (40.4404022, -79.9327476)
house_node = (40.4401195410866,-79.9328209026398)
dun_house = (40.4472292619933,-79.9320683370225)
sec_node = (40.4400874564469,-79.9346059650522)
print(cross_track_distance(*house_node, *al_node2, *al_node1))
print(cross_track_distance(*house_node, *al_node1, *al_node2))
print(cross_track_distance(*dun_house, *al_node2, *al_node1))
print(cross_track_distance(*dun_house, *al_node1, *al_node2))
print(cross_track_distance(*sec_node, *al_node2, *al_node1))
print(cross_track_distance(*sec_node, *al_node1, *al_node2))
# sys.exit()

node_coords_table = pickle.load(open('/home/aaron/walk_list_creator/hash_nodes.pkl', 'rb'))


house_points_file = open('/home/aaron/Downloads/geocoded_addresses.csv', 'r')
num_rows = -1
for line in house_points_file: num_rows += 1
house_points_file.seek(0)
next(house_points_file)
house_points_reader = csv.reader(house_points_file)

blocks = json.load(open('/home/aaron/Downloads/block_output.json', 'r'))

with tqdm(total=num_rows, desc='Matching', unit='rows',
            colour='green') as progress:
    for item in house_points_reader:
        if item[18].strip().upper() != 'PITTSBURGH' or int(item[21]) != 15217:
            continue
        house_lat, house_lon = item[-2], item[-1]
        street_name = item[12].split(' ')[0].upper()
        # node1, node2, id, dist
        best_segment = []
        for start_node in blocks:
            for block in blocks[start_node]:
                if block[2] is None:
                    continue
                possible_street_names = [block[2]['ways'][i][1]['name'].split(' ')[0].upper() for i in range(len(block[2]['ways']))]
                if street_name in possible_street_names:
                    for idx in range(len(block[2]['nodes']) - 1):
                        node_1 = node_coords_table.get(block[2]['nodes'][idx])
                        node_2 = node_coords_table.get(block[2]['nodes'][idx + 1])

                        # if node_1 is None:
                        #     print(block[2]['nodes'][idx])
                        # elif node_2 is None:
                        #     print(block[2]['nodes'][idx + 1])
                        if node_1 is None or node_2 is None:
                            continue

                        house_to_segment = cross_track_distance(float(house_lat), float(house_lon), node_1['lat'], node_1['lon'], node_2['lat'], node_2['lon'])
                        if len(best_segment) == 0 or (len(best_segment) == 4 and abs(house_to_segment) < abs(best_segment[3])):
                            best_segment = [start_node, node_1, node_2, house_to_segment]
        if len(best_segment) != 0:
            print('best block for {} is {}'.format(item, best_segment))
        progress.update()