# -*- encoding: utf-8 -*-

'''
Associate houses with their block segments and save to associated.csv
'''

from copy import deepcopy
import csv
import json
import os
import pickle
from tqdm import tqdm
from typing import NamedTuple
from utils import cross_track_distance

from utils import BASE_DIR


MAX_DISTANCE = 500  # meters from hosue to segment

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords_table = pickle.load(open(os.path.join(BASE_DIR, 'input/hash_nodes.pkl'), 'rb'))

# This file contains the coordinates of every building in the county
print('Loading coordinates of houses...')
house_points_file = open(os.path.join(BASE_DIR, 'input/address_pts.csv'), 'r')
num_rows = -1
for line in house_points_file: num_rows += 1
house_points_file.seek(0)
house_points_reader = csv.DictReader(house_points_file)

# Load the block_output file, containing the blocks returned from the OSM query
print('Loading node and way coordinations query...')
blocks = json.load(open(os.path.join(BASE_DIR, 'input/block_output.json'), 'r'))
block_associations = []


class Segment(NamedTuple):
    '''Define a segment between two nodes on a block relative to a house'''
    start_node_id: int
    sub_node_1: dict
    sub_node_2: dict
    end_node_id: int
    distance: float
    id: int


with tqdm(total=num_rows, desc='Matching', unit='rows', colour='green') as progress:
    for item in house_points_reader:
        progress.update()  # Update the progress bar
        if item['municipality'].strip().upper() != 'PITTSBURGH' or \
                int(item['zip_code']) != 15217:
            continue
        house_lat, house_lon = item['latitude'], item['longitude']
        street_name = item['st_name'].split(' ')[0].upper()

        best_segment = None  # Store the running closest segment to the house

        debug = False

        # Iterate through the blocks looking for possible matches
        for start_node in blocks:
            for block in blocks[start_node]:
                if block[2] is None:
                    continue  # This is a duplicate, as described in the README
                possible_street_names = [
                    block[2]['ways'][i][1]['name'].split(' ')[0].upper()
                    for i in range(len(block[2]['ways']))]

                if street_name in possible_street_names:
                    # Iterate through each block segment (block may curve)
                    for i in range(len(block[2]['nodes']) - 1):
                        node_1 = node_coords_table.get(block[2]['nodes'][i])
                        node_2 = node_coords_table.get(block[2]['nodes'][i + 1])

                        if node_1 is None or node_2 is None:
                            continue

                        house_to_segment = cross_track_distance(
                            float(house_lat), float(house_lon),
                            node_1['lat'], node_1['lon'],
                            node_2['lat'], node_2['lon'])

                        if debug:
                            print('nodes {} and {}, distance {:.2f}.'.format(
                                node_1['id'], node_2['id'], house_to_segment))

                        if best_segment is None or \
                                (best_segment is not None and abs(house_to_segment) < abs(best_segment.distance)):
                            if debug: print('Replacing best segment with distance {:.2f}'.format(-1 if best_segment is None else best_segment.distance))
                            best_segment = Segment(
                                start_node_id=int(start_node),
                                sub_node_1=deepcopy(node_1),
                                sub_node_2=deepcopy(node_2),
                                end_node_id=block[0],
                                distance=abs(house_to_segment),
                                id=block[1])

        if best_segment is not None and best_segment.distance <= MAX_DISTANCE:
            block_associations.append(
                [house_lat, house_lon,
                 str(best_segment.start_node_id) + str(best_segment.end_node_id) + str(best_segment.id)])
        if debug: print('best block for {}, {} is {}.'.format(house_lat, house_lon, best_segment))

print('Writing...')
output_writer = csv.writer(open(os.path.join(BASE_DIR, 'associated.csv'), 'w'))
output_writer.writerow(['Lat', 'Lon', 'BlockID'])
output_writer.writerows(block_associations)
