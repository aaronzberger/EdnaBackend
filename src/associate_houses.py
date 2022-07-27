# -*- encoding: utf-8 -*-

'''
Associate houses with their block segments and save to associated.csv
'''

import csv
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from config import (HouseAssociationDict, SegmentDict, address_pts_file, block_output_file,
                    blocks_file, blocks_file_t, house_t, node_coords_file,
                    node_list_t, node_t)
from gps_utils import Point, along_track_distance, cross_track_distance, great_circle_distance

SEPARATE_SIDES = False
MAX_DISTANCE = 500  # meters from house to segment
DEBUG = False

# Load the hash table containing node coordinates hashed by ID
print('Loading node coordinates table...')
node_coords: dict[str, node_t] = json.load(open(node_coords_file))

# This file contains the coordinates of every building in the county
print('Loading coordinates of houses...')
house_points_file = open(address_pts_file)
num_rows = -1
for line in house_points_file:
    num_rows += 1
house_points_file.seek(0)
house_points_reader = csv.DictReader(house_points_file)

# Load the block_output file, containing the blocks returned from the OSM query
print('Loading node and way coordinations query...')
blocks: dict[str, node_list_t | house_t] = json.load(open(block_output_file))

# Map segment IDs to a dict containting the addresses and node IDs
segments_by_id: blocks_file_t = {}


@dataclass
class Segment():
    '''Define a segment between two nodes on a block relative to a house'''
    start_node_id: str
    sub_node_1: node_t
    sub_node_2: node_t
    end_node_id: str
    distance: float
    id: int
    side: bool
    all_nodes: list[str]


with tqdm(total=num_rows, desc='Matching', unit='rows', colour='green') as progress:
    for item in house_points_reader:
        progress.update()  # Update the progress bar
        if item['municipality'].strip().upper() != 'PITTSBURGH' or \
                int(item['zip_code']) != 15217:
            continue
        house_pt = Point(float(item['latitude']), float(item['longitude']))
        street_name = item['st_name'].split(' ')[0].upper()

        best_segment: Optional[Segment] = None  # Store the running closest segment to the house

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
                        try:
                            # TODO: Do not parse to str, change type in origin file by modifying preprocess_data.py
                            node_1 = node_coords[str(block[2]['nodes'][i])]
                            node_2 = node_coords[str(block[2]['nodes'][i + 1])]
                        except KeyError:
                            continue

                        house_to_segment = cross_track_distance(
                            house_pt,
                            Point(node_1['lat'], node_1['lon']),
                            Point(node_2['lat'], node_2['lon']))

                        if DEBUG:
                            print('nodes {} and {}, distance {:.2f}.'.format(
                                node_1['id'], node_2['id'], house_to_segment))

                        if best_segment is None or \
                                (best_segment is not None and abs(house_to_segment) < abs(best_segment.distance)):
                            if DEBUG:
                                print('Replacing best segment with distance {:.2f}'.format(
                                    -1 if best_segment is None else best_segment.distance))
                            best_segment = Segment(
                                start_node_id=start_node,
                                sub_node_1=deepcopy(node_1),
                                sub_node_2=deepcopy(node_2),
                                end_node_id=str(block[0]),
                                distance=abs(house_to_segment),
                                id=block[1],
                                side=True if house_to_segment > 0 else False,
                                all_nodes=[str(id) for id in block[2]['nodes']])

        if best_segment is not None and best_segment.distance <= MAX_DISTANCE:
            # Create the segment ID from the two nodes, ID, and direction if necessary
            segment_id = str(best_segment.start_node_id) + ':' + str(best_segment.end_node_id) + \
                         ':' + str(best_segment.id)
            if SEPARATE_SIDES:
                segment_id += ':' + str(best_segment.side)

            # If this segment has not been inserted yet, generate an entry
            if segment_id not in segments_by_id:
                # Create the list of sub points in this segment
                all_nodes_coords: list[node_t] = []
                for id in best_segment.all_nodes:
                    try:
                        coords = node_coords[id]
                    except KeyError:
                        continue
                    all_nodes_coords.append(coords)

                if best_segment.all_nodes.index(best_segment.start_node_id) != 0:
                    all_nodes_coords = list(reversed(all_nodes_coords))

                # Place this segment in the table
                segments_by_id[segment_id] = SegmentDict(
                    addresses={},
                    nodes=all_nodes_coords
                )

            all_points = segments_by_id[segment_id]['nodes']
            sub_nodes = [all_points.index(best_segment.sub_node_1),
                         all_points.index(best_segment.sub_node_2)]

            # Calculate distances to the start and end of the block
            distance_to_start: float = 0
            distance_to_end: float = 0

            # Calculate the distance from the start of the block to the beginning of this house's sub-segment
            for first, second in itertools.pairwise(all_points[:min(sub_nodes) + 1]):
                distance_to_start += great_circle_distance(
                    Point(first['lat'], first['lon']), Point(second['lat'], second['lon']))

            # Split this sub-segment's length between the two distances, based on this house's location
            sub_start = all_points[min(sub_nodes)]
            sub_end = all_points[max(sub_nodes)]
            distances = along_track_distance(
                p1=house_pt,
                p2=Point(sub_start['lat'], sub_start['lon']),
                p3=Point(sub_end['lat'], sub_end['lon']))
            distance_to_start += distances[0]
            distance_to_end += distances[1]

            # Lastly, calculate the distance from the end of this house's sub-segment to the end of the block
            for first, second in itertools.pairwise(all_points[min(sub_nodes) + 1:]):
                distance_to_end += great_circle_distance(
                    Point(first['lat'], first['lon']), Point(second['lat'], second['lon']))

            output_house = HouseAssociationDict(
                lat=house_pt.lat, lon=house_pt.lon,
                distance_to_start=round(distance_to_start),
                distance_to_end=round(distance_to_end),
                side=best_segment.side,
                distance_to_road=round(best_segment.distance)
            )

            # Add the house to the list of addresses in the output
            segments_by_id[segment_id]['addresses'][item['full_address']] = output_house

        if DEBUG:
            print('best block for {}, {} is {}.'.format(house_pt.lat, house_pt.lon, best_segment))

print('Writing...')
json.dump(segments_by_id, open(blocks_file, 'w', encoding='utf-8'), indent=4)
