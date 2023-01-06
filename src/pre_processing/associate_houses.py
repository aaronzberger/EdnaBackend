'''
Associate houses with blocks. Generate blocks.json and houses.json
'''

# TODO: See 5889 Aylesboro. If ALD is within bounds of block, prefer that block instead of the one that minimizes distance

import csv
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

from src.config import (HouseInfo, Point, Block,
                        address_pts_file, block_output_file, blocks_file,
                        blocks_file_t, houses_file, houses_file_t,
                        node_coords_file, pt_id, SmallPoint)
from src.gps_utils import (along_track_distance, cross_track_distance,
                           great_circle_distance, bearing)
from tqdm import tqdm

SEPARATE_SIDES = False
MAX_DISTANCE = 300  # meters from house to segment
DEBUG = False


# Load the hash table containing node coordinates hashed by ID
print('Loading node coordinates table...')
node_coords: dict[str, SmallPoint] = json.load(open(node_coords_file))

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
blocks: dict[str, Any] = json.load(open(block_output_file))

# Map segment IDs to a dict containting the addresses and node IDs
segments_by_id: blocks_file_t = {}
houses_to_id: houses_file_t = {}


@dataclass
class Segment():
    '''Define a segment between two nodes on a block relative to a house'''
    start_node_id: str
    sub_node_1: SmallPoint
    sub_node_2: SmallPoint
    end_node_id: str
    distance: float
    id: int
    side: bool
    all_nodes: list[str]
    type: Literal['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']


with tqdm(total=num_rows, desc='Matching', unit='rows', colour='green') as progress:
    for item in house_points_reader:
        progress.update()  # Update the progress bar
        if item['municipality'].strip().upper() != 'PITTSBURGH' or \
                int(item['zip_code']) != 15217:
            continue
        house_pt = Point(lat=float(item['latitude']), lon=float(item['longitude']), type='house', id=None)
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
                            Point(lat=node_1['lat'], lon=node_1['lon'], type='node', id=None),
                            Point(lat=node_2['lat'], lon=node_2['lon'], type='node', id=None))

                        if DEBUG:
                            print('nodes {} and {}, distance {:.2f}.'.format(
                                  pt_id(node_1), pt_id(node_2), house_to_segment))

                        if best_segment is None or \
                                (best_segment is not None and abs(house_to_segment) < abs(best_segment.distance)):
                            if DEBUG:
                                print('Replacing best segment with distance {:.2f}'.format(
                                    -1 if best_segment is None else best_segment.distance))

                            used_info = possible_street_names.index(street_name)

                            best_segment = Segment(
                                start_node_id=start_node,
                                sub_node_1=deepcopy(node_1),
                                sub_node_2=deepcopy(node_2),
                                end_node_id=str(block[0]),
                                distance=abs(house_to_segment),
                                id=block[1],
                                side=True if house_to_segment > 0 else False,
                                all_nodes=[str(id) for id in block[2]['nodes']],
                                type=block[2]['ways'][used_info][1]['highway'])

        if best_segment is not None and best_segment.distance <= MAX_DISTANCE:
            # Create the segment ID from the two nodes, ID, and direction if necessary
            segment_id = str(best_segment.start_node_id) + ':' + str(best_segment.end_node_id) + \
                         ':' + str(best_segment.id)
            if SEPARATE_SIDES:
                segment_id += ':' + str(best_segment.side)

            # If this segment has not been inserted yet, generate an entry
            if segment_id not in segments_by_id:
                # Create the list of sub points in this segment
                all_nodes: list[SmallPoint] = []
                for id in best_segment.all_nodes:
                    try:
                        coords = node_coords[id]
                    except KeyError:
                        continue
                    all_nodes.append(coords)

                if best_segment.all_nodes.index(best_segment.start_node_id) != 0:
                    all_nodes = list(reversed(all_nodes))

                # Calculate the bearings from each side of the block
                b_start = bearing(Point(lat=all_nodes[0]['lat'], lon=all_nodes[0]['lon'], type='node', id=None),
                                  Point(lat=all_nodes[1]['lat'], lon=all_nodes[1]['lon'], type='node', id=None))

                b_end = bearing(Point(lat=all_nodes[-1]['lat'], lon=all_nodes[-1]['lon'], type='node', id=None),
                                Point(lat=all_nodes[-2]['lat'], lon=all_nodes[-2]['lon'], type='node', id=None))

                # Place this segment in the table
                segments_by_id[segment_id] = Block(
                    addresses={},
                    nodes=all_nodes,
                    bearings=(b_start, b_end),
                    type=best_segment.type
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
                    Point(lat=first['lat'], lon=first['lon'], type='other', id=None),
                    Point(lat=second['lat'], lon=second['lon'], type='other', id=None))

            # Split this sub-segment's length between the two distances, based on this house's location
            sub_start = all_points[min(sub_nodes)]
            sub_end = all_points[max(sub_nodes)]
            distances = along_track_distance(
                p1=house_pt,
                p2=Point(lat=sub_start['lat'], lon=sub_start['lon'], type='other', id=None),
                p3=Point(lat=sub_end['lat'], lon=sub_end['lon'], type='other', id=None))
            distance_to_start += distances[0]
            distance_to_end += distances[1]

            # Lastly, calculate the distance from the end of this house's sub-segment to the end of the block
            for first, second in itertools.pairwise(all_points[min(sub_nodes) + 1:]):
                distance_to_end += great_circle_distance(
                    Point(lat=first['lat'], lon=first['lon'], type='other', id=None),
                    Point(lat=second['lat'], lon=second['lon'], type='other', id=None))

            output_house = HouseInfo(
                lat=house_pt['lat'], lon=house_pt['lon'],
                distance_to_start=round(distance_to_start),
                distance_to_end=round(distance_to_end),
                side=best_segment.side,
                distance_to_road=round(best_segment.distance),
                subsegment=(min(sub_nodes), max(sub_nodes))
            )

            # Add the house to the list of addresses in the output
            segments_by_id[segment_id]['addresses'][item['full_address']] = output_house

            # Add this association to the houses file
            houses_to_id[item['full_address']] = segment_id

        if DEBUG:
            print('best block for {}, {} is {}.'.format(house_pt['lat'], house_pt['lon'], best_segment))

print('Writing...')
json.dump(segments_by_id, open(blocks_file, 'w'), indent=4)
json.dump(houses_to_id, open((houses_file), 'w'), indent=4)
