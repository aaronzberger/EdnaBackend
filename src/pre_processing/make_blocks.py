'''
Associate houses with blocks. Take in block_output.json and generate blocks.json and houses.json
'''

# TODO: See Covode Pl and Wendover Pl (both have same-named streets, to which all the houses on pl have been wrongly assigned)

import csv
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Any, Optional

from src.config import (BASE_DIR, HouseInfo, Point, Block,
                        address_pts_file, block_output_file, blocks_file,
                        blocks_file_t, houses_file, houses_file_t,
                        node_coords_file, node_list_t, ALD_BUFFER)
from src.gps_utils import (along_track_distance, cross_track_distance,
                           great_circle_distance)
from tqdm import tqdm

from src.viz_utils import display_blocks

MAX_DISTANCE = 300  # meters from house to segment
DEBUG = False

'-----------------------------------------------------------------------------------------'
'                                     Load files                                          '
'-----------------------------------------------------------------------------------------'
# region Load files
# Load the table containing node coordinates by ID
print('Loading node coordinates table...')
node_coords: dict[str, Point] = json.load(open(node_coords_file))

# Load the file (unorganized) containing house coordinates (and info)
print('Loading coordinates of houses...')
house_points_file = open(address_pts_file)
num_houses = -1
for line in house_points_file:
    num_houses += 1
house_points_file.seek(0)
house_points_reader = csv.DictReader(house_points_file)

# Load the block_output file, containing the blocks returned from the OSM query
print('Loading node and way coordinations query...')
blocks: dict[str, Any] = json.load(open(block_output_file))
# endregion

# Map segment IDs to a dict containting the addresses and node IDs
segments_by_id: blocks_file_t = {}
houses_to_id: houses_file_t = {}


@dataclass
class Segment():
    '''Define an alternate Block with more geographic information'''
    sub_node_1: Point
    sub_node_2: Point
    ctd: float  # cross track distance (to road)
    ald_offset: float  # along track distance offset (to segment bounds)
    side: bool


def distance_along_path(path: node_list_t) -> float:
    '''
    Find the distance through a list of Points

    Parameters:
        path (node_list_t): the navigation path to follow

    Returns:
        float: the distance through the path
    '''
    distance = 0
    for first, second in itertools.pairwise(path):
        distance += great_circle_distance(
            Point(lat=first['lat'], lon=first['lon']),  # type: ignore
            Point(lat=second['lat'], lon=second['lon']))  # type: ignore
    return distance


# We will need to temporarily store the street names for each block as well
block_to_street_names = {}


# First, create all the blocks
with tqdm(total=len(blocks), desc='Reading blocks', unit='rows', colour='green') as progress:
    for start_node in blocks:
        for block in blocks[start_node]:
            if block[2] is None:
                continue
            segment_id = str(start_node) + ':' + str(block[0]) + ':' + str(block[1])

            # Create the list of sub points in this segment
            all_node_ids = [str(i) for i in block[2]['nodes']]
            all_nodes: list[Point] = []
            for id in all_node_ids:
                try:
                    coords = node_coords[id]
                except KeyError:
                    continue
                all_nodes.append(coords)

            segments_by_id[segment_id] = Block(
                addresses={},
                nodes=all_nodes,
                # NOTE: For now, assume there's one way per segment
                type=block[2]['ways'][0][1]['highway']
            )

            # Add the street name to the block
            block_to_street_names[segment_id] = list(set([
                    block[2]['ways'][i][1]['name'].split(' ')[0].upper()
                    for i in range(len(block[2]['ways']))]))
        progress.update(1)


# Next, add the houses to the blocks
with tqdm(total=num_houses, desc='Associating houses', unit='rows', colour='green') as progress:
    for item in house_points_reader:
        progress.update(1)
        if item['municipality'].strip().upper() != 'PITTSBURGH' or \
                int(item['zip_code']) != 15217:
            continue
        house_pt = Point(lat=float(item['latitude']), lon=float(item['longitude']), type='house')  # type: ignore
        street_name = item['st_name'].split(' ')[0].upper()

        best_segment: Optional[Segment] = None  # Store the running closest segment to the house
        best_segment_id: Optional[str] = None

        # Find the closest segment to the house
        for segment_id in segments_by_id:
            if street_name in block_to_street_names[segment_id]:
                segment = segments_by_id[segment_id]

                # Iterate through each block segment (block may curve)
                for i, j in itertools.pairwise(range(len(segment['nodes']))):
                    node_1 = Point(lat=segment['nodes'][i]['lat'], lon=segment['nodes'][i]['lon'])
                    node_2 = Point(lat=segment['nodes'][j]['lat'], lon=segment['nodes'][j]['lon'])

                    ctd = cross_track_distance(house_pt, node_1, node_2)
                    alds = along_track_distance(house_pt, node_1, node_2)

                    # The distance from the point this house projects onto the line created by the segment to the segment (0 if within bounds)
                    house_offset = max(0, max(alds) - great_circle_distance(node_1, node_2) - ALD_BUFFER)

                    # If this segment is better than the best segment, insert it
                    if best_segment is None or \
                            house_offset < best_segment.ald_offset or \
                            (house_offset == best_segment.ald_offset and abs(ctd) < best_segment.ctd):
                        if DEBUG:
                            print('Replacing best segment with distance {:.2f}'.format(
                                -1 if best_segment is None else best_segment.ctd))

                        best_segment = Segment(
                            sub_node_1=deepcopy(node_1),
                            sub_node_2=deepcopy(node_2),
                            ctd=abs(ctd),
                            ald_offset=house_offset,
                            side=True if ctd > 0 else False)

                        best_segment_id = segment_id

        if best_segment is not None and best_segment.ctd <= MAX_DISTANCE:
            # region Create house to insert into table
            all_points = segments_by_id[best_segment_id]['nodes']
            sub_nodes = [all_points.index(best_segment.sub_node_1),
                         all_points.index(best_segment.sub_node_2)]

            # Calculate distances to the start and end of the block
            distance_to_start: float = 0
            distance_to_end: float = 0

            # Calculate the distance from the start of the block to the beginning of this house's sub-segment
            distance_to_start += distance_along_path(all_points[:min(sub_nodes) + 1])

            # Split this sub-segment's length between the two distances, based on this house's location
            sub_start = all_points[min(sub_nodes)]
            sub_end = all_points[max(sub_nodes)]
            distances = along_track_distance(
                p1=house_pt,
                p2=Point(lat=sub_start['lat'], lon=sub_start['lon']),  # type: ignore
                p3=Point(lat=sub_end['lat'], lon=sub_end['lon']))  # type: ignore
            distance_to_start += distances[0]
            distance_to_end += distances[1]

            # Lastly, calculate the distance from the end of this house's sub-segment to the end of the block
            distance_to_end += distance_along_path(all_points[min(sub_nodes) + 1:])

            output_house = HouseInfo(
                lat=house_pt['lat'], lon=house_pt['lon'],
                distance_to_start=round(distance_to_start),
                distance_to_end=round(distance_to_end),
                side=best_segment.side,
                distance_to_road=round(best_segment.ctd),
                subsegment=(min(sub_nodes), max(sub_nodes))
            )
            # endregion

            # Add the house to the segments output
            segments_by_id[best_segment_id]['addresses'][item['full_address']] = output_house

            # Add this association to the houses file
            houses_to_id[item['full_address']] = best_segment_id


# Write the output to files
print('Writing...')
json.dump(segments_by_id, open(blocks_file, 'w'), indent=4)
json.dump(houses_to_id, open((houses_file), 'w'), indent=4)

display_blocks(segments_by_id).save(os.path.join(BASE_DIR, 'viz', 'segments.html'))
