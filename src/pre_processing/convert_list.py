# TODO: Fix with updated type hints and modify to new format

import csv
import json
import os
import sys

from src.config import (BASE_DIR, Node, block_output_file, blocks_file,
                        node_coords_file)
from src.gps_utils import Point, along_track_distance
from termcolor import colored
from tqdm import tqdm

print('Loading associations')
house_associations: dict[str, dict[str, list[str | Node]]] = json.load(open(blocks_file))

# This file contains the addresses of the requested Squirrel Hill houses
print('Loading requested houses...')
requested_houses_file = open(os.path.join(BASE_DIR, 'example_lists/requested.csv'), 'r')
num_requested_houses = -1
for line in requested_houses_file:
    num_requested_houses += 1
requested_houses_file.seek(0)
requested_reader = csv.DictReader(requested_houses_file)

# Load the block_output file, containing the blocks returned from the OSM query
print('Loading node and way coordinations query...')
all_way_nodes = json.load(open(block_output_file))

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords: dict[str, Node] = json.load(open(node_coords_file, 'r'))

walk_list = {
    'addresses': [],
    'route': []
}

block_order = []

# First and last houses on each block
block_ends = {}
last_address = None
with tqdm(total=num_requested_houses, desc='Matching', unit='houses', colour='green') as progress:
    for house in requested_reader:
        progress.update()
        formatted_address = str(house['House']) + ' ' + house['Street'].upper()

        # Eliminate duplicates
        if formatted_address == last_address:
            continue
        last_address = formatted_address

        # Find this house (and it's coordinates and segments)
        found = False

        for block in house_associations:
            if formatted_address in block['addresses']:
                found = True
                walk_list['addresses'].append([formatted_address, ])

        for item in house_associations:
            if formatted_address == item['Address']:
                found = True
                walk_list["addresses"].append([formatted_address, item['Lat'], item['Lon']])

                # Create or update this house's block
                block_id = item['BlockID'][:-1]
                if len(block_order) == 0 or block_id != block_order[-1]:
                    block_order.append(block_id)
                    block_ends[block_id] = [
                        (float(item['Lat']), float(item['Lon']),
                            item['Segment Node 1'], item['Segment Node 2'], formatted_address), None]
                else:
                    block_ends[block_id][1] = (
                        float(item['Lat']), float(item['Lon']),
                        item['Segment Node 1'], item['Segment Node 2'], formatted_address)
                break
        if not found:
            print(colored('Warning: Could not find {} in associations'.format(formatted_address), color='yellow'))

# Calculate direction and add route nodes
for block in block_order:
    # Find the way nodes
    way_nodes = []
    try:
        for end_node in all_way_nodes[str(block[:9])]:
            if str(end_node[0]) == block[9:18]:
                way_nodes = end_node[2]['nodes']
    except ValueError:
        for end_node in all_way_nodes[str(block[9:18])]:
            if end_node[0] == block[:9]:
                way_nodes = end_node[2]['nodes']
    except Exception:
        print(colored('FAIL: Failed to find way nodes for ID {}'.format(block), color='red'))
        sys.exit()

    # Convert way nodes to str (this will be fixed soon in preprocess_data.py)
    way_nodes = [str(i) for i in way_nodes]

    print('this block has start {} and end {} and way nodes'.format(
        block_ends[block][0][4], block_ends[block][1][4]), way_nodes)

    # Get the index of each block end's segment within the entire block
    try:
        h1_hops_to_start = min(way_nodes.index(block_ends[block][0][2]), way_nodes.index(block_ends[block][0][3]))
        h2_hops_to_start = min(way_nodes.index(block_ends[block][1][2]), way_nodes.index(block_ends[block][1][3]))
    except ValueError:
        print(colored('FAIL: For some reason, segment node wasn\'t in block', color='red'))
        sys.exit()

    if h1_hops_to_start < h2_hops_to_start:
        # First house comes before second house, so this is the correct ordering
        walk_list['route'] += way_nodes
    elif h1_hops_to_start > h2_hops_to_start:
        walk_list['route'] += way_nodes[::-1]
    else:
        # Executed if these houses are on the same segment
        start_house_coords = block_ends[block][0][0:2]
        end_house_coords = block_ends[block][1][0:2]

        # Switch the segment nodes if needed so they are oriented the same way as the way nodes
        if way_nodes.index(block_ends[block][0][2]) > way_nodes.index(block_ends[block][0][3]):
            temp = block_ends[block][0][2]
            block_ends[block][0][2] = block_ends[block][0][3]
            block_ends[block][0][3] = temp

        segment_start_coords = node_coords[block_ends[block][0][2]]
        segment_start_pt = Point(lat=segment_start_coords['lat'], lon=segment_start_coords['lon'])
        segment_end_coords = node_coords[block_ends[block][0][3]]
        segment_end_pt = Point(lat=segment_end_coords['lat'], lon=segment_end_coords['lon'])

        # ALD to beginning of segment relative to start
        h1_to_p1, _ = along_track_distance(
            p1=segment_start_pt,
            p2=Point(*start_house_coords),
            p3=Point(*end_house_coords))

        h2_to_p1, _ = along_track_distance(
            p1=segment_end_pt,
            p2=Point(*start_house_coords),
            p3=Point(*end_house_coords))

        if h1_to_p1 < h2_to_p1:
            walk_list['route'] += way_nodes
        else:
            walk_list['route'] += way_nodes[::-1]

# Remove duplicate intersection nodes from the route
without_duplicates = []
last_item = None
for item in walk_list['route']:
    if item != last_item:
        without_duplicates.append(item)
    last_item = item
walk_list['route'] = without_duplicates

# Write the final walk list
filename = os.path.join(BASE_DIR, 'example_lists', 'list.json')
print('Writing walk list to {}'.format(filename))
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(walk_list, f, ensure_ascii=False, indent=4)
