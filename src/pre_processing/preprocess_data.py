"""
Process the data from OSM. Take in overpass.json and write block_output.json
"""

import json
import uuid
from src.config import block_output_file, overpass_file, UUID_NAMESPACE

def associate_ways_and_nodes(json_file):
    response: Model = json.load(open(overpass_file))
    node_dict = {}
    ways = []
    for item in response['elements']:
        if item['type'] == 'node':
            node_dict[item['id']] = {'lat': item['lat'], 'lon': item['lon']}
        elif item['type'] == 'way':
            ways.append(item)
    for way in ways:
        way['node_coords'] = []
        for node in way['nodes']:
            way['node_coords'].append(node_dict[node])
    return ways

def find_intersections(ways):
    intersections = set()
    way_dict = {}
    for way in ways:
        for node_id in way['nodes']:
            if node_id in way_dict.keys():
                way_dict[node_id] += 1
            else:
                way_dict[node_id] = 1
    for id in way_dict.keys():
        if way_dict[id] > 1:
            intersections.add(id)
    return intersections

def make_blocks(ways, intersections):
    blocks = []
    for way in ways:
        current_block_points = [way['node_coords'][0]]
        for i in range(1, len(way['nodes'])):
            current_block_points.append(way['node_coords'][i])
            if (way['nodes'][i] in intersections) or (i == len(way['nodes']) - 1):
                uuid_input = str(current_block_points)
                block_uuid = str(uuid.uuid5(UUID_NAMESPACE, uuid_input))
                blocks.append({'id': block_uuid, 'nodes': current_block_points, 'type': way['tags']['highway']})
                current_block_points = [way['node_coords'][i]]
    return blocks
    
if __name__ == "__main__":
    ways = associate_ways_and_nodes(overpass_file)
    intersections = find_intersections(ways)
    blocks = make_blocks(ways, intersections)
    json.dump(blocks, open(block_output_file, "w", encoding="utf-8"), indent=4)