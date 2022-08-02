'''
Save a json that maps node IDs to their GPS coordinates
'''

import json
import os

from src.config import BASE_DIR, Node, node_coords_file

# Returned from OSM query of all nodes and ways in a region
read_file = os.path.join(BASE_DIR, 'input', 'squirrel_hill.json')
loaded = json.load(open(read_file))

node_coords: dict[str, Node] = {}

print('Loading items from {} into a hash table'.format(read_file))
for item in loaded['elements']:
    if item['type'] == 'node':
        # This is the only ID casting from int to str. Forward, IDs are str only
        node_coords[str(item['id'])] = {
            'lat': item['lat'],
            'lon': item['lon']
        }

# Save the file as input for the data preparation in associate_houses.py
print('Saving node table to {}'.format(node_coords_file))
json.dump(node_coords, open(node_coords_file, 'w', encoding='utf-8'), indent=4)
