'''
Save a json that maps node IDs to their GPS coordinates
'''

import json
import os

from config import BASE_DIR, node_t

# Returned from OSM query of all nodes and ways in a region
read_file = os.path.join(BASE_DIR, 'input', 'squirrel_hill.json')
loaded = json.load(open(read_file))

node_coords: dict[str, node_t] = {}

print('Loading items from {} into hash table'.format(read_file))
for item in loaded['elements']:
    if item['type'] == 'node':
        # This is the only ID conversion from int to str. Forward, IDs are str only
        node_coords[str(item['id'])] = {
            'lat': item['lat'],
            'lon': item['lon']
        }

# Save the file as input for the data preparation in associate_houses.py
save_file = os.path.join(BASE_DIR, 'store', 'node_coords.json')
print('Saving node table to {}'.format(save_file))
with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(node_coords, f, indent=4)
