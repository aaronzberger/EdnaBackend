'''
Group together node and block IDs and coordinates into qgis/blocks.csv
'''

import csv
import json
import os

from config import BASE_DIR, block_output_file, node_coords_file, node_t

file = json.load(open(block_output_file))

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords: dict[str, node_t] = json.load(open(node_coords_file))

# Create the qgis directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'qgis'), exist_ok=True)

with open(os.path.join(BASE_DIR, 'qgis', 'blocks.csv'), 'w') as write:
    writer = csv.DictWriter(write, fieldnames=['ID', 'BlockID', 'Latitude', 'Longitude'])
    writer.writeheader()
    for start_node_list in file:
        for block in file[start_node_list]:
            if block[2] is not None:
                for node in block[2]['nodes']:
                    try:
                        coords = node_coords[node]
                        writer.writerow({
                            'ID': node,
                            'BlockID': '{}{}{}'.format(start_node_list, block[0], block[1]),
                            'Latitude': coords['lat'],
                            'Longitude': coords['lon']
                        })
                    except TypeError:
                        pass
