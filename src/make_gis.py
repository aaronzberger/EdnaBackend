import csv
import json
import pickle

file = json.load(open('/home/aaron/walk_list_creator/input/block_output.json', 'r'))

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords_table = pickle.load(open('/home/aaron/walk_list_creator/input/hash_nodes.pkl', 'rb'))

with open('/home/aaron/walk_list_creator/gis_blocks.csv', 'w') as write:
    writer = csv.DictWriter(write, fieldnames=['ID', 'BlockID', 'Latitude', 'Longitude'])
    writer.writeheader()
    for start_node_list in file:
        for block in file[start_node_list]:
            if block[2] is not None:
                for node in block[2]['nodes']:
                    try:
                        coords = node_coords_table.get(node)
                        writer.writerow({
                            'ID': node,
                            'BlockID': '{}{}{}'.format(start_node_list, block[0], block[1]),
                            'Latitude': coords['lat'],
                            'Longitude': coords['lon']
                        })
                    except TypeError:
                        pass