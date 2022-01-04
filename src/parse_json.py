'''
Save a pickle containing a table of all nodes hashed by ID, read from the OSM query
'''

import json
import pickle

from hash_table import Hash_Table


# Returned from OSM query of all nodes and ways in a region
read_file = '/home/aaron/walk_list_creator/squirrelhill.json'
loaded = json.load(open(read_file))

# Save the file as input for the data preparation in associate_houses.py
save_file = '/home/aaron/walk_list_creator/input/hash_nodes.pkl'

hash_table = Hash_Table()

print('Loading items from {} into hash table'.format(read_file))
for i in range(len(loaded['elements'])):
    if loaded['elements'][i]['type'] == 'node':
        hash_table.insert(loaded['elements'][i])

print('Saving hash table to {}'.format(save_file))
with open(save_file, 'wb') as output:
    pickle.dump(hash_table, output)