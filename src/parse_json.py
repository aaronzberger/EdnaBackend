'''
Save a pickle containing a hash table of all nodes hashed by ID
'''

import json
import os
import pickle

from hash_table import Hash_Table
from gps_utils import BASE_DIR


# Returned from OSM query of all nodes and ways in a region
read_file = os.path.join(BASE_DIR, 'input/squirrel_hill.json')
loaded = json.load(open(read_file))

# Save the file as input for the data preparation in associate_houses.py
save_file = os.path.join(BASE_DIR, 'input/hash_nodes.pkl')

hash_table = Hash_Table()

print('Loading items from {} into hash table'.format(read_file))
for i in range(len(loaded['elements'])):
    if loaded['elements'][i]['type'] == 'node':
        hash_table.insert(loaded['elements'][i])

print('Saving hash table to {}'.format(save_file))
with open(save_file, 'wb') as output:
    pickle.dump(hash_table, output)
