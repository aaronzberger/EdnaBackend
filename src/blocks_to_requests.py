import json
import os
import pickle

from utils import BASE_DIR

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords_table = pickle.load(open(os.path.join(BASE_DIR, 'input/hash_nodes.pkl'), 'rb'))

output = {}

blocks = json.load(open(os.path.join(BASE_DIR, 'blocks.json')))

failed = total = 0
for block_id in blocks.keys():
    total += 1
    start_node = node_coords_table.get(int(block_id[:block_id.find(':')]))
    if start_node is None:
        failed += 1
        print('fail at start', blocks[block_id][0][0], block_id[:block_id.find(':')])
        continue
    start_node_coords = [start_node['lat'], start_node['lon']]

    end_node = node_coords_table.get(int(block_id[block_id.find(':') + 1:block_id.find(':', block_id.find(':') + 1)]))
    if end_node is None:
        print('fail at end', blocks[block_id][0][0], block_id[block_id.find(':') + 1:block_id.find(':', block_id.find(':') + 1)])
        failed += 1
        continue
    end_node_coords = [end_node['lat'], end_node['lon']]

    output[block_id] = {
        'start': start_node_coords,
        'end': end_node_coords,
        'num_houses': len(blocks[block_id][0]),
        'nodes': blocks[block_id][1]
    }

filename = os.path.join(BASE_DIR, 'requests.json')
json.dump(output, open(filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
print('Wrote final json to {}, skipped {} total blocks out of {}'.format(filename, failed, total))
