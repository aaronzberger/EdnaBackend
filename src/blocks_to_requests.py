import json
import os

from config import BASE_DIR, RequestDict, blocks_file_t, node_t, requests_file_t

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords: dict[str, node_t] = json.load(open(os.path.join(BASE_DIR, 'store', 'node_coords.json'), 'r'))

output: requests_file_t = {}

blocks: blocks_file_t = json.load(open(os.path.join(BASE_DIR, 'blocks.json')))

num_failed = total = 0
for block_id in blocks.keys():
    total += 1
    try:
        start_node = node_coords[block_id[:block_id.find(':')]]
    except KeyError:
        num_failed += 1
        continue

    try:
        end_node = node_coords[block_id[block_id.find(':') + 1:block_id.find(':', block_id.find(':') + 1)]]
    except KeyError:
        num_failed += 1
        continue

    output[block_id] = RequestDict(
        start=start_node,
        end=end_node,
        num_houses=len(blocks[block_id]['addresses']),
        nodes=blocks[block_id]['nodes']
    )

filename = os.path.join(BASE_DIR, 'requests.json')
json.dump(output, open(filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
print('Wrote final json to {}, skipped {} total blocks out of {}'.format(filename, num_failed, total))
