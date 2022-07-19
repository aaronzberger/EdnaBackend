import os
from typing import TypedDict

BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'

node_distance_table_file = os.path.join(BASE_DIR, 'store', 'node_distances.json')
node_coords_file = os.path.join(BASE_DIR, 'store', 'node_coords.json')
address_pts_file = os.path.join(BASE_DIR, 'input', 'address_pts.csv')
block_output_file = os.path.join(BASE_DIR, 'input', 'block_output.json')
blocks_file = os.path.join(BASE_DIR, 'blocks.json')
requests_file = os.path.join(BASE_DIR, 'requests.json')

# Maximum distance between two nodes where they should be stored
MAX_NODE_STORAGE_DISTANCE = 800
ARBITRARY_LARGE_DISTANCE = 10000
MAX_TIMELINE_MINS = 180
WALKING_M_PER_S = 1

# JSON type hints
node_t = dict[str, float]
house_t = dict[str, node_t]
node_list_t = list[node_t]


class RequestDict(TypedDict):
    start: node_t
    end: node_t
    num_houses: int
    nodes: node_list_t


class SegmentDict(TypedDict):
    addresses: house_t
    nodes: node_list_t


# JSON store file type hints
requests_file_t = dict[str, RequestDict]
blocks_file_t = dict[str, SegmentDict]

MINS_PER_HOUSE = 1.5
