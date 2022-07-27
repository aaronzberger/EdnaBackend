import os
from typing import TypedDict

BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'

node_distance_table_file = os.path.join(BASE_DIR, 'store', 'node_distances.json')
segment_distance_matrix_file = os.path.join(BASE_DIR, 'store', 'segment_distance_matrix.json')
node_coords_file = os.path.join(BASE_DIR, 'store', 'node_coords.json')
address_pts_file = os.path.join(BASE_DIR, 'input', 'address_pts.csv')
block_output_file = os.path.join(BASE_DIR, 'input', 'block_output.json')
blocks_file = os.path.join(BASE_DIR, 'blocks.json')
requests_file = os.path.join(BASE_DIR, 'requests.json')
associated_file = os.path.join(BASE_DIR, 'associated.csv')

# Maximum distance between two nodes where they should be stored
MAX_NODE_STORAGE_DISTANCE = 1600
ARBITRARY_LARGE_DISTANCE = 10000
MAX_TIMELINE_MINS = 180
WALKING_M_PER_S = 1
MINS_PER_HOUSE = 2
CLUSTERING_CONNECTED_THRESHOLD = 100  # Meters where blocks are connected
KEEP_APARTMENTS = False
DIFFERENT_SEGMENT_ADDITION = 30
DIFFERENT_SIDE_ADDITION = 10

# JSON type hints
node_t = dict[str, float]
house_t = dict[str, node_t]
node_list_t = list[node_t]


class RequestDict(TypedDict):
    start: node_t
    end: node_t
    num_houses: int
    nodes: node_list_t


class HouseAssociationDict(TypedDict):
    lat: float
    lon: float
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int


class SegmentDict(TypedDict):
    addresses: dict[str, HouseAssociationDict]
    nodes: node_list_t


# JSON store file type hints
requests_file_t = dict[str, RequestDict]
blocks_file_t = dict[str, SegmentDict]
