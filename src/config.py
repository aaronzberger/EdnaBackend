from typing import TypedDict


BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'

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
