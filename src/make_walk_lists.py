# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy
from sys import argv

from sklearn_extra.cluster import KMedoids
from termcolor import colored

from gps_utils import SubBlock
from src.config import (BASE_DIR, CLUSTERING_CONNECTED_THRESHOLD,
                        KEEP_APARTMENTS, blocks_file, blocks_file_t,
                        houses_file, houses_file_t, Block, Point, pt_id)
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.distances.blocks import BlockDistances
from src.optimize import Optimizer
# from src.post_process import PostProcess
from src.viz_utils import (display_clustered_blocks, display_blocks,
                           display_walk_lists)
# from src.walkability_scorer import score

all_blocks: blocks_file_t = json.load(open(blocks_file))

'-----------------------------------------------------------------------------------------'
'                                Handle universe file                                     '
' The universe file is the list of voters to target for these routes. It should be a CSV  '
' file in the format [Voter ID, Address, City, Zip]                                       '
'-----------------------------------------------------------------------------------------'
# region: Handle universe file
if len(argv) == 2:
    # Ensure the provided file exists
    if not os.path.exists(argv[1]):
        raise FileExistsError('Usage: make_walk_lists.py [UNIVERSE FILE]')

    reader = csv.DictReader(open(argv[1]))
    houses_to_id: houses_file_t = json.load(open(houses_file))
    requested_blocks: blocks_file_t = {}
    total_houses = failed_houses = 0

    # Process each requested house
    for house in reader:
        formatted_address = house['Address'].upper()
        total_houses += 1
        if formatted_address not in houses_to_id:
            failed_houses += 1
            continue
        block_id = houses_to_id[formatted_address]
        house_info = deepcopy(all_blocks[block_id]['addresses'][formatted_address])

        if block_id in requested_blocks:
            requested_blocks[block_id]['addresses'][formatted_address] = house_info
        else:
            requested_blocks[block_id] = deepcopy(all_blocks[block_id])
            requested_blocks[block_id]['addresses'] = {formatted_address: house_info}
    print('Failed on {} of {} houses'.format(failed_houses, total_houses))
else:
    requested_blocks: blocks_file_t = json.load(open(blocks_file))

# After this point, the original blocks variable should never be used, so delete it
all_blocks.clear()
del all_blocks
# endregion

display_blocks(requested_blocks).save(os.path.join(BASE_DIR, 'viz', 'segments.html'))

# Generate node distance matrix
NodeDistances(requested_blocks)

# Generate block distance matrix
BlockDistances(requested_blocks)

# Initialize calculator for mixed distances
MixDistances()

'-----------------------------------------------------------------------------------------'
'                                      Cluster                                            '
' Using K-medoids, we cluster the blocks and designate a center node for each cluster     '
' Clusters are used for partitioning the space into more reasonable and optimizable areas '
'-----------------------------------------------------------------------------------------'
# region: Cluster
# Cluster blocks using kmedoids
distance_matrix = BlockDistances.get_distance_matrix()
km = KMedoids(metric='precomputed', max_iter=100).fit(distance_matrix)
labels: list[int] = km.labels_

# Expand labels into a list of block groups
clustered_blocks: list[blocks_file_t] = [
    {b_id: b_info for i, (b_id, b_info) in enumerate(requested_blocks.items()) if labels[i] == k}  # Blocks in cluster k
    for k in range(max(labels))]


def cluster_to_houses(cluster: blocks_file_t) -> list[Point]:
    '''Convert a list of blocks to its corresponding list of houses'''
    points: list[Point] = []

    for block in cluster.values():
        for address, house_data in block['addresses'].items():
            # TODO: Move this conditional outward so we can get rid of this whole method
            if not KEEP_APARTMENTS and ' APT ' in address:
                continue

            # TODO: Maybe do this earlier to get rid of this method
            points.append(Point(lat=house_data['lat'], lon=house_data['lon'],
                                id=address, type='house'))

    return points


clustered_points: list[list[Point]] = [cluster_to_houses(c) for c in clustered_blocks]

centers = [c[0] for c in clustered_points]
display_clustered_blocks(requested_blocks, labels, centers).save(os.path.join(BASE_DIR, 'viz', 'clusters.html'))
# endregion

start = Point(lat=40.4418183, lon=-79.9198965, type='node', id=None)
area = clustered_points[1] + clustered_points[2] + clustered_points[6]
area_blocks = deepcopy(clustered_blocks[1])
area_blocks.update(clustered_blocks[2])
area_blocks.update(clustered_blocks[6])

# Generate house distance matrix
HouseDistances(area_blocks, start)


unique_intersections: list[Point] = []
unique_intersection_ids: set[str] = set()
for b_id, block in area_blocks.items():
    if pt_id(block['nodes'][0]) not in unique_intersection_ids:
        unique_intersections.append(block['nodes'][0])
        unique_intersection_ids.add(pt_id(block['nodes'][0]))
    if pt_id(block['nodes'][-1]) not in unique_intersection_ids:
        unique_intersections.append(block['nodes'][-1])
        unique_intersection_ids.add(pt_id(block['nodes'][-1]))

'-----------------------------------------------------------------------------------------'
'                                      Optimize                                           '
' Run the optimizer on the subset of the universe, providing a startting location for the '
' group canvas problem and nothing for the turf split problem                             '
'-----------------------------------------------------------------------------------------'
# region: Optimize
optimizer = Optimizer(area, num_lists=10, starting_locations=start)
solution = optimizer.optimize()

if solution is None:
    print(colored('Failed to generate lists', color='red'))
    sys.exit()

optimizer.visualize()
# endregion

'-----------------------------------------------------------------------------------------'
'                                      Post-Process                                       '
' Details coming soon...                                                                  '
'-----------------------------------------------------------------------------------------'
# region: Post-Process

# depot = Point(-1, -1, id='depot')
# points = area + unique_intersections + [depot]
# post_processor = PostProcess(requested_blocks, points=area, canvas_start=start)
# walk_lists: list[list[SubBlock]] = []
# for i, tour in enumerate(solution['tours']):
#     # Do not count the starting location service at the start or end
#     tour['stops'] = tour['stops'][1:-1]

#     if len(tour['stops']) == 0:
#         print('List {} has 0 stops'.format(i))
#         continue

#     walk_lists.append(post_processor.post_process(tour))

# list_visualizations = display_walk_lists(walk_lists)
# for i, walk_list in enumerate(list_visualizations):
#     walk_list.save(os.path.join(BASE_DIR, 'viz', 'walk_lists', '{}.html'.format(i)))

# scores = [score(start, start, lis) for lis in walk_lists]
# total_crossings = {
#     'motorway': 0,
#     'trunk': 0,
#     'primary': 0,
#     'secondary': 0,
#     'tertiary': 0,
#     'unclassified': 0,
#     'residential': 0,
#     'service': 0,
#     'other': 0
# }
# for s in scores:
#     for key, value in s['road_crossings'].items():
#         total_crossings[key] += value
# print('all crossings', total_crossings)
# print('distance', sum([s['distance'] for s in scores]))
# print('num segments', sum([s['segments'] for s in scores]))
# print('num houses', sum([s['num_houses'] for s in scores]))

# output = {}
# output['segments'] = []
# for sub in walk_lists[0]:
#     segment = {}
#     segment['nodes'] = []
#     for nav_pt in sub.navigation_points:
#         segment['nodes'].append({'coordinates': {'lat': nav_pt.lat, 'lon': nav_pt.lon}})
#     segment['houses'] = []
#     for house in sub.houses:
#         segment['houses'].append({'address': house.id, 'coordinates': {'lat': house.lat, 'lon': house.lon}})
#     output['segments'].append(segment)

# json.dump(output, open('app_list.json', 'w'), indent=2)
# # for s in walk_lists[0]:
# #     print('segment:', s.segment.id)
# #     print('start:', s.start)
# #     print('end:', s.end)
# #     print('extremum:', s.extremum)
# #     print('houses:', [h.id for h in s.houses])
# #     print('nav pts:', s.navigation_points)


# def modify_labels(segments: list[Segment], labels: list[int]) -> list[int]:
#     '''Apply DFS to split clusters into multiple clusters if they are not fully connected'''
#     clusters: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
#                                      for k in range(max(labels))]

#     def dfs(segment: Segment, cluster: list[Segment], visited: set[str]):
#         '''Depth-first search on a connected tree, tracking all visited nodes'''
#         if segment.id in visited:
#             return

#         visited.add(segment.id)

#         # Continuously update the visited set until it includes all segments connected to the original segment
#         distances = [BlockDistances.get_distance(s, segment) for s in cluster]
#         distances = [d for d in distances if d is not None]
#         neighbors = [cluster[i] for i in range(len(cluster)) if distances[i] < CLUSTERING_CONNECTED_THRESHOLD]
#         for neighbor in neighbors:
#             dfs(neighbor, cluster, visited)

#     def split_cluster(cluster: list[Segment]):
#         '''Split a cluster recursively until all sub-clusters are fully connected'''
#         sub_cluster: set[str] = set()
#         dfs(cluster[0], cluster, sub_cluster)

#         # Check if there are non-connected sub-clusters
#         if len(sub_cluster) < len(cluster):
#             # Change the indices of the subcluster to a new cluster
#             indices = [i for i in range(len(segments)) if segments[i].id in sub_cluster]
#             new_idx = max(labels) + 1
#             for idx in indices:
#                 labels[idx] = new_idx

#             # Continously split the remaining parts of the cluster until each are fully connected
#             split_cluster(cluster=[segment for segment in cluster if segment.id not in sub_cluster])

#     for cluster in clusters:
#         split_cluster(cluster)

#     return labels

# If needed, run post-processing on the created labels
# labels = modify_labels(segments, labels)

# endregion