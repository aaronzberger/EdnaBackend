from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy
from sys import argv

import kmedoids
from termcolor import colored

from gps_utils import Point
from src.config import (BASE_DIR, CLUSTERING_CONNECTED_THRESHOLD,
                        KEEP_APARTMENTS, blocks_file, blocks_file_t,
                        houses_file, houses_file_t)
from src.distances.houses import HouseDistances
from src.distances.nodes import NodeDistances
from src.distances.segments import SegmentDistances
from src.optimize import Optimizer
from src.post_process import PostProcess
from src.timeline_utils import Segment, SubSegment
from src.viz_utils import display_clustered_segments, display_segments

blocks: blocks_file_t = json.load(open(blocks_file))

'-----------------------------------------------------------------------------------------'
'                                Handle universe file                                     '
'-----------------------------------------------------------------------------------------'
if len(argv) == 2:
    if not os.path.exists(argv[1]):
        raise FileExistsError('Usage: make_walk_lists.py [UNIVERSE FILE]')
    reader = csv.DictReader(open(argv[1]))
    houses_to_id: houses_file_t = json.load(open(houses_file))
    requested_blocks: blocks_file_t = {}
    total_houses = failed_houses = 0
    for house in reader:
        formatted_address = house['Address'].upper()
        total_houses += 1
        if formatted_address not in houses_to_id:
            failed_houses += 1
            continue
        block_id = houses_to_id[formatted_address]
        house_info = deepcopy(blocks[block_id]['addresses'][formatted_address])

        if block_id in requested_blocks:
            requested_blocks[block_id]['addresses'][formatted_address] = house_info
        else:
            requested_blocks[block_id] = deepcopy(blocks[block_id])
            requested_blocks[block_id]['addresses'] = {formatted_address: house_info}
    print('Failed on {} of {} houses'.format(failed_houses, total_houses))
else:
    requested_blocks = deepcopy(blocks)

# After this point, the original blocks variable should never be used, so delete it for error finding
blocks.clear()
del blocks

# Convert the blocks to segments
segments = [Segment(
    id=i, start=Point(d['nodes'][0]['lat'], d['nodes'][0]['lon']),
    end=Point(d['nodes'][-1]['lat'], d['nodes'][-1]['lon']), num_houses=len(d['addresses']),
    navigation_points=[Point(k['lat'], k['lon']) for k in d['nodes']])
        for i, d in requested_blocks.items()]

display_segments(segments).save(os.path.join(BASE_DIR, 'viz', 'segments.html'))

# Generate node distance matrix
NodeDistances(segments)

# Generate segment distance matrix
SegmentDistances(segments)

# Cluster segments using kmedoids
distance_matrix = SegmentDistances.get_distance_matrix()
km: kmedoids.KMedoidsResult = kmedoids.fasterpam(diss=distance_matrix, medoids=10, max_iter=100, random_state=0)
labels: list[int] = km.labels

clustered_segments: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
                                           for k in range(max(labels))]


def cluster_to_houses(cluster: list[Segment]) -> list[Point]:
    '''Convert a list of segments to its corresponding list of houses'''
    points: list[Point] = []

    for segment in cluster:
        for address in requested_blocks[segment.id]['addresses']:
            if not KEEP_APARTMENTS and ' APT ' in address:
                continue
            points.append(Point(requested_blocks[segment.id]['addresses'][address]['lat'],
                                requested_blocks[segment.id]['addresses'][address]['lon'],
                                id=address))

    return points


clustered_points: list[list[Point]] = [cluster_to_houses(c) for c in clustered_segments]

centers = [c[0] for c in clustered_points]
display_clustered_segments(segments, labels, centers).save(os.path.join(BASE_DIR, 'viz', 'clusters.html'))

start = Point(40.4418183, -79.9198965)
area = clustered_points[5] + clustered_points[6]
area_segments = clustered_segments[5] + clustered_segments[6]
# Generate house distance matrix
HouseDistances(area_segments, start)

# Run the optimizer
optimizer = Optimizer(area, num_lists=12, starting_location=start)
solution = optimizer.optimize()

if solution is None:
    print(colored('Failed to generate lists', color='red'))
    sys.exit()

optimizer.visualize()

# Post-process
post_processor = PostProcess(segments, points=area, canvas_start=start)
walk_lists: list[list[SubSegment]] = []
for tour in solution['tours']:
    # Do not count the starting location service at the start or end
    tour['stops'] = tour['stops'][1:-1]

    walk_lists.append(post_processor.post_process(tour))

for s in walk_lists[0]:
    print('segment:', s.segment.id)
    print('start:', s.start)
    print('end:', s.end)
    print('extremum:', s.extremum)
    print('houses:', [h.id for h in s.houses])
    print('nav pts:', s.navigation_points)


def modify_labels(segments: list[Segment], labels: list[int]) -> list[int]:
    '''Apply DFS to split clusters into multiple clusters if they are not fully connected'''
    clusters: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
                                     for k in range(max(labels))]

    def dfs(segment: Segment, cluster: list[Segment], visited: set[str]):
        '''Depth-first search on a connected tree, tracking all visited nodes'''
        if segment.id in visited:
            return

        visited.add(segment.id)

        # Continuously update the visited set until it includes all segments connected to the original segment
        distances = [SegmentDistances.get_distance(s, segment) for s in cluster]
        distances = [d for d in distances if d is not None]
        neighbors = [cluster[i] for i in range(len(cluster)) if distances[i] < CLUSTERING_CONNECTED_THRESHOLD]
        for neighbor in neighbors:
            dfs(neighbor, cluster, visited)

    def split_cluster(cluster: list[Segment]):
        '''Split a cluster recursively until all sub-clusters are fully connected'''
        sub_cluster: set[str] = set()
        dfs(cluster[0], cluster, sub_cluster)

        # Check if there are non-connected sub-clusters
        if len(sub_cluster) < len(cluster):
            # Change the indices of the subcluster to a new cluster
            indices = [i for i in range(len(segments)) if segments[i].id in sub_cluster]
            new_idx = max(labels) + 1
            for idx in indices:
                labels[idx] = new_idx

            # Continously split the remaining parts of the cluster until each are fully connected
            split_cluster(cluster=[segment for segment in cluster if segment.id not in sub_cluster])

    for cluster in clusters:
        split_cluster(cluster)

    return labels

# If needed, run post-processing on the created labels
# labels = modify_labels(segments, labels)
