from __future__ import annotations

import json
import os

import kmedoids

from config import (BASE_DIR, CLUSTERING_CONNECTED_THRESHOLD, requests_file,
                    requests_file_t)
from gps_utils import Point
from optimize import Optimizer
from timeline_utils import NodeDistances, Segment, SegmentDistances
from viz_utils import display_segments

DISPLAY_VERBOSE = True
LOAD_CENTERS = False  # Set to true if you are providing custom starting point locations

requests: requests_file_t = json.load(open(requests_file))

# Create the list of all segments
segments = [Segment(id=i, start=Point(d['start']['lat'], d['start']['lon']),
            end=Point(d['end']['lat'], d['end']['lon']),
            num_houses=d['num_houses'], all_points=[Point(k['lat'], k['lon']) for k in d['nodes']])
            for i, d in zip(requests.keys(), requests.values())]

if DISPLAY_VERBOSE:
    display_segments(segments).save(os.path.join(BASE_DIR, 'viz', 'segments.html'))

'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'
NodeDistances(segments)

'----------------------------------------------------------------------------------'
'                                Segment Clustering                                '
'----------------------------------------------------------------------------------'
SegmentDistances(segments)

# Cluster the segments using kmedoids
distance_matrix = SegmentDistances.get_distance_matrix()
km: kmedoids.KMedoidsResult = kmedoids.fasterpam(diss=distance_matrix, medoids=10, max_iter=100, random_state=0)
labels = km.labels

clusters: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
                                 for k in range(max(labels))]

# area = list(itertools.chain.from_iterable(clusters))
center = Point(40.4418183, -79.9198965)
optimizer = Optimizer(clusters[3] + clusters[2], num_lists=3, starting_location=center)
optimizer.optimize()
optimizer.visualize()


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
        neighbors = [cluster[i] for i in range(len(cluster)) if distances[i] is not None
                     and distances[i] < CLUSTERING_CONNECTED_THRESHOLD]
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
