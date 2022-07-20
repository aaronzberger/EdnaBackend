from __future__ import annotations

import itertools
import json
import os
import pickle
from typing import Optional

import numpy as np
import sklearn.cluster
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from config import (ARBITRARY_LARGE_DISTANCE, BASE_DIR, CLUSTERING_CONNECTED_THRESHOLD, requests_file,
                    requests_file_t)
from gps_utils import Point
from timeline_utils import NodeDistances, Segment, SegmentDistances, Timeline
from viz_utils import (display_clustered_segments, display_requests,
                       display_segments, display_walk_lists,
                       generate_timelines)

DISPLAY_VERBOSE = True
LOAD_CENTERS = False  # Set to true if you are providing custom starting point locations

requests: requests_file_t = json.load(open(requests_file))

# Create the list of all segments
segments = [Segment(id=i, start=Point(*d['start'].values()), end=Point(*d['end'].values()),
            num_houses=d['num_houses'], all_points=[Point(*k.values()) for k in d['nodes']])
            for i, d in zip(requests.keys(), requests.values())]

if DISPLAY_VERBOSE:
    display_segments(segments).save(os.path.join(BASE_DIR, 'viz', 'segments.html'))

'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'
node_distances = NodeDistances(segments)

'----------------------------------------------------------------------------------'
'                                Segment Clustering                                '
'----------------------------------------------------------------------------------'
segment_distances = SegmentDistances(segments)


def cluster_segments(segments: list[Segment]) -> list[int]:
    def distance_metric(s1: list[Segment], s2: list[Segment]) -> float:
        try:
            dist = SegmentDistances.get_distance(s1[0], s2[0])
            return dist if dist is not None else ARBITRARY_LARGE_DISTANCE
        except KeyError:
            dist = SegmentDistances.get_distance(s2[0], s1[0])
            return dist if dist is not None else ARBITRARY_LARGE_DISTANCE

    # Perform the actual clustering
    formatted_matrix = squareform(pdist(np.expand_dims(segments, axis=1), metric=distance_metric))
    clustered: sklearn.cluster.KMeans = sklearn.cluster.KMeans(
        n_clusters=20, random_state=0, n_init=100).fit(formatted_matrix)
    return clustered.labels_


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


labels = cluster_segments(segments)
labels = modify_labels(segments, labels)

'----------------------------------------------------------------------------------------'
'                                Starting Point Selection                                '
'----------------------------------------------------------------------------------------'
print('Beginning starting point selection...')


def select_starting_locations(segments: list[Segment], labels: list[int]) -> list[Point]:
    centers_file = os.path.join(BASE_DIR, 'store', 'centers.json')
    if os.path.exists(centers_file) and LOAD_CENTERS:
        loaded = json.load(open(centers_file))
        return [Point(*item.values()) for item in loaded]
    else:
        centers: list[Point] = []
        clusters: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
                                         for k in range(max(labels))]
        for cluster in clusters:
            # Center is the one closest to all the other nodes
            all_points: list[Point] = list(itertools.chain.from_iterable((s.start, s.end) for s in cluster))
            distances: list[float] = []
            for point in all_points:
                point_distances = [node_distances.get_distance(point, i) for i in all_points]
                existent_distances = [i for i in point_distances if i is not None]

                # Adding the large distance also pushes the centers away from edges and towards the middle of neighborhoods
                distances.append(ARBITRARY_LARGE_DISTANCE if len(existent_distances) == 0 else sum(existent_distances))
            centers.append(all_points[distances.index(min(distances))])

        # Convert points to their json-exportable format
        json.dump([pt.__dict__ for pt in centers], open(centers_file, 'w', encoding='utf-8'), indent=4)
        return centers


centers = select_starting_locations(segments, labels)

node_distances.add_nodes(centers)

if DISPLAY_VERBOSE:
    display_clustered_segments(segments, labels, centers).save(os.path.join(BASE_DIR, 'viz', 'clustered.html'))

'--------------------------------------------------------------------------------'
'                                Request Ordering                                '
'--------------------------------------------------------------------------------'
print('Beginning request ordering...')


# def order_requests(requests: list[Segment]) -> list[Segment]:
#     '''Variable ordering for the timeline construction'''
#     # Order by house density
#     def score(segment: Segment) -> float:
#         return segment.num_houses / segment.length
#     scores = [score(s) for s in requests]
#     return [s for _, s in sorted(zip(scores, requests), key=lambda pair: pair[0], reverse=True)]

def order_requests(requests: list[Segment], labels: list[int], centers: list[Point]) -> list[Segment]:
    '''Order by closeness to center, then density'''
    assert max(labels) == len(centers)
    ordered_requests: list[Segment] = []

    def density(segment: Segment) -> float:
        return segment.num_houses / segment.length

    # def distance_to_center(center: Point, request: Segment) -> float:
    #     distance_from_start = NodeDistances.get_distance(request.start, center)
    #     if distance_from_start is None:
    #         distance_from_start = get_distance(request.start, center)
    #     distance_from_end = NodeDistances.get_distance(request.end, center)
    #     if distance_from_end is None:
    #         distance_from_end = get_distance(request.end, center)
    #     return min(distance_from_start, distance_from_end)

    clusters: list[list[Segment]] = [[requests[i] for i in range(len(requests)) if labels[i] == k]
                                     for k in range(max(labels))]
    for cluster in clusters:
        density_scores = [density(s) for s in cluster]
        ordered_requests.extend([s for _, s in sorted(zip(density_scores, cluster),
                                 key=lambda pair: pair[0], reverse=True)])

    return ordered_requests


ordered_requests = order_requests(segments, labels, centers)

if DISPLAY_VERBOSE:
    display_requests(ordered_requests).save(os.path.join(BASE_DIR, 'viz', 'requests.html'))

'-----------------------------------------------------------------------------------'
'                                Timeline Population                                '
'-----------------------------------------------------------------------------------'
print('Beginning timeline population... ')

# List of lists of TimelineData
timelines: list[Timeline] = []

# Add starting locations
for center in centers:
    timelines.append(Timeline(start=center, end=center))

timeline_file = os.path.join(BASE_DIR, 'timelines.pkl')
if os.path.exists(timeline_file):
    timelines = pickle.load(open(timeline_file, 'rb'))
else:
    with tqdm(total=len(ordered_requests), desc='Populating', unit='requests', colour='green') as progress:
        for request in ordered_requests:
            progress.update()
            # Look through each delta
            min_delta = ARBITRARY_LARGE_DISTANCE
            min_timeline: Optional[int] = None
            min_slot: Optional[int] = None
            for i, timeline in enumerate(timelines):
                bids = [timeline.get_bid(request, i) for i in range(len(timeline.deltas))]
                # If no timeline can fit this request, continue
                existent_bids = [b for b in bids if b is not None]
                if len(existent_bids) == 0:
                    continue
                # TODO: Implement a stop after a certain number of failed requests (or full iterations through)
                min_bid = min(existent_bids)
                if min_bid < min_delta:
                    min_delta = min_bid
                    min_timeline = i
                    min_slot = bids.index(min_bid)

            # It didn't fit into any timeline
            if min_timeline is None or min_slot is None:
                continue
            timelines[min_timeline].insert(request, min_slot)
        # with open(timeline_file, 'wb') as output:
        #     pickle.dump(timelines, output)

generate_timelines(timelines)

# Always display the final walk lists
test = display_walk_lists(timelines)
test.save(os.path.join(BASE_DIR, 'viz', 'lists.html'))
