from __future__ import annotations

import itertools
import json
import os
import pickle
import sys

import numpy as np
import sklearn.cluster
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from config import (ARBITRARY_LARGE_DISTANCE, BASE_DIR, requests_file,
                    requests_file_t)
from gps_utils import Point
from timeline_utils import NodeDistances, Segment, Timeline
from viz_utils import (display_clustered_segments, display_requests,
                       display_segments, display_walk_lists,
                       generate_timelines)

DISPLAY_VERBOSE = False

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
# TODO: Maybe to Json
node_distances = NodeDistances(segments=segments)

'----------------------------------------------------------------------------------'
'                                Segment Clustering                                '
'----------------------------------------------------------------------------------'
print('Beginning segment clustering...')


def generate_segment_distances(segments: list[Segment]) -> dict[str, dict[str, float]]:
    # TODO: Somehow enforce a policy that all clusters must be fully connected (no outliers)
    segment_distance_matrix_file = os.path.join(BASE_DIR, 'store', 'segment_distance_matrix.json')
    if os.path.exists(segment_distance_matrix_file):
        print('Segment distance matrix pickle found.')
        matrix = json.load(open(segment_distance_matrix_file))
    else:
        print('No segment distance matrix found. Generating now...')
        with tqdm(total=(len(segments) ** 2) / 2, desc='Processing', unit='iters', colour='green') as progress:
            matrix: dict[str, dict[str, float]] = {}

            # Similar code as in NodeDistances creation
            def insert_pair(s1: Segment, s2: Segment):
                try:
                    matrix[s2.id][s1.id]
                except KeyError:
                    routed_distances = \
                        [node_distances.get_distance(i, j) for i, j in
                            [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]
                    existing_distances = [i for i in routed_distances if i]
                    matrix[s1.id][s2.id] = ARBITRARY_LARGE_DISTANCE if len(existing_distances) == 0 \
                        else min(existing_distances)

            for segment in segments:
                matrix[segment.id] = {}
                for other_segment in segments:
                    insert_pair(segment, other_segment)
                    progress.update()

        print('Saving segment distance matrix to {}'.format(segment_distance_matrix_file), flush=True)
        json.dump(matrix, open(segment_distance_matrix_file, 'w', encoding='utf-8'), indent=4)
    return matrix


segment_distance_matrix = generate_segment_distances(segments)


def cluster_segments(segments: list[Segment], distances: dict[str, dict[str, float]]) -> list[int]:
    def distance_metric(s1: list[Segment], s2: list[Segment]) -> float:
        try:
            return distances[s1[0].id][s2[0].id]
        except KeyError:
            return distances[s2[0].id][s1[0].id]

    # Perform the actual clustering
    formatted_matrix = squareform(pdist(np.expand_dims(segments, axis=1), metric=distance_metric))
    clustered: sklearn.cluster.KMeans = sklearn.cluster.KMeans(
        n_clusters=11, random_state=0, n_init=100).fit(formatted_matrix)
    print(clustered.labels_)
    return clustered.labels_


def modify_labels(segments: list[Segment], labels: list[int]) -> list[int]:
    # TODO: Implement
    return labels


labels = cluster_segments(segments, segment_distance_matrix)

'----------------------------------------------------------------------------------------'
'                                Starting Point Selection                                '
'----------------------------------------------------------------------------------------'
print('Beginning starting point selection...')


def select_starting_locations(segments: list[Segment], labels: list[int]) -> list[Point]:
    centers_file = os.path.join(BASE_DIR, 'store', 'centers.json')
    if os.path.exists(centers_file):
        return json.load(open(centers_file))
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
                existent_distances = [i for i in point_distances if i]
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


def order_requests(requests: list[Segment]) -> list[Segment]:
    '''Variable ordering for the timeline construction'''
    # Order by house density
    def score(segment: Segment) -> float:
        return segment.num_houses / segment.length
    scores = [score(s) for s in requests]
    return [s for _, s in sorted(zip(scores, requests), key=lambda pair: pair[0], reverse=True)]


ordered_requests = order_requests(segments)

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

ordered_requests *= 10

timeline_file = os.path.join(BASE_DIR, 'timelines.pkl')
if os.path.exists(timeline_file):
    timelines = pickle.load(open(timeline_file, 'rb'))
else:
    with tqdm(total=len(ordered_requests), desc='Populating', unit='requests', colour='green') as progress:
        for request in ordered_requests:
            progress.update()
            # Look through each delta
            min_delta = ARBITRARY_LARGE_DISTANCE
            min_timeline = min_slot = -1
            for i, timeline in enumerate(timelines):
                bids = [timeline.get_bid(request, i) for i in range(len(timeline.deltas))]
                # If no timeline can fit this request, we are done
                existent_bids = [b for b in bids if b is not None]
                if len(existent_bids) == 0:
                    continue
                # TODO: continue for a few

                min_bid = min(existent_bids)
                if min_bid < min_delta:
                    min_delta = min_bid
                    min_timeline = i
                    min_slot = bids.index(min_bid)

            # It didn't fit into any timeline
            # TODO: Ben says make optional
            if min_timeline == -1:
                continue
            timelines[min_timeline].insert(request, min_slot)
        with open(timeline_file, 'wb') as output:
            pickle.dump(timelines, output)

generate_timelines(timelines)

# Always display the final walk lists
test = display_walk_lists(timelines)
test.save(os.path.join(BASE_DIR, 'viz', 'lists.html'))
