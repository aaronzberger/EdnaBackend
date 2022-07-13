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

from gps_utils import BASE_DIR, Point
from timeline_utils import Segment, Timeline, NodeDistances
from viz_utils import (display_clustered_segments, display_requests,
                       display_segments, display_walk_lists)

DISPLAY_VERBOSE = True


requests = json.load(open(os.path.join(BASE_DIR, 'requests.json')))

# Create the list of all segments
segments = [Segment(id=i, start=Point(*d['start']), end=Point(*d['end']),
            num_houses=d['num_houses'], all_points=[Point(*k) for k in d['nodes']])
            for i, d in zip(requests.keys(), requests.values())]

if DISPLAY_VERBOSE:
    display_segments(segments).save(os.path.join(BASE_DIR, 'segments.html'))

'-----------------------------------------------------------------------------------------'
'                                Node Distances Generation                                '
'-----------------------------------------------------------------------------------------'
node_distances = NodeDistances(segments=segments)

'----------------------------------------------------------------------------------'
'                                Segment Clustering                                '
'----------------------------------------------------------------------------------'
print('Beginning segment clustering...')


def cluster_segments(segments: list[Segment]):
    # TODO: Somehow enforce a policy that all clusters must be fully connected (no outliers)
    segment_distance_matrix_file = os.path.join(BASE_DIR, 'segment_distance_matrix.pkl')
    if os.path.exists(segment_distance_matrix_file):
        print('Segment distance matrix pickle found.')
        segment_distance_matrix = pickle.load(open(segment_distance_matrix_file, 'rb'))
    else:
        print('No segment distance matrix pickle found. Generating now...')
        with tqdm(total=(len(segments) ** 2) / 2, desc='Processing', unit='iters', colour='green') as progress:
            def distance_metric(s1: Segment, s2: Segment):
                progress.update()
                s1_ids = s1.extract_end_ids().values()
                s2_ids = s2.extract_end_ids().values()
                return min([node_distances.get_distance_by_id(i, j) for i, j in
                           [s1_ids[0], s2_ids[0], (s1_ids[0], s2_ids[1]),
                           (s1_ids[1], s2_ids[0]), (s1_ids[1], s2_ids[1])]])
            segment_distance_matrix = squareform(pdist(np.expand_dims(segments, axis=1), metric=distance_metric))
            print('Saving segment distance matrix to {}'.format(segment_distance_matrix_file))
            with open(segment_distance_matrix_file, 'wb') as output:
                pickle.dump(segment_distance_matrix, output)

    # Perform the actual clustering
    clustered = sklearn.cluster.KMeans(n_clusters=11, random_state=0, n_init=100).fit(segment_distance_matrix)
    return segments, clustered.labels_


def modify_labels(segments: list[Segment], labels):
    # TODO: Implement
    return labels


segments, labels = cluster_segments(segments)

'----------------------------------------------------------------------------------------'
'                                Starting Point Selection                                '
'----------------------------------------------------------------------------------------'
print('Beginning starting point selection...')


def select_starting_locations(segments, labels):
    pickle_path = os.path.join(BASE_DIR, 'centers.pkl')
    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))
    else:
        centers = []
        clusters = [[segments[i] for i in range(len(segments)) if labels[i] == k] for k in range(max(labels))]
        for cluster in clusters:
            # Center is the one closest to all the other nodes
            all_points = list(itertools.chain.from_iterable((s.start, s.end) for s in cluster))
            all_ids = list(itertools.chain.from_iterable(s.extract_end_ids() for s in cluster))
            distances = []
            for id in all_ids:
                distances.append(sum([node_distances.get_distance_by_id(id, i) for i in all_ids]))
                # distances.append(sum([get_distance_p(point, i) for i in all_points]))
            centers.append(all_points[distances.index(min(distances))])
        with open(os.path.join(pickle_path), 'wb') as output:
            pickle.dump(centers, output)
        return centers


centers = select_starting_locations(segments, labels)

if DISPLAY_VERBOSE:
    display_clustered_segments(segments, labels, centers).save(os.path.join(BASE_DIR, 'clustered.html'))

'--------------------------------------------------------------------------------'
'                                Request Ordering                                '
'--------------------------------------------------------------------------------'
print('Beginning request ordering...')


def order_requests(requests: list[Segment]) -> list[Segment]:
    '''Variable ordering for the timeline construction'''
    # Order by house density
    def score(segment: Segment):
        return segment.num_houses / segment.length
    scores = [score(s) for s in requests]
    return [s for _, s in sorted(zip(scores, requests), key=lambda pair: pair[0], reverse=True)]


ordered_requests = order_requests(segments)

if DISPLAY_VERBOSE:
    display_requests(ordered_requests).save(os.path.join(BASE_DIR, 'requests.html'))

'-----------------------------------------------------------------------------------'
'                                Timeline Population                                '
'-----------------------------------------------------------------------------------'
print('Beginning timeline population... ')

# List of lists of TimelineData
timelines = []

# Add starting locations
for center in centers:
    timelines.append(Timeline(start=center, end=center))

ordered_requests *= 10

with tqdm(total=len(ordered_requests), desc='Populating', unit='requests', colour='green') as progress:
    for request in ordered_requests:
        progress.update()
        # Look through each delta
        min_delta = sys.float_info.max
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
        if min_timeline == -1:
            continue
        timelines[min_timeline].insert(request, min_slot)

# Always display the final walk lists
display_walk_lists(timelines).save(os.path.join(BASE_DIR, 'lists.html'))
