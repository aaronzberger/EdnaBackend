from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from typing import Optional

import kmedoids
import numpy as np
from sklearn.preprocessing import normalize
from termcolor import colored
from optimize import optimize_cluster

from config import (ARBITRARY_LARGE_DISTANCE, BASE_DIR,
                    CLUSTERING_CONNECTED_THRESHOLD, requests_file,
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
km: kmedoids.KMedoidsResult = kmedoids.fasterpam(diss=distance_matrix, medoids=20, max_iter=100, random_state=0)
labels = km.labels

clusters: list[list[Segment]] = [[segments[i] for i in range(len(segments)) if labels[i] == k]
                                 for k in range(max(labels))]
optimize_cluster(segments)
sys.exit()


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
                point_distances = [NodeDistances.get_distance(point, i) for i in all_points]
                existent_distances = [i for i in point_distances if i is not None]

                # Adding the large distance also pushes the centers away from edges and towards the middle of neighborhoods
                distances.append(ARBITRARY_LARGE_DISTANCE if len(existent_distances) == 0 else sum(existent_distances))
            centers.append(all_points[distances.index(min(distances))])

        # Convert points to their json-exportable format
        json.dump([pt.__dict__ for pt in centers], open(centers_file, 'w', encoding='utf-8'), indent=4)
        return centers


centers = select_starting_locations(segments, labels)

NodeDistances.add_nodes(centers)

if DISPLAY_VERBOSE:
    display_clustered_segments(segments, labels, centers).save(os.path.join(BASE_DIR, 'viz', 'clustered.html'))

'--------------------------------------------------------------------------------'
'                                Request Ordering                                '
'--------------------------------------------------------------------------------'
print('Beginning request ordering...')


class RequestHandler():
    @classmethod
    def __init__(cls, requests: list[Segment], labels: list[int], centers: list[Point], max_fails: int = 5):
        '''Order by closeness to center, then density'''
        assert max(labels) == len(centers), \
            'Mismatch: received {} clusters but {} centers'.format(max(labels), len(centers))
        cls.max_fails = max_fails
        cls.ordered_requests: list[Segment] = []

        clusters: list[list[Segment]] = [[requests[i] for i in range(len(requests)) if labels[i] == k]
                                         for k in range(max(labels))]
        for cluster in clusters:
            densities: list[float] = normalize([[s.num_houses / s.length] for s in cluster])
            lengths: list[float] = normalize([[s.length] for s in cluster])
            scores = [0.7 * d + 0.3 * l for d, l in zip(densities, lengths)]
            cls.ordered_requests.extend([s for _, s in sorted(zip(scores, cluster),
                                        key=lambda pair: pair[0], reverse=True)])

        # densities: list[float] = normalize([[s.num_houses / s.length] for s in requests])
        # lengths: list[float] = normalize([[s.length] for s in requests])
        # scores = [0.7 * d + 0.3 * l for d, l in zip(densities, lengths)]
        # cls.ordered_requests = [s for _, s in sorted(zip(scores, requests),
        #                         key=lambda pair: pair[0], reverse=True)]

        cls.last_idx = 0
        cls.running_idx = 0

        cls.fails: list[int] = [0] * len(cls.ordered_requests)

    @classmethod
    def get_request(cls) -> Optional[Segment]:
        try:
            return cls.ordered_requests[cls.running_idx]
        except IndexError:
            # This means there are no more requests
            return None
        finally:
            cls.last_idx = cls.running_idx
            cls.running_idx = 0 if cls.running_idx == len(cls.ordered_requests) - 1 else cls.running_idx + 1

    @classmethod
    def report_success(cls):
        '''Report that the last request was inserted successfully'''
        del cls.ordered_requests[cls.last_idx]
        del cls.fails[cls.last_idx]

        if cls.running_idx == len(cls.ordered_requests):
            cls.running_idx = 0

    @classmethod
    def report_fail(cls):
        '''Report that the last request was not inserted'''
        cls.fails[cls.last_idx] += 1
        if cls.fails[cls.last_idx] >= cls.max_fails:
            cls.report_success()


RequestHandler(segments, labels, centers)

if DISPLAY_VERBOSE:
    display_requests(RequestHandler.ordered_requests).save(os.path.join(BASE_DIR, 'viz', 'requests.html'))

'-----------------------------------------------------------------------------------'
'                                Timeline Population                                '
'-----------------------------------------------------------------------------------'
print('Beginning timeline population... ')

# List of lists of TimelineData
timelines: list[Timeline] = []

action_report: list[str] = []

# Add starting locations
for center in centers:
    timelines.append(Timeline(start=center, end=center))

timeline_file = os.path.join(BASE_DIR, 'timelines.pkl')
if os.path.exists(timeline_file):
    timelines = pickle.load(open(timeline_file, 'rb'))
else:
    requests_inserted = 0
    print('Populating: ' + colored('{:<4}'.format(requests_inserted), color='green') +
          'requests inserted, ' + colored('{:<4}'.format(len(RequestHandler.ordered_requests)), color='yellow') +
          'remaining to insert ', end='', flush=True)
    request = RequestHandler.get_request()
    while request is not None:
        print('\rPopulating: ' + colored('{:<4}'.format(requests_inserted), color='green') +
              'requests inserted, ' + colored('{:<4}'.format(len(RequestHandler.ordered_requests)), color='yellow') +
              'remaining to insert ', end='', flush=True)
        min_delta = ARBITRARY_LARGE_DISTANCE
        min_timeline: Optional[int] = None
        min_slot: Optional[int] = None
        action_report.append('-------------------------------------\n')
        action_report.append('Now attempting to insert request {}\n'.format(request.id))

        # Check if this request fits in each timeline
        for i, timeline in enumerate(timelines):
            bids = timeline.get_bids(request)
            existent_bids = [b for b in bids if b is not None]
            if len(existent_bids) == 0:
                action_report.append('Tried to insert segment {} in timeline {} but it didn\'t fit\n'.format(request.id, i))
                # It does not fit in this timeline, so move to the next
                continue

            min_bid = min(existent_bids)
            if min_bid < min_delta:
                min_delta = min_bid
                min_timeline = i
                min_slot = bids.index(min_bid)

        # It doesn't fit into any timeline
        if min_timeline is None or min_slot is None:
            RequestHandler.report_fail()
            action_report.append('Failed to insert segment {} into any timeline. Moving on.\n'.format(request.id))
            request = RequestHandler.get_request()
            continue

        timelines[min_timeline].insert(request, min_slot)
        action_report.append('Successfully inserted segment {} in timeline {} at index {}\n'.format(
            request.id, min_timeline, min_slot))
        RequestHandler.report_success()
        requests_inserted += 1

        request = RequestHandler.get_request()

    # with open(timeline_file, 'wb') as output:
    #     pickle.dump(timelines, output)

print()
generate_timelines(timelines)

# Always display the final walk lists
test = display_walk_lists(timelines)
test.save(os.path.join(BASE_DIR, 'viz', 'lists.html'))

with open(os.path.join(BASE_DIR, 'report.txt'), 'w') as report_file:
    report_file.writelines(action_report)
