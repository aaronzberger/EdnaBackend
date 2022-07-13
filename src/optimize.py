from __future__ import annotations
from copy import deepcopy

import itertools
import json
import os
import pickle
from dataclasses import dataclass, field
import sys

import folium
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sklearn.cluster as cluster
from tqdm import tqdm
import matplotlib
import matplotlib.cm as cm

from utils import BASE_DIR, lat_lon_to_x_y, x_y_to_lat_lon

from route import get_distance_p, get_distance


requests = json.load(open(os.path.join(BASE_DIR, 'requests.json')))


# Solely for type hinting
class TimelineData():
    pass


@dataclass
class Point(TimelineData):
    lat: float
    lon: float


@dataclass
class Segment(TimelineData):
    id: str
    start: Point
    end: Point
    num_houses: int
    all_points: list[Point]

    def __post_init__(self):
        self.length = 0
        for first, second in itertools.pairwise(self.all_points):
            self.length += get_distance_p(first, second)


segments = [Segment(id=i, start=Point(*d['start']), end=Point(*d['end']),
            num_houses=d['num_houses'], all_points=[Point(*k) for k in d['nodes']])
            for i, d in zip(requests.keys(), requests.values())]


def extract_node_ids(s: str):
    first_break = s.find(':')
    return s[:first_break], s[first_break + 1:s.find(':', first_break + 1)]


all_nodes = list(itertools.chain.from_iterable((i.start, i.end) for i in segments))
all_ids = list(itertools.chain.from_iterable(extract_node_ids(i.id) for i in segments))

node_distance_table_file = os.path.join(BASE_DIR, 'node_distance_matrix.pkl')
if os.path.exists(node_distance_table_file):
    print('Node distance matrix file found.')
    node_distance_table = pickle.load(node_distance_table_file, 'rb')
else:
    print('No node distance table file found. Generating now...')
    node_distance_table = {}
    with tqdm(total=len(all_ids) ** 2 / 2, desc='Generating Matrix', unit='iters', colour='green') as progress:
        def distance_metric(p1, p2):
            progress.update()
            return get_distance_p(p1, p2)
        for node, id in zip(all_nodes, all_ids):
            node_distance_table[id] = {}
            for other_node, other_id in zip(all_nodes, all_ids):
                if other_id not in node_distance_table:
                    node_distance_table[id][other_id] = distance_metric(node, other_node)

        # node_distance_table = squareform(pdist(np.expand_dims(segments, axis=1), metric=distance_metric))

        print('Saving to {}'.format(node_distance_table_file))
        with open(node_distance_table_file, 'wb') as output:
            pickle.dump(node_distance_table, output)


def get_hashed_distance(id1: str, id2: str):
    try:
        return node_distance_table[id1][id2]
    except KeyError:
        return node_distance_table[id2][id1]


@dataclass
class Timeline():
    start: Point
    end: Point
    deltas: list[float] = field(default_factory=lambda: [0.0])
    segments: list[Segment] = field(default_factory=list)
    total_time: float = 0.0

    def calculate_time(self):
        self.total_time = 0.0
        self.total_time += sum([segment.length * (1/60) for segment in self.segments])
        self.total_time += sum([segment.num_houses * 1.5 for segment in self.segments])
        self.total_time += sum([dist * (1/60) for dist in self.deltas])

    def _insert(self, segment: Segment, index: int):
        self.segments.insert(index, segment)
        del self.deltas[index]
        point_before = self.start if index == 0 else self.segments[index - 1].end
        self.deltas.insert(index, get_distance_p(point_before, segment.start))
        point_after = self.end if index == len(self.segments) - 1 else self.segments[index + 1].start
        self.deltas.insert(index + 1, get_distance_p(segment.end, point_after))
        self.calculate_time()

    def insert(self, segment: Segment, index: int) -> bool:
        # Test if we can fit this segment
        assert index < len(self.deltas) and index >= 0, "Index is {} with {} deltas, {} segments".format(
            index, len(self.deltas), len(self.segments))
        theoretical_timeline = deepcopy(self)
        theoretical_timeline._insert(segment, index)
        if theoretical_timeline.total_time > 180:
            return False

        self._insert(segment, index)
        return True

    def get_bid(self, segment: Segment, index: int) -> float | None:
        theoretical_timeline = deepcopy(self)
        possible = theoretical_timeline.insert(segment, index)
        delta_delta = sum(theoretical_timeline.deltas) - sum(self.deltas)
        if not possible or delta_delta > 50:
            return None
        return delta_delta


def cluster_segments(segments: list[Segment]):
    # TODO: Somehow enforce a policy that all clusters must be fully connected (no outliers)
    segment_distance_matrix_file = os.path.join(BASE_DIR, 'segment_distance_matrix.pkl')
    if os.path.exists(segment_distance_matrix_file):
        print('Segment distance matrix pickle found.')
        segment_distance_matrix = pickle.load(open(segment_distance_matrix_file, 'rb'))
    else:
        print('No segment distance matrix pickle found. Generating now...')
        with tqdm(total=(len(segments) ** 2) / 2, desc='Processing', unit='iters', colour='green') as progress:
            def distance_metric(s1, s2):
                progress.update()
                return min([node_distance_table[i][j] for i, j in
                           [(s1[0].start, s2[0].start), (s1[0].start, s2[0].end),
                            (s1[0].end, s2[0].start), (s1[0].end, s2[0].end)]])
            segment_distance_matrix = squareform(pdist(np.expand_dims(segments, axis=1), metric=distance_metric))
            print('Saving hash table to {}'.format(os.path.join(BASE_DIR, 'pdist.pkl')))
            with open(os.path.join(BASE_DIR, 'pdist.pkl'), 'wb') as output:
                pickle.dump(segment_distance_matrix, output)

    clustered = cluster.KMeans(n_clusters=11, random_state=0, n_init=100).fit(segment_distance_matrix)
    return segments, clustered.labels_


def modify_labels(segments: list[Segment], labels):
    return labels


def display_segments(segments: list[Segment]):
    all_lats = list(itertools.chain.from_iterable((i.start.lat, i.end.lat) for i in segments))
    all_lons = list(itertools.chain.from_iterable((i.start.lon, i.end.lon) for i in segments))

    m = folium.Map(location=[(min(all_lats) + max(all_lats)) / 2,
                             (min(all_lons) + max(all_lons)) / 2],
                   zoom_start=13)

    min_houses, max_houses = min([s.num_houses for s in segments]), max([s.num_houses for s in segments])

    for segment in segments:
        weight = 4 + ((segment.num_houses - min_houses) / (max_houses - min_houses)) * 8
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=weight,
            color='blue',
            opacity=0.6
        ).add_to(m)

    return m


def display_clustered_segments(segments: list[Segment], labels, centers=[]):
    all_lats = list(itertools.chain.from_iterable((i.start.lat, i.end.lat) for i in segments))
    all_lons = list(itertools.chain.from_iterable((i.start.lon, i.end.lon) for i in segments))

    m = folium.Map(location=[(min(all_lats) + max(all_lats)) / 2,
                             (min(all_lons) + max(all_lons)) / 2],
                   zoom_start=13)

    norm = matplotlib.colors.Normalize(vmin=min(labels), vmax=max(labels))
    cmap = cm.get_cmap('hsv')

    for segment, label in zip(segments, labels):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=matplotlib.colors.rgb2hex(cmap(norm(label))[:3]),
            opacity=0.6
        ).add_to(m)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(centers) - 1)

    for i in range(len(centers)):
        folium.Circle(
            [centers[i].lat, centers[i].lon],
            weight=10,
            color=matplotlib.colors.rgb2hex(cmap(norm(i))[:3]),
            opacity=1.0,
            radius=30
        ).add_to(m)

    return m


def make_arrow(p1: Point, p2: Point):
    '''Calculate the coordinates of an arrow from p1 to p2'''
    # Convert to XY
    x1, y1, zone_num, zone_let = lat_lon_to_x_y(p1.lat, p1.lon)
    x2, y2, zone_num1, zone_let1 = lat_lon_to_x_y(p2.lat, p2.lon)

    return np.arctan2([x1, y1], [x2, y2])


def display_walk_list(timeline: Timeline, m: folium.Map, color):
    # folium.Marker(
    #     location=[timeline.start.lat, timeline.start.lon],
    #     icon=folium.Icon(icon='play', color='green')
    # ).add_to(m)

    folium.Marker(
        location=[timeline.end.lat, timeline.end.lon],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    for i, segment in enumerate(timeline.segments):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=color,
            opacity=0.6
        ).add_to(m)

        if i != 0 and i != len(timeline.segments) - 1:
            folium.RegularPolygonMarker(
                location=[segment.end.lat, segment.end.lon],
                number_of_sides=3,
                fill_color=color,
                radius=10,
                rotation=make_arrow(segment.start, segment.end)
            ).add_to(m)

    return m


def display_walk_lists(timelines: list[Timeline]):
    all_lats = list(itertools.chain.from_iterable(
        (i.start.lat, i.end.lat) for timeline in timelines for i in timeline.segments))
    all_lons = list(itertools.chain.from_iterable(
        (i.start.lon, i.end.lon) for timeline in timelines for i in timeline.segments))

    m = folium.Map(location=[(min(all_lats) + max(all_lats)) / 2,
                             (min(all_lons) + max(all_lons)) / 2],
                   zoom_start=13)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(timelines) - 1)
    cmap = cm.get_cmap('hsv')

    for i, timeline in enumerate(timelines):
        color = matplotlib.colors.rgb2hex(cmap(norm(i))[:3])
        m = display_walk_list(timeline, m, color)

    return m


def display_requests(requests: list[Segment]):
    all_lats = list(itertools.chain.from_iterable((i.start.lat, i.end.lat) for i in requests))
    all_lons = list(itertools.chain.from_iterable((i.start.lon, i.end.lon) for i in requests))

    m = folium.Map(location=[(min(all_lats) + max(all_lats)) / 2,
                             (min(all_lons) + max(all_lons)) / 2],
                   zoom_start=13)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(requests) - 1)
    cmap = cm.get_cmap('RdYlGn')

    for i, segment in enumerate(requests):
        color = matplotlib.colors.rgb2hex(cmap(norm(i))[:3])
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=color,
            opacity=0.6
        ).add_to(m)

    return m


def select_starting_locations(segments, labels):
    print('Starting centering...', end=' ', flush=True)
    pickle_path = os.path.join(BASE_DIR, 'centers.pkl')
    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))
    else:
        centers = []
        clusters = [[segments[i] for i in range(len(segments)) if labels[i] == k] for k in range(max(labels))]
        for cluster in clusters:
            # Center is the one closest to all the other nodes
            all_points = list(itertools.chain.from_iterable((s.start, s.end) for s in cluster))
            all_ids = list(itertools.chain.from_iterable(extract_node_ids(s.id) for s in cluster))
            distances = []
            for id in all_ids:
                distances.append(sum([get_hashed_distance(id, i) for i in all_ids]))
                # distances.append(sum([get_distance_p(point, i) for i in all_points]))
            centers.append(all_points[distances.index(min(distances))])
        with open(os.path.join(pickle_path), 'wb') as output:
            pickle.dump(centers, output)
        print('Done centering')
        return centers


segments, labels = cluster_segments(segments)
centers = select_starting_locations(segments, labels)
display_clustered_segments(segments, labels, centers).save('clustered.html')

display_segments(segments).save('segments.html')

print('Beginning request ordering... ')


# Request ordering
def order_requests(requests: list[Segment]) -> list[Segment]:
    '''Variable ordering for the timeline construction'''
    # For now, just order by house density
    def score(segment: Segment):
        return segment.num_houses / segment.length
    scores = [score(s) for s in requests]
    return [s for _, s in sorted(zip(scores, requests), key=lambda pair: pair[0], reverse=True)]


ordered_requests = order_requests(segments)
display_requests(ordered_requests).save('requests.html')


print('Beginning timeline population... ')
################################### Timeline Population ###################################
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

display_walk_lists(timelines).save('lists.html')
