from __future__ import annotations

import itertools

import folium
import matplotlib
import matplotlib.cm as cm
import numpy as np

from gps_utils import Point, pt_to_utm
from timeline_utils import Segment, Timeline


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


def display_walk_list(timeline: Timeline, m: folium.Map, color):
    def make_arrow(p1: Point, p2: Point):
        '''Calculate the coordinates of an arrow from p1 to p2'''
        # Convert to XY
        x1, y1, _, _ = pt_to_utm(p1)
        x2, y2, _, _ = pt_to_utm(p2)

        return np.arctan2([x1, y1], [x2, y2])
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
