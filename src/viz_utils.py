from __future__ import annotations

import itertools

import folium
import matplotlib
import matplotlib.cm as cm

from gps_utils import angle_between_pts
from timeline_utils import Segment, Timeline


def generate_starter_map(segments: list[Segment] | None,
                         lats: list[float] | None = None, lons: list[float] | None = None):
    # Store all the latitudes and longitudes
    if lats is None:
        lats = list(itertools.chain.from_iterable((i.start.lat, i.end.lat) for i in segments))
    if lons is None:
        lons = list(itertools.chain.from_iterable((i.start.lon, i.end.lon) for i in segments))

    return folium.Map(location=[(min(lats) + max(lats)) / 2,
                                (min(lons) + max(lons)) / 2],
                      zoom_start=10)


class ColorMap():
    def __init__(self, min: float, max: float, cmap: str = 'hsv'):
        self.norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        self.cmap = cm.get_cmap(cmap)

    def get(self, value):
        return matplotlib.colors.rgb2hex(self.cmap(self.norm(value))[:3])


def display_segments(segments: list[Segment]):
    m = generate_starter_map(segments)

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


def display_clustered_segments(segments: list[Segment], labels: list, centers: list = []):
    assert len(centers) == max(labels)

    m = generate_starter_map(segments)
    cmap = ColorMap(0, max(labels))

    for segment, label in zip(segments, labels):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=cmap.get(label),
            opacity=0.6
        ).add_to(m)

    for i in range(len(centers)):
        folium.Circle(
            [centers[i].lat, centers[i].lon],
            weight=10,
            color=cmap.get(i),
            opacity=1.0,
            radius=30
        ).add_to(m)

    return m


def display_walk_list(timeline: Timeline, m: folium.Map, color):
    folium.Marker(
        location=[timeline.start.lat, timeline.start.lon],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    for i, segment in enumerate(timeline.segments):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=color,
            opacity=0.6
        ).add_to(m)

        if i == 0:
            folium.RegularPolygonMarker(
                location=[segment.start.lat, segment.start.lon],
                number_of_sides=3,
                color='green',
                radius=15,
                rotation=angle_between_pts(segment.start, segment.end)
            ).add_to(m)

        folium.RegularPolygonMarker(
            location=[segment.start.lat, segment.start.lon],
            number_of_sides=3,
            color=color,
            radius=15,
            rotation=angle_between_pts(segment.start, segment.end)
        ).add_to(m)

    return m


def display_walk_lists(timelines: list[Timeline]):
    lats = list(itertools.chain.from_iterable(
        (i.start.lat, i.end.lat) for timeline in timelines for i in timeline.segments))
    lons = list(itertools.chain.from_iterable(
        (i.start.lon, i.end.lon) for timeline in timelines for i in timeline.segments))

    m = generate_starter_map(segments=None, lats=lats, lons=lons)
    cmap = ColorMap(0, len(timelines) - 1)

    for i, timeline in enumerate(timelines):
        m = display_walk_list(timeline, m, cmap.get(i))

    return m


def display_requests(requests: list[Segment]):
    m = generate_starter_map(requests)
    cmap = ColorMap(0, len(requests) - 1, cmap='RdYlGn')

    for i, segment in enumerate(requests):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=cmap.get(i),
            opacity=0.6
        ).add_to(m)

    return m
