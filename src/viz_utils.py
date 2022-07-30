from __future__ import annotations

import itertools
from typing import Optional

import folium
import matplotlib
import matplotlib.cm as cm
from folium.features import DivIcon

from src.gps_utils import Point
from src.timeline_utils import Segment


def generate_starter_map(segments: Optional[list[Segment]] = None,
                         lats: Optional[list[float]] = None, lons: Optional[list[float]] = None) -> folium.Map:
    # Store all the latitudes and longitudes
    if lats is None and lons is None and segments is not None:
        lats = list(itertools.chain.from_iterable((i.start.lat, i.end.lat) for i in segments))
        lons = list(itertools.chain.from_iterable((i.start.lon, i.end.lon) for i in segments))
    # Check if it's not the only other possible correct arguments
    elif not ((lats is not None and lons is not None) and segments is None):
        raise RuntimeError('Must pass either \'lats\' and \'lons\' argument, or \'segments\', not both or neither')

    if len(lats) == 0:
        raise RuntimeError(
            'Unable to create map with no coordinates. Passed in data had no points.')

    return folium.Map(location=[(min(lats) + max(lats)) / 2,
                                (min(lons) + max(lons)) / 2],
                      zoom_start=10)


class ColorMap():
    def __init__(self, min: float, max: float, cmap: str = 'hsv'):
        self.norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        self.cmap = cm.get_cmap(cmap)

    def get(self, value: float) -> str:
        '''Get the hex code as a string from the value'''
        return matplotlib.colors.rgb2hex(self.cmap(self.norm(value))[:3])

    def get_reverse(self, value: float) -> str:
        '''Get the opposite color for this value'''
        rgb = self.cmap(self.norm(value))[:3]
        return matplotlib.colors.rgb2hex((1 - rgb[0], 1 - rgb[1], 1 - rgb[2]))


def display_segments(segments: list[Segment]) -> folium.Map:
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


def display_clustered_segments(segments: list[Segment],
                               labels: list[int], centers: Optional[list[Point]]) -> folium.Map:
    m = generate_starter_map(segments)
    cmap = ColorMap(0, max(labels))

    for segment, label in zip(segments, labels):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=cmap.get(label),
            opacity=0.6
        ).add_to(m)

    if centers:
        assert len(centers) == max(labels)
        for i in range(len(centers)):
            folium.Circle(
                [centers[i].lat, centers[i].lon],
                weight=10,
                color=cmap.get(i),
                opacity=1.0,
                radius=30
            ).add_to(m)
            folium.Marker(
                location=[centers[i].lat, centers[i].lon],
                icon=DivIcon(
                    icon_size=(50, 50),
                    icon_anchor=(5, 12),  # left-right, up-down
                    html='<div style="font-size: 15pt">{}</div>'.format(i)
                )
            ).add_to(m)

    return m


def display_house_orders(walk_lists: list[list[Point]]) -> folium.Map:
    lats = [i.lat for walk_list in walk_lists for i in walk_list]
    lons = [i.lon for walk_list in walk_lists for i in walk_list]
    m = generate_starter_map(lats=lats, lons=lons)

    cmap = ColorMap(0, len(walk_lists) - 1, cmap='RdYlGn')

    for i, walk_list in enumerate(walk_lists):
        text_color = cmap.get(i)
        for j, house in enumerate(walk_list[:-1]):
            folium.Marker(
                location=[house.lat, house.lon],
                icon=DivIcon(
                    icon_size=(25, 25),
                    icon_anchor=(10, 10),  # left-right, up-down
                    html='<div style="font-size: 15pt; color:{}">{}</div>'.format(text_color, j)
                )
            ).add_to(m)

    return m
