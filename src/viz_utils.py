from __future__ import annotations

import itertools
from typing import Optional

import folium
import matplotlib
import matplotlib.cm as cm
from folium.features import DivIcon

from src.config import blocks_file_t, Point
from src.gps_utils import SubBlock


def generate_starter_map(blocks: Optional[blocks_file_t] = None,
                         lats: Optional[list[float]] = None, lons: Optional[list[float]] = None) -> folium.Map:
    # Store all the latitudes and longitudes
    if lats is None and lons is None and blocks is not None:
        lats = list(itertools.chain.from_iterable(
            (i['nodes'][0]['lat'], i['nodes'][-1]['lat']) for i in blocks.values()))
        lons = list(itertools.chain.from_iterable(
            (i['nodes'][0]['lon'], i['nodes'][-1]['lon']) for i in blocks.values()))
    # Check if it's not the only other possible correct arguments
    elif not ((lats is not None and lons is not None) and blocks is None):
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


def display_blocks(blocks: blocks_file_t) -> folium.Map:
    m = generate_starter_map(blocks)

    num_houses_per_block = [len(b['addresses']) for b in blocks.values()]
    min_houses, max_houses = min(num_houses_per_block), max(num_houses_per_block)

    for block in blocks.values():
        weight = 4 + ((len(block['addresses']) - min_houses) / (max_houses - min_houses)) * 8
        folium.PolyLine(
            [[p['lat'], p['lon']] for p in block['nodes']],
            weight=weight,
            color='blue',
            opacity=0.6
        ).add_to(m)

    return m


def display_clustered_blocks(blocks: blocks_file_t,
                             labels: list[int], centers: Optional[list[Point]]) -> folium.Map:
    m = generate_starter_map(blocks)
    cmap = ColorMap(0, max(labels))

    for block, label in zip(blocks.values(), labels):
        folium.PolyLine(
            [[p['lat'], p['lon']] for p in block['nodes']],
            weight=8,
            color=cmap.get(label),
            opacity=0.6
        ).add_to(m)

    if centers:
        assert len(centers) == max(labels)
        for i in range(len(centers)):
            folium.Circle(
                [centers[i]['lat'], centers[i]['lon']],
                weight=10,
                color=cmap.get(i),
                opacity=1.0,
                radius=30
            ).add_to(m)
            folium.Marker(
                location=[centers[i]['lat'], centers[i]['lon']],
                icon=DivIcon(
                    icon_size=(50, 50),
                    icon_anchor=(5, 12),  # left-right, up-down
                    html='<div style="font-size: 15pt">{}</div>'.format(i)
                )
            ).add_to(m)

    return m


def display_house_orders(walk_lists: list[list[Point]], cmap: Optional[ColorMap] = None, dcs: Optional[list[list[Optional[tuple[float, float]]]]] = None) -> folium.Map:
    lats = [i['lat'] for walk_list in walk_lists for i in walk_list]
    lons = [i['lon'] for walk_list in walk_lists for i in walk_list]
    m = generate_starter_map(lats=lats, lons=lons)

    if cmap is None:
        cmap = ColorMap(0, len(walk_lists) - 1, cmap='tab10')

    for i, walk_list in enumerate(walk_lists):
        text_color = cmap.get(i)
        for j, house in enumerate(walk_list[:-1]):
            folium.Marker(
                location=[house['lat'], house['lon']],
                icon=DivIcon(
                    icon_size=(25, 25),
                    icon_anchor=(10, 10),  # left-right, up-down
                    html='<div style="font-size: 15pt; color:{}">{}</div>'.format(text_color, j)
                )
            ).add_to(m)
        
        if dcs is not None:
            for j, dc in enumerate(dcs[i]):
                points = [[walk_list[j]['lat'], walk_list[j]['lon']],
                          [walk_list[j + 1]['lat'], walk_list[j + 1]['lon']]]
                display = "None" if dc is None else "{}, {}".format(dc[0], dc[1])
                folium.PolyLine(points, color=text_color, weight=10, opacity=0.5, tooltip=display).add_to(m)
    return m


def display_walk_lists(walk_lists: list[list[SubBlock]]) -> list[folium.Map]:
    points = [list(itertools.chain.from_iterable([s.houses for s in walk_list])) for walk_list in walk_lists]
    cmap = ColorMap(0, len(walk_lists) - 1, cmap='RdYlGn')

    list_visualizations: list[folium.Map] = []

    for i, walk_list in enumerate(walk_lists):
        m = display_house_orders([points[i]], cmap=cmap)

        for subsegment in walk_list:
            folium.PolyLine(
                [[p['lat'], p['lon']] for p in subsegment.navigation_points],
                weight=8,
                color='#212121',
                opacity=0.7
            ).add_to(m)

        list_visualizations.append(m)

    return list_visualizations
