from __future__ import annotations

import itertools
from statistics import mean
from typing import Optional

import folium
import matplotlib
import matplotlib.cm as cm
from folium.features import DivIcon
from PIL import Image, ImageDraw, ImageFont

from gps_utils import Point, angle_between_pts, middle
from timeline_utils import Segment, Timeline


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
            'Unable to create map with no coordinates. Timeline population likely produced empty timelines.')

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
        return matplotlib.colors.rgb2hex([1 - rgb[0], 1 - rgb[1], 1 - rgb[2]])


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

    return m


def display_walk_list(timeline: Timeline, id: int, m: folium.Map, color: str) -> folium.Map:
    folium.Marker(
        location=[timeline.start.lat, timeline.start.lon],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    if len(timeline.segments) != 0:
        folium.Marker(
            location=[timeline.start.lat, timeline.start.lon],
            icon=DivIcon(
                icon_size=(50, 50),
                icon_anchor=(5, 40),  # left-right, up-down
                html='<div style="font-size: 15pt; color:red">{}</div>'.format(id)
            )
        ).add_to(m)

    for i, segment in enumerate(timeline.segments):
        folium.PolyLine(
            [[p.lat, p.lon] for p in segment.all_points],
            weight=8,
            color=color,
            opacity=0.6
        ).add_to(m)

        mid = middle(segment.start, segment.end)
        folium.Marker(
            location=[mid.lat, mid.lon],
            icon=DivIcon(
                icon_size=(50, 50),
                icon_anchor=(5, 12),  # left-right, up-down
                html='<div style="font-size: 15pt">{}</div>'.format(i + 1)
            )
        ).add_to(m)

        folium.RegularPolygonMarker(
            location=[segment.start.lat, segment.start.lon],
            number_of_sides=3,
            color=color,
            radius=15,
            rotation=angle_between_pts(segment.start, segment.all_points[1])
        ).add_to(m)

    return m


def display_walk_lists(timelines: list[Timeline]) -> folium.Map:
    lats = list(itertools.chain.from_iterable(
        (i.start.lat, i.end.lat) for timeline in timelines for i in timeline.segments))
    lons = list(itertools.chain.from_iterable(
        (i.start.lon, i.end.lon) for timeline in timelines for i in timeline.segments))

    m = generate_starter_map(lats=lats, lons=lons)
    cmap = ColorMap(0, len(timelines) - 1)

    for i in range(len(timelines)):
        m = display_walk_list(timelines[i], i, m, cmap.get(i))

    return m


def display_requests(requests: list[Segment]) -> folium.Map:
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


def generate_timelines(timelines: list[Timeline]):
    HEIGHT_MARGIN = 30
    BETWEEN_MARGIN = 10
    ROW_WIDTH = 30
    WIDTH_MARGIN = 30
    MIN_TO_PIX = 5

    width = WIDTH_MARGIN * 2 + MIN_TO_PIX * 180 + 20
    height = HEIGHT_MARGIN * 2 + len(timelines) * (ROW_WIDTH + BETWEEN_MARGIN)

    cmap = ColorMap(0, len(timelines) - 1)

    out = Image.new('RGB', (width, height), (255, 255, 255))
    drawer = ImageDraw.Draw(out)

    TIMELINE_START = WIDTH_MARGIN + 10

    for i, timeline in enumerate(timelines):
        report = timeline.generate_report()
        insertion_order = [i['insertion_order'] for i in report.values()]

        row_top = HEIGHT_MARGIN + i * BETWEEN_MARGIN + i * ROW_WIDTH
        row_bottom = row_top + ROW_WIDTH
        drawer.line([(WIDTH_MARGIN, row_top), (WIDTH_MARGIN, row_bottom)], fill=(50, 50, 50), width=5)
        drawer.line([(width - WIDTH_MARGIN, row_top), (width - WIDTH_MARGIN, row_bottom)], fill=(50, 50, 50), width=5)

        segment_times, delta_times = timeline.get_segment_times()
        for route_order, segment_time, delta_time, order in \
                zip(range(len(segment_times)), segment_times, delta_times, insertion_order):
            drawer.rectangle([(TIMELINE_START + segment_time[0] * MIN_TO_PIX, row_top + 13),
                              (TIMELINE_START + segment_time[1] * MIN_TO_PIX, row_bottom)], fill=cmap.get(i))

            drawer.line([(TIMELINE_START + mean(delta_time) * MIN_TO_PIX, row_top + 13),
                         (TIMELINE_START + mean(delta_time) * MIN_TO_PIX, row_bottom)], fill=(255, 255, 255), width=10)

            drawer.regular_polygon(bounding_circle=(TIMELINE_START + mean(delta_time) * MIN_TO_PIX, row_bottom - 4, 5),
                                   n_sides=3, fill=(0, 0, 0), outline=(0, 0, 0))

            drawer.text((TIMELINE_START + mean(segment_time) * MIN_TO_PIX - 1, row_top + 3),
                        str(order), fill=(0, 0, 0), font=ImageFont.load_default())

            drawer.text((TIMELINE_START + mean(segment_time) * MIN_TO_PIX - 1, row_top + 18),
                        str(route_order + 1), fill=cmap.get_reverse(i), font=ImageFont.load_default())

            drawer.text((13, row_top + 15), str(i), fill=(0, 0, 0), font=ImageFont.load_default())

    out.show()
