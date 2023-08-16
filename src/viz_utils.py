from __future__ import annotations

import csv
import itertools
import json
import os
from decimal import Decimal
from random import randint
from typing import Optional

import folium
import humanhash
import matplotlib
import matplotlib.cm as cm
from folium.features import DivIcon
from termcolor import colored
from tqdm import tqdm

from src.config import (
    BASE_DIR,
    DEPOT,
    USE_COST_METRIC,
    Point,
    address_pts_file,
    blocks_file,
    blocks_file_t,
)
from src.gps_utils import SubBlock


def generate_starter_map(
    blocks: Optional[blocks_file_t] = None,
    lats: Optional[list[float]] = None,
    lons: Optional[list[float]] = None,
) -> folium.Map:
    # Store all the latitudes and longitudes
    if lats is None and lons is None and blocks is not None:
        for (key, value) in blocks.items():
            if len(value["nodes"]) < 1:
                print(key, value)
        try:
            lats = list(
                itertools.chain.from_iterable(
                    (i["nodes"][0]["lat"], i["nodes"][-1]["lat"]) for i in blocks.values()
                )
            )
        except:
            print("wtf")

        lons = list(
            itertools.chain.from_iterable(
                (i["nodes"][0]["lon"], i["nodes"][-1]["lon"]) for i in blocks.values()
            )
        )
    # Check if it's not the only other possible correct arguments
    elif not ((lats is not None and lons is not None) and blocks is None):
        raise RuntimeError(
            "Must pass either 'lats' and 'lons' argument, or 'segments', not both or neither"
        )

    if len(lats) == 0:
        raise RuntimeError(
            "Unable to create map with no coordinates. Passed in data had no points."
        )

    return folium.Map(
        location=[(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2],
        zoom_start=30
    )


class ColorMap:
    def __init__(self, min: float, max: float, cmap: str = "hsv"):
        self.norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        self.cmap = cm.get_cmap(cmap)

    def get(self, value: float) -> str:
        """Get the hex code as a string from the value"""
        return matplotlib.colors.rgb2hex(self.cmap(self.norm(value))[:3])

    def get_reverse(self, value: float) -> str:
        """Get the opposite color for this value"""
        rgb = self.cmap(self.norm(value))[:3]
        return matplotlib.colors.rgb2hex((1 - rgb[0], 1 - rgb[1], 1 - rgb[2]))


def display_blocks(
    blocks: blocks_file_t, track_added: bool = False
) -> folium.Map | tuple[folium.Map, set[tuple[float, float]]]:
    # Seed the hash consistently for better visualization
    os.environ["PYTHONHASHSEED"] = "0"

    m = generate_starter_map(blocks)
    cmap = ColorMap(0, len(blocks.values()))

    num_houses_per_block = [len(b["addresses"]) for b in blocks.values()]
    min_houses, max_houses = min(num_houses_per_block), max(num_houses_per_block)

    added_houses: set[tuple[float, float]] = set()

    for b_id, block in blocks.items():
        index = randint(0, len(blocks))
        word = humanhash.humanize(str(hash(b_id)), words=1)
        weight = (
            4 + ((len(block["addresses"]) - min_houses) / (max_houses - min_houses)) * 8
        )
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in block["nodes"]],
            weight=weight,
            color=cmap.get(index),
            opacity=0.6,
            tooltip=word,
        ).add_to(m)

        for address, house in block["addresses"].items():
            lat = float(Decimal(house["lat"]).quantize(Decimal("0.00001")))
            lon = float(Decimal(house["lon"]).quantize(Decimal("0.00001")))
            if (lat, lon) in added_houses:
                print(colored("Duplicate house found: {}".format((lat, lon)), "red"))
            else:
                added_houses.add((lat, lon))

            folium.Circle(
                [house["lat"], house["lon"]],
                weight=10,
                color=cmap.get(index),
                opacity=1.0,
                radius=1,
                tooltip="{}: {}".format(address, word),
            ).add_to(m)

    return m if not track_added else (m, added_houses)


def display_blocks_and_unassociated(
    blocks: blocks_file_t, all_houses: list[Point]
) -> folium.Map:
    # Get the map with all the blocks
    m, added_pts = display_blocks(blocks, track_added=True)

    # Read the universe file and find the unassociated houses
    for pt in all_houses:
        lat = float(Decimal(pt["lat"]).quantize(Decimal("0.00001")))
        lon = float(Decimal(pt["lon"]).quantize(Decimal("0.00001")))
        if (lat, lon) not in added_pts:
            folium.Marker(
                location=[pt["lat"], pt["lon"]],
                icon=folium.Icon(color="red", icon="home", prefix="fa"),
                tooltip=pt["id"],
            ).add_to(m)

    return m


def display_clustered_blocks(
    blocks: blocks_file_t, labels: list[int], centers: Optional[list[Point]]
) -> folium.Map:
    m = generate_starter_map(blocks)
    cmap = ColorMap(0, max(labels))

    for block, label in zip(blocks.values(), labels):
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in block["nodes"]],
            weight=8,
            color=cmap.get(label),
            opacity=0.6,
        ).add_to(m)

    if centers:
        assert len(centers) == max(labels)
        for i in range(len(centers)):
            folium.Circle(
                [centers[i]["lat"], centers[i]["lon"]],
                weight=10,
                color=cmap.get(i),
                opacity=1.0,
                radius=30,
            ).add_to(m)
            folium.Marker(
                location=[centers[i]["lat"], centers[i]["lon"]],
                icon=DivIcon(
                    icon_size=(50, 50),
                    icon_anchor=(5, 12),  # left-right, up-down
                    html='<div style="font-size: 15pt">{}</div>'.format(i),
                ),
            ).add_to(m)

    return m


def display_house_orders(
    walk_lists: list[list[Point]],
    cmap: Optional[ColorMap] = None,
    dcs: Optional[list[list[Optional[tuple[float, float]]]]] = None,
) -> folium.Map:
    """
    Display the house orders in a map.

    Parameters
    ----------
        walk_lists: A list of lists of houses. Each list is a walk, and each house is a dictionary with 'lat' and 'lon' keys.
        cmap: A colormap to use for the blocks. If None, a new one will be created.
        dcs: A list of lists of tuples of (distance, cost) for each house. If None, no DCs will be displayed.

    Returns
    -------
        A folium map with the house orders displayed.
    """
    lats = [i["lat"] for walk_list in walk_lists for i in walk_list]
    lons = [i["lon"] for walk_list in walk_lists for i in walk_list]
    m = generate_starter_map(lats=lats, lons=lons)

    if cmap is None:
        cmap = ColorMap(0, len(walk_lists) - 1, cmap="tab10")

    for i, walk_list in enumerate(walk_lists):
        text_color = cmap.get(i)
        for j, house in enumerate(walk_list[:-1]):
            folium.Marker(
                location=[house["lat"], house["lon"]],
                icon=DivIcon(
                    icon_size=(25, 25),
                    icon_anchor=(10, 10),  # left-right, up-down
                    html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                        text_color, j
                    ),
                ),
            ).add_to(m)

        if dcs is not None:
            for j, dc in enumerate(dcs[i]):
                points = [
                    [walk_list[j]["lat"], walk_list[j]["lon"]],
                    [walk_list[j + 1]["lat"], walk_list[j + 1]["lon"]],
                ]
                if USE_COST_METRIC:
                    display = "None" if dc is None else "{}, {}".format(dc[0], dc[1])
                else:
                    display = "None" if dc is None else "{}".format(dc[0])
                folium.PolyLine(
                    points, color=text_color, weight=5, opacity=0.5, tooltip=display
                ).add_to(m)

                # Display distance, cost text over the line
                mid_point = [
                    (points[0][0] + points[1][0]) / 2,
                    (points[0][1] + points[1][1]) / 2,
                ]
                folium.Marker(
                    location=mid_point,
                    icon=DivIcon(
                        icon_size=(25, 25),
                        icon_anchor=(10, 10),  # left-right, up-down
                        html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                            text_color, display
                        ),
                    ),
                ).add_to(m)
    return m


def display_walk_list(walk_list: list[SubBlock], color: str) -> folium.Map:
    """
    Display a walk list on a map.

    Parameters
    ----------
        walk_list (list[SubBlock]): The walk list to display
        color (str): The color to use for the walk list

    Returns
    -------
        folium.Map: The map with the walk list displayed
    """
    points = list(itertools.chain.from_iterable([s.houses for s in walk_list]))
    lats = [i["lat"] for i in points]
    lons = [i["lon"] for i in points]
    m = generate_starter_map(lats=lats, lons=lons)

    house_counter = 0

    for sub_block in walk_list:
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in sub_block.navigation_points],
            weight=8,
            color=color,
            opacity=0.7,
        ).add_to(m)
        for house in sub_block.houses:
            folium.Marker(
                location=[house["lat"], house["lon"]],
                # On hover, the icon displays the address
                icon=DivIcon(
                    icon_size=(25, 25),
                    icon_anchor=(10, 12),  # left-right, up-down
                    html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                        color, house_counter
                    ),
                ),
                tooltip=house["id"],
            ).add_to(m)
            house_counter += 1

    return m


def display_individual_walk_lists(walk_lists: list[list[SubBlock]]) -> list[folium.Map]:
    """
    Display a list of walk lists, each in its own map.

    Parameters
    ----------
        walk_lists (list[list[SubBlock]]): A list of walk lists

    Returns
    -------
        list[folium.Map]: A map with all the walk lists
    """
    cmap = ColorMap(0, len(walk_lists) - 1, cmap="tab10")

    maps = []
    for i, walk_list in enumerate(walk_lists):
        maps.append(display_walk_list(walk_list, cmap.get(i)))

    return maps


def display_walk_lists(walk_lists: list[list[SubBlock]]) -> folium.Map:
    """
    Display a list of walk lists, all in one map.

    Parameters
    ----------
        walk_lists (list[list[SubBlock]]): A list of walk lists

    Returns
    -------
        folium.Map: A map with all the walk lists
    """
    points = [
        list(itertools.chain.from_iterable([s.houses for s in walk_list]))
        for walk_list in walk_lists
    ]
    points = list(itertools.chain.from_iterable(points))

    lats = [i["lat"] for walk_list in walk_lists for i in walk_list for i in i.houses]
    lons = [i["lon"] for walk_list in walk_lists for i in walk_list for i in i.houses]
    m = generate_starter_map(lats=lats, lons=lons)

    cmap = ColorMap(0, len(walk_lists) - 1, cmap="tab10")

    for i, walk_list in enumerate(walk_lists):
        house_counter = 1
        color = cmap.get(i)
        for sub_block in walk_list:
            folium.PolyLine(
                [[p["lat"], p["lon"]] for p in sub_block.navigation_points],
                weight=6,
                color=color,
                opacity=0.7,
            ).add_to(m)
            for house in sub_block.houses:
                # Make a circle
                folium.Circle(
                    location=[house["lat"], house["lon"]],
                    radius=11,
                    weight=1,
                    color=color,
                    fill=False,
                ).add_to(m)
                folium.Marker(
                    location=[house["lat"], house["lon"]],
                    icon=DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(3, 5),  # left-right, up-down
                        html='<div style="font-size: 6pt; color:{}">{}</div>'.format(
                            color, house_counter
                        ),
                    ),
                ).add_to(m)

                house_counter += 1

    # Place the depot
    folium.Circle(
        location=[DEPOT["lat"], DEPOT["lon"]],
        radius=11,
        color="red",
        fillOpacity=1,
        fill=True,
    ).add_to(m)

    return m


if __name__ == "__main__":
    # Load the file (unorganized) containing house coordinates (and info)
    print("Loading coordinates of houses...")
    house_points_file = open(address_pts_file)
    num_houses = -1
    for _ in house_points_file:
        num_houses += 1
    house_points_file.seek(0)
    house_points_reader = csv.DictReader(house_points_file)

    min_lat, min_lon, max_lat, max_lon = (
        40.5147085,
        -80.2215597,
        40.6199697,
        -80.0632736,
    )

    all_houses: list[Point] = []
    for row in tqdm(house_points_reader, total=num_houses):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            continue
        all_houses.append(
            Point(
                lat=float(Decimal(lat).quantize(Decimal("0.00001"))),
                lon=float(Decimal(lon).quantize(Decimal("0.00001"))),
                id=row["full_address"],
            )
        )

    all_blocks: blocks_file_t = json.load(open(blocks_file))

    # Visualize the pre-processing association
    display_blocks_and_unassociated(all_blocks, all_houses).save(
        os.path.join(BASE_DIR, "viz", "blocks_and_unassociated.html")
    )

    # Visualize the pre-processing association
    # display_blocks(all_blocks).save(os.path.join(BASE_DIR, "viz", "blocks.html"))
    # print(colored("Saved blocks.html", "green"))
