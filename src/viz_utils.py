from __future__ import annotations

import csv
import itertools
import json
import math
import os
from decimal import Decimal
from random import randint
import sys
from typing import Optional

import folium
import humanhash
import matplotlib
from folium.features import DivIcon
from termcolor import colored
from tqdm import tqdm

from src.config import (
    BASE_DIR,
    DEPOT,
    STYLE_COLOR,
    USE_COST_METRIC,
    NodeType,
    Point,
    address_pts_file,
    blocks_file,
    blocks_file_t,
    generate_pt_id,
    node_coords_file,
    house_to_voters_file,
    turnout_predictions_file
)
from src.gps_utils import SubBlock


def generate_starter_map(
    blocks: Optional[blocks_file_t] = None,
    lats: Optional[list[float]] = None,
    lons: Optional[list[float]] = None,
) -> folium.Map:
    # Store all the latitudes and longitudes
    if lats is None and lons is None and blocks is not None:
        lats = list(
            itertools.chain.from_iterable(
                (i["nodes"][0]["lat"], i["nodes"][-1]["lat"]) for i in blocks.values()
            )
        )
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
        # zoom_start=30
    )


class ColorMap:
    def __init__(self, min: float, max: float, cmap: str = "hsv"):
        self.norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        self.cmap = matplotlib.colormaps[cmap]

    def get(self, value: float) -> str:
        """Get the hex code as a string from the value."""
        return matplotlib.colors.rgb2hex(self.cmap(self.norm(value))[:3])

    def get_reverse(self, value: float) -> str:
        """Get the opposite color for this value."""
        rgb = self.cmap(self.norm(value))[:3]
        return matplotlib.colors.rgb2hex((1 - rgb[0], 1 - rgb[1], 1 - rgb[2]))


def display_targeting_voters(voters):
    lats = [v.coords["lat"] for v in voters if v != -1]
    lons = [v.coords["lon"] for v in voters if v != -1]
    m = generate_starter_map(lats=lats, lons=lons)

    for voter in voters:
        if voter.coords["lat"] == -1 or voter.coords["lon"] == -1:
            continue
        # The text is the address on the first line, then each voter and their value on the next
        text = f"{voter.address_line_1}<br>"
        for voter_person in voter.people:
            text += f"{voter_person['name']}: <b>{round(voter_person['value'] * 100)}%</b><br>"
        folium.Circle(
            [voter.coords["lat"], voter.coords["lon"]],
            weight=10,
            color="#0F6BF5",
            opacity=1.0,
            radius=1,
            tooltip=text,
        ).add_to(m)

    m.fit_bounds(m.get_bounds())

    return m


def display_blocks(
    blocks: blocks_file_t,
) -> tuple[folium.Map, set[tuple[float, float]]]:
    # Seed the hash consistently for better visualization
    os.environ["PYTHONHASHSEED"] = "0"

    m = generate_starter_map(blocks)
    cmap = ColorMap(0, len(blocks.values()))

    num_houses_per_block = [len(b["addresses"]) for b in blocks.values()]
    min_houses, max_houses = min(num_houses_per_block), max(num_houses_per_block)
    num_duplicate_coords = 0

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

        for house_info in block["addresses"].values():
            lat = float(Decimal(house_info["lat"]).quantize(Decimal("0.000001")))
            lon = float(Decimal(house_info["lon"]).quantize(Decimal("0.000001")))

            if (lat, lon) in added_houses:
                num_duplicate_coords += 1
            else:
                added_houses.add((lat, lon))

            folium.Circle(
                [house_info["lat"], house_info["lon"]],
                weight=10,
                color=cmap.get(index),
                opacity=1.0,
                radius=1,
                tooltip="{}: {}, {}".format(house_info["display_address"], word, house_info["side"]),
            ).add_to(m)

    print(
        colored(
            "Of {} addresses, {} were duplicates (too close to render both).".format(
                sum(num_houses_per_block), num_duplicate_coords
            ),
            "yellow",
        )
    )
    print(
        "This is likely fine, since storefronts or other close buildings may be in the same location. Debug if necessary."
    )

    return m, added_houses


def display_blocks_and_unassociated(
    blocks: blocks_file_t, all_houses: list[Point]
) -> folium.Map:
    # Get the map with all the blocks
    m, added_pts = display_blocks(blocks)

    # Read the universe file and find the unassociated houses
    for pt in all_houses:
        lat = float(Decimal(pt["lat"]).quantize(Decimal("0.000001")))
        lon = float(Decimal(pt["lon"]).quantize(Decimal("0.000001")))
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


def display_distance_matrix(points: list[Point], distances_file: str) -> folium.Map:
    lats = [i["lat"] for i in points]
    lons = [i["lon"] for i in points]
    m = generate_starter_map(lats=lats, lons=lons)

    distances = json.load(open(distances_file))

    # Distances is a flattened matrix
    num_elements = math.sqrt(len(distances["distances"]))
    assert num_elements == int(num_elements)

    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            # Skip 9 out of every 10 elements randomly
            if randint(0, 19) != 0:
                continue
            if i == j:
                continue
            distance = distances["distances"][int(i * num_elements + j)]
            folium.PolyLine(
                [[p1["lat"], p1["lon"]], [p2["lat"], p2["lon"]]],
                weight=5,
                color="red",
                opacity=0.5,
                tooltip=distance,
            ).add_to(m)

    return m


def display_house_orders(
    walk_lists: list[list[Point]],
    cmap: Optional[ColorMap] = None,
    dcs: Optional[list[list[tuple[float, float]]]] = None,
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

                # # Display distance, cost text over the line
                # mid_point = [
                #     (points[0][0] + points[1][0]) / 2,
                #     (points[0][1] + points[1][1]) / 2,
                # ]
                # folium.Marker(
                #     location=mid_point,
                #     icon=DivIcon(
                #         icon_size=(25, 25),
                #         icon_anchor=(10, 10),  # left-right, up-down
                #         html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                #             text_color, display
                #         ),
                #     ),
                # ).add_to(m)
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
    # points = list(itertools.chain.from_iterable([s.houses for s in walk_list]))
    # lats = [i["lat"] for i in points]
    # lons = [i["lon"] for i in points]
    # m = generate_starter_map(lats=lats, lons=lons)

    house_to_voters = json.load(open(house_to_voters_file))
    turnout_predictions = json.load(open(turnout_predictions_file))

    m = folium.Map()

    house_counter = 0

    for sub_block in walk_list:
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in sub_block.navigation_points],
            weight=8,
            color=color,
            opacity=0.7,
        ).add_to(m)

        if len(sub_block.houses) == 0:
            continue

        display_num = ""
        tooltip = ""
        last_point = None
        for house, next_house in itertools.pairwise(sub_block.houses):

            voter_info = house_to_voters[house["id"]]

            house_counter += 1
            point = generate_pt_id(house)
            if point != last_point:
                display_num = str(house_counter)
                last_point = point

                # Add the address to the tooltip, but delete any unit number
                tooltip = voter_info["display_address"].split("Unit")[0]

            # Add the voter names to the tooltip
            for person in voter_info["voter_info"]:
                turnout: float = turnout_predictions[person["voter_id"]]
                tooltip += f"<br>{person['name'].title()} ({person['party']}): <b>{round(turnout * 100)}%</b>"

            if point == generate_pt_id(next_house):
                continue

            if display_num != str(house_counter):
                display_num += f"-{str(house_counter)}"

            folium.Marker(
                location=[house["lat"], house["lon"]],
                # On hover, the icon displays the address
                icon=DivIcon(
                    icon_size=(60, 30),
                    icon_anchor=(10, 12),  # left-right, up-down
                    html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                        color, display_num
                    ),
                ),
                tooltip=tooltip,
            ).add_to(m)

        house_counter += 1

        # Add the last house
        voter_info = house_to_voters[sub_block.houses[-1]["id"]]
        point = generate_pt_id(sub_block.houses[-1])
        if point != last_point:
            display_num = str(house_counter)
            last_point = point

            # Add the address to the tooltip
            tooltip = voter_info["display_address"]

        # Add the voter names to the tooltip
        for person in voter_info["voter_info"]:
            turnout: float = float(turnout_predictions[person["voter_id"]])
            tooltip += f"<br>{person['name'].title()} ({person['party']}): <b>{round(turnout * 100)}%</b>"

        if display_num != str(house_counter):
            if display_num == "":
                display_num = str(house_counter)
            else:
                display_num += f"-{str(house_counter)}"
        if len(sub_block.houses) == 0:
            continue
        folium.Marker(
            location=[sub_block.houses[-1]["lat"], sub_block.houses[-1]["lon"]],
            # On hover, the icon displays the address
            icon=DivIcon(
                icon_size=(60, 30),
                icon_anchor=(10, 12),  # left-right, up-down
                html='<div style="font-size: 15pt; color:{}">{}</div>'.format(
                    color, display_num
                ),
            ),
            tooltip=tooltip,
        ).add_to(m)

    # Fit the map to the bounds
    m.fit_bounds(m.get_bounds())

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
    # cmap = ColorMap(0, len(walk_lists) - 1, cmap="tab10")

    maps = []
    for walk_list in walk_lists:
        maps.append(display_walk_list(walk_list, STYLE_COLOR))

    return maps


if __name__ == "__main__":
    # Load the file (unorganized) containing house coordinates (and info)
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
    for row in tqdm(
        house_points_reader, total=num_houses, colour="green", desc="Loading houses"
    ):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            continue
        all_houses.append(
            Point(
                lat=float(Decimal(lat).quantize(Decimal("0.000001"))),
                lon=float(Decimal(lon).quantize(Decimal("0.000001"))),
                id=row["full_address"],
                type=NodeType.house,
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
