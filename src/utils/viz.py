from __future__ import annotations

import itertools
import json
import math
import os
from decimal import Decimal
from random import randint
from typing import Optional

import folium
import humanhash
import matplotlib
from folium.features import DivIcon
from nptyping import NDArray
from termcolor import colored
from tqdm import tqdm

from src.config import (
    BASE_DIR,
    BLOCK_DB_IDX,
    ABODE_DB_IDX,
    STYLE_COLOR,
    USE_COST_METRIC,
    VOTER_DB_IDX,
    Voter,
    Abode,
    InternalPoint,
    blocks_file_t,
    generate_pt_id,
    turnout_predictions_file,
)
from src.utils.db import Database
from src.utils.gps import SubBlock


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
    m = folium.Map()

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


def display_custom_area(depot: InternalPoint, abodes: list[InternalPoint]):
    m = folium.Map()

    folium.Circle(
        [depot["lat"], depot["lon"]],
        weight=10,
        color="#0F6BF5",
        opacity=1.0,
        radius=1,
        tooltip="Depot",
    ).add_to(m)

    for abode in abodes:
        folium.Circle(
            [abode["lat"], abode["lon"]],
            weight=10,
            color="#000000",
            opacity=1.0,
            radius=1,
            tooltip=abode["id"],
        ).add_to(m)

    m.fit_bounds(m.get_bounds())

    m.save(os.path.join(BASE_DIR, "viz", "custom_area.html"))


def display_blocks() -> set[tuple[float, float]]:
    # Seed the hash consistently for better visualization
    os.environ["PYTHONHASHSEED"] = "0"

    # Retrieve the blocks from the database
    db = Database()
    blocks = db.get_multiple_dict(db.get_keys(BLOCK_DB_IDX), BLOCK_DB_IDX)

    m = folium.Map()
    cmap = ColorMap(0, len(blocks.values()))

    num_houses_per_block = [len(b["abodes"]) for b in blocks.values()]
    min_houses, max_houses = min(num_houses_per_block), max(num_houses_per_block)
    num_duplicate_coords = 0

    added_houses: set[tuple[float, float]] = set()

    for b_id, block in tqdm(blocks.items(), desc="Displaying blocks", unit="blocks", colour="green"):
        index = randint(0, len(blocks))
        word = humanhash.humanize(str(hash(b_id)), words=1)
        weight = (
            4 + ((len(block["abodes"]) - min_houses) / (max_houses - min_houses)) * 8
        )
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in block["nodes"]],
            weight=weight,
            color=cmap.get(index),
            opacity=0.6,
            tooltip=word,
        ).add_to(m)

        for abode_geography in block["abodes"].values():
            lat = float(Decimal(abode_geography["point"]["lat"]).quantize(Decimal("0.000001")))
            lon = float(Decimal(abode_geography["point"]["lon"]).quantize(Decimal("0.000001")))

            abode: Abode = db.get_dict(abode_geography["id"], ABODE_DB_IDX)

            if (lat, lon) in added_houses:
                num_duplicate_coords += 1
            else:
                added_houses.add((lat, lon))

            folium.Circle(
                [abode_geography["point"]["lat"], abode_geography["point"]["lon"]],
                weight=10,
                color=cmap.get(index),
                opacity=1.0,
                radius=1,
                tooltip="{}: {}, {}".format(
                    abode["display_address"], word, abode_geography["side"]
                ),
            ).add_to(m)

    print(
        colored(
            "Of {} houses, {} were duplicates (too close to render both).".format(
                sum(num_houses_per_block), num_duplicate_coords
            ),
            "yellow",
        ) +
        "This is likely fine, since storefronts or other close buildings may be in the same location. Debug if necessary."
    )

    m.fit_bounds(m.get_bounds())

    m.save(os.path.join(BASE_DIR, "viz", "blocks.html"))

    return added_houses


def display_visited_and_unvisited(
    visited: list[InternalPoint],
    unvisited: list[InternalPoint],
    tooltips: dict[str, str],
) -> folium.Map:
    m = folium.Map()

    for pt in visited:
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            icon=folium.Icon(color="green", icon="check", prefix="fa"),
            tooltip=tooltips[pt["id"]],
        ).add_to(m)

    for pt in unvisited:
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            icon=folium.Icon(color="red", icon="times", prefix="fa"),
            tooltip=tooltips[pt["id"]],
        ).add_to(m)

    m.fit_bounds(m.get_bounds())

    return m


def display_blocks_and_unassociated(
    blocks: blocks_file_t, all_houses: list[InternalPoint]
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


def display_clustered_points(points: list[InternalPoint], labels: list[int]):
    m = folium.Map()

    cmap = ColorMap(0, max(labels))

    for point, label in zip(points, labels):
        folium.Circle(
            [point["lat"], point["lon"]],
            weight=10,
            color=cmap.get(label),
            opacity=1.0,
            radius=1,
        ).add_to(m)

    m.fit_bounds(m.get_bounds())

    m.save(os.path.join(BASE_DIR, "viz", "clusters_points.html"))


def display_clustered_blocks(
    block_ids: list[str],
    labels: list[int],
    centers: Optional[list[InternalPoint]] = None,
):
    # Retrieve the blocks from the database
    db = Database()
    blocks = db.get_multiple_dict(block_ids, BLOCK_DB_IDX)
    m = folium.Map()
    cmap = ColorMap(0, max(labels))

    for block, label in zip(blocks.values(), labels):
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in block["nodes"]],
            weight=8,
            color=cmap.get(label),
            opacity=0.6,
            tooltip=label,
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

    m.fit_bounds(m.get_bounds())

    m.save(os.path.join(BASE_DIR, "viz", "clusters.html"))


def display_distance_matrix_from_file(
    points: list[InternalPoint], distances_file: str
) -> folium.Map:
    m = folium.Map()

    distances = json.load(open(distances_file))

    # Distances is a flattened matrix
    num_elements = math.sqrt(len(distances["distances"]))
    assert num_elements == int(num_elements)

    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            # Skip most of the lines
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

    m.fit_bounds(m.get_bounds())

    return m


def display_distance_matrix(points: list[InternalPoint], distances: NDArray):
    m = folium.Map()

    for i, p1 in enumerate(tqdm(points, desc="Displaying distance matrix", unit="points", colour="green")):
        for j, p2 in enumerate(points):
            # Skip most of the lines
            if randint(0, 9999) != 0:
                continue
            if i == j:
                continue
            distance = distances[i][j]
            folium.PolyLine(
                [[p1["lat"], p1["lon"]], [p2["lat"], p2["lon"]]],
                weight=5,
                color="red",
                opacity=0.5,
                tooltip=distance,
            ).add_to(m)

    m.fit_bounds(m.get_bounds())

    m.save(os.path.join(BASE_DIR, "viz", "distance_matrix.html"))


def display_house_orders(
    walk_lists: list[list[InternalPoint]],
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
    m = folium.Map()

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

    m.fit_bounds(m.get_bounds())

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
    db = Database()
    turnout_predictions = json.load(open(turnout_predictions_file))

    m = folium.Map()

    house_counter = 0

    for sub_block in walk_list:
        folium.PolyLine(
            [[p["lat"], p["lon"]] for p in sub_block.nodes],
            weight=8,
            color=color,
            opacity=0.7,
        ).add_to(m)

        if len(sub_block.abodes) == 0:
            continue

        display_num = ""
        tooltip = ""
        last_point = None
        for abode, next_abode in itertools.pairwise(sub_block.abodes):
            if isinstance(abode["voter_ids"], list):
                voters: list[Voter] = [
                    db.get_dict(v, VOTER_DB_IDX) for v in abode["voter_ids"]
                ]
            else:
                # TODO: Deal with apartments
                voters: list[Voter] = []

            house_counter += 1
            point_id = generate_pt_id(abode["point"])
            if point_id != last_point:
                display_num = str(house_counter)
                last_point = point_id

                # Add the address to the tooltip, but delete any unit number
                tooltip = abode["display_address"].split("Unit")[0]

            # Add the voter names to the tooltip
            for voter in voters:
                turnout: float = turnout_predictions[voter["id"]]
                tooltip += f"<br>{voter['name'].title()} ({voter['party']}): <b>{round(turnout * 100)}%</b>"

            if point_id == generate_pt_id(next_abode["point"]):
                continue

            if display_num != str(house_counter):
                display_num += f"-{str(house_counter)}"

            folium.Marker(
                location=[abode["point"]["lat"], abode["point"]["lon"]],
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
        abode: Abode = db.get_dict(sub_block.abodes[-1]["abode_id"], ABODE_DB_IDX)
        if isinstance(abode["voter_ids"], list):
            voters: list[Voter] = [
                db.get_dict(v, VOTER_DB_IDX) for v in abode["voter_ids"]
            ]
        else:
            # TODO: Deal with apartments
            voters: list[Voter] = []

        point_id: str = generate_pt_id(sub_block.abodes[-1]["point"])
        if point_id != last_point:
            display_num = str(house_counter)
            last_point = point_id

            # Add the address to the tooltip
            tooltip = abode["display_address"]

        # Add the voter names to the tooltip
        for voter in voters:
            turnout: float = float(turnout_predictions[voter["id"]])
            tooltip += f"<br>{voter['name'].title()} ({voter['party']}): <b>{round(turnout * 100)}%</b>"

        if display_num != str(house_counter):
            if display_num == "":
                display_num = str(house_counter)
            else:
                display_num += f"-{str(house_counter)}"
        if len(sub_block.abodes) == 0:
            continue
        folium.Marker(
            location=[sub_block.abodes[-1]["point"]["lat"], sub_block.abodes[-1]["point"]["lon"]],
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
