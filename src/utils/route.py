import heapq
import itertools
import json
import os
from collections import defaultdict
from decimal import Decimal
import sys
from typing import Any

import polyline
import requests
from termcolor import colored

from src.config import (
    DISTANCE_TO_ROAD_MULTIPLIER,
    Block,
    PlaceGeography,
    Point,
    adjacency_list_file,
    blocks_file,
    blocks_file_t,
    coords_node_file,
    node_coords_file,
    node_list_t,
)
from src.utils.gps import great_circle_distance

SERVER = "http://172.17.0.2:5000"


def get_distance(start: Point, end: Point) -> float:
    """
    Get the distance on foot (in meters) between two points.

    Parameters
    ----------
        start (Point): the starting point
        end (Point): the ending point

    Returns
    -------
        float: the distance (in meters), on foot, to walk from start to end
    """
    loc = "{},{};{},{}".format(start["lon"], start["lat"], end["lon"], end["lat"])
    url = SERVER + "/route/v1/walking/" + loc
    try:
        r = requests.get(url, timeout=5)
    except Exception as e:
        raise RuntimeError("Request to OSRM server failed. Is it running?") from e
    if r.status_code == 200:
        return r.json()["routes"][0]["distance"]
    raise RuntimeError("Could not contact OSRM server")


def get_route(start: Point, end: Point) -> dict[str, Any]:
    """
    Get the full route on foot between two points.

    Parameters
    ----------
        start (Point): the starting point
        end (Point): the ending point

    Returns
    -------
        dict:
            'route' (list): the route, as given by polyline
            'start_point' (list): the starting point in the format [lat, lon]
            'end_point' (list): the ending point in the format [lat, lon]
            'distance' (float): the distance from start to end
    """
    loc = "{},{};{},{}".format(start["lon"], start["lat"], end["lon"], end["lat"])
    url = SERVER + "/route/v1/walking/" + loc
    r = requests.get(url, timeout=5)
    if r.status_code != 200:
        return {}

    res = r.json()

    routes = polyline.decode(res["routes"][0]["geometry"])
    start_point = [
        res["waypoints"][0]["location"][1],
        res["waypoints"][0]["location"][0],
    ]
    end_point = [res["waypoints"][1]["location"][1], res["waypoints"][1]["location"][0]]
    distance = res["routes"][0]["distance"]

    return {
        "route": routes,
        "start_point": start_point,
        "end_point": end_point,
        "distance": distance,
    }


class RouteMaker:
    _blocks: blocks_file_t = json.load(open(blocks_file))
    _node_coords = json.load(open(node_coords_file))
    _node_table = {}
    _adjacency_list = {}

    @classmethod
    def id(cls, lat: Decimal, lon: Decimal) -> str:
        """
        Get the node ID for a given latitude and longitude.

        Parameters
        ----------
            lat (Decimal): the latitude
            lon (Decimal): the longitude

        Returns
        -------
            str: the node ID
        """
        return str(lat) + ":" + str(lon)

    @classmethod
    def __init__(cls):
        # Create a hash table, hashing by the first 4 decimal degrees of the latitude and longitude (within 11.1 meters)
        if os.path.exists(coords_node_file):
            cls._node_table = json.load(open(coords_node_file))
        else:
            print("Creating node table...", end=" ")
            for node_id, node in cls._node_coords.items():
                lat = Decimal(node["lat"]).quantize(Decimal("0.0001"))
                lon = Decimal(node["lon"]).quantize(Decimal("0.0001"))
                if cls.id(lat, lon) not in cls._node_table:
                    cls._node_table[cls.id(lat, lon)] = []
                cls._node_table[cls.id(lat, lon)].append((node_id, node))
            json.dump(cls._node_table, open(coords_node_file, "w"))
            print("Done")

        # Create an adjacency list for the graph created by the blocks
        if os.path.exists(adjacency_list_file):
            cls._adjacency_list = json.load(open(adjacency_list_file))
        else:
            print("Creating adjacency list...", end=" ")
            for block_id in cls._blocks.keys():
                node_1, node_2, _ = block_id.split(":")
                if node_1 not in cls._adjacency_list:
                    cls._adjacency_list[node_1] = []
                if node_2 not in cls._adjacency_list:
                    cls._adjacency_list[node_2] = []
                cls._adjacency_list[node_1].append(node_2)
                cls._adjacency_list[node_2].append(node_1)
            json.dump(cls._adjacency_list, open(adjacency_list_file, "w"))
            print("Done")

    @classmethod
    def djikstras(cls, start_node_id: str, end_node_id: str) -> tuple[list[str], float]:
        """
        Run Djikstra's algorithm to find the shortest path between two nodes
        Halt if no paths of length <= 5 are found.

        Parameters
        ----------
            start_node_id (str): the starting node
            end_node_id (str): the ending node

        Returns
        -------
            list[str]: the shortest path from start_node to end_node
            float: the distance of the shortest path
        """

        def block_distance(nodes: node_list_t):
            length = 0
            for first, second in itertools.pairwise(nodes):
                length += great_circle_distance(first, second)
            return length

        def get_block(node_1: str, node_2: str):
            """Get the block, and the appropriate ID, for two given nodes."""
            for i in range(3):
                id1 = node_1 + ":" + node_2 + ":" + str(i)
                if id1 in cls._blocks:
                    return id1, cls._blocks[id1]
                id2 = node_2 + ":" + node_1 + ":" + str(i)
                if id2 in cls._blocks:
                    return id2, cls._blocks[id2]

        # Initialize the distance and previous nodes
        distances = defaultdict(lambda: float("inf"))
        previous = defaultdict(lambda: None)
        edges_from_start = defaultdict(lambda: None)
        distances[start_node_id] = 0
        edges_from_start[start_node_id] = 0

        # Initialize the heap
        heap = [(0, start_node_id)]

        # Run Djikstra's algorithm
        while len(heap) > 0:
            _, node = heapq.heappop(heap)

            # If we've reached the end node, break
            if node == end_node_id:
                break

            # Otherwise, update the distances of the neighbors
            for neighbor in cls._adjacency_list[node]:
                _, block = get_block(node, neighbor)
                if block is None:
                    print("Block not found between {} and {}".format(node, neighbor))
                distance = distances[node] + block_distance(block["nodes"])
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = node
                    edges_from_start[neighbor] = edges_from_start[node] + 1
                    heapq.heappush(heap, (distance, neighbor))

        # Now, reconstruct the path
        path = []
        node = end_node_id
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()

        assert path[0] == start_node_id, f"Path start {path[0]} does not match start node {start_node_id} (end node {end_node_id}, path{path})"
        assert path[-1] == end_node_id, f"Path end {path[-1]} does not match end node {end_node_id}"
        # TODO: Can be removed later
        for first, second in itertools.pairwise(path):
            assert second in cls._adjacency_list[first]

        # Calculate the distance
        distance = 0
        for first, second in itertools.pairwise(path):
            distance += great_circle_distance(
                cls._node_coords[first], cls._node_coords[second]
            )

        # These are the nodes that make up the blocks, so convert them to blocks
        blocks = []
        for first, second in itertools.pairwise(path):
            block_id, _ = get_block(first, second)
            blocks.append(block_id)

        return blocks, distance

    @classmethod
    def get_route(cls, start: Point, end: Point):
        # If the init method has not been called, call it
        if len(cls._node_table) == 0:
            cls.__init__()

        # Take the closest node to the start and end points
        start_lat = Decimal(start["lat"]).quantize(Decimal("0.0001"))
        start_lon = Decimal(start["lon"]).quantize(Decimal("0.0001"))

        end_lat = Decimal(end["lat"]).quantize(Decimal("0.0001"))
        end_lon = Decimal(end["lon"]).quantize(Decimal("0.0001"))

        if (
            cls.id(start_lat, start_lon) not in cls._node_table
            or cls.id(end_lat, end_lon) not in cls._node_table
        ):
            print("Could not find start or end node")
            return []

        start_node = min(
            cls._node_table[cls.id(start_lat, start_lon)],
            key=lambda x: great_circle_distance(start, x[1]),
        )
        end_node = min(
            cls._node_table[cls.id(end_lat, end_lon)],
            key=lambda x: great_circle_distance(end, x[1]),
        )

        # NOTE: If needed, run this route and only run the Djikstra's on the intermediate steps that are needed
        # route = get_route(start_node[1], end_node[1])

        # Now, run Djikstra's algorithm to find the shortest path between the start and end nodes
        # Limit the search to the blocks that are within d(start, end) of one of the nodes
        # Limit the number of iterations to 5, so this runs in O(n) time
        try:
            path, distance = cls.djikstras(start_node[0], end_node[0])
        except (TypeError, AssertionError):
            print(colored(f"Failed to run djikstras from start node {start_node[0]} to end node {end_node[0]}", "red"))
            return None

        return path, distance

    @classmethod
    def get_route_cost(
        cls, start: PlaceGeography, block1: Block, end: PlaceGeography, block2: Block
    ):
        """
        Get the cost of a route between two houses. Each street crossing cost is added
        exactly according to how one would walk optimally from the start to end house (thus,
        there may be 0 cost even if the houses are on different blocks).

        Parameters
        ----------
            start (HouseInfo): the start point
            block1 (Block): the first block
            end (HouseInfo): the end point
            block2 (Block): the second block
        """
        # First, run Djikstra's algorithm to find the shortest path between the start and end nodes
        start_1 = min(
            cls._node_table[
                cls.id(
                    Decimal(block1["nodes"][0]["lat"]).quantize(Decimal("0.0001")),
                    Decimal(block1["nodes"][0]["lon"]).quantize(Decimal("0.0001")),
                )
            ],
            key=lambda x: great_circle_distance(block1["nodes"][0], x[1]),
        )[0]

        end_1 = min(
            cls._node_table[
                cls.id(
                    Decimal(block1["nodes"][-1]["lat"]).quantize(Decimal("0.0001")),
                    Decimal(block1["nodes"][-1]["lon"]).quantize(Decimal("0.0001")),
                )
            ],
            key=lambda x: great_circle_distance(block1["nodes"][-1], x[1]),
        )[0]

        start_2 = min(
            cls._node_table[
                cls.id(
                    Decimal(block2["nodes"][0]["lat"]).quantize(Decimal("0.0001")),
                    Decimal(block2["nodes"][0]["lon"]).quantize(Decimal("0.0001")),
                )
            ],
            key=lambda x: great_circle_distance(block2["nodes"][0], x[1]),
        )[0]

        end_2 = min(
            cls._node_table[
                cls.id(
                    Decimal(block2["nodes"][-1]["lat"]).quantize(Decimal("0.0001")),
                    Decimal(block2["nodes"][-1]["lon"]).quantize(Decimal("0.0001")),
                )
            ],
            key=lambda x: great_circle_distance(block2["nodes"][-1], x[1]),
        )[0]

        # Tuples of route, distance for each possible starting and ending location
        end_routes = [
            cls.djikstras(i, j)
            for i, j in [
                (start_1, start_2),
                (start_1, end_2),
                (end_1, start_2),
                (end_1, end_2),
            ]
        ]

        # TODO: Ensure Djikstra's returned correctly

        distances_to_road = (
            start["distance_to_road"] + end["distance_to_road"]
        ) * DISTANCE_TO_ROAD_MULTIPLIER

        # The format of the list is: [start_start, start_end, end_start, end_end]
        distances: list[float] = [
            end_routes[0][1]
            + start["distance_to_start"]
            + end["distance_to_start"]
            + distances_to_road,
            end_routes[1][1]
            + start["distance_to_start"]
            + end["distance_to_end"]
            + distances_to_road,
            end_routes[2][1]
            + start["distance_to_end"]
            + end["distance_to_start"]
            + distances_to_road,
            end_routes[3][1]
            + start["distance_to_end"]
            + end["distance_to_end"]
            + distances_to_road,
        ]

        route = end_routes[distances.index(min(distances))]

        # For now, use a greedy strategy. That is, only cross when necessary to get onto the next block
        cost = 0

        # Algorithm to determine if a crossing is necessary:
        #   There is a middle point, and n adjacent points, and a single point (where we are, which can be artbitrarily decided as a point on the corresponding side of the block)
        #   At each intersection, calculate the angle between the middle point and the adjacent points
        #   Calculate the angle between the middle point and the single point
        #   If the angle is in the range between the angle to the previous node and the angle to the next node, then we don't need to cross
        #   If we don't cross, the current point can be kept
        #   If we do cross, the new point is a point 1 degree away from the angle to the next node, in the opposite direction of the angle to the previous node


if __name__ == "__main__":
    # Call on a specific house
    house_1_info = PlaceGeography(
        {
            "lat": 40.5494095015337,
            "lon": -80.1919223610264,
            "distance_to_start": 53,
            "distance_to_end": 82,
            "side": False,
            "distance_to_road": 18,
            "subsegment": [1, 2],
        }
    )

    block1 = Block(
        {
            "addresses": {
                "413 CHESTNUT RD": {
                    "lat": 40.5495888473418,
                    "lon": -80.1916865917007,
                    "distance_to_start": 82,
                    "distance_to_end": 53,
                    "side": False,
                    "distance_to_road": 19,
                    "subsegment": [1, 2],
                },
                "421 CHESTNUT RD": {
                    "lat": 40.5498797591104,
                    "lon": -80.1912777518144,
                    "distance_to_start": 129,
                    "distance_to_end": 6,
                    "side": False,
                    "distance_to_road": 21,
                    "subsegment": [2, 3],
                },
                "411 CHESTNUT RD": {
                    "lat": 40.5495374878237,
                    "lon": -80.1917390879274,
                    "distance_to_start": 74,
                    "distance_to_end": 61,
                    "side": False,
                    "distance_to_road": 18,
                    "subsegment": [1, 2],
                },
                "415 CHESTNUT RD": {
                    "lat": 40.549633328746,
                    "lon": -80.1916216066886,
                    "distance_to_start": 89,
                    "distance_to_end": 46,
                    "side": False,
                    "distance_to_road": 20,
                    "subsegment": [1, 2],
                },
                "407 CHESTNUT RD": {
                    "lat": 40.5494095015337,
                    "lon": -80.1919223610264,
                    "distance_to_start": 53,
                    "distance_to_end": 82,
                    "side": False,
                    "distance_to_road": 18,
                    "subsegment": [1, 2],
                },
                "417 CHESTNUT RD": {
                    "lat": 40.5497490533519,
                    "lon": -80.1914598474343,
                    "distance_to_start": 108,
                    "distance_to_end": 27,
                    "side": False,
                    "distance_to_road": 20,
                    "subsegment": [1, 2],
                },
                "419 CHESTNUT RD": {
                    "lat": 40.5498196749462,
                    "lon": -80.1913533315585,
                    "distance_to_start": 120,
                    "distance_to_end": 15,
                    "side": False,
                    "distance_to_road": 20,
                    "subsegment": [2, 3],
                },
                "409 CHESTNUT RD": {
                    "lat": 40.5495111555207,
                    "lon": -80.1918418101735,
                    "distance_to_start": 66,
                    "distance_to_end": 69,
                    "side": False,
                    "distance_to_road": 22,
                    "subsegment": [1, 2],
                },
                "415 1/2 CHESTNUT RD": {
                    "lat": 40.5498489234183,
                    "lon": -80.1918536352896,
                    "distance_to_start": 90,
                    "distance_to_end": 45,
                    "side": False,
                    "distance_to_road": 51,
                    "subsegment": [1, 2],
                },
            },
            "nodes": [
                {"lat": 40.5489748, "lon": -80.1922636},
                {"lat": 40.5491672, "lon": -80.1919728},
                {"lat": 40.54967, "lon": -80.191213},
                {"lat": 40.5497717, "lon": -80.1910593},
            ],
            "type": "residential",
        } # type: ignore
    )

    house_2_info = PlaceGeography(
        {
            "lat": 40.551660346094,
            "lon": -80.1929784945178,
            "distance_to_start": 40,
            "distance_to_end": 47,
            "side": False,
            "distance_to_road": 28,
            "subsegment": [0, 1],
        }
    )

    block2 = Block(
        {
            "addresses": {
                "408 EDGEWORTH LN": {
                    "lat": 40.5512499628013,
                    "lon": -80.1925553719882,
                    "distance_to_start": 37,
                    "distance_to_end": 50,
                    "side": True,
                    "distance_to_road": 30,
                    "subsegment": [0, 1],
                },
                "409 EDGEWORTH LN": {
                    "lat": 40.5517668969989,
                    "lon": -80.1927570000412,
                    "distance_to_start": 62,
                    "distance_to_end": 25,
                    "side": False,
                    "distance_to_road": 24,
                    "subsegment": [0, 1],
                },
                "407 EDGEWORTH LN": {
                    "lat": 40.551660346094,
                    "lon": -80.1929784945178,
                    "distance_to_start": 40,
                    "distance_to_end": 47,
                    "side": False,
                    "distance_to_road": 28,
                    "subsegment": [0, 1],
                },
                "406 EDGEWORTH LN": {
                    "lat": 40.5510804070503,
                    "lon": -80.1928346982635,
                    "distance_to_start": 7,
                    "distance_to_end": 80,
                    "side": True,
                    "distance_to_road": 29,
                    "subsegment": [0, 1],
                },
                "410 EDGEWORTH LN": {
                    "lat": 40.551482311351,
                    "lon": -80.1923974797763,
                    "distance_to_start": 64,
                    "distance_to_end": 23,
                    "side": True,
                    "distance_to_road": 19,
                    "subsegment": [0, 1],
                },
            },
            "nodes": [
                {"lat": 40.5512341, "lon": -80.1931227},
                {"lat": 40.5517477, "lon": -80.1923449},
            ],
            "type": "residential",
        } # type: ignore
    )

    RouteMaker()

    RouteMaker.get_route_cost(house_1_info, block1, house_2_info, block2)
