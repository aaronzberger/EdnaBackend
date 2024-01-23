import heapq
import itertools
import json
import os
from collections import defaultdict
from decimal import Decimal
from typing import Any

import polyline
import requests
from termcolor import colored

from src.config import (
    InternalPoint,
    adjacency_list_file,
    coords_node_file,
    NODE_COORDS_DB_IDX,
    BLOCK_DB_IDX,
    Point,
)
from src.utils.gps import great_circle_distance
from src.utils.db import Database

SERVER = "http://172.19.0.2:5000"


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
        r = requests.get(url, timeout=1)
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
    _node_table = {}
    _adjacency_list = {}
    _db = Database()

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
            for node_id, node in cls._db.get_all_dict(NODE_COORDS_DB_IDX).items():
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
            for block_id in cls._db.get_keys(BLOCK_DB_IDX):
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

        def block_distance(nodes: list[Point]):
            length = 0
            for first, second in itertools.pairwise(nodes):
                length += great_circle_distance(first, second)
            return length

        def get_block(node_1: str, node_2: str):
            """Get the block, and the appropriate ID, for two given nodes."""
            for i in range(3):
                id1 = node_1 + ":" + node_2 + ":" + str(i)
                if cls._db.exists(id1, BLOCK_DB_IDX):
                    return id1, cls._db.get_dict(id1, BLOCK_DB_IDX)
                id2 = node_2 + ":" + node_1 + ":" + str(i)
                if cls._db.exists(id2, BLOCK_DB_IDX):
                    return id2, cls._db.get_dict(id2, BLOCK_DB_IDX)

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

        assert (
            path[0] == start_node_id
        ), f"Path start {path[0]} does not match start node {start_node_id} (end node {end_node_id}, path{path})"
        assert (
            path[-1] == end_node_id
        ), f"Path end {path[-1]} does not match end node {end_node_id}"
        # TODO: Can be removed later
        for first, second in itertools.pairwise(path):
            assert second in cls._adjacency_list[first]

        # Calculate the distance
        distance = 0
        for first, second in itertools.pairwise(path):
            distance += great_circle_distance(
                cls._db.get_dict(first, NODE_COORDS_DB_IDX),
                cls._db.get_dict(second, NODE_COORDS_DB_IDX),
            )

        # These are the nodes that make up the blocks, so convert them to blocks
        blocks = []
        for first, second in itertools.pairwise(path):
            block_id, _ = get_block(first, second)
            blocks.append(block_id)

        return blocks, distance

    @classmethod
    def get_route(cls, start: InternalPoint, end: InternalPoint):
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
            print(
                colored(
                    f"Failed to run djikstras from start node {start_node[0]} to end node {end_node[0]}",
                    "red",
                )
            )
            return None

        return path, distance
