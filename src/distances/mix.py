from typing import Optional

from termcolor import colored
import numpy as np
from tqdm import tqdm

from src.config import (
    MAX_STORAGE_DISTANCE,
    Block,
    NodeType,
    Point,
    pt_id,
    BLOCK_DB_IDX,
    PLACE_DB_IDX,
    Singleton
)

from src.distances.houses import HouseDistances, HouseDistancesSnapshot
from src.distances.nodes import NodeDistances, NodeDistancesSnapshot
from src.utils.route import get_distance
from src.utils.db import Database


class MixDistancesSnapshot:
    def __init__(self, node_distances: NodeDistancesSnapshot, house_distances: HouseDistancesSnapshot):
        self._node_distances = node_distances
        self._house_distances = house_distances
        self._db = Database()
        # TODO: Somehow take a snapshot of blocks if necessary

    def get_distance_through_ends(
        self, node: Point, house: Point
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Determine the distances from an intersection point to a house through the two ends of the house's segments.

        Parameters
        ----------
            node (Point): an intersection point
            house (Point): a house point

        Returns
        -------
            Optional[float]: the distance from the intersection to the house through the start of the segment
            Optional[float]: the distance from the intersection to the house through the end of the segment
        """
        block_id = self._db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
        block: Block = self._db.get_dict(block_id, BLOCK_DB_IDX)
        block_start = Point(lat=block["nodes"][0]["lat"], lon=block["nodes"][0]["lon"], type=NodeType.node, id="block_start")
        through_start = self._node_distances.get_distance(node, block_start)

        if through_start is not None:
            through_start += block["places"][pt_id(house)][
                "distance_to_start"
            ]

        block_end = Point(lat=block["nodes"][-1]["lat"], lon=block["nodes"][-1]["lon"], type=NodeType.node, id="block_end")
        through_end = self._node_distances.get_distance(node, block_end)
        if through_end is not None:
            through_end += block["places"][pt_id(house)][
                "distance_to_end"
            ]
        return through_start, through_end

    def get_distance(
        self, p1: Point, p2: Point
    ) -> Optional[tuple[float, float]] | Optional[float]:
        if "type" not in p1 or "type" not in p2:
            raise ValueError(
                "When retrieiving mix distances, both points must have a 'type'"
            )

        if p1["type"] == p2["type"] == NodeType.house:
            return self._house_distances.get_distance(p1, p2)
        elif p1["type"] == p2["type"] == NodeType.node:
            return self._node_distances.get_distance(p1, p2)
        elif p1["type"] == NodeType.node and p2["type"] == NodeType.house:
            through_start, through_end = self.get_distance_through_ends(node=p1, house=p2)
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        elif p1["type"] == NodeType.house and p2["type"] == NodeType.node:
            through_start, through_end = self.get_distance_through_ends(node=p2, house=p1)
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        else:
            if pt_id(p1) == "depot" or pt_id(p2) == "depot":
                return self._house_distances.get_distance(p1, p2)
            raise ValueError(
                f"Getting routed distance between points live is not recommended. Points: {p1}, {p2}"
            )


class MixDistances(metaclass=Singleton):
    def __init__(self, house_distances: HouseDistances, node_distances: NodeDistances):
        self._db = Database()
        self._house_distances = house_distances
        self._node_distances = node_distances

    def get_distance_through_ends(
        self, node: Point, house: Point
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Determine the distances from an intersection point to a house through the two ends of the house's segments.

        Parameters
        ----------
            node (Point): an intersection point
            house (Point): a house point

        Returns
        -------
            Optional[float]: the distance from the intersection to the house through the start of the segment
            Optional[float]: the distance from the intersection to the house through the end of the segment
        """
        block_id = self._db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
        block: Block = self._db.get_dict(block_id, BLOCK_DB_IDX)
        block_start = Point(lat=block["nodes"][0]["lat"], lon=block["nodes"][0]["lon"], type=NodeType.node, id="block_start")
        through_start = self._node_distances.get_distance(node, block_start)

        if through_start is not None:
            through_start += block["places"][pt_id(house)][
                "distance_to_start"
            ]

        block_end = Point(lat=block["nodes"][-1]["lat"], lon=block["nodes"][-1]["lon"], type=NodeType.node, id="block_end")
        through_end = self._node_distances.get_distance(node, block_end)
        if through_end is not None:
            through_end += block["places"][pt_id(house)][
                "distance_to_end"
            ]
        return through_start, through_end

    def get_distance(
        self, p1: Point, p2: Point
    ) -> Optional[tuple[float, float]] | Optional[float]:
        if "type" not in p1 or "type" not in p2:
            raise ValueError(
                "When retrieiving mix distances, both points must have a 'type'"
            )

        if p1["type"] == p2["type"] == NodeType.house:
            return self._house_distances.get_distance(p1, p2)
        elif p1["type"] == p2["type"] == NodeType.node:
            return self._node_distances.get_distance(p1, p2)
        elif p1["type"] == NodeType.node and p2["type"] == NodeType.house:
            through_start, through_end = self.get_distance_through_ends(node=p1, house=p2)
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        elif p1["type"] == NodeType.house and p2["type"] == NodeType.node:
            through_start, through_end = self.get_distance_through_ends(node=p2, house=p1)
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        else:
            print(
                colored(
                    "Warning: getting routed distance between points live is not recommended",
                    color="yellow",
                )
            )
            return get_distance(p1, p2)

    def snapshot(self) -> MixDistancesSnapshot:
        return MixDistancesSnapshot(
            node_distances=self._node_distances.snapshot(),
            house_distances=self._house_distances.snapshot(),
        )

    def get_distance_matrix(self, points: list[Point]):
        """
        Get the distance matrix for a list of points

        Parameters
        ----------
            points (list[Point]): a list of points
        """
        matrix = np.empty((len(points), len(points)), dtype=np.float32)

        snapshot = self.snapshot()

        for i, p1 in enumerate(tqdm(points)):
            for j, p2 in enumerate(points):
                d_or_dc = snapshot.get_distance(p1, p2)

                # Account for distance and cost stored on houses
                d = d_or_dc if type(d_or_dc) is not tuple else d_or_dc[0]

                # Account for none values
                d = d if d is not None else MAX_STORAGE_DISTANCE

                matrix[i][j] = d

        return matrix
