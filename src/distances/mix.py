# TODO: Accept requested_blocks from main script instead of always defaulting

import json
from typing import Optional

from termcolor import colored

from src.config import (
    NODE_TOO_FAR_DISTANCE,
    Block,
    NodeType,
    Point,
    pt_id,
    BLOCK_DB_IDX,
    PLACE_DB_IDX,
)

from src.distances.houses import HouseDistances
from src.distances.nodes import NodeDistances
from src.utils.route import get_distance
from src.utils.db import Database


class MixDistances:
    @classmethod
    def __init__(cls):
        cls._db = Database()

    @classmethod
    def get_distance_through_ends(
        cls, node: Point, house: Point
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
        block_id = cls._db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
        block: Block = cls._db.get_dict(block_id, BLOCK_DB_IDX)
        block_start = Point(lat=block["nodes"][0]["lat"], lon=block["nodes"][0]["lon"], type=NodeType.node, id="block_start")
        through_start = NodeDistances.get_distance(node, block_start)

        if through_start is not None:
            through_start += block["places"][pt_id(house)][
                "distance_to_start"
            ]

        block_end = Point(lat=block["nodes"][-1]["lat"], lon=block["nodes"][-1]["lon"], type=NodeType.node, id="block_end")
        through_end = NodeDistances.get_distance(node, block_end)
        if through_end is not None:
            through_end += block["places"][pt_id(house)][
                "distance_to_end"
            ]
        return through_start, through_end

    @classmethod
    def get_distance(
        cls, p1: Point, p2: Point
    ) -> Optional[tuple[float, float]] | Optional[float]:
        if "type" not in p1 or "type" not in p2:
            raise ValueError(
                "When retrieiving mix distances, both points must have a 'type'"
            )

        if p1["type"] == p2["type"] == NodeType.house:
            return HouseDistances.get_distance(p1, p2)
        elif p1["type"] == p2["type"] == NodeType.node:
            return NodeDistances.get_distance(p1, p2)
        elif p1["type"] == NodeType.node and p2["type"] == NodeType.house:
            through_start, through_end = cls.get_distance_through_ends(node=p1, house=p2)
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        elif p1["type"] == NodeType.house and p2["type"] == NodeType.node:
            through_start, through_end = cls.get_distance_through_ends(node=p2, house=p1)
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
