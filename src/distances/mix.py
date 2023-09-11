# TODO: Accept requested_blocks from main script instead of always defaulting

import json
from typing import Optional

from termcolor import colored

from src.config import (
    NODE_TOO_FAR_DISTANCE,
    NodeType,
    Point,
    blocks_file,
    requested_blocks_file,
    blocks_file_t,
    house_id_to_block_id_file,
    pt_id,
)

from src.distances.houses import HouseDistances
from src.distances.nodes import NodeDistances
from src.utils.route import get_distance


class MixDistances:
    @classmethod
    def __init__(cls):
        cls.blocks: blocks_file_t = json.load(open(blocks_file))
        cls.requested_blocks: blocks_file_t = json.load(open(requested_blocks_file))
        cls.house_id_to_block_id: dict[str, str] = json.load(open(house_id_to_block_id_file))

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
        block_id = cls.house_id_to_block_id[house["id"]]
        block_start = Point(lat=cls.blocks[block_id]["nodes"][0]["lat"], lon=cls.blocks[block_id]["nodes"][0]["lon"], type=NodeType.node, id="block_start")
        through_start = NodeDistances.get_distance(node, block_start)

        if through_start is not None:
            through_start += cls.requested_blocks[block_id]["addresses"][pt_id(house)][
                "distance_to_start"
            ]

        block_end = Point(lat=cls.blocks[block_id]["nodes"][-1]["lat"], lon=cls.blocks[block_id]["nodes"][-1]["lon"], type=NodeType.node, id="block_end")
        through_end = NodeDistances.get_distance(node, block_end)
        if through_end is not None:
            through_end += cls.requested_blocks[block_id]["addresses"][pt_id(house)][
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
