import itertools
import math
from statistics import mean
from typing import Optional

from tqdm import tqdm

from src.config import (
    DIFFERENT_BLOCK_COST,
    DIFFERENT_SIDE_COST,
    DISTANCE_TO_ROAD_MULTIPLIER,
    USE_COST_METRIC,
    AnyPoint,
    Block,
    Point,
    pt_id,
    PLACE_DB_IDX,
    HOUSE_DISTANCE_MATRIX_DB_IDX,
    generate_place_id_pair,
)
from src.distances.nodes import NodeDistances
from src.utils.route import get_distance
from src.utils.db import Database


def store(distance: int, cost: int) -> int:
    """
    Convert a distance and cost into storage form (compressing into one variable)

    Parameters:
        distance (int): the distance to store (must be less than 10000)
        cost (int): the cost to store (must be less than 10000)

    Returns:
        int: the storable integer representing the inputs
    """
    if distance >= 10000 or cost >= 10000:
        raise ValueError("Cannot store value more than 9999 in house distance matrix")

    return distance * 10000 + cost


def unstore(value: int) -> tuple[int, int]:
    """
    Convert a stored value into the original values for time and cost

    Parameters:
        value (int): the value stored in the table

    Returns:
        (int, int): a tuple of the distance and cost encoded by this value
    """
    return (value // 10000, value % 10000)


class HouseDistances:
    MAX_STORAGE_DISTANCE = 1600
    _db = Database()

    @classmethod
    def _insert_point(cls, pt: Point, b: Block):
        b_houses = b["houses"]

        if len(b_houses) == 0:
            return

        # Calculate the distances between the segment endpoints
        dc_to_start = NodeDistances.get_distance(pt, b["nodes"][0])
        if dc_to_start is None:
            dc_to_start = (
                (get_distance(pt, b["nodes"][0]), 0)
                if USE_COST_METRIC
                else get_distance(pt, b["nodes"][0])
            )
        dc_to_end = NodeDistances.get_distance(pt, b["nodes"][-1])
        if dc_to_end is None:
            dc_to_end = (
                (get_distance(pt, b["nodes"][-1]), 0)
                if USE_COST_METRIC
                else get_distance(pt, b["nodes"][-1])
            )

        # if pt_id(pt) not in cls._house_matrix:
        #     cls._house_matrix[pt_id(pt)] = {pt_id(pt): store(0, 0)} if USE_COST_METRIC else {pt_id(pt): 0}

        for address, info in b_houses.items():
            if USE_COST_METRIC:
                through_start = (
                    dc_to_start[0] + info["distance_to_start"],
                    dc_to_start[1],
                )
                through_end = (dc_to_end[0] + info["distance_to_end"], dc_to_end[1])

                distance, cost = (
                    through_start if through_start[0] < through_end[0] else through_end
                )

                cls._db.set_str(
                    generate_place_id_pair(pt_id(pt), address),
                    str(store(round(distance), round(cost))),
                    HOUSE_DISTANCE_MATRIX_DB_IDX,
                )
            else:
                through_start = dc_to_start + info["distance_to_start"]
                through_end = dc_to_end + info["distance_to_end"]

                cls._db.set_str(
                    generate_place_id_pair(pt_id(pt), address),
                    str(round(min(through_start, through_end))),
                    HOUSE_DISTANCE_MATRIX_DB_IDX,
                )

    @classmethod
    def _insert_pair(cls, b1: Block, b1_id: str, b2: Block, b2_id: str):
        def crossing_penalty(block: Block) -> int:
            try:
                return DIFFERENT_SIDE_COST[block["type"]]
            except KeyError:
                print(
                    "Unable to find penalty for crossing {} street. Adding none".format(
                        b1["type"]
                    )
                )
                return 0

        b1_houses = b1["houses"]
        b2_houses = b2["houses"]

        if len(b1_houses) == 0 or len(b2_houses) == 0:
            return

        # Check if the segments are the same
        if b1_id == b2_id:
            for (id_1, info_1), (id_2, info_2) in itertools.product(
                b1_houses.items(), b2_houses.items()
            ):
                if id_1 == id_2:
                    cls._db.set_str(
                        generate_place_id_pair(id_1, id_2),
                        str(store(0, 0)),
                        HOUSE_DISTANCE_MATRIX_DB_IDX,
                    )
                    # cls._house_matrix[id_1][id_2] = store(0, 0) if USE_COST_METRIC else 0
                else:
                    distance_to_road = (
                        info_1["distance_to_road"] + info_2["distance_to_road"]
                    ) * DISTANCE_TO_ROAD_MULTIPLIER
                    # For primary and secondary roads, crossing in the middle of the block is not possible
                    # Go to the nearest crosswalk and back
                    if b1["type"] in ["motorway", "trunk", "primary", "secondary"]:
                        distance = min(
                            [
                                info_1["distance_to_start"]
                                + info_2["distance_to_start"]
                                + distance_to_road,
                                info_1["distance_to_end"]
                                + info_2["distance_to_end"]
                                + distance_to_road,
                            ]
                        )
                    else:
                        # Simply use the difference of the distances to the start
                        distance = abs(
                            info_1["distance_to_start"] - info_2["distance_to_start"]
                        )

                        if info_1["side"] != info_2["side"]:
                            distance += distance_to_road

                    if not USE_COST_METRIC:
                        cls._db.set_str(
                            generate_place_id_pair(id_1, id_2),
                            str(round(distance)),
                            HOUSE_DISTANCE_MATRIX_DB_IDX,
                        )
                    else:
                        cost = 0
                        if info_1["side"] != info_2["side"]:
                            cost += crossing_penalty(b1)

                        cls._db.set_str(
                            generate_place_id_pair(id_1, id_2),
                            str(store(round(distance), cost)),
                            HOUSE_DISTANCE_MATRIX_DB_IDX,
                        )
            return

        # Calculate the distances between the segment endpoints
        end_distances = [
            NodeDistances.get_distance(i, j)
            for i, j in [
                (b1["nodes"][0], b2["nodes"][0]),
                (b1["nodes"][0], b2["nodes"][-1]),
                (b1["nodes"][-1], b2["nodes"][0]),
                (b1["nodes"][-1], b2["nodes"][-1]),
            ]
        ]
        end_distances = [d for d in end_distances if d is not None]

        # If this pair is too far away, don't add to the table.
        if len(end_distances) != 4 or min(end_distances) > cls.MAX_STORAGE_DISTANCE:
            return

        # Iterate over every possible pair of houses
        for (id_1, info_1), (id_2, info_2) in itertools.product(
            b1_houses.items(), b2_houses.items()
        ):
            distances_to_road = (
                info_1["distance_to_road"] + info_2["distance_to_road"]
            ) * DISTANCE_TO_ROAD_MULTIPLIER

            # The format of the list is: [start_start, start_end, end_start, end_end]
            distances: list[float] = [
                end_distances[0]
                + info_1["distance_to_start"]
                + info_2["distance_to_start"]
                + distances_to_road,
                end_distances[1]
                + info_1["distance_to_start"]
                + info_2["distance_to_end"]
                + distances_to_road,
                end_distances[2]
                + info_1["distance_to_end"]
                + info_2["distance_to_start"]
                + distances_to_road,
                end_distances[3]
                + info_1["distance_to_end"]
                + info_2["distance_to_end"]
                + distances_to_road,
            ]

            # Add the street crossing penalties
            if USE_COST_METRIC:
                # TODO: This isn't quite right. Need a way to check if two houses on different blocks are on the same side /
                # how many streets to cross, etc. This will be a large change
                cost = mean([crossing_penalty(b1), crossing_penalty(b2)])
                cost += DIFFERENT_BLOCK_COST
                distance = min(distances)

                cls._db.set_str(
                    generate_place_id_pair(id_1, id_2),
                    str(store(round(distance), round(cost))),
                    HOUSE_DISTANCE_MATRIX_DB_IDX,
                )
            else:
                cls._db.set_str(
                    generate_place_id_pair(id_1, id_2),
                    str(round(min(distances)) + DIFFERENT_BLOCK_COST),
                    HOUSE_DISTANCE_MATRIX_DB_IDX,
                )

    @classmethod
    def _update(cls, blocks: dict[str, Block], depot: Optional[Point] = None):
        """
        Update the block distance table by adding any missing blocks

        Parameters
        ----------
            blocks (dict[str, Block]): the blocks to confirm are in the table, and to add if they are not
            depot (Optional[Point]): the depot to add to the table

        Notes
        -----
            Time complexity: O(n^2), where n is the number of blocks
        """
        with tqdm(
            total=len(blocks) ** 2, desc="Updating houses", unit="pairs", colour="green"
        ) as progress:
            for b1_id, b1 in blocks.items():
                if depot is not None:
                    cls._insert_point(depot, b1)
                for b2_id, b2 in blocks.items():
                    cls._insert_pair(b1, b1_id, b2, b2_id)
                    progress.update()

    @classmethod
    def __init__(cls, block_ids, depot: Optional[Point] = None):
        blocks: dict[str, Block] = {}
        for block_id in block_ids:
            block = cls._db.get_dict(block_id, PLACE_DB_IDX)

            if block is None:
                raise KeyError(
                    "Block with ID {} not found in database.".format(block_id)
                )

            blocks[block_id] = block

        cls._update(blocks, depot)

        # There should always be a square number of elements in the distance matrix
        assert (
            math.sqrt(len(cls._db.get_keys(HOUSE_DISTANCE_MATRIX_DB_IDX))) % 1 == 0
        ), "The number of entries in the distance matrix is not a square number"

    @classmethod
    def get_distance(
        cls, p1: AnyPoint, p2: AnyPoint
    ) -> Optional[tuple[float, float] | float]:
        """
        Get the distance between two houses by their coordinates.

        Parameters
        ----------
            p1 (Point): the first point
            p2 (Point): the second point

        Returns
        -------
            tuple[float, float] | float | None: distance, cost between the two points (if using the cost metric)
                or just the distance (if not using the cost metric), or None if the distance is too far
        """
        pair_1, pair_2 = generate_place_id_pair(
            pt_id(p1), pt_id(p2)
        ), generate_place_id_pair(pt_id(p2), pt_id(p1))

        pair_1_r = cls._db.get_str(pair_1, HOUSE_DISTANCE_MATRIX_DB_IDX)
        if pair_1_r is not None:
            return unstore(int(pair_1_r)) if USE_COST_METRIC else int(pair_1_r)

        pair_2_r = cls._db.get_str(pair_2, HOUSE_DISTANCE_MATRIX_DB_IDX)
        if pair_2_r is not None:
            return unstore(int(pair_2_r)) if USE_COST_METRIC else int(pair_2_r)

        return None
