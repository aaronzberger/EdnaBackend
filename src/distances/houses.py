from copy import deepcopy
import itertools
from statistics import mean
from typing import Optional

from tqdm import tqdm

from src.config import (
    DIFFERENT_BLOCK_COST,
    DIFFERENT_SIDE_COST,
    DISTANCE_TO_ROAD_MULTIPLIER,
    MAX_STORAGE_DISTANCE,
    USE_COST_METRIC,
    Point,
    Block,
    InternalPoint,
    pt_id,
    BLOCK_DB_IDX,
    generate_abode_id_pair,
)
from src.distances.nodes import NodeDistances
from src.utils.route import get_distance
from src.utils.db import Database


# NOTE/TODO: In the future, this could be made more efficient by directly accessing bits:
# e.g. 16 bits for distance, 16 bits for cost, 32 bits for the total value
def store(distance: int, cost: int) -> int:
    """
    Convert a distance and cost into storage form (compressing into one variable)

    Parameters:
        distance (int): the distance to store (must be less than 10000)
        cost (int): the cost to store (must be less than 10000)

    Returns:
        int: the storable integer representing the inputs
    """
    if distance > 9999 or cost > 9999:
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


class HouseDistancesSnapshot:
    def __init__(self, snapshot: dict[str, str]):
        self.snapshot = snapshot

    def get_distance(self, p1: Point, p2: Point) -> Optional[tuple[float, float]]:
        p1_id, p2_id = pt_id(p1), pt_id(p2)
        id_pair_1, id_pair_2 = generate_abode_id_pair(
            p1_id, p2_id
        ), generate_abode_id_pair(p2_id, p1_id)

        if id_pair_1 in self.snapshot:
            return unstore(int(self.snapshot[id_pair_1]))
        if id_pair_2 in self.snapshot:
            return unstore(int(self.snapshot[id_pair_2]))
        return None


class HouseDistances:
    def _insert_point(self, pt: InternalPoint, b: Block):
        b_houses = b["abodes"]

        if len(b_houses) == 0:
            return

        # Calculate the distances between the segment endpoints
        distance_from_start = self._node_distances_snapshot.get_distance(
            pt, b["nodes"][0]
        )
        if distance_from_start is None:
            distance_from_start = get_distance(pt, b["nodes"][0])

        distance_from_end = self._node_distances_snapshot.get_distance(
            pt, b["nodes"][-1]
        )
        if distance_from_end is None:
            distance_from_end = get_distance(pt, b["nodes"][-1])

        for address, info in b_houses.items():
            through_start = distance_from_start + info["distance_to_start"]
            through_end = distance_from_end + info["distance_to_end"]

            distance = min(through_start, through_end)

            if distance < MAX_STORAGE_DISTANCE:
                self.distance_matrix[
                    generate_abode_id_pair(pt_id(pt), address)
                ] = store(round(distance), 0)

    def _crossing_penalty(self, block: Block) -> int:
        try:
            return DIFFERENT_SIDE_COST[block["type"]]
        except KeyError:
            print(
                "Unable to find penalty for crossing {} street. Adding none".format(
                    block["type"]
                )
            )
            return 0

    def _insert_block(self, b: Block):
        # Check if the segments are the same
        for (id_1, info_1), (id_2, info_2) in itertools.product(
            b["abodes"].items(), b["abodes"].items()
        ):
            if id_1 == id_2:
                self.distance_matrix[generate_abode_id_pair(id_1, id_2)] = store(0, 0)
            else:
                distance_to_road = (
                    info_1["distance_to_road"] + info_2["distance_to_road"]
                ) * DISTANCE_TO_ROAD_MULTIPLIER
                # For primary and secondary roads, crossing in the middle of the block is not possible
                # Go to the nearest crosswalk and back
                if b["type"] in ["motorway", "trunk", "primary", "secondary"]:
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
                    if distance < MAX_STORAGE_DISTANCE:
                        self.distance_matrix[
                            generate_abode_id_pair(id_1, id_2)
                        ] = store(round(distance), 0)
                else:
                    cost = 0
                    if info_1["side"] != info_2["side"]:
                        cost += self._crossing_penalty(b)

                    if distance < MAX_STORAGE_DISTANCE:
                        self.distance_matrix[
                            generate_abode_id_pair(id_1, id_2)
                        ] = store(round(distance), round(cost))

    def _insert_pair(self, b1: Block, b1_id: str, b2: Block, b2_id: str):
        b1_houses = b1["abodes"]
        b2_houses = b2["abodes"]

        if len(b1_houses) == 0 or len(b2_houses) == 0:
            return

        # Check if the segments are the same
        if b1_id == b2_id:
            self._insert_block(b1)
            return

        # Calculate the distances between the segment endpoints
        # TODO: Could this be made more efficient by caching get_distance calls?
        end_distances = [
            self._node_distances_snapshot.get_distance(i, j)
            for i, j in [
                (b1["nodes"][0], b2["nodes"][0]),
                (b1["nodes"][0], b2["nodes"][-1]),
                (b1["nodes"][-1], b2["nodes"][0]),
                (b1["nodes"][-1], b2["nodes"][-1]),
            ]
        ]
        end_distances = [d for d in end_distances if d is not None]

        # If this pair is too far away, don't add to the table.
        if len(end_distances) != 4 or min(end_distances) > MAX_STORAGE_DISTANCE:
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

            distance = min(distances)

            # Add the street crossing penalties
            if USE_COST_METRIC:
                # TODO: This isn't quite right. Need a way to check if two houses on different blocks are on the same side /
                # how many streets to cross, etc. This will be a large change
                cost = mean([self._crossing_penalty(b1), self._crossing_penalty(b2)])
                cost += DIFFERENT_BLOCK_COST

                if distance < MAX_STORAGE_DISTANCE:
                    self.distance_matrix[generate_abode_id_pair(id_1, id_2)] = store(
                        round(distance), round(cost)
                    )
            else:
                if distance < MAX_STORAGE_DISTANCE:
                    self.distance_matrix[generate_abode_id_pair(id_1, id_2)] = store(
                        round(distance), 0
                    )

    def _update(
        self, blocks: dict[str, Block], depots: Optional[list[InternalPoint]] = None
    ):
        """
        Update the block distance table by adding any missing blocks

        Parameters
        ----------
            blocks (dict[str, Block]): the blocks to confirm are in the table, and to add if they are not
            depots (Optional[list[Point]]): the depots to add to the table

        Notes
        -----
            Time complexity: O(n^2), where n is the number of blocks
        """
        with tqdm(
            total=len(blocks) ** 2, desc="Updating houses", unit="pairs", colour="green"
        ) as progress:
            for b1_id, b1 in blocks.items():
                if depots is not None:
                    for depot in depots:
                        # Insert the point and itself
                        self.distance_matrix[
                            generate_abode_id_pair(pt_id(depot), pt_id(depot))
                        ] = store(0, 0)

                        self._insert_point(depot, b1)
                for b2_id, b2 in blocks.items():
                    self._insert_pair(b1, b1_id, b2, b2_id)
                    progress.update()

    def __init__(
        self,
        block_ids: list[str],
        node_distances: NodeDistances,
        depots: Optional[list[InternalPoint]] = None,
    ):
        self._db = Database()
        self.distance_matrix: dict[str, str] = {}

        # To speed up runtime retrieval, take a snapshot of the node distances object
        self._node_distances_snapshot = node_distances.snapshot()

        blocks: dict[str, Block] = {}
        for block_id in block_ids:
            block = self._db.get_dict(block_id, BLOCK_DB_IDX)

            if block is None:
                raise KeyError(
                    "Block with ID {} not found in database.".format(block_id)
                )

            blocks[block_id] = block

        self._update(blocks, depots)

    def get_distance(
        self, p1: Point, p2: Point
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
        pair_1, pair_2 = generate_abode_id_pair(
            pt_id(p1), pt_id(p2)
        ), generate_abode_id_pair(pt_id(p2), pt_id(p1))

        pair_1_r = self.distance_matrix.get(pair_1)
        if pair_1_r is not None:
            return unstore(int(pair_1_r))

        pair_2_r = self.distance_matrix.get(pair_2)
        if pair_2_r is not None:
            return unstore(int(pair_2_r))

        return None

    def snapshot(self) -> HouseDistancesSnapshot:
        """
        Get a snapshot of the current house distances table

        Returns
        -------
            HouseDistancesSnapshot: a snapshot of the current house distances table
        """
        print(
            f"Taking snapshot of house distances table with {len(self.distance_matrix)} entries"
        )
        snapshot = deepcopy(self.distance_matrix)
        return HouseDistancesSnapshot(snapshot)
