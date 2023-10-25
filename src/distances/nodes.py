import math
from typing import Optional

from tqdm import tqdm

from src.config import (
    BLOCK_DB_IDX,
    NODE_DISTANCE_MATRIX_DB_IDX,
    AnyPoint,
    Point,
    generate_pt_id,
    generate_pt_id_pair,
)
from src.utils.db import Database
from src.utils.gps import great_circle_distance
from src.utils.route import get_distance


class NodeDistancesSnapshot:
    def __init__(self, snapshot: dict[str, float]):
        self.snapshot = snapshot

    def get_distance(self, p1: AnyPoint, p2: AnyPoint) -> Optional[float]:
        p1_id, p2_id = generate_pt_id(p1), generate_pt_id(p2)
        id_pair_1, id_pair_2 = generate_pt_id_pair(p1_id, p2_id), generate_pt_id_pair(
            p2_id, p1_id
        )

        if id_pair_1 in self.snapshot:
            return self.snapshot[id_pair_1]
        if id_pair_2 in self.snapshot:
            return self.snapshot[id_pair_2]
        return None


class NodeDistances:
    """
    Helper class to store and retrieve distances between nodes.

    Ultimately, the in-memory database is used for retrieval and storage. This class
    updates the database with unseen nodes and retrieves distances safely.
    """

    MAX_STORAGE_DISTANCE = 1600
    _db = Database()

    @classmethod
    def _insert_pair(cls, node_1: Point, node_2: Point):
        """
        Insert a pair of nodes into the node distance matrix.

        Parameters
        ----------
            node_1 (Point): the first node
            node_2 (Point): the second node
        """
        # If this pair already exists in the database, skip it
        node_1_id, node_2_id = generate_pt_id(node_1), generate_pt_id(node_2)
        pair_1, pair_2 = generate_pt_id_pair(node_1_id, node_2_id), generate_pt_id_pair(
            node_2_id, node_1_id
        )
        if cls._db.exists(pair_1, NODE_DISTANCE_MATRIX_DB_IDX) or cls._db.exists(
            pair_2, NODE_DISTANCE_MATRIX_DB_IDX
        ):
            return

        # Calculate a fast great circle distance
        distance = great_circle_distance(node_1, node_2)

        # Only calculate and insert the routed distance if needed
        if distance <= cls.MAX_STORAGE_DISTANCE:
            cls._db.set_str(
                generate_pt_id_pair(generate_pt_id(node_1), generate_pt_id(node_2)),
                str(get_distance(node_1, node_2)),
                NODE_DISTANCE_MATRIX_DB_IDX,
            )

    @classmethod
    def _update(cls, nodes: list[Point]):
        """
        Update the node distance table by adding any missing points

        Parameters
        ----------
            nodes (list[Point]): the nodes to confirm are in the table, and to add if they are not

        Notes
        -----
            Time complexity: O(n^2), where n is the number of nodes
        """
        with tqdm(
            total=math.comb(len(nodes), 2),
            desc="Updating nodes",
            unit="pairs",
            colour="green",
        ) as progress:
            for i, node in enumerate(nodes):
                for other_node in nodes[i:]:
                    cls._insert_pair(node, other_node)
                    progress.update()

    @classmethod
    def __init__(cls, block_ids: set[str], skip_update: bool = False):
        if skip_update:
            return

        needed_nodes_set: set[str] = set()
        needed_nodes: list[Point] = []
        for block_id in block_ids:
            block = cls._db.get_dict(block_id, BLOCK_DB_IDX)

            if block is None:
                raise KeyError(
                    "Block with ID {} not found in database.".format(block_id)
                )

            if generate_pt_id(block["nodes"][0]) not in needed_nodes_set:
                needed_nodes.append(block["nodes"][0])
                needed_nodes_set.add(generate_pt_id(block["nodes"][0]))
            if generate_pt_id(block["nodes"][-1]) not in needed_nodes_set:
                needed_nodes.append(block["nodes"][-1])
                needed_nodes_set.add(generate_pt_id(block["nodes"][-1]))

        # Add any necessary nodes to the matrix
        cls._update(list(needed_nodes))

    @classmethod
    def get_distance(cls, p1: AnyPoint, p2: AnyPoint) -> Optional[float]:
        """
        Get the distance between two nodes by their coordinates.

        Parameters
        ----------
            p1 (Point): the first point
            p2 (Point): the second point

        Returns
        -------
            float | None: distance between the two points if it exists, None otherwise
        """
        p1_id, p2_id = generate_pt_id(p1), generate_pt_id(p2)
        id_pair_1, id_pair_2 = generate_pt_id_pair(p1_id, p2_id), generate_pt_id_pair(
            p2_id, p1_id
        )

        pair_1_r = cls._db.get_str(id_pair_1, NODE_DISTANCE_MATRIX_DB_IDX)
        if pair_1_r is not None:
            return float(pair_1_r)

        pair_2_r = cls._db.get_str(id_pair_2, NODE_DISTANCE_MATRIX_DB_IDX)
        if pair_2_r is not None:
            return float(pair_2_r)

        return None

    @classmethod
    def snapshot(cls):
        """
        Create a snapshot of the current distance matrix, and return it
        """
        snapshot = cls._db.get_all_dict(NODE_DISTANCE_MATRIX_DB_IDX)
        return NodeDistancesSnapshot(snapshot)

    @classmethod
    def get_multiple(cls, pts: list[tuple[AnyPoint, AnyPoint]]) -> list[Optional[float]]:
        """
        Get the distance between multiple pairs of nodes by their coordinates.

        Parameters
        ----------
            pts (list[tuple[AnyPoint, AnyPoint]]): the pairs of points

        Returns
        -------
            list[Optional[float]]: the distances between the pairs of points
        """
        ids: list[tuple[str, str]] = []
        for p1, p2 in pts:
            p1_id, p2_id = generate_pt_id(p1), generate_pt_id(p2)
            id_pair_1, id_pair_2 = generate_pt_id_pair(p1_id, p2_id), generate_pt_id_pair(
                p2_id, p1_id
            )
            ids.append((id_pair_1, id_pair_2))

        # Flatten the list of tuples
        flattened: list[str] = [id for pair in ids for id in pair]
        results = cls._db.get_multiple_str(flattened, NODE_DISTANCE_MATRIX_DB_IDX)

        # Split the results back into pairs
        distances = []
        for i in range(0, len(results), 2):
            if results[i] is not None:
                distances.append(float(results[i]))
            elif results[i + 1] is not None:
                distances.append(float(results[i + 1]))
            else:
                distances.append(None)

        return distances
