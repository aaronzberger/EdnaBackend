from typing import Optional

from termcolor import colored
import numpy as np
from tqdm import tqdm

from src.config import (
    MAX_STORAGE_DISTANCE,
    Block,
    NodeType,
    InternalPoint,
    pt_id,
    BLOCK_DB_IDX,
    ABODE_DB_IDX,
)

from src.distances.houses import AbodeDistances, AbodeDistancesSnapshot
from src.distances.nodes import NodeDistances, NodeDistancesSnapshot
from src.utils.route import get_distance
from src.utils.db import Database


class MixDistancesSnapshot:
    def __init__(
        self,
        node_distances: NodeDistancesSnapshot,
        abode_distances: AbodeDistancesSnapshot,
    ):
        self._node_distances = node_distances
        self._abode_distances = abode_distances
        self._db = Database()
        # TODO: Somehow take a snapshot of blocks if necessary

    def get_distance_through_ends(
        self, node: InternalPoint, abode_point: InternalPoint
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Determine the distances from an intersection point to a abode through the two ends of the abode's segments.

        Parameters
        ----------
            node (Point): an intersection point
            abode_point (Point): an abode point

        Returns
        -------
            Optional[float]: the distance from the intersection to the abode through the start of the segment
            Optional[float]: the distance from the intersection to the abode through the end of the segment
        """
        block_id = self._db.get_dict(abode_point["id"], ABODE_DB_IDX)["block_id"]
        block: Block = self._db.get_dict(block_id, BLOCK_DB_IDX)
        block_start = InternalPoint(
            lat=block["nodes"][0]["lat"],
            lon=block["nodes"][0]["lon"],
            type=NodeType.node,
            id="block_start",
        )
        through_start = self._node_distances.get_distance(node, block_start)

        if through_start is not None:
            through_start += block["abodes"][pt_id(abode_point)]["distance_to_start"]

        block_end = InternalPoint(
            lat=block["nodes"][-1]["lat"],
            lon=block["nodes"][-1]["lon"],
            type=NodeType.node,
            id="block_end",
        )
        through_end = self._node_distances.get_distance(node, block_end)
        if through_end is not None:
            through_end += block["abodes"][pt_id(abode_point)]["distance_to_end"]
        return through_start, through_end

    def get_distance(
        self, p1: InternalPoint, p2: InternalPoint
    ) -> Optional[tuple[float, float]] | Optional[float]:
        if "type" not in p1 or "type" not in p2:
            raise ValueError(
                "When retrieiving mix distances, both points must have a 'type'"
            )

        if p1["type"] == p2["type"] == NodeType.abode:
            return self._abode_distances.get_distance(p1, p2)
        elif p1["type"] == p2["type"] == NodeType.node:
            return self._node_distances.get_distance(p1, p2)
        elif p1["type"] == NodeType.node and p2["type"] == NodeType.abode:
            through_start, through_end = self.get_distance_through_ends(
                node=p1, abode_point=p2
            )
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        elif p1["type"] == NodeType.abode and p2["type"] == NodeType.node:
            through_start, through_end = self.get_distance_through_ends(
                node=p2, abode_point=p1
            )
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
                return self._abode_distances.get_distance(p1, p2)
            raise ValueError(
                f"Getting routed distance between points live is not recommended. Points: {p1}, {p2}"
            )


class MixDistances:
    def __init__(self, abode_distances: AbodeDistances, node_distances: NodeDistances):
        self._db = Database()
        self._abode_distances = abode_distances
        self._node_distances = node_distances

    def get_distance_through_ends(
        self, node: InternalPoint, abode_point: InternalPoint
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Determine the distances from an intersection point to a abode through the two ends of the abode's segments.

        Parameters
        ----------
            node (InternalPoint): an intersection point
            abode_point (InternalPoint): an abode point

        Returns
        -------
            Optional[float]: the distance from the intersection to the abode through the start of the segment
            Optional[float]: the distance from the intersection to the abode through the end of the segment
        """
        block_id = self._db.get_dict(abode_point["id"], ABODE_DB_IDX)["block_id"]
        block: Block = self._db.get_dict(block_id, BLOCK_DB_IDX)
        block_start = InternalPoint(
            lat=block["nodes"][0]["lat"],
            lon=block["nodes"][0]["lon"],
            type=NodeType.node,
            id="block_start",
        )
        through_start = self._node_distances.get_distance(node, block_start)

        if through_start is not None:
            through_start += block["abodes"][pt_id(abode_point)]["distance_to_start"]

        block_end = InternalPoint(
            lat=block["nodes"][-1]["lat"],
            lon=block["nodes"][-1]["lon"],
            type=NodeType.node,
            id="block_end",
        )
        through_end = self._node_distances.get_distance(node, block_end)
        if through_end is not None:
            through_end += block["abodes"][pt_id(abode_point)]["distance_to_end"]
        return through_start, through_end

    def get_distance(
        self, p1: InternalPoint, p2: InternalPoint
    ) -> Optional[tuple[float, float]] | Optional[float]:
        if "type" not in p1 or "type" not in p2:
            raise ValueError(
                "When retrieiving mix distances, both points must have a 'type'"
            )

        if p1["type"] == p2["type"] == NodeType.abode:
            return self._abode_distances.get_distance(p1, p2)
        elif p1["type"] == p2["type"] == NodeType.node:
            return self._node_distances.get_distance(p1, p2)
        elif p1["type"] == NodeType.node and p2["type"] == NodeType.abode:
            through_start, through_end = self.get_distance_through_ends(
                node=p1, abode_point=p2
            )
            if through_start is None and through_end is None:
                return None
            elif through_start is None:
                return through_end
            elif through_end is None:
                return through_start
            else:
                return min(through_start, through_end)
        elif p1["type"] == NodeType.abode and p2["type"] == NodeType.node:
            through_start, through_end = self.get_distance_through_ends(
                node=p2, abode_point=p1
            )
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
            abode_distances=self._abode_distances.snapshot(),
        )

    def get_distance_matrix(self, points: list[InternalPoint]):
        """
        Get the distance matrix for a list of points

        Parameters
        ----------
            points (list[Point]): a list of points
        """
        matrix = np.empty((len(points), len(points)), dtype=np.float32)

        snapshot = self.snapshot()

        num_empty_distances = 0

        for i, p1 in enumerate(
            tqdm(points, desc="Building mix distance matrix", unit="points", colour="green")
        ):
            for j, p2 in enumerate(points):
                d_or_dc = snapshot.get_distance(p1, p2)

                # Account for distance and cost stored on abodes
                d = d_or_dc if type(d_or_dc) is not tuple else d_or_dc[0]

                if d is None:
                    num_empty_distances += 1

                # Account for none values
                d = d if d is not None else MAX_STORAGE_DISTANCE

                matrix[i][j] = d

        portion_empty = num_empty_distances / (len(points) ** 2)

        if portion_empty > 0.2:
            print(
                colored(
                    f"Warning: {portion_empty * 100:.2f}% of the distance matrix is empty when snapshotting",
                    color="yellow",
                )
                + "This may be fine, but the cluster is likely larger than normal. Look at the cluster visualization to check."
            )

        return matrix
