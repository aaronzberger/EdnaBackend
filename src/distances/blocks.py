from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

from src.config import (
    ARBITRARY_LARGE_DISTANCE,
    Block,
    BLOCK_DISTANCE_MATRIX_DB_IDX,
    generate_id_pair,
    BLOCK_DB_IDX,
)
from src.distances.nodes import NodeDistances
from src.utils.db import Database


class BlockDistancesSnapshot:
    def __init__(self, snapshot: dict[str, float]):
        self.snapshot = snapshot

    def get_distance(self, b1_id: str, b2_id: str) -> Optional[float]:
        pair_1, pair_2 = generate_id_pair(b1_id, b2_id), generate_id_pair(
            b2_id, b1_id
        )
        if pair_1 in self.snapshot:
            return self.snapshot[pair_1]
        if pair_2 in self.snapshot:
            return self.snapshot[pair_2]
        return None


class BlockDistances:
    def _insert_pair(self, b1: Block, b1_id: str, b2: Block, b2_id: str):
        pair_1, pair_2 = generate_id_pair(b1_id, b2_id), generate_id_pair(
            b2_id, b1_id
        )
        if self._db.exists(pair_1, BLOCK_DISTANCE_MATRIX_DB_IDX) or self._db.exists(
            pair_2, BLOCK_DISTANCE_MATRIX_DB_IDX
        ):
            return

        routed_distances = [
            self._snapshot.get_distance(i, j)
            for (i, j) in [
                (b1["nodes"][0], b2["nodes"][0]),
                (b1["nodes"][0], b2["nodes"][-1]),
                (b1["nodes"][-1], b2["nodes"][0]),
                (b1["nodes"][-1], b2["nodes"][-1]),
            ]
        ]

        existing_distances = [i for i in routed_distances if i is not None]
        if len(existing_distances) > 0:
            self._db.set_str(
                pair_1, str(min(existing_distances)), BLOCK_DISTANCE_MATRIX_DB_IDX
            )

    def _update(self, blocks: dict[str, Block]):
        """
        Update the block distance table by adding any missing blocks

        Parameters
        ----------
            blocks (dict[str, Block]): the blocks to confirm are in the table, and to add if they are not

        Notes
        -----
            Time complexity: O(n^2), where n is the number of blocks
        """
        with tqdm(
            total=len(blocks) ** 2, desc="Updating block distance matrix", unit="pairs", colour="green"
        ) as progress:
            for b_id, block in blocks.items():
                for other_b_id, other_block in blocks.items():
                    self._insert_pair(block, b_id, other_block, other_b_id)
                    progress.update()

    def __init__(
        self,
        block_ids: set[str],
        node_distances: NodeDistances,
        skip_update: bool = False,
    ):
        self._db = Database()
        self._node_distances = node_distances

        if skip_update:
            return

        blocks: dict[str, Block] = {}
        for block_id in block_ids:
            block = self._db.get_dict(block_id, BLOCK_DB_IDX)

            if block is None:
                raise KeyError(
                    "Block with ID {} not found in database.".format(block_id)
                )

            # TODO: Decide globally if we should unpack into a Block (Block(**block))), or assume written correctly
            blocks[block_id] = block

        self._snapshot = self._node_distances.snapshot()

        self._update(blocks)

    def get_distance(self, b1_id: str, b2_id: str) -> Optional[float]:
        """
        Get the distance between two blocks by their coordinates

        Parameters:
            b1 (Block): the first block
            b2 (Block): the second block

        Returns:
            float | None: distance between the two blocks if it exists, None otherwise
        """
        pair_1, pair_2 = generate_id_pair(b1_id, b2_id), generate_id_pair(
            b2_id, b1_id
        )

        pair_1_r = self._db.get_str(pair_1, BLOCK_DISTANCE_MATRIX_DB_IDX)
        if pair_1_r is not None:
            return float(pair_1_r)

        pair_2_r = self._db.get_str(pair_2, BLOCK_DISTANCE_MATRIX_DB_IDX)
        if pair_2_r is not None:
            return float(pair_2_r)

        return None

    def get_distance_matrix(self, block_ids: set[str]):
        # Take a snapshot of the block distances
        snapshot = BlockDistancesSnapshot(
            self._db.get_all_dict(BLOCK_DISTANCE_MATRIX_DB_IDX)
        )

        matrix = np.empty((len(block_ids), len(block_ids)), dtype=np.float32)
        with tqdm(
            total=len(block_ids) ** 2,
            desc="Retrieving matrix",
            unit="pairs",
            colour="green",
            leave=False,
        ) as progress:
            for r, block in enumerate(block_ids):
                for c, other_block in enumerate(block_ids):
                    distance = snapshot.get_distance(block, other_block)
                    matrix[r][c] = (
                        ARBITRARY_LARGE_DISTANCE if distance is None else distance
                    )
                    progress.update()
        return matrix
