from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from typing import Optional

import numpy as np
from nptyping import Float32, NDArray, Shape
from tqdm import tqdm

from src.config import (ARBITRARY_LARGE_DISTANCE, Block,
                        block_distance_matrix_file, blocks_file_t)
from src.distances.nodes import NodeDistances


class BlockDistances():
    _block_distances: dict[str, dict[str, float]] = {}
    _blocks: blocks_file_t = {}

    @classmethod
    def _insert_pair(cls, b1: Block, b1_id: str, b2: Block, b2_id: str):
        # If this pair already exists in the opposite order, skip
        try:
            cls._block_distances[b2_id][b1_id]
        except KeyError:
            routed_distances = \
                [NodeDistances.get_distance(i, j) for i, j in
                    [(b1['nodes'][0], b2['nodes'][0]), (b1['nodes'][0], b2['nodes'][-1]),
                     (b1['nodes'][-1], b2['nodes'][0]), (b1['nodes'][-1], b2['nodes'][-1])]]
            existing_distances = [i for i in routed_distances if i is not None]
            if len(existing_distances) > 0:
                cls._block_distances[b1_id][b2_id] = min(existing_distances)

    @classmethod
    def __init__(cls, blocks: blocks_file_t):
        cls._blocks = deepcopy(blocks)

        if os.path.exists(block_distance_matrix_file):
            print('Block distance table file found.')
            cls._block_distances = json.load(open(block_distance_matrix_file))
            need_regeneration = False
            num_samples = min(len(cls._blocks), 100)
            for block_id in random.sample(cls._blocks.keys(), num_samples):
                if block_id not in cls._block_distances:
                    need_regeneration = True
                    break
            if not need_regeneration:
                return
            else:
                print('The saved block distance table did not include all requested blocks. Regenerating...')
        else:
            print('No block distance table file found at {}. Generating now...'.format(block_distance_matrix_file))

        cls._block_distances = {}
        with tqdm(total=len(blocks) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for b_id, block in blocks.items():
                cls._block_distances[b_id] = {}
                for other_b_id, other_block in blocks.items():
                    cls._insert_pair(block, b_id, other_block, other_b_id)
                    progress.update()

            print('Saving to {}'.format(block_distance_matrix_file))
            json.dump(cls._block_distances, open(block_distance_matrix_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, b1_id: str, b2_id: str) -> Optional[float]:
        '''
        Get the distance between two blocks by their coordinates

        Parameters:
            b1 (Block): the first block
            b2 (Block): the second block

        Returns:
            float | None: distance between the two blocks if it exists, None otherwise
        '''
        try:
            return cls._block_distances[b1_id][b2_id]
        except KeyError:
            try:
                return cls._block_distances[b2_id][b1_id]
            except KeyError:
                return None

    @classmethod
    def get_distance_matrix(cls) -> NDArray[Shape[len(_blocks), len(_blocks)], Float32]:
        matrix = np.empty((len(cls._blocks), len(cls._blocks)), dtype=np.float32)
        for r, block in enumerate(cls._blocks):
            for c, other_block in enumerate(cls._blocks):
                distance = cls.get_distance(block, other_block)
                matrix[r][c] = ARBITRARY_LARGE_DISTANCE if distance is None else distance
        return matrix
