from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from typing import Optional

import numpy as np
from nptyping import Float32, NDArray, Shape
from src.config import ARBITRARY_LARGE_DISTANCE, segment_distance_matrix_file
from src.distances.nodes import NodeDistances
from src.timeline_utils import Segment
from tqdm import tqdm


class SegmentDistances():
    _segment_distances: dict[str, dict[str, Optional[float]]] = {}
    _segments: list[Segment] = []

    @classmethod
    def _insert_pair(cls, s1: Segment, s2: Segment):
        # If this pair already exists in the opposite order, skip
        try:
            cls._segment_distances[s2.id][s1.id]
        except KeyError:
            routed_distances = \
                [NodeDistances.get_distance(i, j) for i, j in
                    [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]
            existing_distances = [i for i in routed_distances if i is not None]
            cls._segment_distances[s1.id][s2.id] = None if len(existing_distances) == 0 else min(existing_distances)

    @classmethod
    def __init__(cls, segments: list[Segment]):
        cls._segments = deepcopy(segments)

        if os.path.exists(segment_distance_matrix_file):
            print('Segment distance table file found.')
            cls._segment_distances = json.load(open(segment_distance_matrix_file))
            need_regeneration = False
            num_samples = min(len(cls._segments), 100)
            for segment in random.sample(cls._segments, num_samples):
                if segment.id not in cls._segment_distances:
                    need_regeneration = True
                    break
            if not need_regeneration:
                return
            else:
                print('The saved segment distance table did not include all requested segments. Regenerating...')
        else:
            print('No segment distance table file found at {}. Generating now...'.format(segment_distance_matrix_file))

        cls._segment_distances = {}
        with tqdm(total=len(segments) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for segment in segments:
                cls._segment_distances[segment.id] = {}
                for other_segment in segments:
                    cls._insert_pair(segment, other_segment)
                    progress.update()

            print('Saving to {}'.format(segment_distance_matrix_file))
            json.dump(cls._segment_distances, open(segment_distance_matrix_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, s1: Segment, s2: Segment) -> Optional[float]:
        '''
        Get the distance between two segments

        Parameters:
            s1 (Segment): the first segment
            s2 (Segment): the second segment

        Returns:
            float: distance between the two segments

        Raises:
            KeyError: if the pair does not exist in the table
        '''
        try:
            return cls._segment_distances[s1.id][s2.id]
        except KeyError:
            return cls._segment_distances[s2.id][s1.id]

    @classmethod
    def get_distance_matrix(cls) -> NDArray[Shape[len(_segments), len(_segments)], Float32]:
        matrix = np.empty((len(cls._segments), len(cls._segments)), dtype=np.float32)
        for r, segment in enumerate(cls._segments):
            for c, other_segment in enumerate(cls._segments):
                distance = cls.get_distance(segment, other_segment)
                matrix[r][c] = ARBITRARY_LARGE_DISTANCE if distance is None else distance
        return matrix
