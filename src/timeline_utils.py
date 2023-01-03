from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

from src.config import MINS_PER_HOUSE, WALKING_M_PER_S
from src.gps_utils import Point, great_circle_distance


@dataclass
class Segment():
    id: str
    start: Point
    end: Point
    num_houses: int
    navigation_points: list[Point]
    type: Literal['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']

    def __post_init__(self):
        self.length: float = 0.0
        for first, second in itertools.pairwise(self.navigation_points):
            self.length += great_circle_distance(first, second)
        self.time_to_walk = self.num_houses * MINS_PER_HOUSE + (self.length / WALKING_M_PER_S * (1/60))


@dataclass
class SubSegment():
    segment: Segment

    # Note that start and end could be the same point.
    start: Point
    end: Point

    # Furthest points in each direction the houses span
    extremum: tuple[Point, Point]

    houses: list[Point]
    navigation_points: list[Point]

    def __post_init__(self):
        self.length: float = 0.0
        self.length += great_circle_distance(self.start, self.navigation_points[0])
        for first, second in itertools.pairwise(self.navigation_points):
            self.length += great_circle_distance(first, second)
        self.length += great_circle_distance(self.end, self.navigation_points[-1])

        # TODO: Time to walk depends on walk_method and should likely be iterated through
        self.time_to_walk = len(self.houses) * MINS_PER_HOUSE + (self.length / WALKING_M_PER_S * (1/60))
