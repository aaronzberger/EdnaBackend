from abc import ABC, abstractmethod
from src.config import Point

from src.distances.mix import MixDistances
from typing import TypedDict


class ProblemInfo(TypedDict):
    points: list[Point]
    num_vehicles: int
    num_depots: int
    num_points: int
    starts: list[int]
    ends: list[int]


class Optimizer(ABC):
    """
    Abstract class for an optimizer.

    Attributes
    ----------
        points (list[Point]): the points to be optimized (order matters)
        mix_distances (MixDistances): a computer for distances between the points
        problem_info (ProblemInfo): metadata for the problem
    """
    def __init__(self, block_ids: set[str], place_ids: set[str], voter_ids: set[str]):
        self.block_ids = block_ids
        self.place_ids = place_ids
        self.voter_ids = voter_ids

    def build_problem(self, mix_distances: MixDistances):
        self.mix_distances = mix_distances

    @abstractmethod
    def __call__(self, debug=False, time_limit_s=60):
        """
        Solve the problem. This method must be overridden by subclasses.
        """
        pass

    @classmethod
    def process_solution(cls, solution_file):
        raise NotImplementedError
