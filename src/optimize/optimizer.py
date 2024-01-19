from abc import ABC, abstractmethod
from src.config import Point

from src.distances.mix import MixDistances

from src.optimize.base_solver import BaseSolver


class Optimizer(ABC):
    """
    Abstract class for an optimizer.

    Attributes
    ----------
        points (list[Point]): the points to be optimized (order matters)
        mix_distances (MixDistances): a computer for distances between the points
        problem_info (ProblemInfo): metadata for the problem
    """
    @abstractmethod
    def __init__(self, block_ids: set[str], place_ids: set[str]):
        self.block_ids = block_ids
        self.place_ids = place_ids

    @abstractmethod
    def build_problem(self, mix_distances: MixDistances):
        self.mix_distances = mix_distances

    def __call__(self, debug=False, time_limit_s=60) -> list[list[Point]]:
        """
        Solve the problem which has been constructed.
        """
        return BaseSolver(
            problem_info=self.problem_info, mix_distances=self.mix_distances
        )()
