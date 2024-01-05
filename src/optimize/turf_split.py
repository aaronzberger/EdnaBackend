import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import AgglomerativeClustering

from src.config import MAX_TOURING_TIME, TIME_AT_HOUSE, WALKING_M_PER_S, Point
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize.optimizer import Optimizer, ProblemInfo


class SingleCluster(Optimizer):
    def __post_init__(self):
        self.node_distances = NodeDistances(
            block_ids=self.block_ids, skip_update=True
        )

        # Load block distance matrix
        self.block_distances = BlockDistances(
            block_ids=self.block_ids, node_distances=self.node_distances, skip_update=True
        )

        # Load house distance matrix
        self.house_distances = HouseDistances(
            block_ids=self.block_ids, node_distances=self.node_distances)

        # Load mix distance matrix
        self.mix_distances = MixDistances(
            house_distances=self.house_distances, node_distances=self.node_distances)

    def build_problem(
        self,
        houses: list[Point],
        potential_depots: list[Point],
        num_routes: int,
    ):
        """
        Build a turf split problem.

        Parameters
        ----------
        houses : list[Point]
            The houses to visit.
        potential_depots : list[Point]
            The depots to start from.
        num_routes : int
            The number of routes to create.
        """
        full_distance_matrix = self.mix_distances.get_distance_matrix(
            potential_depots + houses
        )

        centers = self.find_depots(full_distance_matrix)

        # TODO/NOTE This is much less efficient than taking a subset (and duplicating some entries)
        # of the matrix above. This is temporary
        self.distance_matrix = self.mix_distances.get_distance_matrix(centers + houses)
        self.distance_matrix = (
            (self.distance_matrix / WALKING_M_PER_S).round().astype(int).tolist()
        )

        self.points = centers + houses

        self.problem_info = ProblemInfo(
            num_vehicles=num_routes,
            num_depots=num_routes,
            num_points=num_routes + len(houses),
            starts=[i for i in range(num_routes)],
            ends=[i for i in range(num_routes)],
        )

    def find_depots(self, distance_matrix):
        # Simply cluster into num_depots clusters and find the centroids
        num_houses = self.problem_info["num_points"] - self.problem_info["num_depots"]
        house_indices = [i + self.problem_info["num_depots"] for i in range(num_houses)]

        depots = self.points[: self.problem_info["num_depots"]]
        houses = self.points[self.problem_info["num_depots"]:]

        house_distance_matrix = distance_matrix[house_indices]
        house_distance_matrix = house_distance_matrix[:, house_indices]

        clustered = AgglomerativeClustering(
            n_clusters=self.problem_info["num_depots"],
            linkage="complete",
            metric="precomputed",
        ).fit(distance_matrix)

        centers = []
        for cluster in np.unique(clustered.labels_):
            cluster_houses = []
            for label, point in zip(clustered.labels_, houses):
                if label == cluster:
                    cluster_houses.append(point)

            depot_sums = []
            for depot in depots:
                depot_sums.append(
                    [
                        self.mix_distances.get_distance(depot, house)
                        for house in cluster_houses
                    ]
                )

            # Choose the depot which minimizes distance to all these houses
            centers.append(depots[np.argmin(depot_sums)])

        return centers

    def __call__(self, debug=False, time_limit_s=60):
        


class TurfSplit(Optimizer):
    def build_problem(self):
        # TODO: Cluster block_ids and such into clusters, assign number of routes per cluster,
        # and create individual problems
        pass

    def __call__(self, debug=False, time_limit_s=60):
        # TODO: Actually execute the jobs: perhaps in parallel?
        pass
