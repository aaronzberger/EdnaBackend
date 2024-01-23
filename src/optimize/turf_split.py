import math
import random
import sys
import numpy as np
from sklearn.cluster import KMeans

from src.config import (
    BLOCK_DB_IDX,
    NodeType,
    InternalPoint,
    pt_id,
    MAX_STORAGE_DISTANCE,
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances

# from src.optimize.optimizer import Optimizer
from src.optimize.base_solver import BaseSolver, ProblemInfo
from src.utils.db import Database
from src.clustering import Clustering
from src.utils.viz import display_clustered_blocks


class SingleCluster:
    def __init__(self, block_ids: set[str], abode_ids: set[str], num_routes: int):
        """
        Create an individual turf split problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        abode_ids : set[str]
            The abodes to visit.
        num_routes : int
            The number of routes to create.
        """
        self.block_ids = block_ids
        self.abode_ids = abode_ids
        # super().__init__(block_ids=block_ids, abode_ids=abode_ids)

        print(
            f"Setting up a cluster with {len(self.block_ids)} blocks and {len(self.abode_ids)} abodes"
        )

        self.db = Database()

        self.num_routes = num_routes

        # region Load distance matrices
        self.node_distances = NodeDistances(block_ids=self.block_ids, skip_update=True)

        self.block_distances = BlockDistances(
            block_ids=self.block_ids,
            node_distances=self.node_distances,
            skip_update=True,
        )

        self.house_distances = HouseDistances(
            block_ids=self.block_ids, node_distances=self.node_distances
        )

        self.mix_distances = MixDistances(
            house_distances=self.house_distances, node_distances=self.node_distances
        )
        # endregion

        # region Retrieve the abode and node points
        self.abode_points: list[InternalPoint] = []
        self.potential_depots: list[InternalPoint] = []
        inserted_node_ids: set[str] = set()

        for block_id in self.block_ids:
            block = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block is None:
                print(f"Block {block_id} not found in database")
                sys.exit(1)

            # Find which abodes on this block are in the universe
            self.matching_abode_ids = set(block["abodes"].keys()).intersection(
                self.abode_ids
            )

            # Add the abodes to the list of abodes to visit
            self.abode_points.extend(
                InternalPoint(
                    lat=block["abodes"][i]["point"]["lat"],
                    lon=block["abodes"][i]["point"]["lon"],
                    id=i,
                    type=NodeType.house,
                )
                for i in self.matching_abode_ids
            )

            # Add the nodes to the list of potential depots
            for i in [0, -1]:
                if pt_id(block["nodes"][i]) not in inserted_node_ids:
                    self.potential_depots.append(
                        InternalPoint(
                            lat=block["nodes"][i]["lat"],
                            lon=block["nodes"][i]["lon"],
                            id=pt_id(block["nodes"][i]),
                            type=NodeType.node,
                        )
                    )
                    inserted_node_ids.add(pt_id(block["nodes"][i]))
        # endregion

        print(
            f"Found {len(self.abode_points)} abodes and {len(self.potential_depots)} potential depots"
        )

        self.depots = self.find_depots(
            num_depots=num_routes,
            abodes=self.abode_points,
            potential_depots=self.potential_depots,
        )

        print(f"Found {len(self.depots)} depots: {self.depots}")

        self.build_problem(houses=self.abode_points, depots=self.depots)

    def build_problem(
        self,
        houses: list[InternalPoint],
        depots: list[InternalPoint],
    ):
        """
        Build a turf split problem.

        Parameters
        ----------
        houses : list[Point]
            The houses to visit.
        depots : list[Point]
            The depots to start from.
        """
        # # TODO/NOTE This is much less efficient than taking a subset (and duplicating some entries)
        # # of the matrix from find_depots, but it's much easier to implement.
        # self.distance_matrix = self.mix_distances.get_distance_matrix(depots + houses)
        # self.distance_matrix = (
        #     (self.distance_matrix / WALKING_M_PER_S).round().astype(int).tolist()
        # )

        self.points = depots + houses

        self.problem_info = ProblemInfo(
            points=self.points,
            num_vehicles=self.num_routes,
            num_depots=self.num_routes,
            num_points=self.num_routes + len(houses),
            starts=[i for i in range(self.num_routes)],
            ends=[i for i in range(self.num_routes)],
        )

    def find_depots(
        self,
        num_depots: int,
        abodes: list[InternalPoint],
        potential_depots: list[InternalPoint],
    ) -> list[InternalPoint]:
        """
        Find the optimal depots for the given abodes.

        NOTE: In the future, this should be replaced by the actual turf split optimization problem,
        whereby depots are chosen to maximize total number of houses hit in the cluster. For now,
        we use a simple heuristic of choosing centers of a predermined number of clusters.

        Parameters
        ----------
        num_depots : int
            The number of depots to find.
        abodes : list[Point]
            The abodes to find depots for.
        potential_depots : list[Point]
            The potential depots to choose from.

        Returns
        -------
        list[Point]
            The depots.
        """
        distance_matrix = self.mix_distances.get_distance_matrix(self.abode_points)

        clustered = KMeans(n_clusters=num_depots, random_state=0).fit(distance_matrix)

        centers = []
        for cluster in range(num_depots):
            # Re-create the list of houses in this cluster
            cluster_houses = [
                house
                for label, house in zip(clustered.labels_, abodes)
                if label == cluster
            ]

            # Find the depot closest to these houses
            depot_sums = []
            for depot in potential_depots:
                depot_sum = 0
                for house in random.sample(
                    cluster_houses, min(len(cluster_houses), 100)
                ):
                    distance = self.mix_distances.get_distance(depot, house)
                    if isinstance(distance, float):
                        depot_sum += distance
                    else:
                        depot_sum += MAX_STORAGE_DISTANCE
                depot_sums.append(depot_sum)

            print("Finished calculating distances")

            # Choose the depot which minimizes distance to all these houses
            centers.append(potential_depots[np.argmin(depot_sums)])

        return centers

    def __call__(self, debug=False, time_limit_s=60) -> list[list[InternalPoint]]:
        """
        Solve the problem which has been constructed.
        """
        return BaseSolver(
            problem_info=self.problem_info, mix_distances=self.mix_distances
        )(debug=debug, time_limit_s=time_limit_s)


class TurfSplit:
    def __init__(self, block_ids: set[str], abode_ids: set[str], num_routes: int):
        """
        Create a turf split problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        abode_ids : set[str]
            The abodes to visit.
        num_routes : int
            The number of routes to create.
        """
        self.block_ids = block_ids
        self.abode_ids = abode_ids
        # super().__init__(block_ids=block_ids, abode_ids=abode_ids)

        self.num_routes = num_routes

        self.build_problem(
            block_ids=self.block_ids,
            abode_ids=self.abode_ids,
            num_routes=self.num_routes,
        )

    def build_problem(self, block_ids: set[str], abode_ids: set[str], num_routes: int):
        # Cluster into groups of manageable size
        self.clusters = Clustering(block_ids=block_ids, abode_ids=abode_ids)()

        # Display the clusters
        labels = []
        for i, cluster in enumerate(self.clusters):
            labels.extend([i] * len(cluster["block_ids"]))

        ordered_block_ids = []
        for cluster in self.clusters:
            ordered_block_ids.extend(cluster["block_ids"])

        display_clustered_blocks(block_ids=ordered_block_ids, labels=labels)

        # Assign number of routes to each cluster
        self.num_routes = []
        if num_routes == len(self.clusters):
            self.num_routes = [1] * len(self.clusters)
        elif num_routes < len(self.clusters):
            # Choose the tighest clusters to generate on (since they'll likely have the best routes)
            top_clusters = sorted(
                self.clusters, key=lambda x: x["tightness"], reverse=True
            )[:num_routes]
            self.num_routes = [
                1 if cluster in top_clusters else 0 for cluster in self.clusters
            ]
        else:
            # Allocate routes to the tightest clusters
            top_clusters = sorted(self.clusters, key=lambda x: x["tightness"])

            num_to_top_clusters = math.ceil(num_routes / len(self.clusters))
            num_to_other_clusters = math.floor(num_routes / len(self.clusters))
            num_top_clusters = num_routes % len(self.clusters)

            self.num_routes = [
                num_to_top_clusters
                if cluster in top_clusters[:num_top_clusters]
                else num_to_other_clusters
                for cluster in self.clusters
            ]

        for i, cluster in enumerate(self.clusters):
            print(
                f"Cluster {i} has {len(cluster['block_ids'])} blocks, {len(cluster['abode_ids'])} abodes, " +
                f"tightness {cluster['tightness']:.0f}, and thus {self.num_routes[i]} routes"
            )

        self.problems = []

        for cluster, cluster_routes in zip(self.clusters, self.num_routes):
            if cluster_routes > 0:
                self.problems.append(
                    SingleCluster(
                        block_ids=cluster["block_ids"],
                        abode_ids=cluster["abode_ids"],
                        num_routes=cluster_routes,
                    )
                )

        # For access in post-processing
        self.mix_distances = [problem.mix_distances for problem in self.problems]

    def __call__(self, debug=False, time_limit_s=60) -> list[list[list[InternalPoint]]]:
        routes: list[list[list[InternalPoint]]] = []

        for problem in self.problems:
            routes.append(problem(debug=debug, time_limit_s=time_limit_s))

        return routes
