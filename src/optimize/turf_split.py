import math
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.config import BLOCK_DB_IDX, WALKING_M_PER_S, NodeType, Point, pt_id
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize.optimizer import Optimizer
from src.optimize.base_solver import ProblemInfo
from src.utils.db import Database
from src.clustering import Clustering
from src.utils.viz import display_clustered_blocks


class SingleCluster(Optimizer):
    def __init__(self, block_ids: set[str], place_ids: set[str], num_routes: int):
        """
        Create an individual turf split problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        place_ids : set[str]
            The places to visit.
        num_routes : int
            The number of routes to create.
        """
        super().__init__(block_ids=block_ids, place_ids=place_ids)

        self.db = Database()

        self.num_routes = num_routes

        # region Load distance matrices
        self.node_distances = NodeDistances(
            block_ids=self.block_ids, skip_update=True
        )

        self.block_distances = BlockDistances(
            block_ids=self.block_ids, node_distances=self.node_distances, skip_update=True
        )

        self.house_distances = HouseDistances(
            block_ids=self.block_ids, node_distances=self.node_distances)

        self.mix_distances = MixDistances(
            house_distances=self.house_distances, node_distances=self.node_distances)
        # endregion

        # region Retrieve the place and node points
        self.places: list[Point] = []
        self.potential_depots: list[Point] = []
        inserted_node_ids: set[str] = set()

        for block_id in self.block_ids:
            block = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block is None:
                print(f"Block {block_id} not found in database")
                sys.exit(1)

            # Find which places on this block are in the universe
            self.matching_place_ids = set(block["places"].keys()).intersection(self.place_ids)

            # Add the places to the list of places to visit
            self.places.extend(
                Point(lat=block["places"][i]["lat"], lon=block["places"][i]["lon"], id=i, type=NodeType.house) for i in self.matching_place_ids)

            # Add the nodes to the list of potential depots
            for i in [0, -1]:
                if pt_id(block["nodes"][i]) not in inserted_node_ids:
                    self.potential_depots.append(
                        Point(
                            lat=block["nodes"][i]["lat"],
                            lon=block["nodes"][i]["lon"],
                            id=pt_id(block["nodes"][i]),
                            type=NodeType.node,
                        )
                    )
                    inserted_node_ids.add(pt_id(block["nodes"][i]))
        # endregion

        self.depots = self.find_depots(num_depots=num_routes, places=self.places, potential_depots=self.potential_depots)

        self.build_problem(houses=self.places, depots=self.depots)

    def build_problem(
        self,
        houses: list[Point],
        depots: list[Point],
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
        # TODO/NOTE This is much less efficient than taking a subset (and duplicating some entries)
        # of the matrix from find_depots, but it's much easier to implement.
        self.distance_matrix = self.mix_distances.get_distance_matrix(depots + houses)
        self.distance_matrix = (
            (self.distance_matrix / WALKING_M_PER_S).round().astype(int).tolist()
        )

        self.points = depots + houses

        self.problem_info = ProblemInfo(
            num_vehicles=self.num_routes,
            num_depots=self.num_routes,
            num_points=self.num_routes + len(houses),
            starts=[i for i in range(self.num_routes)],
            ends=[i for i in range(self.num_routes)],
        )

    def find_depots(self, num_depots: int, places: list[Point], potential_depots: list[Point]) -> list[Point]:
        """
        Find the optimal depots for the given places.

        NOTE: In the future, this should be replaced by the actual turf split optimization problem,
        whereby depots are chosen to maximize total number of houses hit in the cluster. For now,
        we use a simple heuristic of choosing centers of a predermined number of clusters.

        Parameters
        ----------
        num_depots : int
            The number of depots to find.
        places : list[Point]
            The places to find depots for.
        potential_depots : list[Point]
            The potential depots to choose from.

        Returns
        -------
        list[Point]
            The depots.
        """
        distance_matrix = self.mix_distances.get_distance_matrix(self.places)

        clustered = AgglomerativeClustering(
            n_clusters=num_depots,
            linkage="complete",
            metric="precomputed",
        ).fit(distance_matrix)

        centers = []
        for cluster in np.unique(clustered.labels_):
            # Re-create the list of houses in this cluster
            cluster_houses = []
            for label, house in zip(clustered.labels_, places):
                if label == cluster:
                    cluster_houses.append(house)

            # Find the depot closest to these houses
            depot_sums = []
            for depot in potential_depots:
                depot_sums.append(
                    [
                        self.mix_distances.get_distance(depot, house)
                        for house in cluster_houses
                    ]
                )

            # Choose the depot which minimizes distance to all these houses
            centers.append(potential_depots[np.argmin(np.sum(depot_sums, axis=1))])

        return centers


class TurfSplit(Optimizer):
    def __init__(self, block_ids: set[str], place_ids: set[str], num_routes: int):
        """
        Create a turf split problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        place_ids : set[str]
            The places to visit.
        num_routes : int
            The number of routes to create.
        """
        super().__init__(block_ids=block_ids, place_ids=place_ids)

        self.num_routes = num_routes

        self.build_problem(block_ids=self.block_ids, place_ids=self.place_ids, num_routes=self.num_routes)

    def build_problem(self, block_ids: set[str], place_ids: set[str], num_routes: int):
        # Cluster into groups of manageable size
        self.clusters = Clustering(block_ids=block_ids, place_ids=place_ids)()

        # Display the clusters
        labels = []
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i} has {len(cluster['block_ids'])} blocks and {len(cluster['place_ids'])} places")
            labels.extend([i] * len(cluster["block_ids"]))

        display_clustered_blocks(block_ids=block_ids, labels=labels)
        # display_clustered_blocks(clusters=[cluster["block_ids"] for cluster in self.clusters], labels=list(range(len(self.clusters))))
        sys.exit(0)

        # Assign number of routes to each cluster
        self.num_routes = []
        if num_routes == len(self.clusters):
            self.num_routes = [1] * len(self.clusters)
        elif num_routes < len(self.clusters):
            # Choose the tighest clusters to generate on (since they'll likely have the best routes)
            top_clusters = sorted(self.clusters, key=lambda x: x["tightness"])[:num_routes]
            self.num_routes = [1 if cluster in top_clusters else 0 for cluster in self.clusters]
        else:
            # Allocate routes to the tightest clusters
            top_clusters = sorted(self.clusters, key=lambda x: x["tightness"])

            num_to_top_clusters = math.ceil(num_routes / len(self.clusters))
            num_to_other_clusters = math.floor(num_routes / len(self.clusters))
            num_top_clusters = num_routes % len(self.clusters)

            self.num_routes = [num_to_top_clusters if cluster in top_clusters[:num_top_clusters] else
                               num_to_other_clusters for cluster in self.clusters]

        self.problems = []

        for cluster, num_routes in zip(self.clusters, self.num_routes):
            self.problems.append(SingleCluster(
                block_ids=cluster["block_ids"], place_ids=cluster["place_ids"], num_routes=num_routes))

    def __call__(self, debug=False, time_limit_s=60) -> list[list[Point]]:
        routes: list[list[Point]] = []

        for problem in self.problems:
            routes.extend(problem(debug=debug, time_limit_s=time_limit_s))

        return routes
