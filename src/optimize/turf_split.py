import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import AgglomerativeClustering

from src.config import MAX_TOURING_TIME, TIME_AT_HOUSE, WALKING_M_PER_S, Point
from src.distances.mix import MixDistances
from src.optimize.optimizer import Optimizer, ProblemInfo


def distance_matrix_subset(distance_matrix, selected_indices):
    """
    Given a distance matrix, return a subset of the distance matrix
    corresponding to the selected indices

    Parameters
    ----------
        distance_matrix (list): a list of lists, where each sublist is a row of the distance matrix
        selected_indices (list): a list of indices to keep

    Returns
    -------
        subset_distance_matrix (list): a list of lists, where each sublist is a row of the distance matrix
    """
    subset_distance_matrix = []

    # Remove the extra rows
    subset_distance_matrix = [distance_matrix[i] for i in selected_indices]

    # Remove the extra columns
    for i in range(len(subset_distance_matrix)):
        subset_distance_matrix[i] = [
            subset_distance_matrix[i][j] for j in selected_indices
        ]

    return subset_distance_matrix


class TurfSplit(Optimizer):
    def __init__(
        self,
        houses: list[Point],
        potential_depots: list[Point],
        num_routes: int,
        mix_distances: MixDistances,
    ):
        """
        Create a group canvas problem

        Parameters
        ----------
        houses
        potential_depots
        num_routes
        mix_distances
        """
        super().__init__(mix_distances=mix_distances)

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
        # Build the problem
        manager = pywrapcp.RoutingIndexManager(
            self.problem_info["num_points"],
            self.problem_info["num_vehicles"],
            self.problem_info["starts"],
            self.problem_info["ends"],
        )
        routing = pywrapcp.RoutingModel(manager)

        transit_callback_index = routing.RegisterTransitMatrix(self.distance_matrix)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time constraint
        time_dimension_name = "Time"
        routing.AddDimension(
            transit_callback_index,
            TIME_AT_HOUSE.seconds,  # time at each node
            MAX_TOURING_TIME.seconds,  # walker maximum travel time
            True,  # start cumul to zero
            time_dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(time_dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Allow dropping houses
        penalty = 10000
        for node in range(
            self.problem_info["num_depots"], self.problem_info["num_points"]
        ):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.log_search = debug
        search_parameters.time_limit.seconds = time_limit_s

        solution = routing.SolveWithParameters(search_parameters)

        # Process the solution
        if not solution:
            raise RuntimeError("solution from optimizer was empty.")

        routes = []
        for vehicle_id in range(self.problem_info["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            routes.append(route)
            print(route)

        # Convert to universal format
        house_routes: list[list[Point]] = []
        for route in routes:
            house_route = []
            for node in route:
                house_route.append(self.points[node])
            house_routes.append(house_route)

        return house_routes
