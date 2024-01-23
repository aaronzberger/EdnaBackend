"""
The basic solver that all problem types call to get routes.
"""

from typing import TypedDict
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.config import MAX_TOURING_TIME, TIME_AT_HOUSE, WALKING_M_PER_S, InternalPoint
from src.distances.mix import MixDistances
from src.utils.viz import display_distance_matrix


class ProblemInfo(TypedDict):
    points: list[InternalPoint]
    num_vehicles: int
    num_depots: int
    num_points: int
    starts: list[int]
    ends: list[int]


class BaseSolver:
    def __init__(self, problem_info: ProblemInfo, mix_distances: MixDistances):
        """
        Create a group canvas problem.

        Parameters
        ----------
        problem_info : ProblemInfo
            The problem metadata.
        mix_distances : MixDistances
            The distance matrix to use.
        """
        self.problem_info = problem_info
        self.mix_distances = mix_distances

    def __call__(self, debug=True, time_limit_s=10) -> list[list[InternalPoint]]:
        # Construct the distance matrix
        distance_matrix = self.mix_distances.get_distance_matrix(
            self.problem_info["points"]
        )

        if debug:
            display_distance_matrix(
                points=self.problem_info["points"], distances=distance_matrix
            )

        # Convert distance matrix to input format
        distance_matrix = distance_matrix / WALKING_M_PER_S

        # Add the stopping time to the distance matrix (add to the arriving node)
        house_indices = range(
            self.problem_info["num_vehicles"], self.problem_info["num_points"]
        )
        distance_matrix[house_indices, :] += TIME_AT_HOUSE.seconds
        distance_matrix[:, house_indices] += TIME_AT_HOUSE.seconds
        np.fill_diagonal(distance_matrix, 0)

        # Convert to int
        distance_matrix = distance_matrix.round().astype(int).tolist()

        # Build the problem
        manager = pywrapcp.RoutingIndexManager(
            self.problem_info["num_points"],
            self.problem_info["num_vehicles"],
            self.problem_info["starts"],
            self.problem_info["ends"],
        )
        routing = pywrapcp.RoutingModel(manager)

        transit_callback_index = routing.RegisterTransitMatrix(distance_matrix)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time constraint
        time_dimension_name = "Time"
        routing.AddDimension(
            transit_callback_index,
            0,  # time at each node
            MAX_TOURING_TIME.seconds,  # walker maximum travel time
            True,  # start cumul to zero
            time_dimension_name,
        )

        # Limit the time of the maximum route (not needed?)
        # distance_dimension = routing.GetDimensionOrDie(time_dimension_name)
        # distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Allow dropping houses
        penalty = 10000000
        for node in range(
            self.problem_info["num_vehicles"], self.problem_info["num_points"]
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

        # Convert to universal format
        house_routes: list[list[InternalPoint]] = []
        for route in routes:
            house_route = []
            for node in route:
                house_route.append(self.problem_info["points"][node])
            house_routes.append(house_route)

        return house_routes
