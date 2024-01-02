from src.config import (
    MAX_TOURING_TIME,
    TIME_AT_HOUSE,
    OPTIM_OBJECTIVES,
    WALKING_M_PER_S,
    DistanceMatrix,
    Job,
    Location,
    NodeType,
    PlaceTW,
    Plan,
    Point,
    Problem,
    Service,
    pt_id,
)
from src.distances.mix import MixDistances
from src.optimize.optimizer import Optimizer, ProblemInfo

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class GroupCanvas(Optimizer):
    def __init__(
        self, houses: list[Point], depots: list[Point], mix_distances: MixDistances
    ):
        """
        Create a group canvas problem

        Parameters
        ----------
        houses
        depots
        mix_distances
        """
        super().__init__(mix_distances=mix_distances)

        self.points = depots + houses
        self.start_idx = 0

        num_depots = len(depots)

        self.problem_info = ProblemInfo(
            num_vehicles=num_depots,
            num_depots=num_depots,
            num_points=num_depots + len(houses),
            starts=[i for i in range(len(depots))],
            ends=[i for i in range(len(depots))],
        )

    def __call__(self, debug=False, time_limit_s=60):
        # Construct the distance matrix
        distance_matrix = self.mix_distances.get_distance_matrix(self.points)

        print(
            "Distance matrix has shape {}".format(
                (len(distance_matrix), len(distance_matrix[0]))
            )
        )

        # Convert distance matrix to input format
        distance_matrix = (
            (distance_matrix / WALKING_M_PER_S).round().astype(int).tolist()
        )

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
