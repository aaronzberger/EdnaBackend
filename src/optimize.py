import json
import os
import sys
import time

from folium import Map
from ortools.constraint_solver import (pywrapcp, routing_enums_pb2,
                                       routing_parameters_pb2)
from termcolor import colored

from config import (BASE_DIR, KEEP_APARTMENTS, MINS_PER_HOUSE, WALKING_M_PER_S,
                    blocks_file, blocks_file_t)
from gps_utils import Point
from house_distances import HouseDistances
from timeline_utils import Segment
from viz_utils import display_house_orders


def cluster_to_houses(cluster: list[Segment]) -> dict[str, Point]:
    # Load address_points.csv into a dictionary with address as key
    print('Loading coordinates of houses...')
    blocks: blocks_file_t = json.load(open(blocks_file))

    houses_in_cluster: dict[str, Point] = {}

    for segment in cluster:
        try:
            for address in blocks[segment.id]['addresses']:
                if not KEEP_APARTMENTS and ' APT ' in address:
                    continue
                houses_in_cluster[address] = Point(blocks[segment.id]['addresses'][address]['lat'],
                                                   blocks[segment.id]['addresses'][address]['lon'],
                                                   id=address)
        except KeyError:
            print('Couldn\'t find segment with ID {}'.format(segment.id))

    return houses_in_cluster


class Optimizer():
    MAX_HOUSES_PER_LIST = 90
    TIME_LIMIT = 10
    MAX_TIME_PER_LIST = 180 * 60

    def __init__(self, cluster: list[Segment], num_lists: int, starting_location: Point):
        houses_in_cluster = cluster_to_houses(cluster)
        self.points = list(houses_in_cluster.values())
        self.points.append(starting_location)
        self.start_idx = len(self.points) - 1

        self.num_lists = num_lists

        HouseDistances(cluster, starting_location)

        # Create the routing index manager.
        self.manager = pywrapcp.RoutingIndexManager(len(self.points), self.num_lists, self.start_idx)

        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)

    def get_time_index(self) -> int:
        def time_callback(from_index, to_index):
            '''Returns the time to walk between two nodes.'''
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)

            if from_node == self.start_idx or to_node == self.start_idx:
                return 0

            try:
                distance = round(HouseDistances.get_distance(self.points[from_node], self.points[to_node]) /
                                 WALKING_M_PER_S + MINS_PER_HOUSE * 60)
                return distance
                # return distance if distance < 400 else 1600
            except KeyError:
                if from_node == self.start_idx or to_node == self.start_idx:
                    raise RuntimeError('Unable to find distance from depot to another point, which should never happen')
                # This house is too far away, so return a value greather than the maximum allowed
                return 420

        return self.routing.RegisterTransitCallback(time_callback)

    def add_capacity_constraint(self):
        # Add the capacity constraint, which sets a maximum number of allowed houses per list
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return 1 if from_node != self.start_idx else 0
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.MAX_HOUSES_PER_LIST] * self.num_lists,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

    def add_duration_constraint(self, transit_index: int):
        # Add the duration constraint, which sets a maximum allowed duration per list
        self.routing.AddDimension(
            transit_index,
            self.MAX_TIME_PER_LIST,
            self.MAX_TIME_PER_LIST,
            False,
            'Duration')
        time_dimension: pywrapcp.RoutingDimension = self.routing.GetDimensionOrDie('Duration')
        for i in range(self.num_lists):
            time_dimension.SetSpanUpperBoundForVehicle(self.MAX_TIME_PER_LIST, i)

    def print_solution(self, assignment: pywrapcp.Assignment):
        dropped_nodes = 0
        for node in range(self.routing.Size()):
            if self.routing.IsStart(node) or self.routing.IsEnd(node):
                continue
            if assignment.Value(self.routing.NextVar(node)) == node:
                dropped_nodes += 1
        print(colored('Didn\'t use {} houses'.format(dropped_nodes), color='yellow'))

        total_time = 0
        total_load = 0
        for vehicle_id in range(self.num_lists):
            index = self.routing.Start(vehicle_id)
            route_time = 0
            route_load = 0
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_load += 0 if node_index == 0 else 1
                previous_index = index
                index = assignment.Value(self.routing.NextVar(index))
                route_time += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            print('List {:<2} takes {:<3} minutes to walk and has {:<3} houses'.format(
                vehicle_id, round(route_time / 60), route_load))
            total_time += route_time
            total_load += route_load
        print('In total, {} walk lists hit {} houses.'.format(self.num_lists, total_load))

    def visualize_solution(self, assignment: pywrapcp.Assignment) -> Map:
        walk_lists: list[list[Point]] = []

        # Display routes
        for vehicle_id in range(self.num_lists):
            index = self.routing.Start(vehicle_id)
            walk_lists.append([self.points[self.manager.IndexToNode(index)]])
            while not self.routing.IsEnd(index):
                index = assignment.Value(self.routing.NextVar(index))
                walk_lists[vehicle_id].append(self.points[self.manager.IndexToNode(index)])

        return display_house_orders(walk_lists)

    def optimize(self):
        start_time = time.time()

        transit_callback_index = self.get_time_index()
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        self.add_capacity_constraint()
        self.add_duration_constraint(transit_callback_index)

        # Allow the solver to drop nodes it cannot fit given the above constraints.
        penalty = 1000
        for node in range(0, self.start_idx):
            # Here, add differing penalties based on number of voters in the house
            self.routing.AddDisjunction([self.manager.NodeToIndex(node)], penalty)

        # Assign the first guess
        search_parameters: routing_parameters_pb2.RoutingSearchParameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = self.TIME_LIMIT

        print('Beginning solving')

        # Solve
        assignment: pywrapcp.Assignment = self.routing.SolveWithParameters(search_parameters)

        if assignment is None:
            print(colored('Failed', color='red'))
            sys.exit()

        print(colored('Finished in {:.2f} seconds'.format(time.time() - start_time), color='green'))

        self.print_solution(assignment)
        self.visualize_solution(assignment).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
