import itertools
import json
import math
import os
import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm

from config import BASE_DIR, SegmentDict, blocks_file, blocks_file_t
from gps_utils import Point, along_track_distance, great_circle_distance
from timeline_utils import NodeDistances, Segment, Timeline
from viz_utils import display_house_orders


class HouseDistances():
    _house_distances: dict[str, dict[str, float]] = {}
    _save_file = os.path.join(BASE_DIR, 'store', 'house_distances.json')
    _blocks: blocks_file_t = json.load(open(blocks_file))

    @classmethod
    def _insert_pair(cls, s1: Segment, s2: Segment):
        # If this pair already exists in the opposite order, skip
        end_distances = [NodeDistances.get_distance(i, j) for i, j in
                         [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]
        end_distances = [i for i in end_distances if i is not None]

        # If this pair is too far away, don't add to the table.
        if len(end_distances) == 0 or min(end_distances) > 1600:
            return

        s1_houses = cls._blocks[s1.id]['addresses']
        s1_distances_to_start = []
        for address, info in s1_houses.items():
            distance = 0
            sub_segment_start_idx = int(cls._blocks[s1.id]['addresses'][address]['sub_node_1'])

            # Iterate over the block from the start to this subsegment
            for first, second in itertools.pairwise(s1.all_points[:sub_segment_start_idx + 1]):
                distance += great_circle_distance(first, second)
            distance += along_track_distance(
                p1=Point(s1_houses[address]['lat'], s1_houses[address]['lon']),
                p2=s1.all_points[sub_segment_start_idx], p3=s1.all_points[sub_segment_start_idx + 1])[0]
            s1_distances_to_start

        # try:
        #     cls._house_distances[p2.id][p1.id]
        # except KeyError:
        #     # The distance between these houses is simply the minimum of the distances between the ends of the blocks
        #     # plus the ALD of the houses to those endpoints

        #     endpoints = 
        #     distances = [NodeDistances.get_distance]
        #     # distance = great_circle_distance(p1, p2)
        #     # cls._point_distances[p1.id][p2.id] = great_circle_distance(p1, p2)
        #     # if distance > 800:
        #     #     # Assumimg great circle distance is the hypotenuse of a right triangle, track the outside
        #     #     cls._point_distances[p1.id][p2.id] = math.sqrt(math.pow(distance, 2))
        #     # else:
        #     #     cls._point_distances[p1.id][p2.id] = get_distance(p1, p2)

    @classmethod
    def __init__(cls, cluster: list[Segment]):
        print('Beginning house distances generation... ')

        if os.path.exists(cls._save_file):
            print('House distance table file found.')
            cls._house_distances = json.load(open(cls._save_file))
            return

        print('No house distance table file found at {}. Generating now...'.format(cls._save_file))
        cls._house_distances = {}
        with tqdm(total=len(cluster) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for segment in cluster:
                for other_segment in cluster:
                    cls._insert_pair(segment, other_segment)
                    progress.update()

            print('Saving to {}'.format(cls._save_file))
            json.dump(cls._house_distances, open(cls._save_file, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def get_distance(cls, p1: Point, p2: Point) -> float:
        '''
        Get the distance between two houses by their coordinates

        Parameters:
            p1 (Point): the first point
            p2 (Point): the second point

        Returns:
            float: distance between the two points

        Raises:
            KeyError: if the pair does not exist in the table
        '''
        try:
            return cls._house_distances[p1.id][p2.id]
        except KeyError:
            return cls._house_distances[p2.id][p1.id]


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f'Objective: {assignment.ObjectiveValue()}')
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    print(dropped_nodes)
    # Display routes
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))


def cluster_to_houses(cluster: list[Segment]) -> dict[str, Point]:
    # Load address_points.csv into a dictionary with address as key
    print('Loading coordinates of houses...')
    blocks: blocks_file_t = json.load(open(blocks_file))

    houses_in_cluster: dict[str, Point] = {}

    for segment in cluster:
        try:
            for address in blocks[segment.id]['addresses']:
                houses_in_cluster[address] = Point(blocks[segment.id]['addresses'][address]['lat'],
                                                   blocks[segment.id]['addresses'][address]['lon'],
                                                   id=address)
        except KeyError:
            print('Couldn\'t find segment with ID {}'.format(segment.id))

    return houses_in_cluster


def optimize_cluster(cluster: list[Segment]):
    data = {}

    houses_in_cluster = cluster_to_houses(cluster)
    points = list(houses_in_cluster.values())
    HouseDistances(cluster)

    matrix = np.empty((len(houses_in_cluster) + 1, len(houses_in_cluster) + 1), dtype=int)
    for r, point in enumerate(points):
        for c, other_point in enumerate(points):
            distance = HouseDistances.get_distance(point, other_point)
            matrix[r + 1][c + 1] = round(distance)

    # For arbitrary start and end locations, fill in the first column and row with 0s
    matrix[:, 0] = 0
    matrix[0, :] = 0

    data['distance_matrix'] = matrix
    data['demands'] = [0] + [1] * len(points)  # Normally, this might reflect the number of voters
    data['num_vehicles'] = 15
    data['depot'] = 0
    data['vehicle_capacities'] = [90] * data['num_vehicles']

    start_time = time.time()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Allow to drop nodes.
    penalty = 1000
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    print('Finished in', time.time() - start_time)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)

    walk_lists: list[list[Point]] = []

    # Display routes
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        walk_lists.append([points[manager.IndexToNode(index) - 1]])
        while not routing.IsEnd(index):
            index = assignment.Value(routing.NextVar(index))
            walk_lists[vehicle_id].append(points[manager.IndexToNode(index) - 1])

    display_house_orders(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
