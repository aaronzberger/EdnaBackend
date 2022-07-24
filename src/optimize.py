import itertools
import json
import os
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm

from config import BASE_DIR, MINS_PER_HOUSE, WALKING_M_PER_S, blocks_file, blocks_file_t
from gps_utils import Point
from timeline_utils import NodeDistances, Segment, Timeline
from viz_utils import display_house_orders
from termcolor import colored


class HouseDistances():
    _house_distances: dict[str, dict[str, float]] = {}
    _save_file = os.path.join(BASE_DIR, 'store', 'house_distances.json')
    _blocks: blocks_file_t = json.load(open(blocks_file))

    @classmethod
    def _insert_pair(cls, s1: Segment, s2: Segment):
        s1_houses = cls._blocks[s1.id]['addresses']
        s2_houses = cls._blocks[s2.id]['addresses']

        if len(s1_houses) == 0 or len(s2_houses) == 0:
            return

        # If any combination of houses on these two segments is inserted, they all are
        try:
            cls._house_distances[next(iter(s2_houses))][next(iter(s1_houses))]
            return
        except KeyError:
            pass

        # Check if the segments are the same
        if s1.id == s2.id:
            for (address_1, info_1), (address_2, info_2) in itertools.product(s1_houses.items(), s2_houses.items()):
                if address_1 not in cls._house_distances:
                    cls._house_distances[address_1] = {}

                if address_1 == address_2:
                    cls._house_distances[address_1][address_2] = 0
                else:
                    # Simply use the difference of the distances to the start
                    cls._house_distances[address_1][address_2] = round(
                        abs(info_1['distance_to_start'] - info_2['distance_to_start']))
            return

        # Calculate the distances between the segment endpoints
        end_distances = [NodeDistances.get_distance(i, j) for i, j in
                         [(s1.start, s2.start), (s1.start, s2.end), (s1.end, s2.start), (s1.end, s2.end)]]

        # If this pair is too far away, don't add to the table.
        if None in end_distances or min(end_distances) > 1600:
            return

        # Iterate over every possible pair of houses
        for (address_1, info_1), (address_2, info_2) in itertools.product(s1_houses.items(), s2_houses.items()):
            if address_1 not in cls._house_distances:
                cls._house_distances[address_1] = {}

            start_start = end_distances[0] + info_1['distance_to_start'] + info_2['distance_to_start']
            start_end = end_distances[1] + info_1['distance_to_start'] + info_2['distance_to_end']
            end_start = end_distances[2] + info_1['distance_to_end'] + info_2['distance_to_start']
            end_end = end_distances[3] + info_1['distance_to_end'] + info_2['distance_to_end']

            cls._house_distances[address_1][address_2] = round(min([start_start, start_end, end_start, end_end]))

    @classmethod
    def __init__(cls, cluster: list[Segment]):
        print('Beginning house distances generation... ')

        if os.path.exists(cls._save_file):
            print('House distance table file found. Loading may take a while...')
            cls._house_distances = json.load(open(cls._save_file))
            # cls._house_distances = pickle.load(open(cls._save_file, 'rb'))
            return

        print('No house distance table file found at {}. Generating now...'.format(cls._save_file))
        cls._house_distances = {}
        with tqdm(total=len(cluster) ** 2, desc='Generating', unit='pairs', colour='green') as progress:
            for segment in cluster:
                for other_segment in cluster:
                    cls._insert_pair(segment, other_segment)
                    progress.update()

        print('Saving to {}'.format(cls._save_file))
        # with open(cls._save_file, 'wb') as output:
        #     pickle.dump(cls._house_distances, output)
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
    houses_in_cluster = cluster_to_houses(cluster)
    points = list(houses_in_cluster.values())

    HouseDistances(cluster)

    print('Done generating')

    MAX_HOUSES_PER_LIST = 90
    NUM_WALKERS = 3
    TIME_LIMIT = 10
    MAX_TIME_PER_LIST = 180 * 60

    start_time = time.time()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(points) + 1, NUM_WALKERS, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        '''Returns the time to walk between two nodes.'''
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        # To allow arbitrary starting locations, we assign a ghost depot house at index 0 with 0 distance to all others
        if from_node == 0 or to_node == 0:
            return 0
        try:
            return round(HouseDistances.get_distance(points[from_node - 1], points[to_node - 1]) / WALKING_M_PER_S +
                         MINS_PER_HOUSE * 60)
        except KeyError:
            # This house is too far away, so return a value greather than the maximum allowed
            return MAX_TIME_PER_LIST

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add the capacity constraint, which sets a maximum number of allowed houses per list
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 0 if from_node == 0 else 1
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [MAX_HOUSES_PER_LIST] * NUM_WALKERS,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add the duration constraint, which sets a maximum allowed duration per list
    routing.AddDimension(
        transit_callback_index,
        MAX_TIME_PER_LIST,
        MAX_TIME_PER_LIST,
        False,
        'Duration')
    time_dimension = routing.GetDimensionOrDie('Duration')
    for i in range(NUM_WALKERS):
        time_dimension.SetSpanUpperBoundForVehicle(MAX_TIME_PER_LIST, i)

    # Allow the solver to drop nodes it cannot fit given the above constraints.
    penalty = 1000
    for node in range(1, len(points) + 1):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Assign the first guess
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = TIME_LIMIT

    # Solve
    assignment = routing.SolveWithParameters(search_parameters)

    print(colored('Finished in {:.2f} seconds'.format(time.time() - start_time), color='green'))

    dropped_nodes = 0
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += 1
    print(colored('Didn\'t use {} houses'.format(dropped_nodes), color='yellow'))

    total_time = 0
    total_load = 0
    for vehicle_id in range(NUM_WALKERS):
        index = routing.Start(vehicle_id)
        route_time = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += 0 if node_index == 0 else 1
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        print('List {:<2} takes {:<3} minutes to walk and has {:<3} houses'.format(
            vehicle_id, round(route_time / 60), route_load))
        total_time += route_time
        total_load += route_load
    print('In total, {} walk lists hit {} houses.'.format(NUM_WALKERS, total_load))

    walk_lists: list[list[Point]] = []

    # Display routes
    for vehicle_id in range(NUM_WALKERS):
        index = routing.Start(vehicle_id)
        walk_lists.append([points[manager.IndexToNode(index) - 1]])
        while not routing.IsEnd(index):
            index = assignment.Value(routing.NextVar(index))
            walk_lists[vehicle_id].append(points[manager.IndexToNode(index) - 1])

    display_house_orders(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
