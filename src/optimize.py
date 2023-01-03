import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

from termcolor import colored

from src.config import (BASE_DIR, MAX_ROUTE_DISTANCE, MAX_ROUTE_TIME,
                        MINS_PER_HOUSE, VRP_CLI_PATH, WALKING_M_PER_S, Costs,
                        DistanceMatrix, Fleet, Job, Location, Objective, PlaceTW,
                        Plan, Problem, Profile, Service, Shift, ShiftEnd,
                        ShiftStart, Solution, Statistic, Stop, Time, Tour,
                        Vehicle, VehicleLimits, VehicleProfile, problem_path,
                        solution_path, Place)
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.gps_utils import Point
from src.viz_utils import display_house_orders


class Optimizer():
    distance_matrix_save = os.path.join(BASE_DIR, 'optimize', 'distances.json')

    def __init__(self, cluster: list[Point], num_lists: int,
                 starting_location: Optional[Point] = None, intersections: Optional[list[Point]] = None) -> None:
        self.points = deepcopy(cluster)

        if starting_location is None:
            if intersections is None:
                raise RuntimeError('If starting_location is not provided, intersections must be provided')
            depot = Point(-1, -1, id='depot')
            self.points = self.points + intersections + [depot]
            self.start_idx = len(self.points) - 1
            self.create_area_lists(num_lists)
        else:
            if intersections is not None:
                raise RuntimeError('If starting_location is provided, intersections should not be provided')
            self.points.append(starting_location)
            self.start_idx = len(self.points) - 1
            self.create_group_canvas(num_lists)

    def create_area_lists(self, num_lists: int):
        start_time = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
        depot_to_node_duration = MAX_ROUTE_TIME
        end_time = start_time + MAX_ROUTE_TIME + (2 * depot_to_node_duration)

        full_time_window = [start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), end_time.strftime('%Y-%m-%dT%H:%M:%SZ')]
        node_start_open = (start_time + depot_to_node_duration).strftime('%Y-%m-%dT%H:%M:%SZ')
        node_start_close = (start_time + depot_to_node_duration + timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%SZ')

        node_end_open = (end_time - depot_to_node_duration - timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        node_end_close = (end_time - depot_to_node_duration).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile='person', travelTimes=[], distances=[])

        for pt in self.points:
            for other_pt in self.points:
                if pt.id == 'depot' or other_pt.id == 'depot':
                    if pt.id == other_pt.id == 'depot':
                        distance = time = 0
                    else:
                        if pt.type == 'house' or other_pt.type == 'house':
                            # Make it impossible to traverse from a house to a depot
                            distance = MAX_ROUTE_DISTANCE
                            time = 0
                        else:
                            # Exactly 2 hours to traverse from depot to a node
                            distance = 0
                            time = depot_to_node_duration.seconds
                else:
                    distance = MixDistances.get_distance(pt, other_pt)
                    distance = distance if distance is not None else depot_to_node_duration.seconds * WALKING_M_PER_S
                    time = distance / WALKING_M_PER_S
                distance_matrix['distances'].append(round(distance))
                distance_matrix['travelTimes'].append(round(time))

        json.dump(distance_matrix, open(self.distance_matrix_save, 'w'), indent=2)

        print('Saved distance matrix to {}'.format(self.distance_matrix_save))

        # Create the plan
        jobs: list[Job] = []
        for i, location in enumerate(self.points):
            if location.type == 'node':
                service_start = Service(places=[PlaceTW(
                    location=Location(index=i), duration=60, times=[[node_start_open, node_start_close]])])
                service_end = Service(places=[PlaceTW(
                    location=Location(index=i), duration=60, times=[[node_end_open, node_end_close]])])
                jobs.append(Job(id=location.id, services=[service_start, service_end], value=1))
            else:
                delivery = Service(places=[Place(location=Location(index=i),
                                                 duration=round(MINS_PER_HOUSE * 60))])
                jobs.append(Job(id=location.id, services=[delivery], value=10))

        # Create the fleet
        walker = Vehicle(
            typeId='person',
            vehicleIds=['walker_{}'.format(i) for i in range(num_lists)],
            profile=Profile(matrix='person'),
            costs=Costs(fixed=0, distance=2, time=3),
            shifts=[Shift(start=ShiftStart(earliest=full_time_window[0], location=Location(index=self.start_idx)),
                          end=ShiftEnd(latest=full_time_window[1], location=Location(index=self.start_idx)))],
            capacity=[1],
            limits=VehicleLimits(shiftTime=(end_time - start_time).seconds, maxDistance=MAX_ROUTE_DISTANCE))

        fleet = Fleet(vehicles=[walker], profiles=[VehicleProfile(name='person')])
        objectives = [[Objective(type='maximize-value')], [Objective(type='minimize-cost')]]
        problem = Problem(plan=Plan(jobs=jobs), fleet=fleet, objectives=objectives)

        json.dump(problem, open(os.path.join(BASE_DIR, 'optimize', 'problem.json'), 'w'), indent=2)

    def create_group_canvas(self, num_lists: int):
        start_time = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
        end_time = start_time + MAX_ROUTE_TIME

        full_time_window = [start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), end_time.strftime('%Y-%m-%dT%H:%M:%SZ')]

        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile='person', travelTimes=[], distances=[])
        for pt in self.points:
            for other_pt in self.points:
                distance_cost = HouseDistances.get_distance(pt, other_pt)
                distance_cost = distance_cost if distance_cost is not None else (10000, 10000)
                distance_matrix['distances'].append(round(distance_cost[0] + distance_cost[1]))
                distance_matrix['travelTimes'].append(round(distance_cost[0] / WALKING_M_PER_S))

        json.dump(distance_matrix, open(self.distance_matrix_save, 'w'), indent=2)

        print('Saved distance matrix to {}'.format(self.distance_matrix_save))

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            if i == self.start_idx:
                # The starting location is not a real service
                continue
            service = Service(places=[Place(location=Location(index=i), duration=round(MINS_PER_HOUSE * 60))])
            jobs.append(Job(id=house.id, services=[service], value=1))

        # Create the fleet
        walker = Vehicle(
            typeId='person',
            vehicleIds=['walker_{}'.format(i) for i in range(num_lists)],
            profile=Profile(matrix='person'),
            costs=Costs(fixed=0, distance=1, time=0),
            shifts=[Shift(start=ShiftStart(earliest=full_time_window[0], location=Location(index=self.start_idx)),
                          end=ShiftEnd(latest=full_time_window[1], location=Location(index=self.start_idx)))],
            capacity=[1],
            limits=VehicleLimits(shiftTime=MAX_ROUTE_TIME.seconds, maxDistance=MAX_ROUTE_DISTANCE))

        fleet = Fleet(vehicles=[walker], profiles=[VehicleProfile(name='person')])
        # objectives = [[Objective(type='maximize-value')], [Objective(type='minimize-distance')], [Objective(type='minimize-cost')]]
        objectives = [[Objective(type='maximize-value')], [Objective(type='minimize-cost')], [Objective(type='minimize-tours')]]
        problem = Problem(plan=Plan(jobs=jobs), fleet=fleet, objectives=objectives)

        json.dump(problem, open(os.path.join(BASE_DIR, 'optimize', 'problem.json'), 'w'), indent=2)

    def optimize(self) -> Optional[Solution]:
        start_time = time.time()

        search_deep = False

        p = subprocess.run(
            [VRP_CLI_PATH, 'solve', 'pragmatic', problem_path, '-m', self.distance_matrix_save,
             '-o', solution_path, '-t', '60', '--min-cv', 'sample,200,0.01,true',
             '--search-mode', 'deep' if search_deep else 'broad', '--log'])

        if p.returncode != 0:
            return

        print(colored('Finished in {:.2f} seconds'.format(time.time() - start_time), color='green'))

        solution_dict = json.load(open(solution_path))
        tours: list[Tour] = []
        for tour in solution_dict['tours']:
            stops: list[Stop] = []
            for stop in tour['stops']:
                stops.append(Stop(distance=stop['distance'], load=stop['load'], activities=stop['activities'],
                                  location=Location(**stop['location']), time=Time(**stop['time'])))
            tours.append(Tour(vehicleId=tour['vehicleId'], typeId=tour['typeId'],
                              shiftIndex=tour['shiftIndex'], stops=stops))
        self.solution = Solution(
            statistic=Statistic(**solution_dict['statistic']), tours=tours, unassigned=solution_dict['unassigned'])

        return self.solution

    def visualize(self):
        walk_lists: list[list[Point]] = []

        for i, route in enumerate(self.solution['tours']):
            walk_lists.append([])
            for stop in route['stops'][1:-1]:
                walk_lists[i].append(self.points[stop['location']['index']])

        display_house_orders(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
