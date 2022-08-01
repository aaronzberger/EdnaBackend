import json
import os
import subprocess
import time
from copy import deepcopy
from typing import Optional

from termcolor import colored

from src.config import (BASE_DIR, MINS_PER_HOUSE, VRP_CLI_PATH,
                        WALKING_M_PER_S, Costs, DistanceMatrix, Fleet, Job,
                        Location, Place, Plan, Problem, Profile, Service,
                        Shift, ShiftEnd, ShiftStart, Solution, Statistic, Stop,
                        Time, Tour, Vehicle, VehicleLimits, VehicleProfile,
                        problem_path, solution_path)
from src.distances.houses import HouseDistances
from src.gps_utils import Point
from src.viz_utils import display_house_orders


class Deserializer:
    @classmethod
    def from_dict(cls, dict):
        obj = cls()
        obj.__dict__.update(dict)
        return obj


class Optimizer():
    MAX_HOUSES_PER_LIST = 90
    TIME_LIMIT = 10
    MAX_TIME_PER_LIST = 180 * 60
    distance_matrix_save = os.path.join(BASE_DIR, 'optimize', 'distances.json')

    def __init__(self, cluster: list[Point], num_lists: int, starting_location: Point):
        self.points = deepcopy(cluster)
        self.points.append(starting_location)
        self.start_idx = len(self.points) - 1

        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile='person', travelTimes=[], distances=[])
        for pt in self.points:
            for other_pt in self.points:
                try:
                    distance = HouseDistances.get_distance(pt, other_pt)
                except KeyError:
                    distance = 10000
                distance_matrix['distances'].append(round(distance))
                distance_matrix['travelTimes'].append(round(distance / WALKING_M_PER_S))

        json.dump(distance_matrix, open(self.distance_matrix_save, 'w'), indent=2)

        print('Saved distance matrix to {}'.format(self.distance_matrix_save))

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            if i == self.start_idx:
                # The starting location is not a real service
                continue
            service = Service(places=[Place(location=Location(index=i), duration=round(MINS_PER_HOUSE * 60))])
            jobs.append(Job(id=house.id, services=[service]))

        # Create the fleet
        walker = Vehicle(
            typeId='person',
            vehicleIds=['walker_{}'.format(i) for i in range(num_lists)],
            profile=Profile(matrix='person'),
            costs=Costs(fixed=0, distance=1, time=3),
            shifts=[Shift(start=ShiftStart(earliest='2022-08-01T05:00:00Z', location=Location(index=self.start_idx)),
                          end=ShiftEnd(latest='2022-08-01T08:00:00Z', location=Location(index=self.start_idx)))],
            capacity=[200],
            limits=VehicleLimits(shiftTime=self.MAX_TIME_PER_LIST, maxDistance=3200))

        fleet = Fleet(vehicles=[walker], profiles=[VehicleProfile(name='person')])
        problem = Problem(plan=Plan(jobs=jobs), fleet=fleet)

        json.dump(problem, open(os.path.join(BASE_DIR, 'optimize', 'problem.json'), 'w'), indent=2)

    def optimize(self) -> Optional[Solution]:
        start_time = time.time()

        search_deep = False

        p = subprocess.run(
            [VRP_CLI_PATH, 'solve', 'pragmatic', problem_path, '-m', self.distance_matrix_save,
             '-o', solution_path, '-t', '60', '--min-cv', 'period,5,0.01,true',
             '--search-mode', 'deep' if search_deep else 'broad', '--log'])

        if p.returncode != 0:
            return

        print(colored('Finished in {:.2f} seconds'.format(time.time() - start_time), color='green'))

        solution_dict = json.load(open(solution_path))
        tours: list[Tour] = []
        for tour in solution_dict['tours']:
            stops: list[Stop] = []
            for stop in tour['stops']:
                stops.append(Stop(**stop, location=Location(**stop['location']), time=Time(**stop['time'])))
            tours.append(Tour(**tour, stops=stops))
        self.solution = Solution(
            statistic=Statistic(**solution_dict['statistic']), tours=tours, unassigned=solution_dict['unassigned'])

        return self.solution

    def visualize(self):
        walk_lists: list[list[Point]] = []

        for i, route in enumerate(self.solution['tours']):
            walk_lists.append([])
            for stop in route['stops']:
                walk_lists[i].append(self.points[stop['location']['index']])

        display_house_orders(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
