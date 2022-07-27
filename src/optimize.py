import json
import os
import subprocess
import time

from termcolor import colored

from config import (BASE_DIR, KEEP_APARTMENTS, MINS_PER_HOUSE, WALKING_M_PER_S,
                    Costs, DistanceMatrix, Fleet, Job, Location, Pickup, Place,
                    Plan, Problem, Profile, Shift, ShiftEnd, ShiftStart,
                    Vehicle, VehicleLimits, VehicleProfile, blocks_file,
                    blocks_file_t)
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

    def __init__(self, cluster: list[Segment], num_lists: int, starting_location: Point):
        houses_in_cluster = cluster_to_houses(cluster)
        self.points = list(houses_in_cluster.values())
        self.points.append(starting_location)
        self.start_idx = len(self.points) - 1

        HouseDistances(cluster, starting_location)

        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile='person', travelTimes=[], distances=[])
        for pt in self.points:
            for other_pt in self.points:
                try:
                    distance = HouseDistances.get_distance(pt, other_pt)
                except KeyError:
                    distance = 1600
                distance_matrix['distances'].append(round(distance))
                distance_matrix['travelTimes'].append(round(distance / WALKING_M_PER_S))

        json.dump(distance_matrix, open(self.distance_matrix_save, 'w'), indent=2)

        print('Saved distance matrix to {}'.format(self.distance_matrix_save))

        problem = {}

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            pickup = Pickup(places=[Place(location=Location(index=i), duration=MINS_PER_HOUSE * 60)],
                            demand=[1])
            jobs.append(Job(id=house.id, pickups=[pickup]))

        # Create the fleet
        walker = Vehicle(
            typeId='person',
            vehicleIds=['walker_{}'.format(i) for i in range(num_lists)],
            profile=Profile(matrix='person'),
            costs=Costs(fixed=0, distance=0, time=1),
            shifts=[Shift(start=ShiftStart(earliest='2022-08-01T05:00:00Z', location=Location(index=self.start_idx)),
                          end=ShiftEnd(latest='2022-08-01T08:00:00Z', location=Location(index=self.start_idx)))],
            capacity=[100],
            limits=VehicleLimits(shiftTime=self.MAX_TIME_PER_LIST, maxDistance=3200))

        fleet = Fleet(vehicles=[walker], profiles=[VehicleProfile(name='person')])
        problem = Problem(plan=Plan(jobs=jobs), fleet=fleet)

        json.dump(problem, open(os.path.join(BASE_DIR, 'optimize', 'problem.json'), 'w'), indent=2)

    def optimize(self):
        cli_path = "/Users/aaron/.cargo/bin/vrp-cli"
        problem_path = os.path.join(BASE_DIR, 'optimize', 'problem.json')
        solution_path = os.path.join(BASE_DIR, 'optimize', 'solution.json')

        start_time = time.time()

        # NOTE: modify example to pass matrix, config, initial solution, etc.
        p = subprocess.run(
            [cli_path, 'solve', 'pragmatic', problem_path, '-m', self.distance_matrix_save,
             '-o', solution_path, '-t', '20', '--log'])

        if p.returncode == 0:
            with open(solution_path, 'r') as f:
                solution_str = f.read()
                return json.loads(solution_str, object_hook=Deserializer.from_dict)

        print(colored('Finished in {:.2f} seconds'.format(time.time() - start_time), color='green'))

    def visualize(self):
        solution_path = os.path.join(BASE_DIR, 'optimize', 'solution.json')

        solutions = json.load(open(solution_path))

        walk_lists: list[list[Point]] = []

        for i, route in enumerate(solutions['tours']):
            walk_lists.append([])
            for stop in route['stops']:
                walk_lists[i].append(self.points[stop['location']['index']])

        display_house_orders(walk_lists).save(os.path.join(BASE_DIR, 'viz', 'optimal.html'))
