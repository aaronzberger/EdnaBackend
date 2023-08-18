import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

from termcolor import colored
from tqdm import tqdm

from src.config import (
    BASE_DIR,
    MAX_TOURING_DISTANCE,
    MAX_TOURING_TIME,
    MINS_PER_HOUSE,
    OPTIM_COSTS,
    OPTIM_OBJECTIVES,
    SEARCH_MODE_DEEP,
    TIMEOUT,
    VRP_CLI_PATH,
    WALKING_M_PER_S,
    DistanceMatrix,
    Fleet,
    Job,
    Location,
    NodeType,
    Place,
    PlaceTW,
    Plan,
    Point,
    Problem,
    Profile,
    Service,
    Shift,
    ShiftEnd,
    ShiftStart,
    Solution,
    Statistic,
    Stop,
    Time,
    Tour,
    Vehicle,
    VehicleLimits,
    VehicleProfile,
    problem_path,
    pt_id,
    solution_path,
    house_to_voters_file
)
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances

TIME_STRING_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Optimizer:
    distance_matrix_save = os.path.join(BASE_DIR, "optimize", "distances.json")
    _house_to_voters = json.load(open(house_to_voters_file))

    def __init__(
        self,
        cluster: list[Point],
        num_lists: int,
        starting_locations: Point | list[Point],
    ):
        self.points = deepcopy(cluster)

        # Determine whether to run group canvas or turf split problem
        if isinstance(starting_locations, list):
            depot = Point(lat=-1, lon=-1, type=NodeType.other, id="depot")
            print(
                "There are {} houses and {} starting locations".format(
                    len(self.points), len(starting_locations)
                )
            )
            self.points = self.points + starting_locations + [depot]
            self.start_idx = len(self.points) - 1
            self.create_turf_split(num_lists)
        else:
            self.points.append(starting_locations)
            self.start_idx = len(self.points) - 1
            self.create_group_canvas(num_lists)

    def build_fleet(
        self, shift_time: timedelta, num_vehicles: int, time_window: list[str]
    ):
        """
        Build the fleet (for both the group canvas and turf split problems).

        Parameters
        ----------
            shift_time (timedelta): length of the shift in which the vehicles can operate
            num_vehicles (int): the number of vehicles
            time_window (list[str]): a two-long list of start and end time of the canvas, datetime-formatted
        """
        walker = Vehicle(
            typeId="person",
            vehicleIds=["walker_{}".format(i) for i in range(num_vehicles)],
            profile=Profile(matrix="person"),
            costs=OPTIM_COSTS,
            shifts=[
                Shift(
                    start=ShiftStart(
                        earliest=time_window[0], location=Location(index=self.start_idx)
                    ),
                    end=ShiftEnd(
                        latest=time_window[1], location=Location(index=self.start_idx)
                    ),
                )
            ],
            capacity=[1],
            limits=VehicleLimits(
                shiftTime=shift_time.seconds, maxDistance=MAX_TOURING_DISTANCE
            ),
        )

        return Fleet(vehicles=[walker], profiles=[VehicleProfile(name="person")])

    def create_turf_split(self, num_lists: int):
        """
        Construct the problem file for a turf split to be processed by the VRP solver.

        Parameters
        ----------
            num_lists (int): the number of lists to generate

        Notes
        -----
            See the summary paper for a full explanation of mapping the turf split problem to standard VRP
        """
        start_time = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)

        node_duration = timedelta(seconds=1)
        depot_to_node_duration = MAX_TOURING_TIME - node_duration

        MAX_ROUTE_TIME = (
            MAX_TOURING_TIME + (2 * depot_to_node_duration) + (2 * node_duration)
        )
        end_time = start_time + MAX_ROUTE_TIME

        full_time_window = [
            start_time.strftime(TIME_STRING_FORMAT),
            end_time.strftime(TIME_STRING_FORMAT),
        ]

        # TODO: Fix so that the actual route time isn't impacted by the starting_location_delta

        # Define the times at which a possible starting location can be visited
        node_start_open = (start_time + depot_to_node_duration).strftime(
            TIME_STRING_FORMAT
        )
        node_start_close = (
            start_time + depot_to_node_duration + node_duration
        ).strftime(TIME_STRING_FORMAT)

        # Define the times at which a possible ending location can be visited
        node_end_open = (end_time - depot_to_node_duration - node_duration).strftime(
            TIME_STRING_FORMAT
        )
        node_end_close = (end_time - depot_to_node_duration).strftime(
            TIME_STRING_FORMAT
        )

        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile="person", travelTimes=[], distances=[])

        print(
            "Building problem ... if this is Killed, either increase the memory limit or decrease the area size"
        )

        # Use progress bar
        with tqdm(
            total=len(self.points) ** 2,
            desc="Building problem",
            unit="points",
            leave=False,
            colour="green",
        ) as pbar:
            for pt in self.points:
                for other_pt in self.points:
                    pbar.update(1)
                    if pt_id(pt) == "depot" or pt_id(other_pt) == "depot":
                        if pt_id(pt) == pt_id(other_pt) == "depot":
                            time = cost = 0
                        else:
                            if pt["type"] == NodeType.house or other_pt["type"] == NodeType.house:
                                # It is impossible to traverse between depots, or from a house to a depot
                                time = MAX_ROUTE_TIME.seconds
                                cost = MAX_TOURING_DISTANCE
                            else:
                                # It is possible to travel exactly to one intersection and back
                                time = depot_to_node_duration.seconds
                                cost = 0
                    else:
                        # Calculate the distance between two nodes, two houses, or a house and a node
                        distance_cost = MixDistances.get_distance(pt, other_pt)

                        if type(distance_cost) is tuple:
                            distance, cost = distance_cost
                            time = distance / WALKING_M_PER_S
                        else:
                            cost = (
                                distance_cost
                                if distance_cost is not None
                                else MAX_TOURING_DISTANCE
                            )
                            time = cost / WALKING_M_PER_S

                    distance_matrix["travelTimes"].append(round(time))
                    distance_matrix["distances"].append(round(cost))

        json.dump(distance_matrix, open(self.distance_matrix_save, "w"), indent=2)

        print("Saved distance matrix to {}".format(self.distance_matrix_save))

        # Create the plan
        jobs: list[Job] = []
        for i, location in enumerate(self.points):
            if pt_id(location) == "depot":
                delivery = Service(
                    places=[
                        Place(
                            location=Location(index=i),
                            duration=1,
                        )
                    ]
                )
            elif location["type"] == NodeType.node:
                service_start = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=node_duration.seconds,
                            times=[[node_start_open, node_start_close]],
                        )
                    ]
                )
                service_end = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=node_duration.seconds,
                            times=[[node_end_open, node_end_close]],
                        )
                    ]
                )
                jobs.append(
                    Job(
                        id=pt_id(location),
                        services=[service_start, service_end],
                        value=1,
                    )
                )
            else:
                delivery = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=round(MINS_PER_HOUSE * 60),
                            times=[full_time_window],
                        )
                    ]
                )
                try:
                    value = self._house_to_voters[pt_id(location)]["value"]
                except KeyError:
                    print(colored(f"Unable to find house with ID {pt_id(location)} in voters file. Quitting.", "red"))
                    sys.exit(1)
                jobs.append(Job(id=pt_id(location), services=[delivery], value=1))

        fleet = self.build_fleet(
            shift_time=(end_time - start_time),
            num_vehicles=num_lists,
            time_window=full_time_window,
        )
        problem = Problem(
            plan=Plan(jobs=jobs), fleet=fleet, objectives=OPTIM_OBJECTIVES
        )

        json.dump(
            problem,
            open(os.path.join(BASE_DIR, "optimize", "problem.json"), "w"),
            indent=2,
        )

    def create_group_canvas(self, num_lists: int):
        """
        Construct the problem file for a group canvas to be processed by the VRP solver.

        Parameters
        ----------
            num_lists (int): the number of lists to generate
        """
        start_time = datetime(year=3000, month=1, day=1, hour=0, minute=0, second=0)
        end_time = start_time + MAX_TOURING_TIME

        full_time_window = [
            start_time.strftime(TIME_STRING_FORMAT),
            end_time.strftime(TIME_STRING_FORMAT),
        ]
        # Construct the distance matrix file
        distance_matrix = DistanceMatrix(profile="person", travelTimes=[], distances=[])
        for pt in self.points:
            for other_pt in self.points:
                distance_cost = HouseDistances.get_distance(pt, other_pt)
                if type(distance_cost) is tuple:
                    distance, cost = distance_cost
                    time = distance / WALKING_M_PER_S
                else:
                    cost = (
                        distance_cost
                        if distance_cost is not None
                        else MAX_TOURING_DISTANCE
                    )
                    time = cost / WALKING_M_PER_S
                distance_matrix["travelTimes"].append(round(time))
                distance_matrix["distances"].append(round(cost))

        json.dump(distance_matrix, open(self.distance_matrix_save, "w"), indent=2)
        print("Saved distance matrix to {}".format(self.distance_matrix_save))

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            if i != self.start_idx:  # The starting location is not a real service
                service = Service(
                    places=[
                        Place(
                            location=Location(index=i),
                            duration=round(MINS_PER_HOUSE * 60),
                        )
                    ]
                )
                jobs.append(Job(id=pt_id(house), services=[service], value=1))

        fleet = self.build_fleet(
            shift_time=(end_time - start_time),
            num_vehicles=num_lists,
            time_window=full_time_window,
        )
        problem = Problem(
            plan=Plan(jobs=jobs), fleet=fleet, objectives=OPTIM_OBJECTIVES
        )

        json.dump(
            problem,
            open(os.path.join(BASE_DIR, "optimize", "problem.json"), "w"),
            indent=2,
        )

    def optimize(self) -> Optional[Solution]:
        start_time = time.time()

        p = subprocess.run(
            # [VRP_CLI_PATH, 'solve', 'pragmatic', problem_path, '-m', self.distance_matrix_save,
            #  '-o', solution_path, '-t', '100', '--min-cv', 'sample,300,0.001,true',
            #  '--search-mode', 'deep' if search_deep else 'broad', '--log'])
            [
                VRP_CLI_PATH,
                "solve",
                "pragmatic",
                problem_path,
                "-m",
                self.distance_matrix_save,
                "-o",
                solution_path,
                "-t",
                str(TIMEOUT.seconds),
                "--min-cv",
                "sample,200,0.1,true",
                "--search-mode",
                "deep" if SEARCH_MODE_DEEP else "broad",
                "--log",
            ]
        )

        if p.returncode != 0:
            return

        print(
            colored(
                "Finished in {:.2f} seconds".format(time.time() - start_time),
                color="green",
            )
        )

        solution_dict = json.load(open(solution_path))
        tours: list[Tour] = []
        for tour in solution_dict["tours"]:
            stops: list[Stop] = []
            for stop in tour["stops"]:
                stops.append(
                    Stop(
                        distance=stop["distance"],
                        load=stop["load"],
                        activities=stop["activities"],
                        location=Location(**stop["location"]),
                        time=Time(**stop["time"]),
                    )
                )
            tours.append(
                Tour(
                    vehicleId=tour["vehicleId"],
                    typeId=tour["typeId"],
                    shiftIndex=tour["shiftIndex"],
                    stops=stops,
                )
            )
        self.solution = Solution(
            statistic=Statistic(**solution_dict["statistic"]),
            tours=tours,
            unassigned=solution_dict["unassigned"],
        )

        return self.solution
