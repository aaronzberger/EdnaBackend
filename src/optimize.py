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
    USE_COST_METRIC,
    VRP_CLI_PATH,
    WALKING_M_PER_S,
    Costs,
    DistanceMatrix,
    Fleet,
    Job,
    Location,
    NodeType,
    Objective,
    Place,
    PlaceTW,
    Plan,
    Point,
    Problem,
    Profile,
    Service,
    Shift,
    ShiftDispatch,
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
    distances_path,
    house_to_voters_file,
    problem_path,
    pt_id,
    solution_path,
)
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances

TIME_STRING_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Optimizer:
    _house_to_voters = json.load(open(house_to_voters_file))

    def __init__(
        self,
        cluster: list[Point],
        starting_locations: Point | list[Point],
        num_lists: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        self.points = deepcopy(cluster)

        self.problem_path = (
            problem_path
            if save_path is None
            else os.path.join(save_path, "problem.json")
        )
        self.distances_path = (
            distances_path
            if save_path is None
            else os.path.join(save_path, "distances.json")
        )
        self.solution_path = (
            solution_path
            if save_path is None
            else os.path.join(save_path, "solution.json")
        )

        if num_lists is None:
            depot = deepcopy(starting_locations)
            depot["type"] = NodeType.node
            depot["id"] = "depot"
            self.points.append(depot)
            self.start_idx = len(self.points) - 1
            self.create_completed_group_canvas()
        else:
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
        self, shift_time: timedelta, num_vehicles: int, time_window: list[str], costs: Costs = OPTIM_COSTS
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
            costs=costs,
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
                        elif (
                            pt["type"] == NodeType.house
                            or other_pt["type"] == NodeType.house
                        ):
                            # It is impossible to traverse between depots, or from a house to a depot
                            time = MAX_ROUTE_TIME.seconds
                            cost = MAX_TOURING_DISTANCE
                        elif (
                            pt["type"] == NodeType.node
                            or other_pt["type"] == NodeType.node
                        ):
                            # It is possible to travel exactly to one intersection and back
                            time = depot_to_node_duration.seconds
                            cost = 0
                        else:
                            print(
                                colored(
                                    "All points must be either nodes, houses, or the depot. Quitting",
                                    "red",
                                )
                            )
                    else:
                        if (
                            pt["type"] == NodeType.node
                            and other_pt["type"] == NodeType.node
                        ):
                            # It is impossible to travel between two intersections
                            time = MAX_ROUTE_TIME.seconds
                            cost = MAX_TOURING_DISTANCE

                        # Calculate the distance between two nodes, two houses, or a house and a node
                        distance_cost = MixDistances.get_distance(pt, other_pt)

                        if type(distance_cost) is tuple:
                            # This only holds for houses (if one is not a house, distance is a float)
                            distance, cost = distance_cost
                            time = distance / WALKING_M_PER_S
                        else:
                            cost = (
                                distance_cost
                                if distance_cost is not None
                                else MAX_TOURING_DISTANCE
                            )
                            time = cost / WALKING_M_PER_S

                            if USE_COST_METRIC:
                                cost = 0  # There is no cost between house and node or between two nodes

                    distance_matrix["travelTimes"].append(round(time))
                    distance_matrix["distances"].append(round(cost))

        json.dump(distance_matrix, open(self.distances_path, "w"), indent=2)

        print("Saved distance matrix to {}".format(self.distances_path))

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
                            duration=1,
                            times=[[node_start_open, node_start_close]],
                        )
                    ]
                )
                service_end = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=1,
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
            elif location["type"] == NodeType.house:
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
                    print(
                        colored(
                            f"Unable to find house with ID {pt_id(location)} in voters file. Quitting.",
                            "red",
                        )
                    )
                    sys.exit(1)
                jobs.append(Job(id=pt_id(location), services=[delivery], value=10))
            else:
                print(
                    colored(
                        f"Point {pt_id(location)} has an invalid type. Quitting.",
                        "red",
                    )
                )
                sys.exit(1)

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
            open(self.problem_path, "w"),
            indent=2,
        )

    def build_dispatch(self):
    #     """
    #     Construct the problem file for a turf split to be processed by the VRP solver.

    #     Parameters
    #     ----------
    #         num_lists (int): the number of lists to generate

    #     Notes
    #     -----
    #         See the summary paper for a full explanation of mapping the turf split problem to standard VRP
    #     """
    #     start_time = datetime(year=3000, month=1, day=1, hour=0, minute=0, second=0)
    #     end_time = start_time + MAX_TOURING_TIME

    #     full_time_window = [
    #         start_time.strftime(TIME_STRING_FORMAT),
    #         end_time.strftime(TIME_STRING_FORMAT),
    #     ]
    #     # Construct the distance matrix file
    #     distance_matrix = DistanceMatrix(profile="person", travelTimes=[], distances=[])

    #     print(
    #         "Building problem ... if this is Killed, either increase the memory limit or decrease the area size"
    #     )

    #     # Use progress bar
    #     with tqdm(
    #         total=len(self.points) ** 2,
    #         desc="Building problem",
    #         unit="points",
    #         leave=False,
    #         colour="green",
    #     ) as pbar:
    #         for pt in self.points:
    #             for other_pt in self.points:
    #                 pbar.update(1)
    #                 if pt_id(pt) == "depot" or pt_id(other_pt) == "depot":
    #                     if pt_id(pt) == pt_id(other_pt) == "depot":
    #                         time = distance = 0
    #                     elif (
    #                         pt["type"] == NodeType.house
    #                         or other_pt["type"] == NodeType.house
    #                     ):
    #                         # It is impossible to traverse between depots, or from a house to a depot
    #                         time = MAX_TOURING_TIME.seconds
    #                         distance = MAX_TOURING_DISTANCE
    #                     elif (
    #                         pt["type"] == NodeType.node
    #                         or other_pt["type"] == NodeType.node
    #                     ):
    #                         # It is possible to travel exactly to one intersection and back
    #                         time = distance = 0
    #                     else:
    #                         print(
    #                             colored(
    #                                 "All points must be either nodes, houses, or the depot. Quitting",
    #                                 "red",
    #                             )
    #                         )
    #                 else:
    #                     if (
    #                         pt["type"] == NodeType.node
    #                         and other_pt["type"] == NodeType.node
    #                     ):
    #                         # It is impossible to travel between two intersections
    #                         time = MAX_TOURING_TIME.seconds
    #                         distance = MAX_TOURING_DISTANCE

    #                     # Calculate the distance between two nodes, two houses, or a house and a node
    #                     distance_cost = MixDistances.get_distance(pt, other_pt)

    #                     assert isinstance(distance_cost, float)

    #                     distance = (
    #                         distance_cost
    #                         if distance_cost is not None
    #                         else MAX_TOURING_DISTANCE
    #                     )
    #                     time = distance / WALKING_M_PER_S

    #                 distance_matrix["travelTimes"].append(round(time))
    #                 distance_matrix["distances"].append(round(distance))

    #     json.dump(distance_matrix, open(self.distances_path, "w"), indent=2)

    #     print("Saved distance matrix to {}".format(self.distances_path))

    #     # Create the plan
    #     jobs: list[Job] = []
    #     for i, location in enumerate(self.points):
    #         if pt_id(location) == "depot":
    #             continue
    #         elif location["type"] == NodeType.node:
    #             continue
    #         elif location["type"] == NodeType.house:
    #             delivery = Service(
    #                 places=[
    #                     PlaceTW(
    #                         location=Location(index=i),
    #                         duration=round(MINS_PER_HOUSE * 60),
    #                         times=[full_time_window],
    #                     )
    #                 ]
    #             )
    #             jobs.append(Job(id=pt_id(location), services=[delivery], value=10))
    #         else:
    #             print(
    #                 colored(
    #                     f"Point {pt_id(location)} has an invalid type. Quitting.",
    #                     "red",
    #                 )
    #             )
    #             sys.exit(1)

    #     walker = Vehicle(
    #         typeId="person",
    #         vehicleIds=["walker_{}".format(i) for i in range(200)],
    #         profile=Profile(matrix="person"),
    #         costs=Costs(fixed=0, distance=0, time=1),
    #         shifts=[
    #             ShiftDispatch(
    #                 start=ShiftStart(
    #                     earliest=full_time_window[0], location=Location(index=self.start_idx)
    #                 ),
    #                 end=ShiftEnd(
    #                     latest=full_time_window[1], location=Location(index=self.start_idx)
    #                 ),
    #             )
    #         ],
    #         capacity=[1],
    #         limits=VehicleLimits(
    #             shiftTime=(end_time - start_time).seconds, maxDistance=MAX_TOURING_DISTANCE
    #         ),
    #     )

    #     fleet = Fleet(vehicles=[walker], profiles=[VehicleProfile(name="person")])

    #     optim_objectives = [
    #         [Objective(type="minimize-unassigned")],
    #         [Objective(type="minimize-tours")],
    #         [Objective(type="minimize-cost")]]

    #     problem = Problem(
    #         plan=Plan(jobs=jobs), fleet=fleet, objectives=optim_objectives
    #     )

    #     json.dump(
    #         problem,
    #         open(self.problem_path, "w"),
    #         indent=2,
    #     )
        pass

    def create_completed_group_canvas(self):
        """
        Construct the problem file for a group canvas to be processed by the VRP solver.
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
                if pt_id(pt) == "depot" or pt_id(other_pt) == "depot":
                    distance = MixDistances.get_distance(pt, other_pt)
                else:
                    distance = HouseDistances.get_distance(pt, other_pt)

                if type(distance) is tuple:
                    raise ValueError(
                        "When retrieving distances, both points must be houses"
                    )

                distance = (
                    distance
                    if distance is not None
                    else MAX_TOURING_DISTANCE
                )
                time = distance / WALKING_M_PER_S
                distance_matrix["travelTimes"].append(round(time))
                distance_matrix["distances"].append(round(distance))

        json.dump(distance_matrix, open(self.distances_path, "w"), indent=2)
        print("Saved distance matrix to {}".format(self.distances_path))

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            if i != self.start_idx:  # The starting location is not a real service
                service = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=round(MINS_PER_HOUSE * 60),
                            times=[full_time_window],
                        )
                    ]
                )
                jobs.append(Job(id=pt_id(house), services=[service]))

        optim_cost = Costs(fixed=1, distance=0, time=1)

        # About 60 houses per list
        # TODO: Figure out how to use the actual minimize_tours to decide this.
        num_vehicles = len(self.points) // 60 + 1

        fleet = self.build_fleet(
            shift_time=(end_time - start_time),
            num_vehicles=num_vehicles,
            time_window=full_time_window,
            costs=optim_cost,
        )
        optim_objectives = [
            [Objective(type="minimize-unassigned")],
            [Objective(type="minimize-tours")],
            [Objective(type="minimize-cost")]]

        problem = Problem(
            plan=Plan(jobs=jobs), fleet=fleet, objectives=optim_objectives
        )

        json.dump(
            problem,
            open(self.problem_path, "w"),
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

        json.dump(distance_matrix, open(self.distances_path, "w"), indent=2)
        print("Saved distance matrix to {}".format(self.distances_path))

        # Create the plan
        jobs: list[Job] = []
        for i, house in enumerate(self.points):
            if i != self.start_idx:  # The starting location is not a real service
                service = Service(
                    places=[
                        PlaceTW(
                            location=Location(index=i),
                            duration=round(MINS_PER_HOUSE * 60),
                            times=[full_time_window],
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
            open(self.problem_path, "w"),
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
                self.problem_path,
                "-m",
                self.distances_path,
                "-o",
                self.solution_path,
                "-t",
                str(TIMEOUT.seconds),
                "--min-cv",
                "sample,200,0.001,true",
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

    @classmethod
    def process_solution(cls, solution_file) -> Solution:
        solution_dict = json.load(open(solution_file))
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

        if "unassigned" not in solution_dict:
            solution_dict["unassigned"] = []
        solution = Solution(
            statistic=Statistic(**solution_dict["statistic"]),
            tours=tours,
            unassigned=solution_dict["unassigned"],
        )

        return solution
