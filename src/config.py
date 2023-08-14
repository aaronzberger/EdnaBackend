import os
from datetime import timedelta
from typing import Any, Literal, TypedDict
from dataclasses import dataclass

"----------------------------------------------------------------------------------"
"                                     File Paths                                   "
"----------------------------------------------------------------------------------"

BASE_DIR = "/home/user/WLC"
VRP_CLI_PATH = "/home/user/.cargo/bin/vrp-cli"

AREA_ID = "rosselli"
USE_COST_METRIC = False

street_suffixes_file = os.path.join(BASE_DIR, "src", "street_suffixes.json")

node_distance_table_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "store", "node_distances.json"
)
house_distance_table_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "store", "house_distances.json"
)
block_distance_matrix_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "store", "segment_distance_matrix.json"
)
node_coords_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "store", "node_coords.json"
)
block_output_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "input", "block_output.json"
)
adjacency_list_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "input", "adjacency_list.json"
)
coords_node_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "input", "coords_node.json"
)
overpass_file = os.path.join(BASE_DIR, "regions", AREA_ID, "input", "overpass.json")

# Map addresses to block IDs
houses_file = os.path.join(BASE_DIR, "regions", AREA_ID, "houses.json")
blocks_file = os.path.join(BASE_DIR, "regions", AREA_ID, "blocks.json")

address_pts_file = os.path.join(BASE_DIR, "input", "address_pts.csv")
problem_path = os.path.join(BASE_DIR, "optimize", "problem.json")
solution_path = os.path.join(BASE_DIR, "optimize", "solution.json")

"----------------------------------------------------------------------------------"
"                                  Optimization                                    "
"----------------------------------------------------------------------------------"
SEARCH_MODE_DEEP = False
TIMEOUT = timedelta(seconds=100)


"----------------------------------------------------------------------------------"
"                                     Constants                                    "
"----------------------------------------------------------------------------------"

TURF_SPLIT = False  # Which problem to run

# Maximum distance between two nodes where they should be stored
ARBITRARY_LARGE_DISTANCE = 10000
MAX_TOURING_TIME = timedelta(minutes=180)
MAX_TOURING_DISTANCE = 10000
WALKING_M_PER_S = 0.85
MINS_PER_HOUSE = 1.65
CLUSTERING_CONNECTED_THRESHOLD = 100  # Meters where blocks are connected
KEEP_APARTMENTS = False
DISTANCE_TO_ROAD_MULTIPLIER = 0.5
ALD_BUFFER = 150  # Meters after a block ends where a house is still on the block
DIFFERENT_BLOCK_COST = 25


# Cost of crossing the street (technically, in meters)
DIFFERENT_SIDE_COST = {
    "motorway": 400,
    "trunk": 400,
    "primary": 100,
    "secondary": 60,
    "tertiary": 35,
    "unclassified": 20,
    "residential": 20,
    "service": 10,
}

# Cost of crossing the street (technically, in meters)
ROAD_WIDTH = {
    "motorway": 20,
    "trunk": 16,
    "primary": 12,
    "secondary": 10,
    "tertiary": 8,
    "unclassified": 8,
    "residential": 6,
    "service": 0,
}
"----------------------------------------------------------------------------------"
"                                       Type Hints                                 "
"----------------------------------------------------------------------------------"


# NOTE: the 'type' and 'id' attributes are not required. When using Python 3.11,
# wrap these attributes in the 'typing.NotRequired' hint to eliminate errors on instance creation.
# On earlier versions, either suppress or ignore these errors: they do not affect json export or reading.


class Point(TypedDict):
    lat: float
    lon: float
    type: Literal["house", "node", "other"]
    id: str


def pt_id(p: Point) -> str:
    """
    Get the ID of a point

    Parameters:
        p (Point): the point

    Returns:
        str: the ID, if it was provided upon creation. Otherwise, an ID made up of the rounded coordinates
    """
    return (
        str("{:.7f}".format(p["lat"])) + ":" + str("{:.7f}".format(p["lon"]))
        if "id" not in p or p["id"] is None
        else p["id"]
    )


house_t = dict[str, Point]
node_list_t = list[Point]

# DEPOT = Point(lat=40.4471477, lon=-79.9311578, type='node')  # Kipling and Dunmoyle
# DEPOT = Point(lat=40.4310603, lon=-79.9191268, type='node')  # Shady and Nicholson
# DEPOT = Point(lat=40.4430899, lon=-79.9329246, type='node')  # Maynard and Bennington
DEPOT = Point(lat=40.4362340, lon=-79.9191103, type="node")
NUM_LISTS = 1


class HouseInfo(TypedDict):
    lat: float
    lon: float
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int
    subsegment: tuple[int, int]


class Block(TypedDict):
    addresses: dict[str, HouseInfo]
    nodes: node_list_t
    bearings: tuple[float, float]  # The bearings at the start and end of the block
    type: str


blocks_file_t = dict[str, Block]
houses_file_t = dict[str, str]


"----------------------------------------------------------------------------------"
"                               Output File Type Hints                             "
"----------------------------------------------------------------------------------"


class Person(TypedDict):
    name: str
    age: int


class HousePeople(TypedDict):
    address: str
    coordinates: Point
    voter_info: list[Person]
    subsegment_start: int


"----------------------------------------------------------------------------------"
"                                 Solution Type Hints                              "
"----------------------------------------------------------------------------------"


class Statistic(TypedDict):
    cost: float
    distance: float
    duration: float
    times: dict[str, int]


class Location(TypedDict):
    index: int


class Time(TypedDict):
    arrival: str
    departure: str


class Stop(TypedDict):
    location: Location
    time: Time
    distance: int
    load: list[int]
    activities: list[dict[str, str]]


class Tour(TypedDict):
    vehicleId: str
    typeId: str
    shiftIndex: int
    stops: list[Stop]


class Solution(TypedDict):
    statistic: Statistic
    tours: list[Tour]
    unassigned: list[Any]


"----------------------------------------------------------------------------------"
"                                  Problem Type Hints                              "
"----------------------------------------------------------------------------------"


class Profile(TypedDict):
    matrix: str


class Costs(TypedDict):
    fixed: int
    distance: int
    time: int


class ShiftStart(TypedDict):
    earliest: str
    location: Location


class ShiftEnd(TypedDict):
    latest: str
    location: Location


class Shift(TypedDict):
    start: ShiftStart
    end: ShiftEnd


class VehicleLimits(TypedDict):
    shiftTime: int
    maxDistance: int


class Vehicle(TypedDict):
    typeId: str
    vehicleIds: list[str]
    profile: Profile
    costs: Costs
    shifts: list[Shift]
    capacity: list[int]
    limits: VehicleLimits


class VehicleProfile(TypedDict):
    name: str


class Fleet(TypedDict):
    vehicles: list[Vehicle]
    profiles: list[VehicleProfile]


class PlaceTW(TypedDict):
    location: Location
    duration: int
    times: list[list[str]]


class Place(TypedDict):
    location: Location
    duration: int


class Service(TypedDict):
    places: list[PlaceTW | Place]


class Job(TypedDict):
    id: str
    services: list[Service]
    value: int


class Plan(TypedDict):
    jobs: list[Job]


class Objective(TypedDict):
    type: str


class Problem(TypedDict):
    plan: Plan
    fleet: Fleet
    objectives: list[list[Objective]]


class DistanceMatrix(TypedDict):
    profile: str
    travelTimes: list[int]
    distances: list[int]


"----------------------------------------------------------------------------------"
"                             Optimization Parameters                              "
"----------------------------------------------------------------------------------"
OPTIM_COSTS = Costs(fixed=0, distance=3, time=1)

OPTIM_OBJECTIVES = [
    [Objective(type="maximize-value")],
    [Objective(type="minimize-cost")],
]
