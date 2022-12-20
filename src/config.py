from datetime import timedelta
import os
from typing import Any, TypedDict

'----------------------------------------------------------------------------------'
'                                     File Paths                                   '
'----------------------------------------------------------------------------------'

BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'
VRP_CLI_PATH = "/Users/aaron/.cargo/bin/vrp-cli"

node_distance_table_file = os.path.join(BASE_DIR, 'store', 'node_distances.json')
segment_distance_matrix_file = os.path.join(BASE_DIR, 'store', 'segment_distance_matrix.json')
node_coords_file = os.path.join(BASE_DIR, 'store', 'node_coords.json')
address_pts_file = os.path.join(BASE_DIR, 'input', 'address_pts.csv')
block_output_file = os.path.join(BASE_DIR, 'input', 'block_output.json')
blocks_file = os.path.join(BASE_DIR, 'blocks.json')
associated_file = os.path.join(BASE_DIR, 'associated.csv')
houses_file = os.path.join(BASE_DIR, 'houses.json')
problem_path = os.path.join(BASE_DIR, 'optimize', 'problem.json')
solution_path = os.path.join(BASE_DIR, 'optimize', 'solution.json')

'----------------------------------------------------------------------------------'
'                                     Constants                                    '
'----------------------------------------------------------------------------------'

# Maximum distance between two nodes where they should be stored
ARBITRARY_LARGE_DISTANCE = 10000
MAX_ROUTE_TIME = timedelta(minutes=180)
MAX_ROUTE_DISTANCE = 10000
WALKING_M_PER_S = 0.75
MINS_PER_HOUSE = 1.5
CLUSTERING_CONNECTED_THRESHOLD = 100  # Meters where blocks are connected
KEEP_APARTMENTS = False
DIFFERENT_SIDE_TIME_DIVISION = 3
DIFFERENT_SIDE_COST = {
    'motorway': 400,
    'trunk': 400,
    'primary': 100,
    'secondary': 60,
    'tertiary': 25,
    'unclassified': 15,
    'residential': 15,
    'service': 5
}
'----------------------------------------------------------------------------------'
'                                       Type Hints                                 '
'----------------------------------------------------------------------------------'


class Node(TypedDict):
    lat: float
    lon: float


house_t = dict[str, Node]
node_list_t = list[Node]


class HouseAssociationDict(TypedDict):
    lat: float
    lon: float
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int
    subsegment: tuple[int, int]


class SegmentDict(TypedDict):
    addresses: dict[str, HouseAssociationDict]
    nodes: node_list_t
    type: str


blocks_file_t = dict[str, SegmentDict]
houses_file_t = dict[str, str]


'----------------------------------------------------------------------------------'
'                                 Solution Type Hints                              '
'----------------------------------------------------------------------------------'


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


'----------------------------------------------------------------------------------'
'                                  Problem Type Hints                              '
'----------------------------------------------------------------------------------'


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
