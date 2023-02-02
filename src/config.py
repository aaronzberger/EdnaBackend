import os
from datetime import timedelta
from typing import Any, Literal, TypedDict

'----------------------------------------------------------------------------------'
'                                     File Paths                                   '
'----------------------------------------------------------------------------------'

# BASE_DIR = '/Users/aaron/Documents/GitHub/WLC'
BASE_DIR = '/home/user/WLC'
# VRP_CLI_PATH = "/Users/aaron/.cargo/bin/vrp-cli"
VRP_CLI_PATH = "/home/user/.cargo/bin/vrp-cli"

node_distance_table_file = os.path.join(BASE_DIR, 'store', 'node_distances.json')
block_distance_matrix_file = os.path.join(BASE_DIR, 'store', 'segment_distance_matrix.json')
node_coords_file = os.path.join(BASE_DIR, 'store', 'node_coords.json')
address_pts_file = os.path.join(BASE_DIR, 'input', 'address_pts.csv')
block_output_file = os.path.join(BASE_DIR, 'input', 'block_output.json')
blocks_file = os.path.join(BASE_DIR, 'blocks.json')
associated_file = os.path.join(BASE_DIR, 'associated.csv')

# Map addresses to block IDs
houses_file = os.path.join(BASE_DIR, 'houses.json')

problem_path = os.path.join(BASE_DIR, 'optimize', 'problem.json')
solution_path = os.path.join(BASE_DIR, 'optimize', 'solution.json')

'----------------------------------------------------------------------------------'
'                                     Constants                                    '
'----------------------------------------------------------------------------------'

TURF_SPLIT = False  # Which problem to run

# Maximum distance between two nodes where they should be stored
ARBITRARY_LARGE_DISTANCE = 10000
MAX_TOURING_TIME = timedelta(minutes=180)
MAX_TOURING_DISTANCE = 10000
WALKING_M_PER_S = 0.75
MINS_PER_HOUSE = 1.5
CLUSTERING_CONNECTED_THRESHOLD = 100  # Meters where blocks are connected
KEEP_APARTMENTS = False
DISTANCE_TO_ROAD_MULTIPLIER = 0.5
ALD_BUFFER = 5   # Meters after a block ends where a house is still on the block


# Cost of crossing the street (technically, in meters)
DIFFERENT_SIDE_COST = {
    'motorway': 400,
    'trunk': 400,
    'primary': 100,
    'secondary': 60,
    'tertiary': 35,
    'unclassified': 20,
    'residential': 20,
    'service': 10
}
'----------------------------------------------------------------------------------'
'                                       Type Hints                                 '
'----------------------------------------------------------------------------------'


# NOTE: the 'type' and 'id' attributes are not required. When using Python 3.11,
# wrap these attributes in the 'typing.NotRequired' hint to eliminate errors on instance creation.
# On earlier versions, either suppress or ignore these errors: they do not affect json export or reading.

class Point(TypedDict):
    lat: float
    lon: float
    type: Literal['house', 'node', 'other']
    id: str


def pt_id(p: Point) -> str:
    '''
    Get the ID of a point

    Parameters:
        p (Point): the point

    Returns:
        str: the ID, if it was provided upon creation. Otherwise, an ID made up of the rounded coordinates
    '''
    return str('{:.7f}'.format(p['lat'])) + ':' + str('{:.7f}'.format(p['lon'])) \
        if 'id' not in p or p['id'] is None else p['id']


house_t = dict[str, Point]
node_list_t = list[Point]


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


'----------------------------------------------------------------------------------'
'                             Optimization Parameters                              '
'----------------------------------------------------------------------------------'
OPTIM_COSTS = Costs(fixed=0, distance=3, time=1)

OPTIM_OBJECTIVES = [[Objective(type='maximize-value')],
                    [Objective(type='minimize-cost')]]
# [Objective(type='minimize-tours')][Objective(type='minimize-cost')]