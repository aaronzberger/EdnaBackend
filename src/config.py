import math
import os
import uuid
from datetime import timedelta
from enum import Enum
from typing import Any, Literal, TypedDict

"----------------------------------------------------------------------------------"
"                                     File Paths                                   "
"----------------------------------------------------------------------------------"


BASE_DIR = os.path.abspath(os.path.join(__file__, "../../"))
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
addresses_file = os.path.join(BASE_DIR, "regions", AREA_ID, "addresses.json")

manual_match_input_file = os.path.join(BASE_DIR, "regions", AREA_ID, "manual_match_input.json")

manual_match_output_file = os.path.join(BASE_DIR, "regions", AREA_ID, "input", "manual_match_output.json")

reverse_geocode_file = os.path.join(BASE_DIR, "regions", AREA_ID, "reverse_geocode.json")

house_id_to_block_id_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "house_id_to_block_id.json"
)

id_to_addresses_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "id_to_addresses.json"
)

universe_association = os.path.join(
    BASE_DIR, "regions", AREA_ID, "universe_association.json"
)

requested_blocks_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "requested_blocks.json"
)

house_to_voters_file = os.path.join(
    BASE_DIR, "regions", AREA_ID, "house_to_voters.json"
)

turnout_predictions_file = os.path.join(
    BASE_DIR, "input", "2023_general_predictions.json"
)

blocks_file = os.path.join(BASE_DIR, "regions", AREA_ID, "blocks.json")

address_pts_file = os.path.join(BASE_DIR, "input", "address_pts.csv")
problem_path = os.path.join(BASE_DIR, "optimize", "problem.json")
solution_path = os.path.join(BASE_DIR, "optimize", "solution.json")

"----------------------------------------------------------------------------------"
"                               Problem Parameters                                 "
"----------------------------------------------------------------------------------"


def sigmoid(x: float, k: float, a: float) -> float:
    """Sigmoid function with steepness parameter k and shift parameter a.

    This should be used on voter probabilities when the metric needs to be exaggerated,
    so low probabilities will map even lower and high probabilities even higher.
    """
    return 1 / (1 + math.exp(-k * (x - a)))


def normalized_fn(fn):
    min_value, max_value = fn(0, 1, 0), fn(1, 1, 0)

    def normalized(x: float, k: float, a: float) -> float:
        return (fn(x, k, a) - min_value) / (max_value - min_value)

    return normalized


def exponential(x: float, k: float) -> float:
    """Exponential function with steepness parameter k, scaled to [0, 1].

    This should be used on voter probabilities when the metric needs to be diminished,
    so low probabilities will map higher.
    """
    return (1 - math.exp(-k * x)) / (1 - math.exp(-k))


def voter_value(party: Literal["D", "R", "I"], turnout: float) -> float:
    # We use the normalized sigmoid function to exaggerate the differences between turnout probabilities.
    # Here, more steepness (k) means more exaggeration, and the (a) value is at what probability the function
    # is ambivalent (lower means fewer low-propensity voters are targeted)
    base_value = normalized_fn(sigmoid)(turnout, 10, 0.4)
    return base_value if party in ["D", "I"] else 0


def house_value(voter_values: list[float]) -> float:
    """Calculate the value of a house based on the voters in it."""
    scaling_factor = 1 + (0.2 * len(voter_values))
    return sum(voter_values) / len(voter_values) * scaling_factor


"----------------------------------------------------------------------------------"
"                             Optimization Parameters                              "
"----------------------------------------------------------------------------------"
SEARCH_MODE_DEEP = False
TIMEOUT = timedelta(seconds=100)


"----------------------------------------------------------------------------------"
"                                     Constants                                    "
"----------------------------------------------------------------------------------"

GOOGLE_MAPS_API_KEY = "AIzaSyAPpRP4mPuMlyRP8YiIaEOL_YAms6TpCwM"

UUID_NAMESPACE = uuid.UUID("ccf207c6-3b15-11ee-be56-0242ac120002")

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

# Define the Nodetypes as an enum
NodeType = Enum("NodeType", ["house", "node", "other"])


class Point(TypedDict):
    lat: float
    lon: float
    type: NodeType
    id: str


def pt_id(p: Point) -> str:
    """
    Get the ID of a point.

    Parameters
    ----------
        p (Point): the point

    Returns
    -------
        str: the ID, if it was provided upon creation. Otherwise, an ID made up of the rounded coordinates
    """
    return (
        str("{:.7f}".format(p["lat"])) + ":" + str("{:.7f}".format(p["lon"]))
        if "id" not in p or p["id"] is None
        else p["id"]
    )


house_t = dict[str, Point]
node_list_t = list[Point]

# DEPOT = Point(lat=40.5397171, lon=-80.1763386, type="node", id=None)  # Sewickley
# DEPOT = Point(lat=40.4471477, lon=-79.9311578, type='node')  # Kipling and Dunmoyle
# DEPOT = Point(lat=40.4310603, lon=-79.9191268, type='node')  # Shady and Nicholson
# DEPOT = Point(lat=40.4430899, lon=-79.9329246, type='node')  # Maynard and Bennington
DEPOT = Point(
    lat=40.4362340, lon=-79.9191103, type=NodeType.node, id=""
)  # Forbes and Shady
NUM_LISTS = 1


class HouseInfo(TypedDict):
    display_address: str  # VERY IMPORTANT!!! NEVER USE THIS TO INDEX INTO HOUSES, USE HOUSE UUID INSTEAD
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
    type: str


blocks_file_t = dict[str, Block]


"----------------------------------------------------------------------------------"
"                               Output File Type Hints                             "
"----------------------------------------------------------------------------------"

tracked_elections = Enum("elections", [
    "primary_2023",
    "general_2022",
    "primary_2022",
    "general_2021",
    "primary_2021",
    "general_2020",
    "primary_2020",
    "general_2019",
    "primary_2019"
])


# Mapping from voter file column names to elections
# This is retreived directly from the "Election Map" file from the county
voter_file_mapping = {
    7: tracked_elections.primary_2019,
    8: tracked_elections.general_2019,
    9: tracked_elections.primary_2020,
    10: tracked_elections.general_2020,
    21: tracked_elections.primary_2021,
    22: tracked_elections.general_2021,
    25: tracked_elections.primary_2022,
    26: tracked_elections.general_2022,
    30: tracked_elections.primary_2023
}


class Person(TypedDict):
    name: str
    age: int
    party: Literal["D", "R", "I"]
    voting_history: dict[str, bool]
    voter_id: str
    value: float


class HousePeople(TypedDict):
    display_address: str
    latitude: float
    longitude: float
    voter_info: list[Person]
    value: float
    # NOTE: This previously had subsegment_start, but this will now only be in the post-processing output


voters_file_t = dict[str, HousePeople]


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
